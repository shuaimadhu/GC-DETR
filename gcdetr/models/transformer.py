"""
Transformer encoder and decoder for GC-DETR.

The encoder processes the multi-scale feature tokens with standard
multi-head self-attention (plus feed-forward) layers to produce enriched
context features.

The decoder iteratively refines a set of object queries using:
  - Self-attention among the queries.
  - Cross-attention via :class:`~gcdetr.models.sdda.SpectralDecompositionDeformableAttention`
    (SDDA) against the encoded context features.
  - A feed-forward refinement step.
"""

from __future__ import annotations

import copy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gcdetr.models.sdda import SpectralDecompositionDeformableAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer (self-attention + FFN).

    Args:
        embed_dim: Model embedding dimension.
        num_heads: Number of self-attention heads.
        ffn_dim: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src2, _ = self.self_attn(
            src, src, src, key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        return src


class TransformerEncoder(nn.Module):
    """Stack of :class:`TransformerEncoderLayer`.

    Args:
        layer: A single encoder layer (will be deep-copied ``num_layers`` times).
        num_layers: Number of stacked encoder layers.
    """

    def __init__(self, layer: TransformerEncoderLayer, num_layers: int = 6) -> None:
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.norm = nn.LayerNorm(layer.self_attn.embed_dim)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """Single SDDA-based transformer decoder layer.

    Performs:
    1. Self-attention among object queries.
    2. SDDA cross-attention against encoder output.
    3. Feed-forward network.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of heads for self-attention (SDDA uses its own).
        ffn_dim: Hidden dimension of the feed-forward network.
        sdda_cfg: Keyword arguments forwarded to
            :class:`~gcdetr.models.sdda.SpectralDecompositionDeformableAttention`.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        sdda_cfg: Optional[dict] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        sdda_cfg = sdda_cfg or {}
        sdda_cfg.setdefault("embed_dim", embed_dim)
        sdda_cfg.setdefault("num_heads", num_heads)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # SDDA cross-attention
        self.cross_attn = SpectralDecompositionDeformableAttention(**sdda_cfg)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        object_shape: Optional[Tensor] = None,
        query_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of one decoder layer.

        Args:
            query: Object queries (B, N, C).
            memory: Encoder output / multi-scale features (B, S, C).
            reference_points: Normalized reference points (B, N, num_levels, 2).
            spatial_shapes: (num_levels, 2).
            level_start_index: (num_levels,).
            object_shape: Predicted object sizes (B, N, 2) for SDDA.
            query_key_padding_mask: Padding mask for queries.
            memory_key_padding_mask: Padding mask for encoder output.

        Returns:
            Refined query tensor (B, N, C).
        """
        # Self-attention
        q2, _ = self.self_attn(
            query, query, query, key_padding_mask=query_key_padding_mask
        )
        query = self.norm1(query + self.dropout(q2))

        # Cross-attention via SDDA
        q2 = self.cross_attn(
            query=query,
            reference_points=reference_points,
            input_flatten=memory,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            object_shape=object_shape,
            input_padding_mask=memory_key_padding_mask,
        )
        query = self.norm2(query + q2)

        # FFN
        q2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = self.norm3(query + self.dropout(q2))
        return query


class TransformerDecoder(nn.Module):
    """Stack of :class:`TransformerDecoderLayer`.

    Args:
        layer: A single decoder layer (will be deep-copied ``num_layers`` times).
        num_layers: Number of stacked decoder layers.
        return_intermediate: If *True* return predictions from every layer for
            auxiliary loss computation.
    """

    def __init__(
        self,
        layer: TransformerDecoderLayer,
        num_layers: int = 6,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        object_shape: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        """Run all decoder layers.

        Returns:
            If ``return_intermediate`` is *True*, a list of query tensors
            (one per layer); otherwise a single query tensor.
        """
        intermediate: List[Tensor] = []
        output = query

        for layer in self.layers:
            output = layer(
                query=output,
                memory=memory,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                object_shape=object_shape,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate
        return output
