"""
Channel-projection neck for GC-DETR.

Projects the multi-scale features from the backbone to a common channel
dimension (``embed_dim``), adds 2-D sinusoidal positional encodings, and
flattens the spatial dimensions for use by the transformer encoder and SDDA.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionEmbedding2D(nn.Module):
    """2-D sinusoidal positional encoding.

    Generates position embeddings for feature maps following the scheme in
    *Attention Is All You Need* (Vaswani et al., 2017) extended to two
    spatial dimensions.

    Args:
        embed_dim: Total embedding dimension (must be even).
        temperature: Frequency temperature factor (default 10 000).
        normalize: Whether to normalize position coordinates to [0, 2π]
            before computing sinusoids.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        temperature: float = 10_000.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for 2-D positional encoding"
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, mask: Tensor) -> Tensor:
        """Compute positional encoding from a padding mask.

        Args:
            mask: Boolean padding mask of shape (B, H, W) where *True*
                indicates a padded (invalid) position.

        Returns:
            Positional encoding tensor of shape (B, embed_dim, H, W).
        """
        not_mask = ~mask  # valid positions
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        half_dim = self.embed_dim // 2
        dim_t = torch.arange(half_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half_dim)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, D/2)
        pos_y = y_embed[:, :, :, None] / dim_t  # (B, H, W, D/2)

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1)
        pos_x = pos_x.flatten(-2)  # (B, H, W, D/2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1)
        pos_y = pos_y.flatten(-2)  # (B, H, W, D/2)

        pos = torch.cat([pos_y, pos_x], dim=-1)  # (B, H, W, D)
        pos = pos.permute(0, 3, 1, 2)  # (B, D, H, W)
        return pos


class ChannelProjectionNeck(nn.Module):
    """Multi-scale neck that projects backbone features to ``embed_dim``.

    Each backbone feature level is reduced to ``embed_dim`` channels via a
    1×1 convolution.  Optionally, sinusoidal positional encodings are added
    before flattening to produce the flat token sequence expected by the
    transformer.

    Args:
        in_channels: List of channel sizes for each input feature level
            (typically from the backbone's ``out_channels``).
        embed_dim: Common embedding dimension for the transformer.
        add_pos_encoding: If *True*, sinusoidal position encodings are added
            to each level's feature map before flattening.
    """

    def __init__(
        self,
        in_channels: List[int],
        embed_dim: int = 256,
        add_pos_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.add_pos_encoding = add_pos_encoding

        self.input_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels]
        )
        self.input_norm = nn.ModuleList(
            [nn.GroupNorm(32, embed_dim) for _ in in_channels]
        )

        if add_pos_encoding:
            self.pos_enc = SinusoidalPositionEmbedding2D(embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        features: Dict[str, Tensor],
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project and flatten multi-scale features.

        Args:
            features: Ordered mapping from level name to feature tensor
                (B, C_i, H_i, W_i).
            masks: Optional mapping from level name to boolean padding mask
                (B, H_i, W_i).  If *None* and positional encodings are
                requested, zero-filled masks are created.

        Returns:
            A 4-tuple:

            - **flat_features** (Tensor): Concatenated flattened features of
              shape (B, S_total, embed_dim).
            - **flat_masks** (Tensor): Concatenated padding masks of shape
              (B, S_total) – *True* for padded positions.
            - **spatial_shapes** (Tensor): (num_levels, 2) tensor of (H, W)
              per level.
            - **level_start_index** (Tensor): (num_levels,) cumulative start
              index of each level in the flat sequence.
        """
        level_names = list(features.keys())
        flat_list: List[Tensor] = []
        mask_list: List[Tensor] = []
        shapes: List[Tuple[int, int]] = []

        for i, name in enumerate(level_names):
            feat = features[name]  # (B, C_i, H, W)
            B, _, H, W = feat.shape

            # Project to embed_dim.
            feat = self.input_norm[i](self.input_proj[i](feat))

            # Build mask for this level.
            if masks is not None and name in masks:
                mask = masks[name]
            else:
                mask = feat.new_zeros(B, H, W, dtype=torch.bool)

            # Add positional encoding.
            if self.add_pos_encoding:
                pos = self.pos_enc(mask)  # (B, D, H, W)
                feat = feat + pos

            shapes.append((H, W))
            # Flatten spatial: (B, D, H, W) -> (B, H*W, D)
            flat_list.append(feat.flatten(2).permute(0, 2, 1))
            mask_list.append(mask.flatten(1))  # (B, H*W)

        flat_features = torch.cat(flat_list, dim=1)  # (B, S_total, D)
        flat_masks = torch.cat(mask_list, dim=1)  # (B, S_total)

        spatial_shapes = torch.tensor(shapes, dtype=torch.long, device=flat_features.device)
        level_start_index = torch.cat(
            [
                spatial_shapes.new_zeros(1),
                spatial_shapes.prod(dim=1).cumsum(dim=0)[:-1],
            ]
        )

        return flat_features, flat_masks, spatial_shapes, level_start_index
