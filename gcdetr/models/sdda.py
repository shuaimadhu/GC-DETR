"""
Spectral Decomposition-based Decoupled Deformable Attention (SDDA) for GC-DETR.

SDDA extends multi-scale deformable attention with two key innovations:

1. **Shape-conditioned sampling** – the deformable sampling offsets are
   modulated by the predicted object shape (width and height) so that the
   receptive field of each attention head automatically adapts to the
   geometry of the object being detected.  This is particularly important
   for UAV imagery where objects span a wide range of scales.

2. **Spectral decomposition-based geometric decoupling** – the attention
   weights are computed in a frequency (spectral) domain, separating the
   horizontal and vertical spatial components via learnable 1-D DCT-style
   bases.  This allows different heads to specialise in orthogonal spatial
   frequencies, diversifying the multi-head representation and improving
   feature alignment against complex, cluttered backgrounds.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List


class SpectralDecompositionDeformableAttention(nn.Module):
    """Spectral Decomposition-based Decoupled Deformable Attention (SDDA).

    Args:
        embed_dim: Total embedding dimension.  Must be divisible by
            ``num_heads``.
        num_heads: Number of attention heads.
        num_points: Number of sampling points per head per feature level.
        num_levels: Number of feature-map scales.
        dropout: Dropout probability applied to output projection.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        num_levels: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        self.head_dim = embed_dim // num_heads

        # ------------------------------------------------------------------
        # Projections
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # ------------------------------------------------------------------
        # Shape-conditioned sampling offset prediction.
        # Input to the offset MLP: query (embed_dim) + shape (2: w, h).
        # Output: (num_heads * num_levels * num_points * 2) offsets.
        # ------------------------------------------------------------------
        self.offset_mlp = nn.Sequential(
            nn.Linear(embed_dim + 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_heads * num_levels * num_points * 2),
        )

        # ------------------------------------------------------------------
        # Attention weight prediction with spectral decoupling.
        # We decompose the attention logits along x and y independently
        # using separate linear projections (one per spatial axis), then
        # combine them via an outer product.  This ensures that each head
        # focuses on orthogonal spatial frequency bands.
        # ------------------------------------------------------------------
        self.attn_x = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.attn_y = nn.Linear(embed_dim, num_heads * num_levels * num_points)

        # Learnable spectral basis vectors (DCT-style) per head.
        # Shape: (num_heads, head_dim) – project head features into the
        # 1-D spectral domain.
        self.spectral_basis_x = nn.Parameter(
            torch.empty(num_heads, self.head_dim)
        )
        self.spectral_basis_y = nn.Parameter(
            torch.empty(num_heads, self.head_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Offset MLP: initialise close to zero for stable early training.
        for layer in self.offset_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.zeros_(self.attn_x.weight)
        nn.init.zeros_(self.attn_x.bias)
        nn.init.zeros_(self.attn_y.weight)
        nn.init.zeros_(self.attn_y.bias)

        # Spectral bases – initialise with alternating cosine values so
        # each head starts with a different frequency component.
        with torch.no_grad():
            for h in range(self.num_heads):
                freq = math.pi * (h + 1) / self.num_heads
                freqs = torch.arange(self.head_dim, dtype=torch.float32) * freq
                self.spectral_basis_x[h] = torch.cos(freqs)
                self.spectral_basis_y[h] = torch.sin(freqs)  # orthogonal to x

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bilinear_sample(
        self,
        value: Tensor,
        sampling_locations: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        """Sample values at deformable locations using bilinear interpolation.

        Args:
            value: Flattened multi-scale value tensor of shape
                (B, S_total, num_heads, head_dim) where S_total is the sum of
                H*W across all feature levels.
            sampling_locations: Normalized sampling coordinates in [0, 1]
                with shape (B, Q, num_heads, num_levels, num_points, 2).
            spatial_shapes: (num_levels, 2) tensor of (H, W) per level.
            level_start_index: (num_levels,) start index in the flattened
                sequence for each level.

        Returns:
            Sampled features with shape (B, Q, num_heads, num_levels,
            num_points, head_dim).
        """
        B, Q, H, L, P, _ = sampling_locations.shape
        _, _, _, D = value.shape

        output = torch.zeros(B, Q, H, L, P, D, device=value.device, dtype=value.dtype)

        for lvl in range(L):
            h_lvl, w_lvl = spatial_shapes[lvl, 0].item(), spatial_shapes[lvl, 1].item()
            start = level_start_index[lvl].item()
            end = start + int(h_lvl) * int(w_lvl)

            # value for this level: (B, H*W, num_heads, head_dim)
            v_lvl = value[:, start:end]  # (B, Hl*Wl, num_heads, D)
            v_lvl = v_lvl.reshape(B, int(h_lvl), int(w_lvl), H, D)
            # Permute to (B, num_heads, D, Hl, Wl) for grid_sample
            v_lvl = v_lvl.permute(0, 3, 4, 1, 2)  # (B, num_heads, D, Hl, Wl)
            v_lvl = v_lvl.reshape(B * H, D, int(h_lvl), int(w_lvl))

            # Sampling locations for this level: (B, Q, num_heads, P, 2)
            loc = sampling_locations[:, :, :, lvl, :, :]  # (B, Q, H, P, 2)
            # Reshape to (B*num_heads, Q*P, 1, 2) for grid_sample
            loc = loc.permute(0, 2, 1, 3, 4)  # (B, H, Q, P, 2)
            loc = loc.reshape(B * H, Q * P, 1, 2)
            # grid_sample expects coordinates in [-1, 1]
            loc_norm = loc * 2.0 - 1.0

            sampled = F.grid_sample(
                v_lvl.float(),
                loc_norm.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # (B*H, D, Q*P, 1)
            sampled = sampled.squeeze(-1)  # (B*H, D, Q*P)
            sampled = sampled.reshape(B, H, D, Q, P).permute(0, 3, 1, 4, 2)
            output[:, :, :, lvl, :, :] = sampled

        return output

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        object_shape: Optional[Tensor] = None,
        input_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply SDDA to decoder queries over multi-scale feature maps.

        Args:
            query: Decoder queries with shape (B, Q, C).
            reference_points: Normalized 2-D reference points (cx, cy) for
                each query and scale level, shape (B, Q, num_levels, 2),
                with coordinates in [0, 1].
            input_flatten: Flattened multi-scale feature maps with shape
                (B, S_total, C).
            input_spatial_shapes: (num_levels, 2) – height and width of each
                feature level.
            input_level_start_index: (num_levels,) – flat index of the first
                spatial position for each level.
            object_shape: Predicted object sizes (w, h) in [0, 1] with shape
                (B, Q, 2).  Used to scale the sampling offsets so that the
                effective receptive field of each head matches the object
                geometry.  When *None* a unit shape is assumed (no scaling).
            input_padding_mask: Boolean mask with shape (B, S_total) where
                *True* marks padded positions.  Currently unused; reserved
                for future masking support.

        Returns:
            Context features with the same shape as ``query``: (B, Q, C).
        """
        B, Q, C = query.shape
        S = input_flatten.shape[1]

        # ------------------------------------------------------------------
        # 1. Query and value projections.
        # ------------------------------------------------------------------
        q = self.q_proj(query)  # (B, Q, C)
        v = self.v_proj(input_flatten)  # (B, S, C)

        v = v.reshape(B, S, self.num_heads, self.head_dim)  # (B, S, H, D)

        # ------------------------------------------------------------------
        # 2. Shape-conditioned sampling offsets.
        # ------------------------------------------------------------------
        if object_shape is None:
            object_shape = query.new_ones(B, Q, 2) * 0.1

        # Concatenate query features with object shape.
        q_shape = torch.cat([q, object_shape], dim=-1)  # (B, Q, C+2)
        offsets = self.offset_mlp(q_shape)  # (B, Q, H*L*P*2)
        offsets = offsets.reshape(
            B, Q, self.num_heads, self.num_levels, self.num_points, 2
        )
        # Scale offsets by object shape so larger objects get larger receptive
        # fields.  object_shape is (B, Q, 2): interpret as (w, h) scaling.
        shape_scale = object_shape[:, :, None, None, None, :]  # (B, Q, 1, 1, 1, 2)
        offsets = offsets * shape_scale

        # Absolute sampling locations (clamped to [0, 1]).
        # reference_points: (B, Q, L, 2) -> (B, Q, 1, L, 1, 2) for broadcasting
        # with offsets: (B, Q, H, L, P, 2).
        ref = reference_points[:, :, None, :, None, :]  # (B, Q, 1, L, 1, 2)
        sampling_locs = (ref + offsets).clamp(0.0, 1.0)  # (B, Q, H, L, P, 2)

        # ------------------------------------------------------------------
        # 3. Spectral decomposition-based decoupled attention weights.
        # ------------------------------------------------------------------
        # Project queries along x and y spectral bases independently.
        # q: (B, Q, C) -> reshape to (B, Q, H, D)
        q_heads = q.reshape(B, Q, self.num_heads, self.head_dim)

        # Spectral projection: (B, Q, H, D) x (H, D) -> (B, Q, H)
        # One scalar per head reflecting the spectral energy along each axis.
        spec_x = (q_heads * self.spectral_basis_x).sum(-1)  # (B, Q, H)
        spec_y = (q_heads * self.spectral_basis_y).sum(-1)  # (B, Q, H)
        # Outer combination: amplifies attention where both x and y components
        # are active (i.e., at the spatial frequency matching the head's basis).
        spectral_weight = (spec_x * spec_y).unsqueeze(-1).unsqueeze(-1)  # (B, Q, H, 1, 1)

        # Standard deformable attention weights, split by x / y and combined.
        attn_x = self.attn_x(q)  # (B, Q, H*L*P)
        attn_y = self.attn_y(q)  # (B, Q, H*L*P)
        attn_x = attn_x.reshape(B, Q, self.num_heads, self.num_levels, self.num_points)
        attn_y = attn_y.reshape(B, Q, self.num_heads, self.num_levels, self.num_points)
        # Decouple: attention is the element-wise product of x and y components.
        attn = attn_x * attn_y  # (B, Q, H, L, P)
        # Modulate with spectral energy.
        attn = attn + spectral_weight
        # Softmax over all levels and points jointly.
        attn = attn.reshape(B, Q, self.num_heads, self.num_levels * self.num_points)
        attn = F.softmax(attn, dim=-1)
        attn = attn.reshape(B, Q, self.num_heads, self.num_levels, self.num_points)

        # ------------------------------------------------------------------
        # 4. Weighted aggregation of sampled values.
        # ------------------------------------------------------------------
        sampled = self._bilinear_sample(
            v, sampling_locs, input_spatial_shapes, input_level_start_index
        )  # (B, Q, H, L, P, D)

        # Weighted sum over levels and points.
        # attn: (B, Q, H, L, P) -> unsqueeze D: (B, Q, H, L, P, 1)
        output = (sampled * attn.unsqueeze(-1)).sum(dim=[3, 4])  # (B, Q, H, D)
        output = output.reshape(B, Q, C)

        output = self.dropout(self.out_proj(output))
        return output
