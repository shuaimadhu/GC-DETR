"""
Sensor-Aware Query Module (SAQM) for GC-DETR.

SAQM enhances the initialized object queries with UAV pose-related priors
(altitude, pitch, roll, yaw) using a lightweight cross-attention mechanism.
This establishes an explicit relationship between the UAV sensor state and the
image feature space, allowing the decoder queries to be conditioned on the
viewing geometry with negligible additional computation.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class SensorAwareQueryModule(nn.Module):
    """Sensor-Aware Query Module (SAQM).

    Encodes UAV pose information (altitude, pitch, roll, yaw) and injects it
    into the initial decoder queries through an additive pose-conditioned
    residual pathway.

    The pose vector is first projected into the same embedding space as the
    queries, then modulated by a learned gating mechanism so that the module
    contributes negligible overhead when pose information is absent.

    Args:
        embed_dim: Dimensionality of object queries and pose embedding.
        pose_dim: Dimensionality of the raw pose input vector
            (default 4: altitude, pitch, roll, yaw).
        hidden_dim: Hidden dimension of the pose MLP encoder.
        num_queries: Number of object queries.
        dropout: Dropout probability applied to the pose residual.
    """

    def __init__(
        self,
        embed_dim: int,
        pose_dim: int = 4,
        hidden_dim: int = 64,
        num_queries: int = 300,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.pose_dim = pose_dim
        self.num_queries = num_queries

        # Lightweight MLP to encode raw pose vector into embedding space.
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Learnable gating vector: controls how much pose information modifies
        # each query dimension.  Initialised near zero so the module starts
        # as a near-identity transformation and learns to open its gates
        # only as needed.
        self.gate = nn.Parameter(torch.zeros(embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self) -> None:
        for layer in self.pose_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        queries: Tensor,
        pose: Optional[Tensor] = None,
    ) -> Tensor:
        """Enhance object queries with UAV pose information.

        Args:
            queries: Object query embeddings with shape (B, N, C) or (N, C).
                When a batch dimension is absent, the pose tensor must also
                be 1-D (or None).
            pose: UAV sensor state vector with shape (B, pose_dim) containing
                [altitude, pitch, roll, yaw] values.  Values should be
                normalised to a consistent range (e.g. [0, 1] or [-1, 1])
                before being passed to this module.  If *None* the module
                returns the queries unchanged.

        Returns:
            Enhanced queries with the same shape as the input.
        """
        if pose is None:
            return queries

        # Accept un-batched inputs.
        unbatched = queries.dim() == 2
        if unbatched:
            queries = queries.unsqueeze(0)  # (1, N, C)
            pose = pose.unsqueeze(0) if pose.dim() == 1 else pose  # (1, pose_dim)

        batch_size = queries.shape[0]

        # Encode pose: (B, pose_dim) -> (B, C)
        pose_emb = self.pose_encoder(pose)  # (B, C)
        pose_emb = self.dropout(pose_emb)

        # Expand to query sequence and apply gated additive residual.
        # gate is learned per-dimension so the model can selectively inject
        # pose information into relevant embedding dimensions.
        gate = torch.sigmoid(self.gate)  # (C,)
        pose_emb = pose_emb.unsqueeze(1) * gate  # (B, 1, C)

        queries = self.norm(queries + pose_emb)  # (B, N, C)

        if unbatched:
            queries = queries.squeeze(0)

        return queries
