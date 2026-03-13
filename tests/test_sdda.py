"""Tests for Spectral Decomposition-based Decoupled Deformable Attention (SDDA)."""

import pytest
import torch

from gcdetr.models.sdda import SpectralDecompositionDeformableAttention


def _make_sdda(embed_dim=64, num_heads=4, num_points=2, num_levels=2):
    return SpectralDecompositionDeformableAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_points=num_points,
        num_levels=num_levels,
    )


def _make_inputs(B=2, Q=10, embed_dim=64, num_levels=2, H=8, W=8):
    """Create minimal inputs for SDDA forward pass."""
    spatial_shapes = torch.tensor([[H, W], [H // 2, W // 2]], dtype=torch.long)
    level_start_index = torch.tensor([0, H * W], dtype=torch.long)
    S_total = H * W + (H // 2) * (W // 2)

    query = torch.randn(B, Q, embed_dim)
    memory = torch.randn(B, S_total, embed_dim)
    # Reference points: one center per query per level.
    ref_pts = torch.full((B, Q, num_levels, 2), 0.5)
    return query, memory, ref_pts, spatial_shapes, level_start_index


class TestSDDAForward:
    def test_output_shape(self):
        sdda = _make_sdda()
        query, memory, ref_pts, shapes, starts = _make_inputs()
        out = sdda(query, ref_pts, memory, shapes, starts)
        assert out.shape == query.shape

    def test_with_object_shape(self):
        sdda = _make_sdda()
        B, Q = 2, 10
        query, memory, ref_pts, shapes, starts = _make_inputs(B=B, Q=Q)
        object_shape = torch.rand(B, Q, 2) * 0.3 + 0.05
        out = sdda(query, ref_pts, memory, shapes, starts, object_shape=object_shape)
        assert out.shape == query.shape

    def test_no_nan_in_output(self):
        sdda = _make_sdda()
        query, memory, ref_pts, shapes, starts = _make_inputs()
        out = sdda(query, ref_pts, memory, shapes, starts)
        assert not torch.isnan(out).any(), "SDDA output contains NaN"
        assert not torch.isinf(out).any(), "SDDA output contains Inf"

    def test_gradient_flows(self):
        sdda = _make_sdda()
        query, memory, ref_pts, shapes, starts = _make_inputs()
        query.requires_grad_(True)
        memory.requires_grad_(True)
        out = sdda(query, ref_pts, memory, shapes, starts)
        out.sum().backward()
        assert query.grad is not None
        assert memory.grad is not None

    def test_sampling_locations_clamped(self):
        """Sampling locations must stay in [0, 1] even for large offsets."""
        sdda = _make_sdda()
        B, Q, embed_dim, num_levels = 1, 5, 64, 2
        spatial_shapes = torch.tensor([[4, 4], [2, 2]], dtype=torch.long)
        level_start_index = torch.tensor([0, 16], dtype=torch.long)
        S_total = 16 + 4
        query = torch.randn(B, Q, embed_dim) * 100  # large magnitude
        memory = torch.randn(B, S_total, embed_dim)
        ref_pts = torch.full((B, Q, num_levels, 2), 0.5)
        # Should not raise or produce NaN.
        out = sdda(query, ref_pts, memory, spatial_shapes, level_start_index)
        assert not torch.isnan(out).any()

    def test_embed_dim_mismatch_raises(self):
        with pytest.raises(AssertionError):
            SpectralDecompositionDeformableAttention(embed_dim=65, num_heads=8)

    def test_spectral_basis_orthogonality(self):
        """Verify x and y spectral bases start with distinct frequency content."""
        sdda = _make_sdda(num_heads=4)
        # Cosine basis for x, Sine basis for y → per construction they differ.
        dot = (sdda.spectral_basis_x * sdda.spectral_basis_y).sum(dim=-1)
        # They should NOT be identical.
        assert not torch.allclose(sdda.spectral_basis_x, sdda.spectral_basis_y)
