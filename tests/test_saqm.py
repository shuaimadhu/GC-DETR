"""Tests for the Sensor-Aware Query Module (SAQM)."""

import pytest
import torch
import torch.nn as nn

from gcdetr.models.saqm import SensorAwareQueryModule


@pytest.fixture
def saqm():
    return SensorAwareQueryModule(embed_dim=64, pose_dim=4, hidden_dim=32, num_queries=10)


class TestSAQMForward:
    def test_batched_with_pose(self, saqm):
        B, N, C = 2, 10, 64
        queries = torch.randn(B, N, C)
        pose = torch.rand(B, 4)
        out = saqm(queries, pose)
        assert out.shape == queries.shape, "Output shape must match input shape"
        assert not torch.allclose(out, queries), "SAQM should modify queries when pose is given"

    def test_no_pose_passthrough(self, saqm):
        B, N, C = 2, 10, 64
        queries = torch.randn(B, N, C)
        out = saqm(queries, pose=None)
        assert torch.allclose(out, queries), "SAQM must return queries unchanged when pose is None"

    def test_unbatched_input(self, saqm):
        N, C = 10, 64
        queries = torch.randn(N, C)
        pose = torch.rand(4)
        out = saqm(queries, pose)
        assert out.shape == (N, C), "Unbatched output should be 2-D"

    def test_output_dtype(self, saqm):
        queries = torch.randn(2, 10, 64)
        pose = torch.rand(2, 4)
        out = saqm(queries, pose)
        assert out.dtype == queries.dtype

    def test_gradient_flows(self, saqm):
        queries = torch.randn(2, 10, 64, requires_grad=True)
        pose = torch.rand(2, 4, requires_grad=True)
        out = saqm(queries, pose)
        loss = out.sum()
        loss.backward()
        assert queries.grad is not None
        assert pose.grad is not None

    def test_gate_initially_near_zero_effect(self):
        """Gate is initialised to 0; sigmoid(0)=0.5 so effect is half-strength."""
        saqm = SensorAwareQueryModule(embed_dim=64, pose_dim=4, hidden_dim=32, num_queries=10)
        gate_sigmoid = torch.sigmoid(saqm.gate)
        assert gate_sigmoid.shape == (64,)
        assert (gate_sigmoid > 0).all() and (gate_sigmoid < 1).all()

    def test_pose_dim_flexibility(self):
        """SAQM should accept arbitrary pose_dim values."""
        saqm6 = SensorAwareQueryModule(embed_dim=64, pose_dim=6, hidden_dim=32, num_queries=5)
        queries = torch.randn(2, 5, 64)
        pose = torch.rand(2, 6)
        out = saqm6(queries, pose)
        assert out.shape == (2, 5, 64)


class TestSAQMInit:
    def test_parameter_count(self, saqm):
        total = sum(p.numel() for p in saqm.parameters())
        assert total > 0

    def test_pose_encoder_bias_zero_init(self, saqm):
        for layer in saqm.pose_encoder:
            if isinstance(layer, nn.Linear):
                assert torch.all(layer.bias == 0).item()
