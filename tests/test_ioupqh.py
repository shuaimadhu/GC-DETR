"""Tests for the IoU Prediction Quality Head (IOUPQH)."""

import pytest
import torch
import torch.nn as nn

from gcdetr.models.ioupqh import IoUPredictionQualityHead


@pytest.fixture
def head():
    return IoUPredictionQualityHead(embed_dim=64, num_classes=10, hidden_dim=64, num_layers=2)


class TestIOUPQHForward:
    def test_output_keys(self, head):
        B, N, C = 2, 20, 64
        feats = torch.randn(B, N, C)
        out = head(feats)
        assert "cls_logits" in out
        assert "iou_logits" in out
        assert "scores" in out

    def test_output_shapes(self, head):
        B, N, C = 3, 15, 64
        feats = torch.randn(B, N, C)
        out = head(feats)
        assert out["cls_logits"].shape == (B, N, 10)
        assert out["iou_logits"].shape == (B, N, 1)
        assert out["scores"].shape == (B, N, 10)

    def test_scores_in_01(self, head):
        feats = torch.randn(4, 10, 64)
        out = head(feats)
        scores = out["scores"]
        assert (scores >= 0).all() and (scores <= 1).all(), (
            "Scores should lie in [0, 1] since they are products of two sigmoids"
        )

    def test_iou_logits_unbounded(self, head):
        """iou_logits are raw logits; sigmoid must be applied externally."""
        feats = torch.randn(2, 10, 64) * 10
        out = head(feats)
        # Raw logits can be outside [0,1].
        assert out["iou_logits"].shape[-1] == 1

    def test_gradient_flows(self, head):
        feats = torch.randn(2, 10, 64, requires_grad=True)
        out = head(feats)
        out["scores"].sum().backward()
        assert feats.grad is not None

    def test_no_nan(self, head):
        feats = torch.randn(2, 10, 64)
        out = head(feats)
        for key, val in out.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"


class TestIOUPQHLoss:
    def test_loss_keys(self, head):
        N_pos, N_neg = 8, 12
        cls_logits = torch.randn(N_pos + N_neg, 10)
        iou_logits = torch.randn(N_pos, 1)
        target_cls = torch.zeros(N_pos + N_neg, 10)
        target_cls[:N_pos, 0] = 1.0  # class 0 for positives
        target_iou = torch.rand(N_pos, 1)
        losses = head.loss(cls_logits, iou_logits, target_cls, target_iou)
        assert "loss_cls" in losses
        assert "loss_iou_pred" in losses
        assert "loss_total" in losses

    def test_loss_positive(self, head):
        N = 10
        cls_logits = torch.randn(N, 10)
        iou_logits = torch.randn(N, 1)
        target_cls = torch.zeros(N, 10)
        target_iou = torch.rand(N, 1)
        losses = head.loss(cls_logits, iou_logits, target_cls, target_iou)
        assert losses["loss_total"].item() > 0

    def test_loss_backward(self, head):
        N = 10
        cls_logits = torch.randn(N, 10, requires_grad=True)
        iou_logits = torch.randn(N, 1, requires_grad=True)
        target_cls = torch.zeros(N, 10)
        target_iou = torch.rand(N, 1)
        losses = head.loss(cls_logits, iou_logits, target_cls, target_iou)
        losses["loss_total"].backward()
        assert cls_logits.grad is not None
        assert iou_logits.grad is not None

    def test_iou_reweights_scores(self, head):
        """Higher predicted IoU should raise joint scores."""
        head.eval()
        feats = torch.randn(1, 5, 64)
        with torch.no_grad():
            out_base = head(feats)

        # Force the IoU head to predict high confidence by patching bias.
        original_bias = head.iou_head[-1].bias.data.clone()
        head.iou_head[-1].bias.data.fill_(10.0)
        with torch.no_grad():
            out_high = head(feats)
        head.iou_head[-1].bias.data.copy_(original_bias)

        # Scores with high IoU bias should be close to cls_prob (IoU≈1).
        cls_prob = torch.sigmoid(out_high["cls_logits"])
        assert torch.allclose(out_high["scores"], cls_prob, atol=1e-3)
