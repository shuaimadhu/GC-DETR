"""Integration tests for the full GC-DETR forward pass."""

import pytest
import torch

from gcdetr.models.gc_detr import build_gc_detr, GCDETR


@pytest.fixture(scope="module")
def model():
    """Build a tiny GC-DETR model for fast testing."""
    return build_gc_detr(
        num_classes=5,
        backbone_name="resnet18",
        pretrained_backbone=False,
        embed_dim=64,
        num_queries=20,
        num_encoder_layers=1,
        num_decoder_layers=2,
        num_heads=4,
        ffn_dim=128,
        num_points=2,
        num_levels=2,
        dropout=0.0,
        aux_loss=True,
    )


class TestGCDETRForward:
    def test_output_keys(self, model):
        images = torch.randn(2, 3, 128, 128)
        out = model(images)
        assert "pred_boxes" in out
        assert "pred_logits" in out
        assert "pred_iou" in out
        assert "pred_scores" in out

    def test_output_shapes(self, model):
        B, N, C_cls = 2, 20, 5
        images = torch.randn(B, 3, 128, 128)
        out = model(images)
        assert out["pred_boxes"].shape == (B, N, 4)
        assert out["pred_logits"].shape == (B, N, C_cls)
        assert out["pred_iou"].shape == (B, N, 1)
        assert out["pred_scores"].shape == (B, N, C_cls)

    def test_boxes_in_01(self, model):
        images = torch.randn(2, 3, 128, 128)
        out = model(images)
        boxes = out["pred_boxes"]
        assert (boxes >= 0).all() and (boxes <= 1).all(), "Box coords must be in [0, 1]"

    def test_scores_in_01(self, model):
        images = torch.randn(2, 3, 128, 128)
        out = model(images)
        scores = out["pred_scores"]
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_with_pose(self, model):
        B = 2
        images = torch.randn(B, 3, 128, 128)
        pose = torch.rand(B, 4)
        out = model(images, pose=pose)
        assert out["pred_boxes"].shape == (B, 20, 4)

    def test_without_pose(self, model):
        images = torch.randn(2, 3, 128, 128)
        out = model(images, pose=None)
        assert "pred_boxes" in out

    def test_aux_outputs_present(self, model):
        images = torch.randn(2, 3, 128, 128)
        out = model(images)
        assert "aux_outputs" in out

    def test_no_nan(self, model):
        images = torch.randn(2, 3, 128, 128)
        out = model(images)
        for key in ("pred_boxes", "pred_logits", "pred_iou", "pred_scores"):
            assert not torch.isnan(out[key]).any(), f"NaN in {key}"

    def test_gradient_flows_through_model(self, model):
        images = torch.randn(2, 3, 128, 128, requires_grad=True)
        pose = torch.rand(2, 4, requires_grad=True)
        out = model(images, pose=pose)
        loss = out["pred_boxes"].sum() + out["pred_scores"].sum()
        loss.backward()
        assert images.grad is not None
        assert pose.grad is not None

    def test_single_image_batch(self, model):
        images = torch.randn(1, 3, 128, 128)
        out = model(images)
        assert out["pred_boxes"].shape[0] == 1

    def test_no_aux_loss_model(self):
        m = build_gc_detr(
            num_classes=3,
            backbone_name="resnet18",
            pretrained_backbone=False,
            embed_dim=64,
            num_queries=10,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            ffn_dim=64,
            num_points=2,
            num_levels=2,
            dropout=0.0,
            aux_loss=False,
        )
        images = torch.randn(1, 3, 128, 128)
        out = m(images)
        assert "aux_outputs" not in out
