"""Tests for box utility functions."""

import pytest
import torch

from gcdetr.utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
)


class TestBoxConversions:
    def test_cxcywh_to_xyxy(self):
        boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]])  # full-image box
        xyxy = box_cxcywh_to_xyxy(boxes)
        expected = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        assert torch.allclose(xyxy, expected)

    def test_xyxy_to_cxcywh(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        cxcywh = box_xyxy_to_cxcywh(boxes)
        expected = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        assert torch.allclose(cxcywh, expected)

    def test_round_trip(self):
        boxes = torch.rand(10, 4)
        # Ensure x2>x1, y2>y1 for xyxy format
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:].abs().clamp(min=0.01)
        recovered = box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(boxes)))
        assert torch.allclose(recovered, box_xyxy_to_cxcywh(boxes), atol=1e-5)

    def test_batch_dimensions(self):
        boxes = torch.rand(4, 10, 4)
        out = box_cxcywh_to_xyxy(boxes)
        assert out.shape == boxes.shape


class TestBoxIoU:
    def test_identical_boxes(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        iou, _ = box_iou(boxes, boxes)
        assert torch.allclose(iou, torch.ones(1, 1))

    def test_non_overlapping_boxes(self):
        b1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
        b2 = torch.tensor([[0.6, 0.6, 1.0, 1.0]])
        iou, _ = box_iou(b1, b2)
        assert torch.allclose(iou, torch.zeros(1, 1))

    def test_iou_range(self):
        boxes1 = torch.rand(5, 4)
        boxes1[:, 2:] = boxes1[:, :2] + 0.3
        boxes2 = torch.rand(7, 4)
        boxes2[:, 2:] = boxes2[:, :2] + 0.3
        iou, _ = box_iou(boxes1, boxes2)
        assert (iou >= 0).all() and (iou <= 1).all()


class TestGeneralizedBoxIoU:
    def test_identical_boxes_giou_is_1(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        giou = generalized_box_iou(boxes, boxes)
        assert torch.allclose(giou, torch.ones(1, 1))

    def test_giou_range(self):
        boxes1 = torch.rand(5, 4)
        boxes1[:, 2:] = boxes1[:, :2] + 0.3
        boxes1 = boxes1.clamp(0.0, 1.0)
        boxes1[:, 2] = torch.max(boxes1[:, 0], boxes1[:, 2])
        boxes1[:, 3] = torch.max(boxes1[:, 1], boxes1[:, 3])
        boxes2 = boxes1.flip(0)
        giou = generalized_box_iou(boxes1, boxes2)
        assert (giou >= -1).all() and (giou <= 1).all()
