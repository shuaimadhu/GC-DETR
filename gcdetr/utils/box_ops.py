"""
Box operations for GC-DETR.

Provides utilities for bounding-box format conversions and IoU computations
used throughout the detection pipeline.
"""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        boxes: Bounding boxes in center-size format with shape (..., 4).

    Returns:
        Bounding boxes in corner format with shape (..., 4).
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    Args:
        boxes: Bounding boxes in corner format with shape (..., 4).

    Returns:
        Bounding boxes in center-size format with shape (..., 4).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor):
    """Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: Boxes in (x1, y1, x2, y2) format with shape (N, 4).
        boxes2: Boxes in (x1, y1, x2, y2) format with shape (M, 4).

    Returns:
        iou: IoU matrix of shape (N, M).
        union: Union area matrix of shape (N, M).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute the Generalized IoU (GIoU) between two sets of boxes.

    Reference: https://giou.stanford.edu/

    Args:
        boxes1: Boxes in (x1, y1, x2, y2) format with shape (N, 4).
        boxes2: Boxes in (x1, y1, x2, y2) format with shape (M, 4).

    Returns:
        GIoU matrix of shape (N, M) with values in [-1, 1].
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 must have x2>x1, y2>y1"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 must have x2>x1, y2>y1"

    iou, union = box_iou(boxes1, boxes2)

    enclosing_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enclosing_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enclosing_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enclosing_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enclosing_area = (enclosing_x2 - enclosing_x1).clamp(min=0) * (
        enclosing_y2 - enclosing_y1
    ).clamp(min=0)

    giou = iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)
    return giou
