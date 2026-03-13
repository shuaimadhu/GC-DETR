"""
Miscellaneous utility classes and functions for GC-DETR.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor


class NestedTensor:
    """A tensor bundled with a boolean padding mask.

    ``tensors`` has shape (B, C, H, W) and ``mask`` has shape (B, H, W)
    where ``True`` indicates a *padded* (non-valid) position.
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> "NestedTensor":
        tensors = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return f"NestedTensor(tensors={self.tensors.shape}, mask={self.mask})"


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """Create a :class:`NestedTensor` from a list of tensors with varying sizes.

    All tensors are padded to the largest spatial dimensions in the batch.

    Args:
        tensor_list: List of image tensors each with shape (C, H, W).

    Returns:
        A :class:`NestedTensor` with the batched images and corresponding mask.
    """
    max_h = max(t.shape[-2] for t in tensor_list)
    max_w = max(t.shape[-1] for t in tensor_list)

    batch_size = len(tensor_list)
    channels = tensor_list[0].shape[0]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    batched = torch.zeros(batch_size, channels, max_h, max_w, dtype=dtype, device=device)
    mask = torch.ones(batch_size, max_h, max_w, dtype=torch.bool, device=device)

    for i, t in enumerate(tensor_list):
        h, w = t.shape[-2], t.shape[-1]
        batched[i, :, :h, :w] = t
        mask[i, :h, :w] = False  # valid positions are False

    return NestedTensor(batched, mask)
