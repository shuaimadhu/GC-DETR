"""
GC-DETR: Geometry-Conditioned Real-Time Object Detector.

This module assembles the full GC-DETR pipeline:

  Backbone  →  ChannelProjectionNeck
             → TransformerEncoder
             → SAQM-enhanced object queries
             → TransformerDecoder (SDDA cross-attention)
             → box regression head + IOUPQH detection head

Reference
---------
GC-DETR: Geometry-Conditioned Real-Time Object Detection for UAV-based
Open-Water Scenarios.
"""

from __future__ import annotations

import math
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gcdetr.models.backbone import ResNetBackbone
from gcdetr.models.neck import ChannelProjectionNeck
from gcdetr.models.saqm import SensorAwareQueryModule
from gcdetr.models.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from gcdetr.models.ioupqh import IoUPredictionQualityHead
from gcdetr.utils.box_ops import box_cxcywh_to_xyxy


class GCDETR(nn.Module):
    """GC-DETR end-to-end object detector.

    Args:
        backbone: Feature extraction backbone.
        neck: Multi-scale feature projection neck.
        encoder: Transformer encoder operating on flattened multi-scale tokens.
        decoder: Transformer decoder with SDDA cross-attention.
        saqm: Sensor-Aware Query Module for pose-conditioned query initialisation.
        detection_head: IoU Prediction Quality Head (IOUPQH) producing joint
            classification-quality scores.
        embed_dim: Transformer embedding dimension.
        num_queries: Number of object queries.
        num_classes: Number of object categories.
        num_levels: Number of feature-map scales fed to the decoder.
        aux_loss: If *True*, compute auxiliary detection losses at every
            decoder layer.
    """

    def __init__(
        self,
        backbone: ResNetBackbone,
        neck: ChannelProjectionNeck,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        saqm: SensorAwareQueryModule,
        detection_head: IoUPredictionQualityHead,
        embed_dim: int = 256,
        num_queries: int = 300,
        num_classes: int = 80,
        num_levels: int = 3,
        aux_loss: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.encoder = encoder
        self.decoder = decoder
        self.saqm = saqm
        self.detection_head = detection_head

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.aux_loss = aux_loss

        # Learnable query embeddings (content + position slots).
        self.query_embed = nn.Embedding(num_queries, embed_dim * 2)

        # Box regression MLP: predict (cx, cy, w, h) ∈ [0, 1].
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4),
        )

        self._reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self) -> None:
        # Initialise query embeddings.
        nn.init.normal_(self.query_embed.weight)

        # Box head: final layer bias makes initial predictions cover the
        # full image (cx=0.5, cy=0.5, w=0.5, h=0.5 in sigmoid space).
        for layer in self.box_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.constant_(self.box_head[-1].bias, 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_reference_points(
        self,
        spatial_shapes: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Create normalised reference points on a regular grid.

        For each feature level, reference points are placed at the centre of
        each spatial cell (in normalised [0, 1] coordinates).

        Args:
            spatial_shapes: (num_levels, 2) tensor of (H, W) per level.
            device: Target device.

        Returns:
            Reference points of shape (1, S_total, num_levels, 2) where
            S_total = sum(H_i * W_i).
        """
        ref_pts: List[Tensor] = []
        for lvl, (H, W) in enumerate(spatial_shapes.tolist()):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, int(H), device=device) / H,
                torch.linspace(0.5, W - 0.5, int(W), device=device) / W,
                indexing="ij",
            )
            ref = torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], dim=-1)  # (H*W, 2)
            ref_pts.append(ref)
        ref_pts = torch.cat(ref_pts, dim=0)  # (S_total, 2)
        # Expand to (1, S_total, num_levels, 2): use the same reference for
        # all decoder levels (the SDDA offsets refine from here).
        return ref_pts[None, :, None, :].expand(1, -1, spatial_shapes.shape[0], -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: Tensor,
        pose: Optional[Tensor] = None,
        masks: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Run the full GC-DETR forward pass.

        Args:
            images: Batched input images of shape (B, 3, H, W).
            pose: Optional UAV sensor state (altitude, pitch, roll, yaw)
                of shape (B, 4) with values normalised to [0, 1] or [-1, 1].
                Passed to SAQM for query conditioning.  When *None* the
                module returns unmodified queries.
            masks: Optional per-level padding masks for the backbone features,
                mapping level name → (B, H_i, W_i) boolean tensor.

        Returns:
            Dictionary containing:

            - ``"pred_boxes"`` (Tensor): Box predictions (B, N, 4) in
              (cx, cy, w, h) normalised format.
            - ``"pred_logits"`` (Tensor): Raw classification logits
              (B, N, num_classes).
            - ``"pred_iou"`` (Tensor): IoU quality logits (B, N, 1).
            - ``"pred_scores"`` (Tensor): Joint classification×IoU scores
              (B, N, num_classes); used for NMS ranking at inference time.
            - ``"aux_outputs"`` (list, optional): Per-decoder-layer dicts with
              the same keys, included when ``self.aux_loss`` is *True*.
        """
        B = images.shape[0]
        device = images.device

        # ------------------------------------------------------------------
        # 1. Backbone feature extraction.
        # ------------------------------------------------------------------
        backbone_features = self.backbone(images)

        # Select only the last ``num_levels`` feature maps for the neck so
        # that the number of levels matches the neck's projection modules.
        all_level_names = list(backbone_features.keys())
        selected_names = all_level_names[-self.num_levels :]
        selected_features = {k: backbone_features[k] for k in selected_names}
        selected_masks = (
            {k: masks[k] for k in selected_names if k in masks}
            if masks is not None
            else None
        )

        # ------------------------------------------------------------------
        # 2. Neck: project + flatten to transformer tokens.
        # ------------------------------------------------------------------
        flat_feats, flat_masks, spatial_shapes, level_start_index = self.neck(
            selected_features, selected_masks
        )

        # ------------------------------------------------------------------
        # 3. Transformer encoder.
        # ------------------------------------------------------------------
        memory = self.encoder(flat_feats, src_key_padding_mask=flat_masks)

        # ------------------------------------------------------------------
        # 4. Initialise object queries.
        # ------------------------------------------------------------------
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = query_embed[:, :, : self.embed_dim]  # positional slot
        query_content = query_embed[:, :, self.embed_dim :]  # content slot

        # Inject UAV pose priors via SAQM.
        query_content = self.saqm(query_content, pose)

        queries = query_content + query_pos  # combine positional and content

        # Build per-query reference points (centred at (0.5, 0.5) initially).
        ref_points = (
            query_content.new_zeros(B, self.num_queries, spatial_shapes.shape[0], 2)
            + 0.5
        )

        # ------------------------------------------------------------------
        # 5. Transformer decoder (SDDA cross-attention).
        # ------------------------------------------------------------------
        intermediate = self.decoder(
            query=queries,
            memory=memory,
            reference_points=ref_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            memory_key_padding_mask=flat_masks,
        )

        if not isinstance(intermediate, list):
            intermediate = [intermediate]

        # ------------------------------------------------------------------
        # 6. Prediction heads on the last (and optionally all) decoder layers.
        # ------------------------------------------------------------------
        outputs: List[Dict[str, Tensor]] = []
        for hs in intermediate:
            boxes = self.box_head(hs).sigmoid()  # (B, N, 4)
            head_out = self.detection_head(hs)
            outputs.append(
                {
                    "pred_boxes": boxes,
                    "pred_logits": head_out["cls_logits"],
                    "pred_iou": head_out["iou_logits"],
                    "pred_scores": head_out["scores"],
                }
            )

        result = outputs[-1]
        if self.aux_loss and len(outputs) > 1:
            result["aux_outputs"] = outputs[:-1]

        return result


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_gc_detr(
    num_classes: int = 80,
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = False,
    embed_dim: int = 256,
    num_queries: int = 300,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    num_heads: int = 8,
    ffn_dim: int = 1024,
    num_points: int = 4,
    num_levels: int = 3,
    dropout: float = 0.1,
    aux_loss: bool = True,
) -> GCDETR:
    """Construct a GC-DETR model with the specified hyperparameters.

    Args:
        num_classes: Number of object categories (excluding background).
        backbone_name: ResNet variant name (e.g. ``"resnet50"``).
        pretrained_backbone: Load ImageNet pre-trained backbone weights.
        embed_dim: Transformer embedding dimension.
        num_queries: Number of object queries.
        num_encoder_layers: Depth of the transformer encoder.
        num_decoder_layers: Depth of the transformer decoder.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        num_points: Deformable sampling points per head per level in SDDA.
        num_levels: Number of feature-map scales (must match backbone outputs).
        dropout: Dropout probability used throughout.
        aux_loss: Enable auxiliary per-layer decoder losses.

    Returns:
        Fully initialised :class:`GCDETR` model.
    """
    # Backbone
    backbone = ResNetBackbone(name=backbone_name, pretrained=pretrained_backbone)

    # Neck – project the last ``num_levels`` backbone stages.
    neck = ChannelProjectionNeck(
        in_channels=backbone.out_channels[-num_levels:],
        embed_dim=embed_dim,
        add_pos_encoding=True,
    )

    # Encoder
    enc_layer = TransformerEncoderLayer(
        embed_dim=embed_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout
    )
    encoder = TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

    # Decoder with SDDA
    sdda_cfg = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_points": num_points,
        "num_levels": num_levels,
        "dropout": dropout,
    }
    dec_layer = TransformerDecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        sdda_cfg=sdda_cfg,
        dropout=dropout,
    )
    decoder = TransformerDecoder(
        dec_layer, num_layers=num_decoder_layers, return_intermediate=aux_loss
    )

    # SAQM
    saqm = SensorAwareQueryModule(
        embed_dim=embed_dim,
        pose_dim=4,
        hidden_dim=64,
        num_queries=num_queries,
        dropout=dropout,
    )

    # IOUPQH detection head
    detection_head = IoUPredictionQualityHead(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=embed_dim,
        num_layers=2,
    )

    model = GCDETR(
        backbone=backbone,
        neck=neck,
        encoder=encoder,
        decoder=decoder,
        saqm=saqm,
        detection_head=detection_head,
        embed_dim=embed_dim,
        num_queries=num_queries,
        num_classes=num_classes,
        num_levels=num_levels,
        aux_loss=aux_loss,
    )

    return model
