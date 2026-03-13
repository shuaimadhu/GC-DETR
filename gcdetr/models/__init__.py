"""Models package for GC-DETR."""

from gcdetr.models.saqm import SensorAwareQueryModule
from gcdetr.models.sdda import SpectralDecompositionDeformableAttention
from gcdetr.models.ioupqh import IoUPredictionQualityHead
from gcdetr.models.backbone import ResNetBackbone
from gcdetr.models.neck import ChannelProjectionNeck
from gcdetr.models.gc_detr import GCDETR, build_gc_detr

__all__ = [
    "SensorAwareQueryModule",
    "SpectralDecompositionDeformableAttention",
    "IoUPredictionQualityHead",
    "ResNetBackbone",
    "ChannelProjectionNeck",
    "GCDETR",
    "build_gc_detr",
]
