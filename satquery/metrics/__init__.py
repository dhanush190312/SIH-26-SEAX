"""Benchmark metrics. Pure numpy - no Java, no nltk, no pycocoevalcap needed."""

from .._lazy import lazy_getattr

_TABLE = {
    "VqaAccuracy": ".vqa:VqaAccuracy",
    "CaptionMetrics": ".caption:CaptionMetrics",
    "GroundingMetrics": ".grounding:GroundingMetrics",
    "MultiLabelMetrics": ".multilabel:MultiLabelMetrics",
    "ChangeMaskMetrics": ".multilabel:ChangeMaskMetrics",
}

__all__ = sorted(_TABLE)
__getattr__ = lazy_getattr(__name__, _TABLE)
