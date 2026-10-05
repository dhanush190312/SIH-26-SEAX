"""Predictors.

The test doubles import eagerly (stdlib only). Real adapters (`qwen_vl`,
`earthdial`) pull in torch/transformers, so they load on first attribute access
and are normally reached through `satquery.registry.build_predictor`.
"""

from .._lazy import lazy_getattr
from .dummy import ConstantPredictor, DummyPredictor, OraclePredictor

_TABLE = {
    "QwenVLPredictor": ".qwen_vl:QwenVLPredictor",
    "EarthDialPredictor": ".earthdial:EarthDialPredictor",
}

__all__ = [
    "ConstantPredictor",
    "DummyPredictor",
    "OraclePredictor",
    "QwenVLPredictor",
    "EarthDialPredictor",
]
__getattr__ = lazy_getattr(__name__, _TABLE)
