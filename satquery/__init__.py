"""SatQuery AI - agentic vision-language assistant for satellite imagery.

Import policy: this module and everything it touches must stay importable with
numpy alone. Heavy extras (torch, transformers, rasterio) are imported inside
the functions that need them, so `satquery datasets` and `satquery doctor` work
on a bare laptop.
"""

from .config import SatQueryConfig, get_config, set_config
from .contracts import (
    Box,
    DatasetNotAvailable,
    ImageRef,
    Metric,
    Prediction,
    Predictor,
    RunResult,
    RunSpec,
    Sample,
    SatQueryError,
    TASKS,
    Task,
    UnsupportedTask,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Box",
    "DatasetNotAvailable",
    "ImageRef",
    "Metric",
    "Prediction",
    "Predictor",
    "RunResult",
    "RunSpec",
    "Sample",
    "SatQueryConfig",
    "SatQueryError",
    "TASKS",
    "Task",
    "UnsupportedTask",
    "get_config",
    "set_config",
]
