"""Name -> implementation registries for datasets, metrics and predictors.

This module is the single source of truth for the string names used by the CLI,
the eval matrix and (later) the router's tool registry. Entries are declared as
``"module:ClassName"`` and imported lazily, so listing the available datasets
never imports torch and never touches the disk.

Adding an implementation = adding one line to the tables below. Nothing else in
the codebase may keep its own name->class mapping.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Iterable

from .contracts import Task

__all__ = [
    "DATASETS",
    "METRICS",
    "PREDICTORS",
    "DEFAULT_METRICS",
    "build_dataset",
    "build_metric",
    "build_predictor",
    "list_datasets",
    "list_metrics",
    "list_predictors",
    "default_metrics_for",
    "resolve",
]


# --------------------------------------------------------------------------- #
# Tables. Keys are CLI-visible names; keep them lowercase snake_case.
# --------------------------------------------------------------------------- #

#: dataset name -> (import target, task, default split)
DATASETS: dict[str, tuple[str, Task, str]] = {
    "rsvqa_lr":         ("satquery.datasets.rsvqa:RSVQALRDataset",           "vqa",               "test"),
    "rsvqa_hr":         ("satquery.datasets.rsvqa:RSVQAHRDataset",           "vqa",               "test"),
    "vrsbench_vqa":     ("satquery.datasets.vrsbench:VRSBenchVQADataset",    "vqa",               "test"),
    "vrsbench_caption": ("satquery.datasets.vrsbench:VRSBenchCaptionDataset","caption",           "test"),
    "vrsbench_ground":  ("satquery.datasets.vrsbench:VRSBenchGroundDataset", "ground",            "test"),
    "dior_rsvg":        ("satquery.datasets.dior_rsvg:DiorRSVGDataset",      "ground",            "test"),
    "cdvqa":            ("satquery.datasets.cdvqa:CDVQADataset",             "change_vqa",        "test"),
    "levir_cc":         ("satquery.datasets.levir:LevirCCDataset",           "change_caption",    "test"),
    "levir_mci":        ("satquery.datasets.levir:LevirMCIDataset",          "change_map",        "test"),
    "bigearthnet_mm":   ("satquery.datasets.bigearthnet:BigEarthNetMMDataset","fusion_multilabel", "val"),
}

#: metric name -> import target
METRICS: dict[str, str] = {
    "vqa_accuracy": "satquery.metrics.vqa:VqaAccuracy",
    "caption":      "satquery.metrics.caption:CaptionMetrics",
    "grounding":    "satquery.metrics.grounding:GroundingMetrics",
    "multilabel":   "satquery.metrics.multilabel:MultiLabelMetrics",
    "change_mask":  "satquery.metrics.multilabel:ChangeMaskMetrics",
}

#: predictor name -> import target. ``dummy``/``oracle``/``constant`` are the
#: harness's own test doubles and must never import torch.
PREDICTORS: dict[str, str] = {
    "dummy":      "satquery.models.dummy:DummyPredictor",
    "oracle":     "satquery.models.dummy:OraclePredictor",
    "constant":   "satquery.models.dummy:ConstantPredictor",
    "qwen2_5_vl": "satquery.models.qwen_vl:QwenVLPredictor",
    "earthdial":  "satquery.models.earthdial:EarthDialPredictor",
}

#: Which metrics the harness runs for a task when none are named explicitly.
DEFAULT_METRICS: dict[Task, tuple[str, ...]] = {
    "vqa": ("vqa_accuracy",),
    "change_vqa": ("vqa_accuracy",),
    "caption": ("caption",),
    "change_caption": ("caption",),
    "ground": ("grounding",),
    "change_map": ("change_mask",),
    "fusion_multilabel": ("multilabel",),
}


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def resolve(target: str) -> Any:
    """Import ``"package.module:Attribute"`` and return the attribute."""
    if ":" not in target:
        raise ValueError(f"bad import target {target!r}, expected 'module:Attribute'")
    module_name, attr = target.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - developer error
        raise ImportError(f"{module_name} has no attribute {attr!r}") from exc


def _unknown(kind: str, name: str, known: Iterable[str]) -> KeyError:
    return KeyError(f"unknown {kind} {name!r}; known: {', '.join(sorted(known))}")


def build_dataset(name: str, *, split: str | None = None, **kwargs: Any) -> Any:
    """Instantiate a dataset adapter by registry name."""
    if name not in DATASETS:
        raise _unknown("dataset", name, DATASETS)
    target, _task, default_split = DATASETS[name]
    cls: Callable[..., Any] = resolve(target)
    return cls(split=split or default_split, **kwargs)


def build_metric(name: str, **kwargs: Any) -> Any:
    if name not in METRICS:
        raise _unknown("metric", name, METRICS)
    return resolve(METRICS[name])(**kwargs)


def build_predictor(name: str, **kwargs: Any) -> Any:
    if name not in PREDICTORS:
        raise _unknown("predictor", name, PREDICTORS)
    return resolve(PREDICTORS[name])(**kwargs)


def list_datasets() -> list[tuple[str, Task, str]]:
    """``[(name, task, default_split)]`` without importing any adapter."""
    return [(name, task, split) for name, (_t, task, split) in DATASETS.items()]


def list_metrics() -> list[str]:
    return sorted(METRICS)


def list_predictors() -> list[str]:
    return sorted(PREDICTORS)


def task_of(name: str) -> Task:
    if name not in DATASETS:
        raise _unknown("dataset", name, DATASETS)
    return DATASETS[name][1]


def default_metrics_for(task: Task) -> tuple[str, ...]:
    return DEFAULT_METRICS.get(task, ())
