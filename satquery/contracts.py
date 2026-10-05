"""Frozen data contracts for SatQuery AI.

Every interior component -- dataset adapters, metrics, model predictors, the
eval harness -- speaks *these* types and nothing else. Treat this module as a
published API: renaming a field here breaks every module that reads it.

Conventions that are easy to get wrong, so they are stated once, here:

* Bounding boxes are ALWAYS absolute pixel coordinates in ``(x0, y0, x1, y1)``
  order with ``x1 >= x0`` and ``y1 >= y0``, relative to the *preprocessed* image
  actually handed to the model. Adapters for models that emit normalised or
  ``[0, 1000]``-scaled coordinates (Qwen2.5-VL does) must convert with
  :meth:`Box.from_normalized` before returning a :class:`Prediction`.
* Image arrays are ``(band, height, width)`` float32, band-first, like rasterio.
* Nothing in this module may import numpy, torch, rasterio or pandas. It must
  stay importable on a bare interpreter so the harness can be introspected
  without the heavy extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal, Mapping, Protocol, Sequence, runtime_checkable

__all__ = [
    "Task",
    "TASKS",
    "PAIR_TASKS",
    "Modality",
    "MODALITIES",
    "ImageRef",
    "Box",
    "Sample",
    "Prediction",
    "RunSpec",
    "RunResult",
    "DatasetAdapter",
    "Predictor",
    "Metric",
    "SatQueryError",
    "DatasetNotAvailable",
    "UnsupportedTask",
    "supports_task",
]

# --------------------------------------------------------------------------- #
# Task / modality vocabularies
# --------------------------------------------------------------------------- #

Task = Literal[
    "vqa",               # single image + question -> short answer
    "caption",           # single image -> description
    "ground",            # single image + referring expression -> boxes
    "change_vqa",        # image pair + question -> short answer
    "change_caption",    # image pair -> change description
    "change_map",        # image pair -> binary change mask (+ stats)
    "fusion_multilabel", # optical+SAR pair -> multi-label land cover
]

TASKS: tuple[Task, ...] = (
    "vqa",
    "caption",
    "ground",
    "change_vqa",
    "change_caption",
    "change_map",
    "fusion_multilabel",
)

#: Tasks that consume exactly two co-registered images.
PAIR_TASKS: frozenset[str] = frozenset(
    {"change_vqa", "change_caption", "change_map", "fusion_multilabel"}
)

Modality = Literal["optical", "sar", "unknown"]
MODALITIES: tuple[Modality, ...] = ("optical", "sar", "unknown")


class SatQueryError(Exception):
    """Base class for all SatQuery errors."""


class DatasetNotAvailable(SatQueryError):
    """Raised when a dataset's files are missing on this machine.

    The harness catches this and records the dataset as skipped rather than
    failing the whole run -- benchmarks live on Colab/Kaggle, not the laptop.
    """


class UnsupportedTask(SatQueryError):
    """Raised when a predictor or metric is asked for a task it does not serve."""


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ImageRef:
    """A reference to one image on disk, plus what we believe it is.

    ``modality`` is a *claim* -- either declared by the dataset or inferred by
    :mod:`satquery.io.modality`. The inferring code records provenance in
    ``meta['modality_source']`` so the execution trace can show it.
    """

    image_id: str
    path: str
    modality: Modality = "unknown"
    bands: int | None = None
    height: int | None = None
    width: int | None = None
    timestamp: str | None = None  # ISO date, for bi-temporal ordering
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("ImageRef.image_id must be non-empty")
        if self.modality not in MODALITIES:
            raise ValueError(f"unknown modality {self.modality!r}, expected one of {MODALITIES}")

    def with_modality(self, modality: Modality, source: str = "inferred") -> "ImageRef":
        meta = dict(self.meta)
        meta["modality_source"] = source
        return replace(self, modality=modality, meta=meta)


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in absolute pixel coordinates, ``(x0, y0, x1, y1)``."""

    x0: float
    y0: float
    x1: float
    y1: float
    label: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"degenerate box: {self.as_xyxy()} (need x1>=x0, y1>=y0)")

    # -- constructors -------------------------------------------------------- #

    @classmethod
    def from_xywh(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str | None = None,
        score: float | None = None,
    ) -> "Box":
        return cls(x, y, x + w, y + h, label, score)

    @classmethod
    def from_normalized(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: int,
        height: int,
        scale: float = 1.0,
        label: str | None = None,
        score: float | None = None,
    ) -> "Box":
        """Convert model-space coords to pixels.

        ``scale`` is the model's coordinate range: ``1.0`` for 0-1 normalised
        output, ``1000.0`` for Qwen-style 0-1000 output.
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        return cls(
            x0 / scale * width,
            y0 / scale * height,
            x1 / scale * width,
            y1 / scale * height,
            label,
            score,
        )

    # -- geometry ------------------------------------------------------------ #

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def iou(self, other: "Box") -> float:
        ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return 0.0 if union <= 0 else inter / union

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            min(max(self.x0, 0.0), float(width)),
            min(max(self.y0, 0.0), float(height)),
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            self.label,
            self.score,
        )


@dataclass(frozen=True, slots=True)
class Sample:
    """One evaluation item, uniform across every benchmark we run.

    Which fields carry ground truth depends on ``task``:

    ==================  ========================================================
    task                ground truth fields
    ==================  ========================================================
    vqa, change_vqa     ``answer`` (canonical) and ``references`` (all accepted)
    caption,            ``references`` (one or more reference sentences)
    change_caption
    ground              ``boxes`` (one or more; multi-instance is normal in RS)
    change_map          ``meta['mask_path']``, when the dataset ships masks
    fusion_multilabel   ``labels``
    ==================  ========================================================

    ``meta`` carries per-dataset extras the metrics slice on -- most importantly
    ``question_type`` (RSVQA: count/presence/comparison/rural_urban; CDVQA:
    change-question type), because the challenge scores per-type accuracy.
    """

    sample_id: str
    task: Task
    images: tuple[ImageRef, ...]
    dataset: str = ""
    split: str = ""
    question: str | None = None
    answer: str | None = None
    references: tuple[str, ...] = ()
    boxes: tuple[Box, ...] = ()
    labels: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("Sample.sample_id must be non-empty")
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}, expected one of {TASKS}")
        if not self.images:
            raise ValueError(f"sample {self.sample_id}: at least one image is required")
        if self.task in PAIR_TASKS and len(self.images) != 2:
            raise ValueError(
                f"sample {self.sample_id}: task {self.task} needs exactly 2 images, "
                f"got {len(self.images)}"
            )
        if self.task in ("vqa", "change_vqa", "ground") and not self.question:
            raise ValueError(f"sample {self.sample_id}: task {self.task} needs a question")

    @property
    def question_type(self) -> str:
        """Metric slicing key; ``'all'`` when the dataset does not label types."""
        return str(self.meta.get("question_type", "all"))

    @property
    def all_answers(self) -> tuple[str, ...]:
        """Every accepted answer string (canonical first, de-duplicated)."""
        out: list[str] = []
        for a in (self.answer, *self.references):
            if a and a not in out:
                out.append(a)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class Prediction:
    """What a predictor returns for one :class:`Sample`.

    ``evidence`` is the numeric substrate the composer is later required to cite
    (mask area %, box coords, class probabilities, backscatter stats). Metrics
    ignore it; the trace and the PDF report do not.
    """

    sample_id: str
    task: Task
    text: str | None = None
    boxes: tuple[Box, ...] = ()
    labels: tuple[str, ...] = ()
    label_scores: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    latency_ms: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def failed(cls, sample_id: str, task: Task, error: str, model: str = "") -> "Prediction":
        return cls(sample_id=sample_id, task=task, error=error, model=model)


# --------------------------------------------------------------------------- #
# Run artefacts. This pins the on-disk JSON schema: report.py and the
# before/after regression diffs read exactly these keys.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    dataset: str
    split: str
    task: Task
    predictor: str
    limit: int | None = None
    seed: int = 1337
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "split": self.split,
            "task": self.task,
            "predictor": self.predictor,
            "limit": self.limit,
            "seed": self.seed,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunSpec":
        return cls(
            run_id=str(d["run_id"]),
            dataset=str(d["dataset"]),
            split=str(d["split"]),
            task=d["task"],
            predictor=str(d["predictor"]),
            limit=d.get("limit"),
            seed=int(d.get("seed", 1337)),
            params=dict(d.get("params", {})),
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    spec: RunSpec
    metrics: Mapping[str, float]
    n_samples: int
    n_failed: int = 0
    n_skipped: int = 0
    duration_s: float = 0.0
    started_at: str = ""
    per_sample: tuple[Mapping[str, Any], ...] = ()
    env: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "n_samples": self.n_samples,
            "n_failed": self.n_failed,
            "n_skipped": self.n_skipped,
            "duration_s": round(self.duration_s, 3),
            "started_at": self.started_at,
            "per_sample": [dict(r) for r in self.per_sample],
            "env": dict(self.env),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunResult":
        return cls(
            spec=RunSpec.from_dict(d["spec"]),
            metrics=dict(d.get("metrics", {})),
            n_samples=int(d.get("n_samples", 0)),
            n_failed=int(d.get("n_failed", 0)),
            n_skipped=int(d.get("n_skipped", 0)),
            duration_s=float(d.get("duration_s", 0.0)),
            started_at=str(d.get("started_at", "")),
            per_sample=tuple(dict(r) for r in d.get("per_sample", ())),
            env=dict(d.get("env", {})),
            notes=tuple(d.get("notes", ())),
        )


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class DatasetAdapter(Protocol):
    """Yields :class:`Sample` objects for one (dataset, split, task).

    Concrete adapters subclass :class:`satquery.datasets.base.BaseDataset`,
    which implements the bookkeeping; this Protocol is what the harness types
    against.
    """

    name: str
    split: str
    task: Task

    def available(self) -> bool:
        """True when the files this adapter needs exist locally."""
        ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Sample]: ...


@runtime_checkable
class Predictor(Protocol):
    """A model (or deterministic tool) that answers samples."""

    name: str

    def supports(self, task: Task) -> bool: ...

    def predict(self, sample: Sample) -> Prediction: ...


@runtime_checkable
class Metric(Protocol):
    """Streaming metric: ``reset`` -> ``update`` per sample -> ``compute``.

    ``compute`` returns flat ``{name: value}`` floats. Per-slice breakdowns use
    a ``'/'`` suffix convention, e.g. ``{"accuracy": 0.71, "accuracy/count":
    0.62, "accuracy/presence": 0.88}`` -- the report renders the bare key as the
    headline and the suffixed keys as a breakdown table.
    """

    name: str
    tasks: tuple[Task, ...]

    def reset(self) -> None: ...

    def update(self, sample: Sample, prediction: Prediction) -> None: ...

    def compute(self) -> dict[str, float]: ...


def supports_task(obj: Any, task: Task) -> bool:
    """Best-effort task check that works for predictors and metrics alike."""
    if hasattr(obj, "supports"):
        return bool(obj.supports(task))
    tasks: Sequence[str] = getattr(obj, "tasks", ())
    return task in tasks
