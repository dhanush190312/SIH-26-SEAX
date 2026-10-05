"""Base class + parsing helpers shared by every dataset adapter.

An adapter's whole job is: find the annotation files under ``self.root``, and
yield :class:`~satquery.contracts.Sample` objects. Everything else --
availability checks, deterministic subsampling, caching, ``len()`` -- lives
here so the ten adapters cannot drift apart.

Two rules for adapters:

1. **Never** raise on a missing dataset at import or construction time. Report
   it through :meth:`available`; the harness records "skipped" and moves on.
   Benchmarks live on Colab/Kaggle, not on the dev laptop.
2. Layout knowledge that has *not* been checked against a real download must be
   marked with a ``# VERIFY:`` comment and listed in the adapter's docs page, so
   nobody mistakes a guess for a fact.
"""

from __future__ import annotations

import abc
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..config import SatQueryConfig, get_config
from ..contracts import DatasetNotAvailable, ImageRef, Sample, Task
from ..log import get_logger

__all__ = [
    "BaseDataset",
    "read_json",
    "read_jsonl",
    "read_csv_dicts",
    "first_existing",
    "as_tuple",
]

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Small IO helpers (stdlib only: pandas is an optional extra)
# --------------------------------------------------------------------------- #


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_csv_dicts(path: str | Path) -> Iterator[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def first_existing(*candidates: str | Path) -> Path | None:
    """First path that exists, or ``None``. Datasets ship inconsistent names."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def as_tuple(value: Any) -> tuple[str, ...]:
    """Coerce ``None`` / str / iterable-of-str into a tuple of non-empty strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(v) for v in value if v is not None and str(v) != "")


# --------------------------------------------------------------------------- #
# BaseDataset
# --------------------------------------------------------------------------- #


class BaseDataset(abc.ABC):
    """Iterable of :class:`Sample` for one (dataset, split, task).

    Subclasses set the class attributes and implement :meth:`_load`.
    """

    #: registry key, e.g. ``"rsvqa_lr"``
    name: str = ""
    #: the single task this adapter produces
    task: Task = "vqa"
    #: splits the adapter understands (for a clear error message)
    splits: tuple[str, ...] = ("train", "val", "test")
    #: paths relative to :attr:`root` that must exist for the dataset to be usable
    required_paths: tuple[str, ...] = ()
    #: where to get it, shown in the DatasetNotAvailable message
    download_hint: str = ""
    #: canonical citation / URL, carried into the report
    homepage: str = ""

    def __init__(
        self,
        split: str | None = None,
        *,
        root: str | Path | None = None,
        limit: int | None = None,
        seed: int | None = None,
        config: SatQueryConfig | None = None,
        **options: Any,
    ) -> None:
        self.config = config or get_config()
        self.split = split or self.splits[-1]
        if self.split not in self.splits:
            raise ValueError(
                f"{self.name}: unknown split {self.split!r}, expected one of {self.splits}"
            )
        self._root = Path(root) if root is not None else self.config.dataset_root(self.name)
        self.limit = limit
        self.seed = self.config.seed if seed is None else seed
        self.options: dict[str, Any] = dict(options)
        self._samples: list[Sample] | None = None

    # -- location ------------------------------------------------------------ #

    @property
    def root(self) -> Path:
        return self._root

    def path(self, *parts: str) -> Path:
        return self._root.joinpath(*parts)

    def missing_paths(self) -> list[str]:
        return [p for p in self.required_paths if not self.path(p).exists()]

    def available(self) -> bool:
        if not self._root.exists():
            return False
        return not self.missing_paths()

    def require_available(self) -> None:
        if self.available():
            return
        missing = self.missing_paths() or ["<root>"]
        hint = f"\n  download: {self.download_hint}" if self.download_hint else ""
        raise DatasetNotAvailable(
            f"{self.name} ({self.split}) not found under {self._root}."
            f"\n  missing: {', '.join(missing)}"
            f"\n  point it elsewhere with SATQUERY_DATA_{self.name.upper()}=<dir>{hint}"
        )

    # -- loading ------------------------------------------------------------- #

    @abc.abstractmethod
    def _load(self) -> Iterable[Sample]:
        """Parse annotations and yield samples. Called once, lazily."""

    def samples(self) -> list[Sample]:
        """All samples for this split, after deterministic subsampling."""
        if self._samples is None:
            self.require_available()
            loaded = list(self._load())
            self._validate(loaded)
            self._samples = self._subsample(loaded)
            log.debug("%s/%s: %d samples", self.name, self.split, len(self._samples))
        return self._samples

    def _validate(self, samples: Sequence[Sample]) -> None:
        if not samples:
            log.warning("%s/%s: parsed 0 samples -- check the layout", self.name, self.split)
            return
        seen: set[str] = set()
        for s in samples:
            if s.task != self.task:
                raise ValueError(
                    f"{self.name}: sample {s.sample_id} has task {s.task!r}, "
                    f"adapter declares {self.task!r}"
                )
            if s.sample_id in seen:
                raise ValueError(f"{self.name}: duplicate sample_id {s.sample_id!r}")
            seen.add(s.sample_id)

    def _subsample(self, samples: list[Sample]) -> list[Sample]:
        """Deterministic subsample.

        Seeded shuffle then head, so ``--limit 200`` is a random-but-reproducible
        slice rather than the first 200 rows of a file that is usually sorted by
        image and would otherwise cover one scene only.
        """
        if self.limit is None or self.limit >= len(samples):
            return samples
        rng = random.Random(f"{self.name}:{self.split}:{self.seed}")
        picked = rng.sample(range(len(samples)), self.limit)
        picked.sort()
        return [samples[i] for i in picked]

    # -- python protocol ----------------------------------------------------- #

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples())

    def __len__(self) -> int:
        return len(self.samples())

    def __repr__(self) -> str:
        state = "ok" if self.available() else "MISSING"
        return f"<{type(self).__name__} {self.name}/{self.split} task={self.task} {state} root={self._root}>"

    # -- reporting ----------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        """Metadata for the report / ``satquery datasets`` listing."""
        return {
            "name": self.name,
            "split": self.split,
            "task": self.task,
            "root": str(self._root),
            "available": self.available(),
            "missing": self.missing_paths(),
            "homepage": self.homepage,
            "download_hint": self.download_hint,
        }

    # -- helpers for subclasses --------------------------------------------- #

    def image_ref(
        self,
        image_id: str,
        path: str | Path,
        *,
        modality: str = "unknown",
        timestamp: str | None = None,
        **meta: Any,
    ) -> ImageRef:
        """Build an :class:`ImageRef` with the modality declared by the dataset.

        No file probing here -- construction must stay cheap enough to build
        100k samples; sizes are filled in by the io layer when an image is
        actually read.
        """
        return ImageRef(
            image_id=image_id,
            path=str(path),
            modality=modality,  # type: ignore[arg-type]
            timestamp=timestamp,
            meta=dict(meta),
        )

    def make_sample(self, sample_id: str, **kwargs: Any) -> Sample:
        """Build a :class:`Sample` pre-filled with this adapter's identity."""
        kwargs.setdefault("task", self.task)
        kwargs.setdefault("dataset", self.name)
        kwargs.setdefault("split", self.split)
        meta: dict[str, Any] = dict(kwargs.pop("meta", {}) or {})
        return Sample(sample_id=sample_id, meta=meta, **kwargs)
