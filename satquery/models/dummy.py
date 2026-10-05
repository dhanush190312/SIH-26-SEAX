"""Predictor test doubles. No torch, no GPU, no network.

These exist because an eval harness you cannot trust is worse than no harness:

* :class:`OraclePredictor` returns the ground truth. Every metric must score a
  perfect value on it. If BLEU or acc@0.5 does not hit its ceiling here, the
  *metric* is broken, not the model -- and that is a bug you want to find on the
  laptop, not after a 5-hour Colab run.
* :class:`ConstantPredictor` answers the same thing every time. This is the
  majority-class floor: RSVQA presence questions are ~yes-heavy, so "always
  yes" is the number a real model has to beat before anyone gets excited.
* :class:`DummyPredictor` is deterministic pseudo-random. It exercises the
  plumbing (schemas, per-type slicing, failure paths, report rendering) with
  scores that are neither 0 nor 1.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Mapping, Sequence

from ..contracts import Box, Prediction, Sample, Task, TASKS
from ..text import extract_answer

__all__ = ["DummyPredictor", "OraclePredictor", "ConstantPredictor"]

_FILLER_WORDS = (
    "a residential area with buildings and roads",
    "an agricultural field next to a river",
    "a harbour with several ships and storage tanks",
    "bare land partially covered by vegetation",
    "an urban block with a large parking lot",
)


def _seed_for(sample_id: str, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class _BasePredictor:
    """Shared bookkeeping: name, task support, uniform ``predict`` wrapper."""

    name = "base"
    supported: tuple[Task, ...] = TASKS

    def supports(self, task: Task) -> bool:
        return task in self.supported

    def predict(self, sample: Sample) -> Prediction:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. No-op for the doubles; real adapters free VRAM."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "tasks": list(self.supported), "kind": "test-double"}


class OraclePredictor(_BasePredictor):
    """Returns ground truth. Used to validate metrics, never to report scores.

    ``noise`` deliberately corrupts a fraction of predictions (deterministically,
    keyed by ``sample_id``) so tests can assert a metric *moves* in the right
    direction: ``OraclePredictor(noise=0.5)`` should land near 0.5 accuracy.
    """

    name = "oracle"

    def __init__(self, noise: float = 0.0, seed: int = 1337) -> None:
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be in [0, 1]")
        self.noise = noise
        self.seed = seed

    def _corrupt(self, sample: Sample) -> bool:
        if self.noise <= 0.0:
            return False
        rng = random.Random(_seed_for(sample.sample_id, f"oracle-noise:{self.seed}"))
        return rng.random() < self.noise

    def predict(self, sample: Sample) -> Prediction:
        corrupt = self._corrupt(sample)
        text: str | None = None
        boxes: tuple[Box, ...] = ()
        labels: tuple[str, ...] = ()
        label_scores: dict[str, float] = {}
        evidence: dict[str, Any] = {}

        if sample.task in ("vqa", "change_vqa"):
            text = sample.answer or (sample.references[0] if sample.references else "")
            if corrupt:
                text = "__wrong__"
        elif sample.task in ("caption", "change_caption"):
            text = sample.references[0] if sample.references else ""
            if corrupt:
                text = "completely unrelated sentence about nothing"
        elif sample.task == "ground":
            boxes = tuple(Box(b.x0, b.y0, b.x1, b.y1, b.label, 1.0) for b in sample.boxes)
            if corrupt and boxes:
                # shift every box far enough to drop IoU below 0.5
                boxes = tuple(
                    Box(b.x0 + 10_000, b.y0 + 10_000, b.x1 + 10_000, b.y1 + 10_000, b.label, 1.0)
                    for b in boxes
                )
            text = "; ".join(f"{b.label or 'object'} {b.as_xyxy()}" for b in boxes) or None
        elif sample.task == "fusion_multilabel":
            labels = tuple(sample.labels)
            if corrupt:
                labels = ()
            label_scores = {lab: 1.0 for lab in labels}
            text = ", ".join(labels) or None
        elif sample.task == "change_map":
            mask_path = sample.meta.get("mask_path")
            if mask_path and not corrupt:
                evidence["mask_path"] = str(mask_path)
            text = sample.references[0] if sample.references else None

        return Prediction(
            sample_id=sample.sample_id,
            task=sample.task,
            text=text,
            boxes=boxes,
            labels=labels,
            label_scores=label_scores,
            confidence=0.5 if corrupt else 1.0,
            latency_ms=0.0,
            evidence=evidence,
            model=self.name,
        )


class ConstantPredictor(_BasePredictor):
    """Majority-class / trivial baseline. Answers the same thing every time."""

    name = "constant"

    def __init__(
        self,
        text: str = "yes",
        labels: Sequence[str] = (),
        boxes: Sequence[Box] = (),
    ) -> None:
        self.text = text
        self.labels = tuple(labels)
        self.boxes = tuple(boxes)

    def predict(self, sample: Sample) -> Prediction:
        return Prediction(
            sample_id=sample.sample_id,
            task=sample.task,
            text=self.text,
            boxes=self.boxes,
            labels=self.labels,
            label_scores={lab: 1.0 for lab in self.labels},
            confidence=1.0,
            latency_ms=0.0,
            model=self.name,
        )


class DummyPredictor(_BasePredictor):
    """Deterministic pseudo-random predictor for plumbing tests.

    Given the same ``sample_id`` and ``seed`` it always returns the same
    prediction, so runs are reproducible and diffable. ``accuracy`` biases how
    often it copies the ground truth, which makes report rendering realistic.
    ``fail_rate`` injects predictor errors so the harness's failure accounting
    gets exercised too.
    """

    name = "dummy"

    def __init__(self, seed: int = 1337, accuracy: float = 0.5, fail_rate: float = 0.0) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if not 0.0 <= fail_rate <= 1.0:
            raise ValueError("fail_rate must be in [0, 1]")
        self.seed = seed
        self.accuracy = accuracy
        self.fail_rate = fail_rate

    def predict(self, sample: Sample) -> Prediction:
        rng = random.Random(_seed_for(sample.sample_id, f"dummy:{self.seed}"))
        if rng.random() < self.fail_rate:
            return Prediction.failed(sample.sample_id, sample.task, "dummy failure", self.name)

        be_right = rng.random() < self.accuracy
        text: str | None = None
        boxes: tuple[Box, ...] = ()
        labels: tuple[str, ...] = ()
        evidence: dict[str, Any] = {"note": "dummy predictor, no pixels were read"}

        if sample.task in ("vqa", "change_vqa"):
            truth = sample.answer or (sample.references[0] if sample.references else "yes")
            text = truth if be_right else rng.choice(["no", "yes", "3", "unknown"])
        elif sample.task in ("caption", "change_caption"):
            ref = sample.references[0] if sample.references else "an image"
            text = ref if be_right else rng.choice(_FILLER_WORDS)
        elif sample.task == "ground":
            if sample.boxes and be_right:
                jitter = 2.0
                boxes = tuple(
                    Box(
                        b.x0 + rng.uniform(-jitter, jitter),
                        b.y0 + rng.uniform(-jitter, jitter),
                        b.x1 + rng.uniform(-jitter, jitter),
                        b.y1 + rng.uniform(-jitter, jitter),
                        b.label,
                        round(rng.uniform(0.5, 0.99), 3),
                    )
                    for b in sample.boxes
                )
            else:
                w = float(sample.images[0].width or 512)
                h = float(sample.images[0].height or 512)
                x0, y0 = rng.uniform(0, w * 0.6), rng.uniform(0, h * 0.6)
                boxes = (
                    Box(x0, y0, x0 + w * 0.2, y0 + h * 0.2, None, round(rng.uniform(0.2, 0.8), 3)),
                )
        elif sample.task == "fusion_multilabel":
            pool = list(sample.labels) or ["urban", "water", "forest"]
            if be_right:
                labels = tuple(pool)
            else:
                k = max(1, len(pool) // 2)
                labels = tuple(rng.sample(pool, k)) if pool else ()
            text = ", ".join(labels) or None
        elif sample.task == "change_map":
            mask_path = sample.meta.get("mask_path")
            if mask_path and be_right:
                evidence["mask_path"] = str(mask_path)
            evidence["changed_area_pct"] = round(rng.uniform(0.5, 25.0), 2)
            text = "some buildings appear in the second image"

        return Prediction(
            sample_id=sample.sample_id,
            task=sample.task,
            text=text,
            boxes=boxes,
            labels=labels,
            label_scores={lab: round(rng.uniform(0.5, 1.0), 3) for lab in labels},
            confidence=round(rng.uniform(0.3, 0.95), 3),
            latency_ms=round(rng.uniform(5.0, 50.0), 2),
            evidence=evidence,
            model=self.name,
        )


def _sanity_extract(text: str) -> str:
    """Kept so the doubles and the real adapters share answer post-processing."""
    return extract_answer(text)
