"""Referring-expression grounding metrics: acc@IoU, mean IoU, precision/recall.

Two scoring conventions live side by side here because the literature reports
both and conflating them silently inflates a number by 10-20 points:

* **Top-1 accuracy** (``acc@0.5`` / ``acc@0.7``) -- the VRSBench / DIOR-RSVG
  referring convention. Exactly one prediction is judged, the highest-ranked
  one, and it may hit *any* ground-truth box. Multi-instance ground truth is
  therefore generous on purpose: "the ships" with three boxes counts as located
  when the model boxes any single ship.
* **Detection-style precision/recall** (``precision@t`` / ``recall@t``) from
  greedy score-ordered one-to-one matching, aggregated corpus-wide (micro), not
  averaged per sample. This is the number that punishes a model for spraying
  boxes or for finding one ship out of three -- top-1 accuracy cannot see either
  failure, and multi-instance referring expressions are the norm in RS.

``mean_iou`` is deliberately the *top-1* box's best IoU against any ground-truth
box, so the identity ``acc@t == mean(iou_i >= t)`` holds exactly. A "best IoU
over all predictions" definition would break that link and would rise merely by
predicting more boxes.

Ranking rule for ``score is None``: scored boxes come first in descending score,
unscored boxes come last in the order the model emitted them. Qwen2.5-VL's
native grounding returns boxes with no confidence at all, and that emission
order is the only ranking signal it gives us -- discarding those boxes would
score the fallback grounder at zero for a formatting reason. For the same reason
``score_threshold`` only filters boxes that *have* a score; an unscored box is
never dropped by it.
"""

from __future__ import annotations

from typing import Sequence

from ..contracts import Box, Prediction, Sample, Task, UnsupportedTask

__all__ = ["GroundingMetrics"]


def _fmt_threshold(t: float) -> str:
    """``0.5 -> '0.5'``, ``0.75 -> '0.75'``, ``1.0 -> '1.0'``.

    ``%g`` alone would emit ``acc@1``, which sorts and reads badly next to
    ``acc@0.5`` in a report table, so a bare integer gets its ``.0`` back. The
    keys are part of the on-disk run JSON, so this formatting is API.
    """
    key = f"{t:g}"
    return key if "." in key else f"{key}.0"


def _ranked(boxes: Sequence[Box]) -> list[Box]:
    """Score-descending, unscored last, otherwise emission order preserved.

    ``sorted`` is stable, so equal scores -- and the whole unscored group -- keep
    the order the predictor produced them in.
    """
    return sorted(
        boxes,
        key=lambda b: (b.score is None, -(b.score if b.score is not None else 0.0)),
    )


def _match_one_to_one(
    predictions: Sequence[Box], truth: Sequence[Box], threshold: float
) -> tuple[int, int]:
    """Greedy score-ordered matching -> ``(n_correct_pred, n_matched_truth)``.

    ``predictions`` must already be ranked. Each prediction claims the
    highest-IoU *unclaimed* ground-truth box; if that IoU is below the threshold
    the prediction is a false positive and the box stays available for a
    lower-ranked prediction. Same rule COCO uses, so the numbers stay comparable
    to published detection scores. Matching is one-to-one, so the two returned
    counts are equal -- they are returned separately only because
    :func:`_match_coverage` breaks that equality.
    """
    claimed = [False] * len(truth)
    tp = 0
    for pred in predictions:
        best_j, best_iou = -1, 0.0
        for j, gt in enumerate(truth):
            if claimed[j]:
                continue
            iou = pred.iou(gt)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0 and best_iou >= threshold:
            claimed[best_j] = True
            tp += 1
    return tp, tp


def _match_coverage(
    predictions: Sequence[Box], truth: Sequence[Box], threshold: float
) -> tuple[int, int]:
    """Relaxed matching (``greedy_match=False``).

    Ground-truth boxes are reusable: a prediction is correct if it hits any box,
    and recall becomes the fraction of ground-truth boxes covered by at least
    one prediction -- which is why the truth-side count is tracked separately
    from the prediction-side one. Use this when duplicate boxes are an artefact
    of the decoder rather than real over-detection; it is not comparable to
    published detection numbers.
    """
    covered = [False] * len(truth)
    tp = 0
    for pred in predictions:
        hit = False
        for j, gt in enumerate(truth):
            if pred.iou(gt) >= threshold:
                covered[j] = True
                hit = True
        tp += int(hit)
    return tp, covered.count(True)


def _ratio(num: float, den: float) -> float:
    """Zero, not NaN, for an empty denominator -- reports have to render it."""
    return 0.0 if den <= 0 else num / den


class GroundingMetrics:
    """Streaming grounding metric for the ``ground`` task.

    ``thresholds`` are IoU cut-offs in ``(0, 1]``; each one produces ``acc@<t>``,
    ``precision@<t>`` and ``recall@<t>``. ``score_threshold`` drops
    low-confidence *scored* boxes before anything is measured -- raising it
    trades recall for precision without re-running the model.
    """

    name = "grounding"
    tasks: tuple[Task, ...] = ("ground",)

    def __init__(
        self,
        *,
        thresholds: Sequence[float] = (0.5, 0.7),
        score_threshold: float = 0.0,
        greedy_match: bool = True,
    ) -> None:
        cleaned = sorted({float(t) for t in thresholds})
        if not cleaned:
            raise ValueError("thresholds must contain at least one IoU cut-off")
        for t in cleaned:
            if not 0.0 < t <= 1.0:
                raise ValueError(f"IoU threshold {t!r} out of range, expected (0, 1]")
        self.thresholds: tuple[float, ...] = tuple(cleaned)
        self.score_threshold = float(score_threshold)
        self.greedy_match = bool(greedy_match)
        self.reset()

    # -- streaming API ------------------------------------------------------- #

    def reset(self) -> None:
        self._n = 0
        self._n_empty = 0
        self._n_skipped = 0
        self._iou_sum = 0.0
        self._n_pred = 0
        self._n_truth = 0
        self._hits: dict[float, int] = {t: 0 for t in self.thresholds}
        self._correct: dict[float, int] = {t: 0 for t in self.thresholds}
        self._matched: dict[float, int] = {t: 0 for t in self.thresholds}

    def update(self, sample: Sample, prediction: Prediction) -> None:
        """Score one sample.

        A sample with no ground-truth boxes is *skipped*, not counted as a miss:
        IoU is undefined without a target, and charging the model for a dataset
        defect hides the defect. A failed prediction is the opposite case -- a
        genuine miss, scored as an empty prediction; the harness counts the
        failure itself in ``n_failed``.
        """
        if sample.task not in self.tasks:
            raise UnsupportedTask(
                f"{self.name} scores {self.tasks}, got task {sample.task!r} "
                f"for sample {sample.sample_id!r}"
            )
        truth = tuple(sample.boxes)
        if not truth:
            self._n_skipped += 1
            return

        boxes = tuple(prediction.boxes) if prediction.ok else ()
        kept = _ranked([b for b in boxes if b.score is None or b.score >= self.score_threshold])

        self._n += 1
        self._n_pred += len(kept)
        self._n_truth += len(truth)
        if not kept:
            self._n_empty += 1

        top_iou = max((kept[0].iou(gt) for gt in truth), default=0.0) if kept else 0.0
        self._iou_sum += top_iou

        match = _match_one_to_one if self.greedy_match else _match_coverage
        for t in self.thresholds:
            if top_iou >= t:
                self._hits[t] += 1
            correct, matched = match(kept, truth, t)
            self._correct[t] += correct
            self._matched[t] += matched

    def compute(self) -> dict[str, float]:
        """Flat metric dict. Every key is always present, zero-filled when empty.

        Precision and recall are micro-averaged: box counts are pooled over the
        corpus before dividing, so a sample with eight ships weighs eight times
        a sample with one. Per-sample averaging would let single-instance
        expressions dominate the number.
        """
        n = float(self._n)
        out: dict[str, float] = {}
        for t in self.thresholds:
            out[f"acc@{_fmt_threshold(t)}"] = _ratio(self._hits[t], n)
        out["mean_iou"] = _ratio(self._iou_sum, n)
        for t in self.thresholds:
            key = _fmt_threshold(t)
            out[f"precision@{key}"] = _ratio(self._correct[t], self._n_pred)
            out[f"recall@{key}"] = _ratio(self._matched[t], self._n_truth)
        out["n"] = n
        out["empty_pred_pct"] = 100.0 * _ratio(self._n_empty, n)
        out["n_skipped"] = float(self._n_skipped)
        return out
