"""Multi-label (optical-SAR fusion) and binary change-mask metrics.

Both classes here score a *label vector* or a *pixel mask* rather than a text
answer, so both had to pick conventions the literature reports inconsistently.
The choices are written down because a silently different convention is worth
5-20 points and makes a baseline incomparable.

**MultiLabelMetrics** scores the fusion slot against BigEarthNet-MM's 19-class
nomenclature.

* ``precision`` / ``recall`` / ``f1`` are **micro**: label decisions are pooled
  over the corpus and then divided, so a patch with six labels weighs six times
  a patch with one. They are also emitted as ``micro_*``; the bare keys exist
  because the report renders a bare key as the headline and ``f1/<label>`` as
  its breakdown table, and an orphaned breakdown renders badly.
* ``macro_*`` and ``mAP`` average over **labels with non-zero ground-truth
  support only**. BigEarthNet class frequencies span three orders of magnitude
  ("Marine waters" against "Beaches, dunes, sands"), so on any subsampled split
  some classes have no positive example at all; folding a defined-as-zero score
  for them into the average would make the macro number a function of
  ``--limit`` rather than of the model. On the full val split every class has
  support and this reduces to the ordinary macro average. ``f1/<label>`` is
  still emitted for every label in the vocabulary, zero-support ones included,
  so the exclusion hides nothing -- and ``n_labels_scored`` says how many
  labels the average actually covered.
* Average precision is the **all-points** AP (sklearn's
  ``average_precision_score``, not PASCAL's 11-point interpolation) with **ties
  pooled**: equally scored samples collapse to one point on the PR curve. That
  tie rule is the entire reason a model emitting one constant score gets
  AP = prevalence instead of a flattering number decided by how ``argsort``
  happened to break ties.
* Which labels count as predicted: ``prediction.labels`` when the predictor
  committed to a set, otherwise ``label_scores >= threshold``. Scores drive AP
  either way; a predictor that returns no scores gets a degenerate two-level
  ranking, which is honest -- binary output carries no ranking information and
  its AP should say so.

**ChangeMaskMetrics** scores the change-map slot. ``iou`` / ``f1`` /
``precision`` / ``recall`` are micro over pooled pixels, which is what the
LEVIR-CD change-detection literature reports; ``mean_iou`` is the per-sample
average kept beside it because the two diverge sharply on datasets containing
"no change" pairs -- an empty prediction against an empty truth is a per-sample
IoU of 1.0 and contributes nothing at all to the pooled number.

LEVIR-MCI ships **multi-class** masks (road / building). Everything here
binarises them as "any change", so these are map-level change metrics, not
per-class ones: a model that finds exactly the right pixels but calls a road a
building still scores 1.0. Per-class scoring is a different metric and is out
of Phase 1 scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..contracts import Prediction, Sample, SatQueryError, Task, UnsupportedTask
from ..log import get_logger
from ..text import normalize_answer

__all__ = ["MultiLabelMetrics", "ChangeMaskMetrics", "load_mask", "SKIP_REASONS"]

log = get_logger(__name__)


def _ratio(num: float, den: float) -> float:
    """Zero, not NaN, for an empty denominator -- reports have to render it."""
    return 0.0 if den <= 0 else float(num) / float(den)


def _f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 0.0 if total <= 0 else 2.0 * precision * recall / total


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values)) / float(len(values))


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def _canonical(label: Any) -> str:
    """Fold spelling differences that are not label differences.

    Land-cover class names arrive as ``"Urban fabric"`` from an annotation file
    and ``"urban_fabric"`` from a model decode; scoring those as two classes
    reports a working model at zero for a formatting reason. Number-word folding
    and article dropping are switched off -- they are answer-string tricks and
    have no business rewriting a controlled vocabulary.
    """
    return normalize_answer(str(label), map_numbers=False, drop_articles=False, unify_yes_no=False)


def _slug(canonical: str) -> str:
    """Metric-key form: ``"inland waters"`` -> ``"inland_waters"``.

    Metric keys land in run JSON and report tables, so they stay free of spaces
    and punctuation. Two source spellings of one class collapse to one key --
    which is the point.
    """
    return canonical.replace(" ", "_")


def _label_set(labels: Iterable[Any]) -> set[str]:
    return {c for c in (_canonical(x) for x in labels) if c}


def _average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    """All-points average precision for one label, ties pooled.

    Equivalent to ``sklearn.metrics.average_precision_score``: the sum of
    ``(R_k - R_{k-1}) * P_k`` over the score-ranked list, where every run of
    equal scores collapses into a single point. Without the tie pooling, N
    samples that all scored 0.5 would be ranked by array order and a label could
    score AP = 1.0 purely because the positives came first in the file.

    ``truth`` must contain at least one positive; callers exclude zero-support
    labels rather than inventing a value for them.
    """
    n_pos = float(truth.sum())
    if n_pos <= 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    y = truth[order].astype(np.float64)
    s = scores[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    # Keep only the last index of each run of equal scores.
    last_of_run = np.concatenate([s[1:] != s[:-1], np.array([True])])
    tp, fp = tp[last_of_run], fp[last_of_run]
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    delta = np.diff(np.concatenate([np.array([0.0]), recall]))
    return float(np.sum(delta * precision))


class MultiLabelMetrics:
    """Streaming multi-label metric for the ``fusion_multilabel`` task.

    ``labels`` declares the vocabulary. Pass ``BEN19_LABELS`` to score against a
    fixed nomenclature: every class then gets a stable ``f1/<label>`` key
    whether or not it appeared in the split, and any label outside the
    vocabulary is dropped from *both* sides and counted in ``n_out_of_vocab``
    -- a non-zero count there is nearly always a spelling mismatch, not a model
    inventing classes, and it is the first thing to check when fusion scores
    zero. Left at ``None`` the vocabulary is the sorted union of every label
    seen, which is what lets the metric run on a synthetic fixture without
    importing a dataset adapter.

    ``threshold`` binarises ``label_scores`` and is consulted only when the
    prediction carries scores but no explicit ``labels``.

    One record is kept per sample rather than folded into counters, because
    average precision needs the whole score ranking. That is a few hundred bytes
    a sample -- fine for a benchmark split, not for an unbounded stream.
    """

    name = "multilabel"
    tasks: tuple[Task, ...] = ("fusion_multilabel",)

    def __init__(
        self,
        *,
        labels: Sequence[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        if labels is None:
            self.labels: tuple[str, ...] | None = None
            self._fixed: tuple[str, ...] | None = None
        else:
            canon: list[str] = []
            for lab in labels:
                c = _canonical(lab)
                if c and c not in canon:
                    canon.append(c)
            if not canon:
                raise ValueError("labels was given but contains no usable label names")
            self.labels = tuple(str(x) for x in labels)
            self._fixed = tuple(canon)
        self.threshold = float(threshold)
        self.reset()

    # -- streaming API ------------------------------------------------------- #

    def reset(self) -> None:
        self._records: list[tuple[frozenset[str], frozenset[str], dict[str, float]]] = []
        self._n_oov = 0

    def update(self, sample: Sample, prediction: Prediction) -> None:
        """Score one sample.

        A sample with no ground-truth labels is *scored*, not skipped: an
        all-negative target is well defined for multi-label classification and
        predicting nothing for it is genuinely correct. (The grounding metric
        skips its empty-truth case because IoU has no meaning without a target;
        this metric does not have that problem.)

        A failed prediction is an empty prediction -- a real miss. The harness
        already counts the failure in ``n_failed``; treating it as a skip here
        too would quietly shrink the denominator and flatter the model.
        """
        if sample.task not in self.tasks:
            raise UnsupportedTask(
                f"{self.name} scores {self.tasks}, got task {sample.task!r} "
                f"for sample {sample.sample_id!r}"
            )

        truth = _label_set(sample.labels)
        scores: dict[str, float] = {}
        predicted: set[str] = set()
        if prediction.ok:
            for key, value in (prediction.label_scores or {}).items():
                canon = _canonical(key)
                if canon:
                    scores[canon] = float(value)
            declared = _label_set(prediction.labels)
            predicted = declared or {k for k, v in scores.items() if v >= self.threshold}

        if self._fixed is not None:
            allowed = set(self._fixed)
            self._n_oov += len(truth - allowed) + len(predicted - allowed)
            truth &= allowed
            predicted &= allowed
            scores = {k: v for k, v in scores.items() if k in allowed}

        self._records.append((frozenset(truth), frozenset(predicted), scores))

    # -- computation --------------------------------------------------------- #

    def _vocabulary(self) -> list[str]:
        if self._fixed is not None:
            return list(self._fixed)
        seen: set[str] = set()
        for truth, predicted, scores in self._records:
            seen |= truth
            seen |= predicted
            seen |= set(scores)
        return sorted(seen)

    def _score_vector(self, label: str) -> tuple[np.ndarray, np.ndarray]:
        """``(scores, truth)`` for one label over every recorded sample.

        Predictions without ``label_scores`` fall back to a 1.0/0.0 score, so a
        corpus where only some predictors emitted probabilities still ranks --
        the binary ones simply land in the two extreme tie groups.
        """
        n = len(self._records)
        scores = np.zeros(n, dtype=np.float64)
        truth = np.zeros(n, dtype=bool)
        for i, (gt, predicted, label_scores) in enumerate(self._records):
            truth[i] = label in gt
            if label_scores:
                scores[i] = label_scores.get(label, 0.0)
            else:
                scores[i] = 1.0 if label in predicted else 0.0
        return scores, truth

    def compute(self) -> dict[str, float]:
        """Flat metric dict. Every key is always present and never NaN.

        ``f1`` is the micro F1, repeated under both names on purpose: ``f1`` is
        the headline the report pairs with the ``f1/<label>`` breakdown,
        ``micro_f1`` is the unambiguous name to quote in a table next to
        ``macro_f1``.
        """
        vocab = self._vocabulary()
        n = len(self._records)
        index = {label: i for i, label in enumerate(vocab)}

        tp = np.zeros(len(vocab), dtype=np.float64)
        fp = np.zeros(len(vocab), dtype=np.float64)
        fn = np.zeros(len(vocab), dtype=np.float64)
        n_exact = 0
        for truth, predicted, _scores in self._records:
            if truth == predicted:
                n_exact += 1
            for label in predicted:
                i = index.get(label)
                if i is not None:
                    if label in truth:
                        tp[i] += 1.0
                    else:
                        fp[i] += 1.0
            for label in truth - predicted:
                i = index.get(label)
                if i is not None:
                    fn[i] += 1.0

        micro_p = _ratio(tp.sum(), tp.sum() + fp.sum())
        micro_r = _ratio(tp.sum(), tp.sum() + fn.sum())
        micro_f1 = _f1(micro_p, micro_r)

        per_p = [_ratio(tp[i], tp[i] + fp[i]) for i in range(len(vocab))]
        per_r = [_ratio(tp[i], tp[i] + fn[i]) for i in range(len(vocab))]
        per_f1 = [_f1(per_p[i], per_r[i]) for i in range(len(vocab))]
        # Support > 0. See the module docstring for why zero-support labels are
        # excluded from every average but still reported individually.
        kept = [i for i in range(len(vocab)) if (tp[i] + fn[i]) > 0]

        aps = [_average_precision(*self._score_vector(vocab[i])) for i in kept]

        out: dict[str, float] = {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_precision": _mean([per_p[i] for i in kept]),
            "macro_recall": _mean([per_r[i] for i in kept]),
            "macro_f1": _mean([per_f1[i] for i in kept]),
            "mAP": _mean(aps),
            "exact_match": _ratio(n_exact, n),
            "n": float(n),
            "n_labels": float(len(vocab)),
            "n_labels_scored": float(len(kept)),
            "n_out_of_vocab": float(self._n_oov),
        }
        for i, label in enumerate(vocab):
            out[f"f1/{_slug(label)}"] = per_f1[i]
        return out


# --------------------------------------------------------------------------- #
# Change masks
# --------------------------------------------------------------------------- #

#: Every reason a ``change_map`` sample can drop out of scoring. Emitted as
#: zero-filled ``n_skipped/<reason>`` keys so report columns stay stable across
#: runs and a systematic loading failure is visible instead of merely shrinking
#: ``n``.
SKIP_REASONS: tuple[str, ...] = (
    "no_ground_truth",
    "no_prediction",
    "prediction_failed",
    "load_error",
    "shape_mismatch",
)


class _MaskUnavailable(SatQueryError):
    """A mask could not be read. Caught by the metric and recorded as a skip."""


def _binarise(array: Any, threshold: float) -> np.ndarray:
    """Coerce any mask representation to boolean "changed / not changed".

    Three dtypes, three meanings, and getting them backwards is the classic
    silent failure here:

    * ``bool`` -- already a decision, pass through.
    * integer -- a *class index* map (LEVIR-MCI: 0 background, 1 road,
      2 building). Non-zero is change; thresholding it at 0.5 happens to work
      but at any value above 1 it would silently delete class 1.
    * float -- a probability or score map, so ``>= threshold``. NaN compares
      False and therefore reads as "no change", which is the safe direction.
    """
    a = np.asarray(array)
    if a.dtype == np.bool_:
        return a
    if np.issubdtype(a.dtype, np.integer):
        return a != 0
    return np.asarray(a, dtype=np.float64) >= threshold


def _squeeze_to_2d(mask: np.ndarray) -> np.ndarray:
    """Reduce a binarised mask of any layout to ``(h, w)``.

    Handles ``(1, h, w)`` band-first arrays and ``(h, w, 3)`` RGB PNGs alike by
    collapsing the shorter of the outer axes with ``any`` -- "changed in any
    channel", consistent with the any-change binarisation. The heuristic only
    turns ambiguous for a mask whose spatial size is at most its channel count,
    which no real change mask is.
    """
    a = mask
    while a.ndim > 2:
        axis = 0 if a.shape[0] <= a.shape[-1] else a.ndim - 1
        a = a.any(axis=axis)
    if a.ndim != 2:
        raise _MaskUnavailable(
            f"mask has shape {tuple(np.shape(mask))}, expected two spatial dimensions"
        )
    return a


def _read_mask_file(path: str | Path) -> np.ndarray:
    """Load a mask from disk: ``.npy`` via numpy, everything else via PIL.

    PIL is an optional extra, so a missing Pillow degrades to a skip carrying an
    actionable message instead of an ImportError halfway through a run -- masks
    can always be shipped as ``.npy``. GeoTIFF masks that would need rasterio
    fail down the same path with the same message shape.
    """
    p = Path(path)
    if not p.exists():
        raise _MaskUnavailable(f"mask file not found: {p}")
    if p.suffix.lower() == ".npy":
        try:
            return np.load(p, allow_pickle=False)
        except Exception as exc:  # corrupt, or a pickled object array
            raise _MaskUnavailable(f"could not read {p}: {exc}") from exc
    try:
        from PIL import Image  # optional extra
    except ImportError as exc:
        raise _MaskUnavailable(
            f"reading a {p.suffix or 'binary'} mask needs Pillow (pip install pillow), "
            f"or ship the mask as .npy: {p}"
        ) from exc
    try:
        with Image.open(p) as img:
            array = np.asarray(img)
    except Exception as exc:
        raise _MaskUnavailable(f"could not decode mask {p}: {exc}") from exc
    # An RGBA export carries an opaque alpha channel, and "changed in any
    # channel" would then read every pixel as changed -- recall 1.0, precision
    # equal to prevalence, and nothing in the numbers to say why. Alpha is
    # transparency, never a class, so it never votes.
    if array.ndim == 3 and array.shape[-1] in (2, 4):
        array = array[..., : array.shape[-1] - 1]
    return array


def load_mask(source: Any, *, positive_threshold: float = 0.5) -> np.ndarray:
    """Boolean ``(h, w)`` mask from an in-memory array or a path.

    Public because the change tool and the PDF report must cite a changed-area
    percentage computed under exactly the binarisation the metric scored; two
    copies of these rules would drift and the report would quote a number the
    metric never saw.

    Raises :class:`~satquery.contracts.SatQueryError` when the mask cannot be
    read; :class:`ChangeMaskMetrics` catches that and records a skip.
    """
    if isinstance(source, (str, Path)):
        source = _read_mask_file(source)
    if source is None:
        raise _MaskUnavailable("mask source is None")
    return _squeeze_to_2d(_binarise(source, positive_threshold))


class ChangeMaskMetrics:
    """Streaming binary change-mask metric for the ``change_map`` task.

    Ground truth comes from ``sample.meta["mask_path"]``, the prediction from
    ``prediction.evidence["mask"]`` (an array) or ``["mask_path"]``. A sample
    missing either side is skipped, never scored as zero: a caption-only
    predictor pointed at this metric must read ``n=0, n_skipped=N`` rather than
    a plausible-looking 0.0 that hides the fact nothing was measured. That is
    the opposite of the grounding metric's rule, deliberately -- there, an empty
    prediction is a real miss; here, an absent mask means the tool was never run.
    """

    name = "change_mask"
    tasks: tuple[Task, ...] = ("change_map",)

    def __init__(self, *, positive_threshold: float = 0.5) -> None:
        self.positive_threshold = float(positive_threshold)
        self.reset()

    # -- streaming API ------------------------------------------------------- #

    def reset(self) -> None:
        self._n = 0
        self._tp = 0.0
        self._fp = 0.0
        self._fn = 0.0
        self._ious: list[float] = []
        self._area_errors: list[float] = []
        self._skipped: dict[str, int] = {reason: 0 for reason in SKIP_REASONS}

    def _skip(self, reason: str, sample_id: str, detail: str) -> None:
        self._skipped[reason] += 1
        log.warning("change_mask: skipped %s (%s): %s", sample_id, reason, detail)

    def update(self, sample: Sample, prediction: Prediction) -> None:
        if sample.task not in self.tasks:
            raise UnsupportedTask(
                f"{self.name} scores {self.tasks}, got task {sample.task!r} "
                f"for sample {sample.sample_id!r}"
            )

        truth_path = sample.meta.get("mask_path")
        if not truth_path:
            self._skip("no_ground_truth", sample.sample_id, "sample.meta['mask_path'] is unset")
            return
        if not prediction.ok:
            self._skip("prediction_failed", sample.sample_id, str(prediction.error))
            return

        evidence: Mapping[str, Any] = prediction.evidence or {}
        source = evidence.get("mask")
        if source is None:
            source = evidence.get("mask_path")
        if source is None:
            self._skip(
                "no_prediction",
                sample.sample_id,
                "prediction.evidence has neither 'mask' nor 'mask_path'",
            )
            return

        try:
            truth = load_mask(truth_path, positive_threshold=self.positive_threshold)
            predicted = load_mask(source, positive_threshold=self.positive_threshold)
        except SatQueryError as exc:
            self._skip("load_error", sample.sample_id, str(exc))
            return

        if truth.shape != predicted.shape:
            self._skip(
                "shape_mismatch",
                sample.sample_id,
                f"prediction {predicted.shape} != ground truth {truth.shape}",
            )
            return

        tp = float(np.count_nonzero(truth & predicted))
        fp = float(np.count_nonzero(~truth & predicted))
        fn = float(np.count_nonzero(truth & ~predicted))
        total = float(truth.size)

        self._n += 1
        self._tp += tp
        self._fp += fp
        self._fn += fn
        union = tp + fp + fn
        # Empty prediction against empty truth is perfect agreement on "nothing
        # changed" -- LEVIR-CC/MCI pairs genuinely contain those -- so 0/0 is
        # 1.0 here rather than 0.0.
        self._ious.append(1.0 if union <= 0 else tp / union)
        self._area_errors.append(
            abs(100.0 * _ratio(tp + fp, total) - 100.0 * _ratio(tp + fn, total))
        )

    def compute(self) -> dict[str, float]:
        """Flat metric dict, zero-filled when nothing was scored.

        ``iou`` / ``f1`` / ``precision`` / ``recall`` are pooled over pixels;
        ``mean_iou`` is the per-sample average of the same quantity.

        ``over_detection_ratio`` is predicted-positive pixels over
        ground-truth-positive pixels: 1.0 is calibrated, above 1.0 means the model
        paints more change than exists, below 1.0 means it under-segments. It
        reads 0.0 -- not infinity -- when the corpus contains no changed pixels
        at all, because run JSON cannot carry an infinity; read ``n`` and
        ``recall`` before reading anything into it.
        """
        precision = _ratio(self._tp, self._tp + self._fp)
        recall = _ratio(self._tp, self._tp + self._fn)
        out: dict[str, float] = {
            "iou": _ratio(self._tp, self._tp + self._fp + self._fn),
            "f1": _f1(precision, recall),
            "precision": precision,
            "recall": recall,
            "mean_iou": _mean(self._ious),
            "over_detection_ratio": _ratio(self._tp + self._fp, self._tp + self._fn),
            "changed_area_mae_pct": _mean(self._area_errors),
            "n": float(self._n),
            "n_skipped": float(sum(self._skipped.values())),
        }
        for reason in SKIP_REASONS:
            out[f"n_skipped/{reason}"] = float(self._skipped[reason])
        return out
