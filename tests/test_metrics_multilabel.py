"""Tests for the fusion multi-label and change-mask metrics.

Every expected number in here is hand-computed in the test body or its
docstring rather than captured from a previous run, because the whole point of
these two metrics is that the *convention* is right -- a regression test that
merely pins whatever the code did last is worthless for that.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from satquery.contracts import Prediction, Sample, SatQueryError, UnsupportedTask
from satquery.metrics.multilabel import (
    SKIP_REASONS,
    ChangeMaskMetrics,
    MultiLabelMetrics,
    load_mask,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def fusion_prediction(
    sample_id: str,
    *,
    labels: tuple[str, ...] = (),
    label_scores: dict[str, float] | None = None,
    **kw: Any,
) -> Prediction:
    return Prediction(
        sample_id=sample_id,
        task="fusion_multilabel",
        labels=labels,
        label_scores=label_scores or {},
        **kw,
    )


def mask_prediction(sample_id: str, **evidence: Any) -> Prediction:
    return Prediction(sample_id=sample_id, task="change_map", evidence=evidence)


def assert_finite(metrics: dict[str, float]) -> None:
    """No NaN, no infinity, no non-float -- run JSON cannot carry any of them."""
    for key, value in metrics.items():
        assert isinstance(value, float), f"{key} is {type(value).__name__}, not float"
        assert np.isfinite(value), f"{key} is {value!r}"


@pytest.fixture
def captured_warnings():
    """Records ``satquery`` warnings.

    ``satquery.log`` sets ``propagate=False`` on the package logger, so pytest's
    ``caplog`` (which listens on the root logger) sees nothing. Attaching its
    handler directly to the package logger is the least invasive fix and does
    not touch the frozen logging setup.
    """
    records: list[logging.LogRecord] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("satquery.metrics.multilabel")
    handler = _Sink(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def write_mask(tmp_path: Path) -> Callable[..., Path]:
    """Write a mask as ``.npy`` or (via PIL) ``.png`` and return the path."""

    def _write(name: str, array: np.ndarray) -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".npy":
            np.save(p, array)
            return p
        from PIL import Image

        a = np.asarray(array)
        if a.dtype == np.bool_:
            a = a.astype(np.uint8) * 255
        Image.fromarray(a.astype(np.uint8)).save(p)
        return p

    return _write


def rows_mask(height: int, width: int, rows: slice) -> np.ndarray:
    """Boolean ``(h, w)`` mask with ``rows`` set -- easy pixel counting."""
    m = np.zeros((height, width), dtype=bool)
    m[rows] = True
    return m


# --------------------------------------------------------------------------- #
# MultiLabelMetrics -- micro / macro / per-label
# --------------------------------------------------------------------------- #


WORKED_TRUTH = (("water",), ("water",), ("forest",), ())
WORKED_SCORES = (
    {"water": 0.9, "forest": 0.2},
    {"water": 0.4, "forest": 0.1},
    {"water": 0.4, "forest": 0.8},
    {"water": 0.1, "forest": 0.3},
)


def _worked_example(metric: MultiLabelMetrics, make_sample) -> dict[str, float]:
    """Feed the four-sample corpus the module docstrings' numbers refer to."""
    for i, (truth, scores) in enumerate(zip(WORKED_TRUTH, WORKED_SCORES)):
        sample = make_sample(f"s{i}", task="fusion_multilabel", labels=truth)
        metric.update(sample, fusion_prediction(sample.sample_id, label_scores=scores))
    return metric.compute()


def test_worked_example_micro_macro_and_exact_match(make_sample, assert_metrics_close):
    """Hand-computed on the four-sample corpus, threshold 0.5.

    Thresholding the scores gives predictions ``{water}``, ``{}``, ``{forest}``,
    ``{}`` against truth ``{water}``, ``{water}``, ``{forest}``, ``{}``.

    * water:  tp=1 fp=0 fn=1 -> P=1.0  R=0.5  F1=2/3
    * forest: tp=1 fp=0 fn=0 -> P=1.0  R=1.0  F1=1.0
    * micro:  tp=2 fp=0 fn=1 -> P=1.0  R=2/3  F1=0.8
    * macro F1 = (2/3 + 1)/2 = 5/6
    * exact match on samples 0, 2, 3 -> 0.75 (an all-negative target that is
      predicted empty is genuinely correct, not a skip)
    """
    got = _worked_example(MultiLabelMetrics(), make_sample)
    assert_metrics_close(
        got,
        {
            "precision": 1.0,
            "recall": 2.0 / 3.0,
            "f1": 0.8,
            "micro_precision": 1.0,
            "micro_recall": 2.0 / 3.0,
            "micro_f1": 0.8,
            "macro_precision": 1.0,
            "macro_recall": 0.75,
            "macro_f1": 5.0 / 6.0,
            "exact_match": 0.75,
            "n": 4.0,
            "f1/water": 2.0 / 3.0,
            "f1/forest": 1.0,
        },
    )
    assert_finite(got)


def test_worked_example_map_hand_computed(make_sample, assert_metrics_close):
    """All-points AP per label, then averaged.

    water, truth ``[1, 1, 0, 0]``, scores ``[0.9, 0.4, 0.4, 0.1]``. Samples 1
    and 2 tie at 0.4 -- one positive, one negative -- so tie pooling collapses
    them into one PR point: 0.9(+) gives precision 1.0 at recall 0.5, the pooled
    0.4 group gives precision 2/3 at recall 1.0, and 0.1 adds no recall.
    AP = 0.5*1 + 0.5*(2/3) = 5/6. Without pooling this AP would depend on which
    side of the tie ``argsort`` happened to put first.

    forest, truth ``[0, 0, 1, 0]``, scores ``[0.2, 0.1, 0.8, 0.3]``. The single
    positive ranks first, so AP = 1.0.

    mAP = (5/6 + 1)/2 = 11/12.
    """
    got = _worked_example(MultiLabelMetrics(), make_sample)
    assert_metrics_close(got, {"mAP": 11.0 / 12.0, "n_labels_scored": 2.0})


def test_binary_labels_score_without_label_scores(make_sample, assert_metrics_close):
    """A predictor that commits to a label set but emits no probabilities.

    Two of three samples are exactly right, so micro P = R = F1 = 2/3, and AP
    degenerates to a two-level ranking. ``water``: truth ``[1, 1, 0]``, scores
    ``[1, 0, 0]``; the top group is one true positive (precision 1.0 at recall
    0.5) and the pooled zero group adds the second positive next to a negative
    (precision 2/3 at recall 1.0), so AP = 0.5*1 + 0.5*(2/3) = 5/6.
    ``forest``: truth ``[0, 0, 1]``, scores ``[0, 1, 1]`` -- its one positive is
    tied with a false positive at the top, so precision is 0.5 at recall 1.0 and
    AP = 0.5. Binary output is thus *penalised* rather than silently credited:
    neither label reaches the 1.0 a score-emitting predictor could earn for the
    same decisions.
    """
    metric = MultiLabelMetrics(labels=("water", "forest"))
    cases = ((("water",), ("water",)), (("water",), ("forest",)), (("forest",), ("forest",)))
    for i, (truth, predicted) in enumerate(cases):
        sample = make_sample(f"s{i}", task="fusion_multilabel", labels=truth)
        metric.update(sample, fusion_prediction(sample.sample_id, labels=predicted))
    got = metric.compute()
    assert_metrics_close(
        got,
        {
            "micro_precision": 2.0 / 3.0,
            "micro_recall": 2.0 / 3.0,
            "micro_f1": 2.0 / 3.0,
            "f1/water": 2.0 / 3.0,
            "f1/forest": 2.0 / 3.0,
            "mAP": (5.0 / 6.0 + 0.5) / 2.0,
            "exact_match": 2.0 / 3.0,
        },
    )
    assert_finite(got)


def test_explicit_labels_win_over_thresholded_scores(make_sample, assert_metrics_close):
    """A committed label set is the decision; scores only supply the ranking."""
    metric = MultiLabelMetrics(threshold=0.5)
    sample = make_sample("s0", task="fusion_multilabel", labels=("water",))
    metric.update(
        sample,
        fusion_prediction("s0", labels=("water",), label_scores={"water": 0.1, "forest": 0.99}),
    )
    got = metric.compute()
    assert_metrics_close(got, {"exact_match": 1.0, "micro_f1": 1.0})


def test_threshold_is_honoured(make_sample, assert_metrics_close):
    """Lowering the threshold turns a sub-threshold score into a prediction."""
    sample_labels = ("water",)
    scores = {"water": 0.3}
    strict = MultiLabelMetrics(threshold=0.5)
    loose = MultiLabelMetrics(threshold=0.25)
    for metric in (strict, loose):
        sample = make_sample("s0", task="fusion_multilabel", labels=sample_labels)
        metric.update(sample, fusion_prediction("s0", label_scores=scores))
    assert_metrics_close(strict.compute(), {"recall": 0.0, "exact_match": 0.0})
    assert_metrics_close(loose.compute(), {"recall": 1.0, "exact_match": 1.0})


def test_label_spelling_is_folded(make_sample, assert_metrics_close):
    """``"Urban fabric"`` from an annotation == ``"urban_fabric"`` from a decode.

    Scoring those as two classes reports a working model at zero for a
    formatting reason, which is the single most likely way this metric lies.
    """
    metric = MultiLabelMetrics(labels=("Urban fabric", "Marine waters"))
    sample = make_sample("s0", task="fusion_multilabel", labels=("Urban fabric",))
    metric.update(sample, fusion_prediction("s0", labels=("urban_fabric",)))
    got = metric.compute()
    assert_metrics_close(got, {"micro_f1": 1.0, "f1/urban_fabric": 1.0, "n_out_of_vocab": 0.0})
    assert "f1/marine_waters" in got


def test_out_of_vocabulary_labels_are_counted_not_scored(make_sample, assert_metrics_close):
    """Labels outside a declared vocabulary drop from both sides, visibly."""
    metric = MultiLabelMetrics(labels=("water",))
    sample = make_sample("s0", task="fusion_multilabel", labels=("water", "pasture"))
    metric.update(sample, fusion_prediction("s0", labels=("water", "moors")))
    got = metric.compute()
    assert_metrics_close(got, {"n_out_of_vocab": 2.0, "micro_f1": 1.0, "n_labels": 1.0})
    assert "f1/pasture" not in got


def test_free_vocabulary_is_the_union_of_everything_seen(make_sample):
    metric = MultiLabelMetrics()
    sample = make_sample("s0", task="fusion_multilabel", labels=("water",))
    metric.update(sample, fusion_prediction("s0", labels=("forest",)))
    got = metric.compute()
    assert got["n_labels"] == 2.0
    assert {"f1/water", "f1/forest"} <= set(got)


def test_failed_prediction_is_a_miss_not_a_skip(make_sample, assert_metrics_close):
    """An errored prediction must not shrink the denominator and flatter a model."""
    metric = MultiLabelMetrics()
    sample = make_sample("s0", task="fusion_multilabel", labels=("water",))
    metric.update(sample, Prediction.failed("s0", "fusion_multilabel", "OOM"))
    got = metric.compute()
    assert_metrics_close(got, {"n": 1.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0})
    assert_finite(got)


def test_wrong_task_is_rejected(make_sample):
    metric = MultiLabelMetrics()
    sample = make_sample("s0", task="vqa")
    with pytest.raises(UnsupportedTask):
        metric.update(sample, Prediction(sample_id="s0", task="vqa", text="yes"))


def test_reset_clears_state(make_sample):
    metric = MultiLabelMetrics()
    sample = make_sample("s0", task="fusion_multilabel", labels=("water",))
    metric.update(sample, fusion_prediction("s0", labels=("water",)))
    assert metric.compute()["n"] == 1.0
    metric.reset()
    assert metric.compute()["n"] == 0.0


def test_labels_must_contain_something_usable():
    with pytest.raises(ValueError):
        MultiLabelMetrics(labels=("", "   "))


# --------------------------------------------------------------------------- #
# MultiLabelMetrics -- the degenerate cases the spec calls out
# --------------------------------------------------------------------------- #


def test_no_samples_gives_zeros_not_nan():
    got = MultiLabelMetrics(labels=("water", "forest")).compute()
    assert_finite(got)
    assert got == pytest.approx(
        {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0,
            "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0,
            "mAP": 0.0, "exact_match": 0.0, "n": 0.0,
            "n_labels": 2.0, "n_labels_scored": 0.0, "n_out_of_vocab": 0.0,
            "f1/water": 0.0, "f1/forest": 0.0,
        }
    )


def test_zero_support_label_is_reported_but_excluded_from_averages(
    make_sample, assert_metrics_close
):
    """A class with no positive example must not drag the macro average down.

    ``snow`` never appears in the four-sample corpus. It still gets an
    ``f1/snow`` key -- the exclusion hides nothing -- but the macro average and
    mAP are unchanged from the two-label run, so they stay a function of the
    model rather than of ``--limit``.
    """
    got = _worked_example(MultiLabelMetrics(labels=("water", "forest", "snow")), make_sample)
    assert got["f1/snow"] == 0.0
    assert_metrics_close(
        got,
        {
            "n_labels": 3.0,
            "n_labels_scored": 2.0,
            "macro_f1": 5.0 / 6.0,
            "mAP": 11.0 / 12.0,
        },
    )
    assert_finite(got)


def test_all_equal_scores_give_prevalence_not_a_flattering_ap(make_sample):
    """Ties are pooled, so a constant-score model scores AP = prevalence.

    Without tie pooling the AP would be decided by whether ``argsort`` happened
    to put the two positives first -- 1.0 here, 0.58 with the same corpus
    shuffled. The ordering of the samples is therefore varied deliberately and
    both orders must give 0.5.
    """
    truths = (("water",), ("water",), (), ())
    for order in (truths, tuple(reversed(truths))):
        metric = MultiLabelMetrics(labels=("water",))
        for i, truth in enumerate(order):
            sample = make_sample(f"s{i}", task="fusion_multilabel", labels=truth)
            metric.update(sample, fusion_prediction(sample.sample_id, label_scores={"water": 0.5}))
        got = metric.compute()
        assert got["mAP"] == pytest.approx(0.5), f"order {order}"
        assert_finite(got)


def test_all_negative_corpus_gives_zeros_not_nan(make_sample):
    """Nothing has support, so nothing is averaged -- and nothing is NaN."""
    metric = MultiLabelMetrics(labels=("water",))
    for i in range(3):
        sample = make_sample(f"s{i}", task="fusion_multilabel", labels=())
        metric.update(sample, fusion_prediction(sample.sample_id, label_scores={"water": 0.1}))
    got = metric.compute()
    assert_finite(got)
    assert got["n_labels_scored"] == 0.0
    assert got["mAP"] == 0.0
    assert got["exact_match"] == 1.0  # predicting nothing for nothing is correct


def test_perfect_scores_give_map_one(make_sample, assert_metrics_close):
    metric = MultiLabelMetrics(labels=("water", "forest"))
    for i, truth in enumerate((("water",), ("forest",), ("water", "forest"))):
        scores = {lab: (0.9 if lab in truth else 0.1) for lab in ("water", "forest")}
        sample = make_sample(f"s{i}", task="fusion_multilabel", labels=truth)
        metric.update(sample, fusion_prediction(sample.sample_id, label_scores=scores))
    assert_metrics_close(metric.compute(), {"mAP": 1.0, "micro_f1": 1.0, "exact_match": 1.0})


# --------------------------------------------------------------------------- #
# ChangeMaskMetrics
# --------------------------------------------------------------------------- #


def test_perfect_mask(make_sample, write_mask, assert_metrics_close):
    truth = rows_mask(8, 8, slice(0, 4))
    path = write_mask("gt.npy", truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=truth.copy()))
    got = metric.compute()
    assert_metrics_close(
        got,
        {
            "iou": 1.0,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "mean_iou": 1.0,
            "over_detection_ratio": 1.0,
            "changed_area_mae_pct": 0.0,
            "n": 1.0,
            "n_skipped": 0.0,
        },
    )
    assert_finite(got)


def test_disjoint_masks(make_sample, write_mask, assert_metrics_close):
    """No overlap at all: IoU and F1 are 0, but the areas still match."""
    truth = rows_mask(4, 4, slice(0, 2))
    pred = rows_mask(4, 4, slice(2, 4))
    path = write_mask("gt.npy", truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=pred))
    got = metric.compute()
    assert_metrics_close(
        got,
        {
            "iou": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_iou": 0.0,
            "over_detection_ratio": 1.0,  # same area, entirely wrong place
            "changed_area_mae_pct": 0.0,
            "n": 1.0,
        },
    )
    assert_finite(got)


def test_partial_overlap_hand_computed(make_sample, write_mask, assert_metrics_close):
    """4x4 truth on rows 0-1 (8 px), prediction on rows 1-3 (12 px).

    tp = row 1 = 4, fp = rows 2-3 = 8, fn = row 0 = 4.

    * IoU       = 4 / 16       = 0.25
    * precision = 4 / 12       = 1/3
    * recall    = 4 / 8        = 0.5
    * F1        = 2*(1/3)*0.5 / (1/3 + 0.5) = 0.4
    * over-detection = 12 / 8  = 1.5
    * changed area: predicted 12/16 = 75%, truth 8/16 = 50% -> MAE 25 points
    """
    truth = rows_mask(4, 4, slice(0, 2))
    pred = rows_mask(4, 4, slice(1, 4))
    path = write_mask("gt.npy", truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=pred))
    got = metric.compute()
    assert_metrics_close(
        got,
        {
            "iou": 0.25,
            "precision": 1.0 / 3.0,
            "recall": 0.5,
            "f1": 0.4,
            "mean_iou": 0.25,
            "over_detection_ratio": 1.5,
            "changed_area_mae_pct": 25.0,
            "n": 1.0,
            "n_skipped": 0.0,
        },
    )
    assert_finite(got)


def test_oracle_mask_path_reaches_one(make_sample, write_mask, assert_metrics_close):
    """The oracle path: the prediction echoes ``sample.meta['mask_path']``.

    This is the harness's own upper bound. If it does not read 1.0, mask
    loading or binarisation is broken and every model number below it is
    meaningless -- so it is asserted separately from the in-memory case.
    """
    truth = np.zeros((6, 6), dtype=bool)
    truth[1:4, 2:5] = True
    path = write_mask("levir/gt.png", truth)
    metric = ChangeMaskMetrics()
    for i in range(3):
        sample = make_sample(f"s{i}", task="change_map", meta={"mask_path": str(path)})
        metric.update(sample, mask_prediction(sample.sample_id, mask_path=str(path)))
    got = metric.compute()
    assert_metrics_close(
        got,
        {"iou": 1.0, "f1": 1.0, "mean_iou": 1.0, "n": 3.0, "n_skipped": 0.0},
    )


def test_multiclass_mask_is_binarised_as_any_change(make_sample, write_mask, assert_metrics_close):
    """LEVIR-MCI class-index masks: 0 background, 1 road, 2 building.

    A prediction that calls a road a building still scores 1.0 -- these are
    map-level metrics, not per-class ones, and the test pins that on purpose so
    nobody later reads ``iou`` as class-aware.
    """
    truth = np.zeros((4, 4), dtype=np.uint8)
    truth[0, :] = 1  # road
    truth[1, :] = 2  # building
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, :] = 2  # classes swapped, pixels identical
    pred[1, :] = 1
    path = write_mask("gt.npy", truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=pred))
    assert_metrics_close(metric.compute(), {"iou": 1.0, "f1": 1.0, "n": 1.0})


def test_float_mask_uses_positive_threshold(make_sample, write_mask, assert_metrics_close):
    """A probability map is thresholded, and the threshold moves the score."""
    truth = rows_mask(4, 4, slice(0, 2))
    probs = np.zeros((4, 4), dtype=np.float32)
    probs[0, :] = 0.9
    probs[1, :] = 0.3  # only counted when the threshold drops below 0.3
    path = write_mask("gt.npy", truth)
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})

    strict = ChangeMaskMetrics(positive_threshold=0.5)
    strict.update(sample, mask_prediction("s0", mask=probs))
    assert_metrics_close(strict.compute(), {"iou": 0.5, "precision": 1.0, "recall": 0.5})

    loose = ChangeMaskMetrics(positive_threshold=0.25)
    loose.update(sample, mask_prediction("s0", mask=probs))
    assert_metrics_close(loose.compute(), {"iou": 1.0, "recall": 1.0})


def test_nan_pixels_read_as_no_change(make_sample, write_mask, assert_metrics_close):
    """NaN must not poison the whole mask; it reads as "not changed"."""
    truth = rows_mask(4, 4, slice(0, 2))
    probs = np.full((4, 4), np.nan, dtype=np.float32)
    probs[0:2, :] = 1.0
    path = write_mask("gt.npy", truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=probs))
    got = metric.compute()
    assert_metrics_close(got, {"iou": 1.0, "f1": 1.0})
    assert_finite(got)


def test_band_first_and_rgb_masks_collapse_to_two_dimensions(
    make_sample, write_mask, assert_metrics_close
):
    """``(1, h, w)`` from a tool and ``(h, w, 3)`` from a PNG must both work."""
    truth = rows_mask(4, 4, slice(0, 2))
    path = write_mask("gt.npy", truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=truth[None, ...]))
    metric.update(sample, mask_prediction("s0", mask=np.repeat(truth[..., None], 3, axis=2)))
    assert_metrics_close(metric.compute(), {"iou": 1.0, "n": 2.0})


def test_rgba_png_alpha_channel_does_not_flood_the_mask(
    tmp_path, make_sample, assert_metrics_close
):
    """An opaque alpha channel is transparency, not a class.

    Treating it as one would report recall 1.0 for any prediction whatsoever,
    which is the sort of wrong number that survives a review because it looks
    plausible.
    """
    from PIL import Image

    truth = rows_mask(4, 4, slice(0, 2))
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0:2, :, :3] = 255
    p = tmp_path / "pred.png"
    Image.fromarray(rgba, mode="RGBA").save(p)

    gt = tmp_path / "gt.npy"
    np.save(gt, truth)
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(gt)})
    metric.update(sample, mask_prediction("s0", mask_path=str(p)))
    assert_metrics_close(metric.compute(), {"iou": 1.0, "precision": 1.0, "n": 1.0})


def test_pooled_and_per_sample_iou_diverge_on_no_change_pairs(
    make_sample, write_mask, assert_metrics_close
):
    """An empty/empty pair is per-sample IoU 1.0 and invisible to the pooled one.

    LEVIR-CC/MCI genuinely contain "no change" pairs, so the two numbers must be
    reported side by side; this test is the proof they are computed differently.
    """
    empty_gt = write_mask("empty.npy", np.zeros((4, 4), dtype=bool))
    half_gt = write_mask("half.npy", rows_mask(4, 4, slice(0, 2)))
    metric = ChangeMaskMetrics()
    s_empty = make_sample("s0", task="change_map", meta={"mask_path": str(empty_gt)})
    metric.update(s_empty, mask_prediction("s0", mask=np.zeros((4, 4), dtype=bool)))
    s_half = make_sample("s1", task="change_map", meta={"mask_path": str(half_gt)})
    metric.update(s_half, mask_prediction("s1", mask=rows_mask(4, 4, slice(1, 3))))
    # Second sample: tp=4, fp=4, fn=4 -> IoU 1/3. Pooled sees only that sample.
    got = metric.compute()
    assert_metrics_close(
        got,
        {"iou": 1.0 / 3.0, "mean_iou": (1.0 + 1.0 / 3.0) / 2.0, "n": 2.0},
    )
    assert_finite(got)


def test_empty_corpus_of_no_change_pairs_is_finite(make_sample, write_mask):
    """Nothing changed anywhere: pooled numbers are 0/0, reported as 0.0."""
    empty = write_mask("empty.npy", np.zeros((4, 4), dtype=bool))
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(empty)})
    metric.update(sample, mask_prediction("s0", mask=np.zeros((4, 4), dtype=bool)))
    got = metric.compute()
    assert_finite(got)
    assert got["mean_iou"] == 1.0
    assert got["iou"] == 0.0
    assert got["over_detection_ratio"] == 0.0
    assert got["n"] == 1.0


# --------------------------------------------------------------------------- #
# ChangeMaskMetrics -- skips
# --------------------------------------------------------------------------- #


def test_shape_mismatch_is_a_skip_with_a_warning(
    make_sample, write_mask, captured_warnings, assert_metrics_close
):
    """A wrong-sized prediction must not crash and must not score as zero."""
    path = write_mask("gt.npy", rows_mask(8, 8, slice(0, 4)))
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=np.ones((4, 4), dtype=bool)))
    got = metric.compute()
    assert_metrics_close(
        got, {"n": 0.0, "n_skipped": 1.0, "n_skipped/shape_mismatch": 1.0, "iou": 0.0}
    )
    assert_finite(got)
    assert any("shape_mismatch" in r.getMessage() for r in captured_warnings)


def test_missing_ground_truth_is_a_skip(make_sample, assert_metrics_close):
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map")
    metric.update(sample, mask_prediction("s0", mask=np.ones((4, 4), dtype=bool)))
    assert_metrics_close(
        metric.compute(), {"n": 0.0, "n_skipped": 1.0, "n_skipped/no_ground_truth": 1.0}
    )


def test_prediction_without_a_mask_is_a_skip_not_a_zero(
    make_sample, write_mask, assert_metrics_close
):
    """A text-only predictor pointed at this metric must read ``n=0``.

    Scoring it 0.0 would hide the fact that nothing was measured, which is the
    difference between "the model is bad at change maps" and "the change tool
    never ran".
    """
    path = write_mask("gt.npy", rows_mask(4, 4, slice(0, 2)))
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, Prediction(sample_id="s0", task="change_map", text="a road was built"))
    assert_metrics_close(
        metric.compute(), {"n": 0.0, "n_skipped": 1.0, "n_skipped/no_prediction": 1.0}
    )


def test_failed_prediction_is_a_skip(make_sample, write_mask, assert_metrics_close):
    path = write_mask("gt.npy", rows_mask(4, 4, slice(0, 2)))
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, Prediction.failed("s0", "change_map", "CUDA OOM"))
    assert_metrics_close(
        metric.compute(), {"n": 0.0, "n_skipped": 1.0, "n_skipped/prediction_failed": 1.0}
    )


def test_unreadable_mask_is_a_skip(make_sample, tmp_path, captured_warnings, assert_metrics_close):
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(tmp_path / "nope.npy")})
    metric.update(sample, mask_prediction("s0", mask=np.ones((4, 4), dtype=bool)))
    assert_metrics_close(
        metric.compute(), {"n": 0.0, "n_skipped": 1.0, "n_skipped/load_error": 1.0}
    )
    assert any("not found" in r.getMessage() for r in captured_warnings)


def test_every_skip_reason_key_is_always_present():
    got = ChangeMaskMetrics().compute()
    for reason in SKIP_REASONS:
        assert f"n_skipped/{reason}" in got
    assert_finite(got)


def test_change_mask_rejects_the_wrong_task(make_sample):
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_caption")
    with pytest.raises(UnsupportedTask):
        metric.update(sample, Prediction(sample_id="s0", task="change_caption", text="x"))


def test_change_mask_reset_clears_state(make_sample, write_mask):
    path = write_mask("gt.npy", rows_mask(4, 4, slice(0, 2)))
    metric = ChangeMaskMetrics()
    sample = make_sample("s0", task="change_map", meta={"mask_path": str(path)})
    metric.update(sample, mask_prediction("s0", mask=np.ones((4, 4), dtype=bool)))
    metric.update(sample, mask_prediction("s0", mask=np.ones((2, 2), dtype=bool)))
    assert metric.compute()["n"] == 1.0
    assert metric.compute()["n_skipped"] == 1.0
    metric.reset()
    got = metric.compute()
    assert got["n"] == 0.0 and got["n_skipped"] == 0.0


# --------------------------------------------------------------------------- #
# load_mask, the shared binarisation
# --------------------------------------------------------------------------- #


def test_load_mask_round_trips_npy_and_png(write_mask):
    truth = rows_mask(5, 7, slice(1, 3))
    for name in ("m.npy", "m.png"):
        loaded = load_mask(write_mask(name, truth))
        assert loaded.dtype == np.bool_
        assert loaded.shape == (5, 7)
        assert np.array_equal(loaded, truth), name


def test_load_mask_accepts_arrays_and_rejects_none():
    assert load_mask(np.array([[0, 2], [1, 0]], dtype=np.uint8)).tolist() == [
        [False, True],
        [True, False],
    ]
    with pytest.raises(SatQueryError):
        load_mask(None)


def test_load_mask_reports_a_missing_file_actionably(tmp_path):
    with pytest.raises(SatQueryError, match="not found"):
        load_mask(tmp_path / "absent.npy")


def test_load_mask_rejects_a_one_dimensional_array():
    with pytest.raises(SatQueryError, match="two spatial dimensions"):
        load_mask(np.array([0, 1, 1], dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Registry wiring -- the names the CLI and harness use
# --------------------------------------------------------------------------- #


def test_registry_builds_both_metrics():
    from satquery.registry import build_metric, default_metrics_for

    assert isinstance(build_metric("multilabel"), MultiLabelMetrics)
    assert isinstance(build_metric("change_mask"), ChangeMaskMetrics)
    assert default_metrics_for("fusion_multilabel") == ("multilabel",)
    assert default_metrics_for("change_map") == ("change_mask",)


def test_metrics_satisfy_the_metric_protocol():
    from satquery.contracts import Metric

    assert isinstance(MultiLabelMetrics(), Metric)
    assert isinstance(ChangeMaskMetrics(), Metric)


def test_oracle_predictor_scores_perfectly_on_fusion(make_sample, oracle, assert_metrics_close):
    """The harness's upper bound for the fusion slot.

    ``OraclePredictor`` returns the ground truth, so anything but a perfect
    score means the metric disagrees with the contract about where fusion
    labels live.
    """
    metric = MultiLabelMetrics()
    n = 0
    for i, truth in enumerate((("water",), ("forest", "urban_fabric"), ())):
        sample = make_sample(f"s{i}", task="fusion_multilabel", labels=truth)
        prediction = oracle.predict(sample)
        if prediction.labels or not truth:
            metric.update(sample, prediction)
            n += 1
    assert n == 3, "OraclePredictor did not echo fusion labels"
    assert_metrics_close(metric.compute(), {"micro_f1": 1.0, "exact_match": 1.0, "n": 3.0})
