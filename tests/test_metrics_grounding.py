"""Grounding metric tests.

Every IoU asserted here is computed by hand from the box coordinates and every
coordinate is exactly representable in binary floating point, so the assertions
are exact rather than approximate. Two worked cases carry the file:

* ``(2.5, 0, 12.5, 10)`` against ``(0, 0, 10, 10)``: intersection 7.5*10 = 75,
  union 100 + 100 - 75 = 125, IoU = 0.6 -- above 0.5, below 0.7.
* the three-box multi-instance case in
  :func:`test_multi_instance_precision_recall`, where a duplicate prediction
  must cost precision.
"""

from __future__ import annotations

import pytest

from satquery.contracts import Box, Prediction, Sample, UnsupportedTask
from satquery.metrics.grounding import GroundingMetrics
from satquery.models.dummy import OraclePredictor

GT = Box(0.0, 0.0, 10.0, 10.0, "building")
#: IoU exactly 0.6 against GT -- see module docstring.
IOU_060 = Box(2.5, 0.0, 12.5, 10.0, "building", 0.9)


def _pred(sample: Sample, *boxes: Box) -> Prediction:
    return Prediction(sample_id=sample.sample_id, task="ground", boxes=tuple(boxes))


def _score_one(sample: Sample, *boxes: Box, **kwargs) -> dict[str, float]:
    metric = GroundingMetrics(**kwargs)
    metric.update(sample, _pred(sample, *boxes))
    return metric.compute()


# --------------------------------------------------------------------------- #
# Top-1 accuracy at the two reported thresholds
# --------------------------------------------------------------------------- #


def test_exact_box_hits_every_threshold(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    got = _score_one(sample, Box(0.0, 0.0, 10.0, 10.0, "building", 0.9))
    assert_metrics_close(
        got,
        {
            "acc@0.5": 1.0,
            "acc@0.7": 1.0,
            "mean_iou": 1.0,
            "precision@0.7": 1.0,
            "recall@0.7": 1.0,
            "n": 1.0,
            "empty_pred_pct": 0.0,
        },
    )


def test_iou_060_passes_half_fails_seven_tenths(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    got = _score_one(sample, IOU_060)
    assert got["mean_iou"] == pytest.approx(0.6)
    assert_metrics_close(
        got,
        {
            "acc@0.5": 1.0,
            "acc@0.7": 0.0,
            "precision@0.5": 1.0,
            "recall@0.5": 1.0,
            "precision@0.7": 0.0,
            "recall@0.7": 0.0,
        },
    )


def test_iou_is_exactly_what_the_fixture_claims():
    assert IOU_060.iou(GT) == 0.6


def test_no_predicted_boxes_scores_zero_and_counts_as_empty(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    got = _score_one(sample)
    assert_metrics_close(
        got,
        {
            "acc@0.5": 0.0,
            "acc@0.7": 0.0,
            "mean_iou": 0.0,
            "precision@0.5": 0.0,
            "recall@0.5": 0.0,
            "n": 1.0,
            "empty_pred_pct": 100.0,
        },
    )


def test_empty_pred_pct_is_a_percentage_over_samples(make_sample, assert_metrics_close):
    metric = GroundingMetrics()
    hit = make_sample("s1", task="ground", boxes=(GT,))
    miss = make_sample("s2", task="ground", boxes=(GT,))
    metric.update(hit, _pred(hit, Box(0.0, 0.0, 10.0, 10.0, None, 0.8)))
    metric.update(miss, _pred(miss))
    assert_metrics_close(metric.compute(), {"n": 2.0, "empty_pred_pct": 50.0, "acc@0.5": 0.5})


def test_failed_prediction_is_a_miss_not_a_skip(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    metric = GroundingMetrics()
    # Boxes present but the predictor errored: they must be ignored entirely.
    metric.update(
        sample,
        Prediction(
            sample_id="s1", task="ground", boxes=(Box(0.0, 0.0, 10.0, 10.0, None, 1.0),),
            error="decode failed",
        ),
    )
    assert_metrics_close(
        metric.compute(), {"n": 1.0, "acc@0.5": 0.0, "empty_pred_pct": 100.0, "n_skipped": 0.0}
    )


def test_hit_against_any_ground_truth_box(make_sample, assert_metrics_close):
    """The referring convention: matching the second of two boxes still counts."""
    sample = make_sample(
        "s1", task="ground", boxes=(GT, Box(100.0, 100.0, 110.0, 110.0, "building"))
    )
    got = _score_one(sample, Box(100.0, 100.0, 110.0, 110.0, None, 0.5))
    assert_metrics_close(got, {"acc@0.7": 1.0, "mean_iou": 1.0})


# --------------------------------------------------------------------------- #
# Ranking: score order, None-scored boxes, score_threshold
# --------------------------------------------------------------------------- #


def test_top_1_is_the_highest_scoring_box_not_the_best_box(make_sample, assert_metrics_close):
    """A correct box hidden behind a confident wrong one must not count."""
    sample = make_sample("s1", task="ground", boxes=(GT,))
    wrong = Box(500.0, 500.0, 510.0, 510.0, None, 0.9)
    right = Box(0.0, 0.0, 10.0, 10.0, None, 0.1)
    got = _score_one(sample, right, wrong)
    assert_metrics_close(got, {"acc@0.5": 0.0, "mean_iou": 0.0})
    # ... but detection recall still finds it, which is the point of both views.
    assert_metrics_close(got, {"recall@0.5": 1.0, "precision@0.5": 0.5})


def test_unscored_boxes_are_usable_in_emission_order(make_sample, assert_metrics_close):
    """Qwen native grounding gives no scores; the first box emitted is top-1."""
    sample = make_sample("s1", task="ground", boxes=(GT,))
    first = Box(0.0, 0.0, 10.0, 10.0, None, None)
    second = Box(900.0, 900.0, 910.0, 910.0, None, None)
    assert_metrics_close(_score_one(sample, first, second), {"acc@0.7": 1.0, "mean_iou": 1.0})
    assert_metrics_close(_score_one(sample, second, first), {"acc@0.7": 0.0, "mean_iou": 0.0})


def test_unscored_boxes_sort_after_scored_ones(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    unscored_correct = Box(0.0, 0.0, 10.0, 10.0, None, None)
    scored_wrong = Box(700.0, 700.0, 710.0, 710.0, None, 0.01)
    # Emission order puts the correct unscored box first; the ranking must still
    # promote the scored box, however weak its score.
    got = _score_one(sample, unscored_correct, scored_wrong)
    assert_metrics_close(got, {"acc@0.5": 0.0, "recall@0.5": 1.0})


def test_score_threshold_filters_low_confidence_boxes(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    weak_correct = Box(0.0, 0.0, 10.0, 10.0, None, 0.2)
    assert_metrics_close(_score_one(sample, weak_correct), {"acc@0.7": 1.0})
    filtered = _score_one(sample, weak_correct, score_threshold=0.5)
    assert_metrics_close(filtered, {"acc@0.7": 0.0, "empty_pred_pct": 100.0})


def test_score_threshold_never_drops_unscored_boxes(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    got = _score_one(
        sample, Box(0.0, 0.0, 10.0, 10.0, None, None), score_threshold=0.9
    )
    assert_metrics_close(got, {"acc@0.7": 1.0, "empty_pred_pct": 0.0})


# --------------------------------------------------------------------------- #
# Multi-instance precision / recall
# --------------------------------------------------------------------------- #


def test_multi_instance_precision_recall(make_sample, assert_metrics_close):
    """Hand-built: 3 ground-truth boxes, 3 predictions, one a duplicate.

    P1 exactly matches A, P2 exactly matches B, P3 duplicates A. Greedy
    one-to-one matching has already consumed A, so P3 is a false positive and C
    is never found: tp=2, fp=1, fn=1 -> precision = recall = 2/3. Top-1 is
    exact, so acc stays 1.0 -- exactly the blindness precision/recall exist to
    cover.
    """
    a = Box(0.0, 0.0, 10.0, 10.0, "ship")
    b = Box(20.0, 20.0, 30.0, 30.0, "ship")
    c = Box(40.0, 40.0, 50.0, 50.0, "ship")
    sample = make_sample("s1", task="ground", question="the ships", boxes=(a, b, c))
    got = _score_one(
        sample,
        Box(0.0, 0.0, 10.0, 10.0, "ship", 0.9),
        Box(20.0, 20.0, 30.0, 30.0, "ship", 0.8),
        Box(0.0, 0.0, 10.0, 10.0, "ship", 0.7),
    )
    assert got["precision@0.5"] == pytest.approx(2.0 / 3.0)
    assert got["recall@0.5"] == pytest.approx(2.0 / 3.0)
    assert_metrics_close(got, {"acc@0.5": 1.0, "acc@0.7": 1.0, "mean_iou": 1.0})


def test_duplicate_boxes_cost_precision_under_one_to_one(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    dup = Box(0.0, 0.0, 10.0, 10.0, None, 0.9)
    got = _score_one(sample, dup, Box(0.0, 0.0, 10.0, 10.0, None, 0.8))
    assert_metrics_close(got, {"precision@0.5": 0.5, "recall@0.5": 1.0})


def test_coverage_matching_forgives_duplicates(make_sample, assert_metrics_close):
    """``greedy_match=False``: ground truth is reusable, recall is coverage."""
    a = Box(0.0, 0.0, 10.0, 10.0, "ship")
    b = Box(20.0, 20.0, 30.0, 30.0, "ship")
    c = Box(40.0, 40.0, 50.0, 50.0, "ship")
    sample = make_sample("s1", task="ground", question="the ships", boxes=(a, b, c))
    got = _score_one(
        sample,
        Box(0.0, 0.0, 10.0, 10.0, "ship", 0.9),
        Box(20.0, 20.0, 30.0, 30.0, "ship", 0.8),
        Box(0.0, 0.0, 10.0, 10.0, "ship", 0.7),
        greedy_match=False,
    )
    assert_metrics_close(got, {"precision@0.5": 1.0})
    assert got["recall@0.5"] == pytest.approx(2.0 / 3.0)


def test_partial_hit_below_threshold_is_a_false_positive(make_sample, assert_metrics_close):
    sample = make_sample("s1", task="ground", boxes=(GT,))
    got = _score_one(sample, IOU_060)
    assert_metrics_close(got, {"precision@0.7": 0.0, "recall@0.7": 0.0})


def test_precision_recall_are_micro_averaged(make_sample, assert_metrics_close):
    """Pooled box counts, not a per-sample mean.

    One single-instance sample scored perfectly plus one three-instance sample
    with 1 of 3 found gives 2/4 either way you slice it; averaging per sample
    would give (1 + 1/3)/2 = 0.667 instead.
    """
    metric = GroundingMetrics(thresholds=(0.5,))
    single = make_sample("s1", task="ground", boxes=(GT,))
    metric.update(single, _pred(single, Box(0.0, 0.0, 10.0, 10.0, None, 0.9)))

    truth = tuple(
        Box(40.0 * i, 0.0, 40.0 * i + 10.0, 10.0, "ship") for i in range(3)
    )
    multi = make_sample("s2", task="ground", question="the ships", boxes=truth)
    metric.update(
        multi,
        _pred(
            multi,
            Box(0.0, 0.0, 10.0, 10.0, "ship", 0.9),
            Box(200.0, 200.0, 210.0, 210.0, "ship", 0.8),
            Box(300.0, 300.0, 310.0, 310.0, "ship", 0.7),
        ),
    )
    assert_metrics_close(metric.compute(), {"precision@0.5": 0.5, "recall@0.5": 0.5, "n": 2.0})


def test_mean_iou_follows_top_1_not_the_best_box(make_sample):
    """Documented choice: ``mean_iou`` must not rise by adding more boxes."""
    sample = make_sample("s1", task="ground", boxes=(GT,))
    got = _score_one(sample, IOU_060, Box(0.0, 0.0, 10.0, 10.0, None, 0.1))
    assert got["mean_iou"] == pytest.approx(0.6)


def test_acc_is_the_fraction_of_samples_above_threshold(make_sample):
    """The identity the top-1 definition buys: ``acc@t == mean(iou_i >= t)``."""
    metric = GroundingMetrics()
    exact = make_sample("s1", task="ground", boxes=(GT,))
    partial = make_sample("s2", task="ground", boxes=(GT,))
    miss = make_sample("s3", task="ground", boxes=(GT,))
    metric.update(exact, _pred(exact, Box(0.0, 0.0, 10.0, 10.0, None, 0.9)))
    metric.update(partial, _pred(partial, IOU_060))
    metric.update(miss, _pred(miss, Box(80.0, 80.0, 90.0, 90.0, None, 0.9)))
    got = metric.compute()
    ious = [1.0, 0.6, 0.0]
    assert got["mean_iou"] == pytest.approx(sum(ious) / 3.0)
    for t, key in ((0.5, "acc@0.5"), (0.7, "acc@0.7")):
        assert got[key] == pytest.approx(sum(i >= t for i in ious) / 3.0)


# --------------------------------------------------------------------------- #
# Predictor doubles: ceiling and floor
# --------------------------------------------------------------------------- #


def _score_dataset(metric: GroundingMetrics, samples, predictor) -> dict[str, float]:
    metric.reset()
    for sample in samples:
        metric.update(sample, predictor.predict(sample))
    return metric.compute()


@pytest.fixture
def ground_samples(make_sample) -> tuple[Sample, ...]:
    """A small multi-instance grounding set, deterministic geometry."""
    out = []
    for i in range(6):
        offset = float(10 * i)
        boxes = (
            Box(offset, offset, offset + 20.0, offset + 20.0, "object"),
            Box(offset + 40.0, offset + 40.0, offset + 55.0, offset + 60.0, "object"),
        )
        out.append(
            make_sample(f"g{i}", task="ground", question="the object", boxes=boxes)
        )
    return tuple(out)


def test_oracle_reaches_the_ceiling(ground_samples, assert_metrics_close):
    got = _score_dataset(GroundingMetrics(), ground_samples, OraclePredictor())
    assert_metrics_close(
        got,
        {
            "acc@0.5": 1.0,
            "acc@0.7": 1.0,
            "mean_iou": 1.0,
            "precision@0.5": 1.0,
            "recall@0.5": 1.0,
            "precision@0.7": 1.0,
            "recall@0.7": 1.0,
            "n": float(len(ground_samples)),
            "empty_pred_pct": 0.0,
        },
    )


def test_fully_corrupted_oracle_hits_the_floor(ground_samples, assert_metrics_close):
    got = _score_dataset(GroundingMetrics(), ground_samples, OraclePredictor(noise=1.0))
    assert_metrics_close(
        got,
        {
            "acc@0.5": 0.0,
            "acc@0.7": 0.0,
            "mean_iou": 0.0,
            "precision@0.5": 0.0,
            "recall@0.5": 0.0,
            "n": float(len(ground_samples)),
            # The corrupted oracle still emits boxes, just in the wrong place:
            # a floored score with 0% empty predictions is the signal that the
            # metric is measuring geometry and not missing output.
            "empty_pred_pct": 0.0,
        },
    )


# --------------------------------------------------------------------------- #
# Bookkeeping, degenerate inputs, configuration
# --------------------------------------------------------------------------- #


def test_empty_state_is_zero_not_nan():
    got = GroundingMetrics().compute()
    assert set(got) == {
        "acc@0.5", "acc@0.7", "mean_iou",
        "precision@0.5", "recall@0.5", "precision@0.7", "recall@0.7",
        "n", "empty_pred_pct", "n_skipped",
    }
    assert all(v == 0.0 for v in got.values())


def test_sample_without_ground_truth_is_skipped(make_sample, assert_metrics_close):
    metric = GroundingMetrics()
    scored = make_sample("s1", task="ground", boxes=(GT,))
    unlabelled = make_sample("s2", task="ground", boxes=())
    metric.update(scored, _pred(scored, Box(0.0, 0.0, 10.0, 10.0, None, 1.0)))
    metric.update(unlabelled, _pred(unlabelled))
    assert_metrics_close(metric.compute(), {"n": 1.0, "n_skipped": 1.0, "acc@0.5": 1.0})


def test_reset_clears_state(make_sample, assert_metrics_close):
    metric = GroundingMetrics()
    sample = make_sample("s1", task="ground", boxes=(GT,))
    metric.update(sample, _pred(sample, Box(0.0, 0.0, 10.0, 10.0, None, 1.0)))
    metric.reset()
    assert_metrics_close(metric.compute(), {"n": 0.0, "acc@0.5": 0.0, "mean_iou": 0.0})


def test_wrong_task_is_refused(make_sample):
    metric = GroundingMetrics()
    sample = make_sample("s1", task="vqa")
    with pytest.raises(UnsupportedTask):
        metric.update(sample, Prediction(sample_id="s1", task="vqa", text="2"))


def test_custom_thresholds_shape_the_keys(make_sample, assert_metrics_close):
    metric = GroundingMetrics(thresholds=(0.75, 0.25, 0.75))
    sample = make_sample("s1", task="ground", boxes=(GT,))
    metric.update(sample, _pred(sample, IOU_060))
    got = metric.compute()
    assert metric.thresholds == (0.25, 0.75)
    assert_metrics_close(got, {"acc@0.25": 1.0, "acc@0.75": 0.0})
    assert "acc@0.5" not in got


def test_integral_threshold_keeps_a_decimal_point(make_sample, assert_metrics_close):
    """``acc@1.0``, never ``acc@1``: the key shape is part of the run JSON."""
    metric = GroundingMetrics(thresholds=(1.0,))
    sample = make_sample("s1", task="ground", boxes=(GT,))
    metric.update(sample, _pred(sample, Box(0.0, 0.0, 10.0, 10.0, None, 0.9)))
    got = metric.compute()
    assert_metrics_close(got, {"acc@1.0": 1.0, "precision@1.0": 1.0, "recall@1.0": 1.0})
    assert "acc@1" not in got


@pytest.mark.parametrize("bad", [(), (0.0,), (1.5,), (-0.5, 0.5)])
def test_invalid_thresholds_rejected(bad):
    with pytest.raises(ValueError):
        GroundingMetrics(thresholds=bad)


def test_metric_protocol_shape():
    metric = GroundingMetrics()
    assert metric.name == "grounding"
    assert metric.tasks == ("ground",)
    assert all(isinstance(v, float) for v in metric.compute().values())
