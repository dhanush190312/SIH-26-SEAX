"""Tests for :class:`satquery.metrics.vqa.VqaAccuracy`.

The contract this file defends is stated in ``satquery/models/dummy.py``: the
oracle must score 1.0, the constant predictor must reproduce the majority-class
floor you can compute by hand, and half-corrupted predictions must land near 0.5.
If any of those slips, every baseline number the project reports is wrong in a
direction nobody notices.
"""

from __future__ import annotations

import pytest

from satquery.contracts import Metric, Prediction, UnsupportedTask
from satquery.metrics.vqa import VqaAccuracy
from satquery.models.dummy import ConstantPredictor, OraclePredictor
from satquery.registry import build_metric

# question_type values as the RSVQA-LR/HR and CDVQA adapters emit them.
MIXED = (
    ("yes", "presence"),
    ("no", "presence"),
    ("2", "count"),
    ("17", "count"),
    ("yes", "comparison"),
    ("urban", "rural_urban"),
    ("0m2 - 10m2", "area"),
)


def _pred(sample, text, *, error=None):
    return Prediction(
        sample_id=sample.sample_id,
        task=sample.task,
        text=text,
        error=error,
        model="test",
    )


def _run(metric, samples, texts):
    for sample, text in zip(samples, texts, strict=True):
        metric.update(sample, _pred(sample, text))
    return metric.compute()


@pytest.fixture
def vqa_samples(make_sample):
    """Build ``[(answer, question_type), ...]`` into samples of one task."""

    def _build(specs, task: str = "vqa", **kw):
        return [
            make_sample(
                sample_id=f"s{i}",
                task=task,
                answer=answer,
                meta={"question_type": qtype},
                **kw,
            )
            for i, (answer, qtype) in enumerate(specs)
        ]

    return _build


# --------------------------------------------------------------------------- #
# Metric-validation contract: oracle / noisy oracle / constant
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("task", ["vqa", "change_vqa"])
def test_oracle_scores_perfect_on_mixed_types(vqa_samples, oracle, assert_metrics_close, task):
    samples = vqa_samples(MIXED, task=task)
    metric = VqaAccuracy()
    for sample in samples:
        metric.update(sample, oracle.predict(sample))
    got = metric.compute()
    assert_metrics_close(
        got,
        {
            "accuracy": 1.0,
            "average_accuracy": 1.0,
            "answered_pct": 100.0,
            "n": float(len(samples)),
            "accuracy/count": 1.0,
            "accuracy/area": 1.0,
            "n/presence": 2.0,
        },
    )


def test_oracle_with_half_noise_lands_near_half(vqa_samples):
    # 200 samples: the corruption is hash-keyed per sample_id, so this is exact,
    # but the assertion is a band because the point is that the metric *moves*.
    specs = [MIXED[i % len(MIXED)] for i in range(200)]
    samples = vqa_samples(specs)
    metric = VqaAccuracy()
    predictor = OraclePredictor(noise=0.5)
    for sample in samples:
        metric.update(sample, predictor.predict(sample))
    got = metric.compute()
    assert 0.4 <= got["accuracy"] <= 0.6
    assert got["n"] == 200.0
    # "__wrong__" is a wrong answer, not a missing one.
    assert got["answered_pct"] == 100.0


def test_constant_yes_reproduces_the_majority_class_floor(vqa_samples, assert_metrics_close):
    # Hand-built floor: 7 of 10 answers are "yes" -> 0.70 overall. Presence is
    # yes-heavy (6/8) while comparison is balanced (1/2), so the unweighted
    # average over types is 0.625 and *below* overall accuracy. That gap is the
    # reason RSVQA papers report both numbers.
    specs = [("yes", "presence")] * 6 + [("no", "presence")] * 2 + [("yes", "comparison"), ("no", "comparison")]
    samples = vqa_samples(specs)
    metric = VqaAccuracy()
    constant = ConstantPredictor(text="yes")
    for sample in samples:
        metric.update(sample, constant.predict(sample))
    assert_metrics_close(
        metric.compute(),
        {
            "accuracy": 0.7,
            "average_accuracy": 0.625,
            "accuracy/presence": 0.75,
            "accuracy/comparison": 0.5,
            "n/presence": 8.0,
            "n/comparison": 2.0,
            "n": 10.0,
            "answered_pct": 100.0,
        },
    )


def test_constant_predictor_fixture_matches_ad_hoc_instance(vqa_samples, constant):
    samples = vqa_samples([("yes", "presence"), ("no", "presence")])
    metric = VqaAccuracy()
    for sample in samples:
        metric.update(sample, constant.predict(sample))
    assert metric.compute()["accuracy"] == 0.5


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #


def test_per_type_slicing_is_exact(vqa_samples, assert_metrics_close):
    samples = vqa_samples([("2", "count"), ("3", "count"), ("yes", "presence")])
    got = _run(VqaAccuracy(), samples, ["2", "9", "yes"])
    assert_metrics_close(
        got,
        {
            "accuracy": 2.0 / 3.0,
            "average_accuracy": 0.75,  # mean(0.5, 1.0), not 2/3
            "accuracy/count": 0.5,
            "accuracy/presence": 1.0,
            "n/count": 2.0,
            "n/presence": 1.0,
        },
    )
    assert sorted(k for k in got if k.startswith("accuracy/")) == [
        "accuracy/count",
        "accuracy/presence",
    ]


def test_missing_question_type_collapses_to_all(make_sample):
    sample = make_sample(sample_id="s0", answer="2")  # no meta -> Sample.question_type == "all"
    metric = VqaAccuracy()
    metric.update(sample, _pred(sample, "2"))
    got = metric.compute()
    assert got["accuracy/all"] == 1.0
    assert got["n/all"] == 1.0
    assert got["average_accuracy"] == got["accuracy"] == 1.0


def test_slice_by_none_makes_one_bucket(vqa_samples):
    samples = vqa_samples([("2", "count"), ("yes", "presence")])
    got = _run(VqaAccuracy(slice_by=None), samples, ["2", "no"])
    assert got["accuracy/all"] == 0.5
    assert "accuracy/count" not in got


def test_slice_by_custom_meta_key(make_sample):
    samples = [
        make_sample(sample_id="a", answer="yes", meta={"question_type": "presence", "sensor": "s2"}),
        make_sample(sample_id="b", answer="no", meta={"question_type": "presence", "sensor": "s1"}),
    ]
    got = _run(VqaAccuracy(slice_by="sensor"), samples, ["yes", "yes"])
    assert got["accuracy/s2"] == 1.0
    assert got["accuracy/s1"] == 0.0
    assert got["average_accuracy"] == 0.5


def test_slice_key_never_contains_the_breakdown_separator(make_sample):
    sample = make_sample(sample_id="s0", answer="urban", meta={"question_type": "rural/urban area"})
    metric = VqaAccuracy()
    metric.update(sample, _pred(sample, "urban"))
    got = metric.compute()
    assert "accuracy/rural_urban_area" in got
    assert not any(k.count("/") > 1 for k in got)


# --------------------------------------------------------------------------- #
# Prediction-side cleanup
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "The answer is: two.",
        "two",
        "Two",
        " 2 ",
        "The answer is 2",
        "answer: two",
    ],
)
def test_chatty_prediction_scores_as_the_short_answer(vqa_samples, text):
    samples = vqa_samples([("2", "count")])
    assert _run(VqaAccuracy(), samples, [text])["accuracy"] == 1.0


def test_strict_mode_skips_extraction_and_marks_chatty_wrong(vqa_samples):
    samples = vqa_samples([("2", "count")])
    assert _run(VqaAccuracy(strict=True), samples, ["The answer is: two."])["accuracy"] == 0.0
    # Bare answers still score in strict mode -- normalisation always applies.
    assert _run(VqaAccuracy(strict=True), samples, ["Two."])["accuracy"] == 1.0


def test_yes_no_synonyms_and_casing_are_normalised(vqa_samples):
    samples = vqa_samples([("yes", "presence"), ("no", "presence")])
    assert _run(VqaAccuracy(), samples, ["YES!", "Nope"])["accuracy"] == 1.0


def test_any_accepted_answer_counts(make_sample):
    sample = make_sample(sample_id="s0", answer="0m2 - 10m2", references=("small", "tiny"))
    metric = VqaAccuracy()
    metric.update(sample, _pred(sample, "tiny"))
    assert metric.compute()["accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# Failure accounting: wrong, never skipped
# --------------------------------------------------------------------------- #


def test_failed_prediction_is_wrong_not_skipped(vqa_samples, assert_metrics_close):
    samples = vqa_samples([("2", "count"), ("yes", "presence")])
    metric = VqaAccuracy()
    metric.update(samples[0], Prediction.failed(samples[0].sample_id, samples[0].task, "OOM"))
    metric.update(samples[1], _pred(samples[1], "yes"))
    assert_metrics_close(
        metric.compute(),
        {
            "accuracy": 0.5,
            "n": 2.0,
            "n/count": 1.0,
            "accuracy/count": 0.0,
            "answered_pct": 50.0,
        },
    )


def test_errored_prediction_with_text_is_not_credited(vqa_samples):
    samples = vqa_samples([("2", "count")])
    metric = VqaAccuracy()
    metric.update(samples[0], _pred(samples[0], "2", error="truncated decode"))
    got = metric.compute()
    assert got["accuracy"] == 0.0
    assert got["answered_pct"] == 0.0


@pytest.mark.parametrize("text", [None, "", "   ", "???", "..."])
def test_empty_predictions_are_wrong_and_unanswered(vqa_samples, text):
    samples = vqa_samples([("2", "count")])
    got = _run(VqaAccuracy(), samples, [text])
    assert got["accuracy"] == 0.0
    assert got["answered_pct"] == 0.0
    assert got["n"] == 1.0


def test_empty_ground_truth_does_not_match_an_empty_prediction(make_image_ref):
    from satquery.contracts import Sample

    sample = Sample(
        sample_id="s0",
        task="vqa",
        images=(make_image_ref(),),
        question="how many buildings?",
        answer="???",  # normalises to the empty string
    )
    metric = VqaAccuracy()
    metric.update(sample, _pred(sample, ""))
    assert metric.compute()["accuracy"] == 0.0


def test_non_string_prediction_text_does_not_crash(vqa_samples):
    samples = vqa_samples([("2", "count")])
    prediction = Prediction(sample_id=samples[0].sample_id, task="vqa", text=2)  # type: ignore[arg-type]
    metric = VqaAccuracy()
    metric.update(samples[0], prediction)
    assert metric.compute()["accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# Numeric tolerance
# --------------------------------------------------------------------------- #


def test_numeric_tolerance_is_off_by_default(vqa_samples):
    samples = vqa_samples([("2", "count")])
    assert _run(VqaAccuracy(), samples, ["3"])["accuracy"] == 0.0


def test_numeric_tolerance_credits_off_by_one_counts(vqa_samples):
    samples = vqa_samples([("2", "count"), ("2", "count")])
    got = _run(VqaAccuracy(numeric_tolerance=1), samples, ["3", "4"])
    assert got["accuracy"] == 0.5  # |3-2| <= 1 counts, |4-2| does not


def test_numeric_tolerance_accepts_word_numbers_and_chatty_text(vqa_samples):
    samples = vqa_samples([("2", "count")])
    assert _run(VqaAccuracy(numeric_tolerance=1), samples, ["The answer is: three."])["accuracy"] == 1.0


def test_numeric_tolerance_ignores_non_numeric_answers(vqa_samples):
    samples = vqa_samples([("yes", "presence")])
    assert _run(VqaAccuracy(numeric_tolerance=100), samples, ["no"])["accuracy"] == 0.0


def test_numeric_tolerance_never_conflates_rsvqa_range_classes(vqa_samples):
    # Both strings contain digits; neither is a count. A parser that grabbed the
    # first number would score these as equal at tol=100 and silently inflate
    # RSVQA-HR area accuracy.
    samples = vqa_samples([("0m2 - 10m2", "area"), ("between 10 and 100", "comparison")])
    got = _run(VqaAccuracy(numeric_tolerance=1000), samples, ["10m2 - 100m2", "between 100 and 1000"])
    assert got["accuracy"] == 0.0


def test_percent_answers_compare_numerically(vqa_samples):
    samples = vqa_samples([("20%", "area")])
    assert _run(VqaAccuracy(numeric_tolerance=2), samples, ["21%"])["accuracy"] == 1.0


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError):
        VqaAccuracy(numeric_tolerance=-1)


# --------------------------------------------------------------------------- #
# Streaming contract
# --------------------------------------------------------------------------- #


def test_compute_on_no_samples_is_all_zeros():
    got = VqaAccuracy().compute()
    assert got == {
        "accuracy": 0.0,
        "average_accuracy": 0.0,
        "accuracy_lenient": 0.0,
        "n": 0.0,
        "answered_pct": 0.0,
    }


def test_reset_clears_all_state(vqa_samples):
    samples = vqa_samples([("2", "count")])
    metric = VqaAccuracy()
    _run(metric, samples, ["2"])
    metric.reset()
    assert metric.compute()["n"] == 0.0
    assert "accuracy/count" not in metric.compute()


def test_compute_is_idempotent(vqa_samples):
    samples = vqa_samples(MIXED)
    metric = VqaAccuracy()
    _run(metric, samples, ["yes", "no", "2", "17", "yes", "urban", "0m2 - 10m2"])
    assert metric.compute() == metric.compute()


def test_score_is_stateless(vqa_samples):
    samples = vqa_samples([("2", "count")])
    metric = VqaAccuracy()
    assert metric.score(samples[0], _pred(samples[0], "2")) == 1.0
    assert metric.score(samples[0], _pred(samples[0], "9")) == 0.0
    assert metric.compute()["n"] == 0.0  # score() must not accumulate


def test_wrong_task_raises_unsupported_task(make_sample):
    sample = make_sample(sample_id="c0", task="caption")
    prediction = _pred(sample, "a residential area")
    with pytest.raises(UnsupportedTask):
        VqaAccuracy().update(sample, prediction)


def test_registry_and_lazy_package_export_the_same_class():
    import satquery.metrics as metrics_pkg

    assert build_metric("vqa_accuracy").__class__ is VqaAccuracy
    assert metrics_pkg.VqaAccuracy is VqaAccuracy
    assert build_metric("vqa_accuracy", numeric_tolerance=1, strict=True).describe() == {
        "name": "vqa_accuracy",
        "tasks": ["vqa", "change_vqa"],
        "slice_by": "question_type",
        "numeric_tolerance": 1.0,
        "strict": True,
    }


def test_satisfies_the_metric_protocol():
    metric = VqaAccuracy()
    assert isinstance(metric, Metric)
    assert metric.name == "vqa_accuracy"
    assert metric.tasks == ("vqa", "change_vqa")
