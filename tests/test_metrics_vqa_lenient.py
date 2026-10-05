"""Tests for ``VqaAccuracy.accuracy_lenient``.

The lenient score exists to keep the before/after adaptation delta honest: an
un-tuned baseline answers in sentences, a tuned model answers tersely, and a
strict-only comparison reports "learned the output format" as "learned the task".
These tests pin the two properties that make the number trustworthy -- it is
never below strict accuracy, and it never credits a numeric substring match.
"""

from __future__ import annotations

import pytest

from satquery.contracts import ImageRef, Prediction, Sample
from satquery.metrics.vqa import VqaAccuracy

IMAGES = (ImageRef("i1", "i1.png", "optical"),)


def _sample(sample_id: str, answer: str, question_type: str = "count") -> Sample:
    return Sample(
        sample_id=sample_id,
        task="vqa",
        images=IMAGES,
        question="how many buildings?",
        answer=answer,
        meta={"question_type": question_type},
    )


def _score(pairs: list[tuple[Sample, str]]) -> dict[str, float]:
    metric = VqaAccuracy()
    for sample, text in pairs:
        metric.update(sample, Prediction(sample.sample_id, "vqa", text=text))
    return metric.compute()


@pytest.mark.parametrize(
    "text, strict_hit",
    [
        ("2", True),                          # already terse
        ("two", True),                         # number word folds to the digit
        ("The answer is: two.", True),         # preamble stripped by extract_answer
        ("There are 2 buildings.", False),     # verbose: strict says no, lenient says yes
        ("two buildings", False),
        ("I count 2 buildings in the image.", False),
    ],
)
def test_verbose_answers_are_lenient_hits_but_not_always_strict(text: str, strict_hit: bool) -> None:
    got = _score([(_sample("s1", "2"), text)])
    assert got["accuracy"] == (1.0 if strict_hit else 0.0)
    assert got["accuracy_lenient"] == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "There are 12 buildings.",   # 2 must not be found inside 12
        "22",
        "0.2 hectares",
        "no buildings",
    ],
)
def test_lenient_never_credits_a_substring_or_wrong_count(text: str) -> None:
    got = _score([(_sample("s1", "2"), text)])
    assert got["accuracy"] == 0.0
    assert got["accuracy_lenient"] == 0.0, f"{text!r} must not be credited against gold '2'"


def test_lenient_is_never_below_strict_across_a_mixed_batch() -> None:
    pairs = [
        (_sample("a", "2"), "2"),
        (_sample("b", "7"), "There are 7 buildings."),
        (_sample("c", "yes", "presence"), "Yes, there is a road."),
        (_sample("d", "no", "presence"), "yes"),
        (_sample("e", "3"), ""),
        (_sample("f", "5"), "There are 15 buildings."),
    ]
    got = _score(pairs)
    assert got["accuracy"] == pytest.approx(1 / 6)
    assert got["accuracy_lenient"] == pytest.approx(3 / 6)
    assert got["accuracy_lenient"] >= got["accuracy"]


def test_lenient_is_sliced_per_question_type() -> None:
    pairs = [
        (_sample("a", "2", "count"), "There are 2 buildings."),
        (_sample("b", "4", "count"), "nine"),
        (_sample("c", "yes", "presence"), "Yes."),
    ]
    got = _score(pairs)
    assert got["accuracy_lenient/count"] == pytest.approx(0.5)
    assert got["accuracy_lenient/presence"] == pytest.approx(1.0)
    assert got["accuracy/count"] == pytest.approx(0.0)


def test_multi_token_gold_matches_only_as_a_contiguous_run() -> None:
    sample = Sample(
        sample_id="s1",
        task="vqa",
        images=IMAGES,
        question="how large?",
        answer="between 10 and 100",
        meta={"question_type": "comparison"},
    )
    hit = _score([(sample, "The area is between 10 and 100 square metres.")])
    assert hit["accuracy_lenient"] == 1.0
    miss = _score([(sample, "between 10 and 1000")])
    assert miss["accuracy_lenient"] == 0.0


def test_failed_prediction_is_wrong_under_both_scores() -> None:
    sample = _sample("s1", "2")
    metric = VqaAccuracy()
    metric.update(sample, Prediction.failed("s1", "vqa", "cuda oom"))
    got = metric.compute()
    assert got["accuracy"] == 0.0
    assert got["accuracy_lenient"] == 0.0
    assert got["answered_pct"] == 0.0


def test_oracle_ceiling_holds_for_the_lenient_score(make_sample, oracle) -> None:
    samples = [
        make_sample("s0", "vqa", answer="yes", meta={"question_type": "presence"}),
        make_sample("s1", "vqa", answer="2", meta={"question_type": "count"}),
        make_sample("s2", "vqa", answer="urban", meta={"question_type": "rural_urban"}),
    ]
    metric = VqaAccuracy()
    for sample in samples:
        metric.update(sample, oracle.predict(sample))
    got = metric.compute()
    assert got["accuracy"] == 1.0
    assert got["accuracy_lenient"] == 1.0
