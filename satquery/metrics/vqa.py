"""VQA accuracy -- the headline number for RSVQA-LR/HR, VRSBench-VQA and CDVQA.

Exact match is the published convention for all four benchmarks, which makes the
*normalisation* the whole game: comparing raw strings scores ``"Yes."`` against
``"yes"`` as wrong and understates a model by several points. Both sides
therefore go through :func:`satquery.text.normalize_answer`, and the prediction
additionally through :func:`satquery.text.extract_answer` first, because VLMs
answer in sentences. ``strict=True`` disables that second step, which is how you
measure how much of a score is formatting cleanup rather than model skill.

Three conventions, stated here because each one is easy to get wrong and each one
moves the number a reviewer will quote:

* RSVQA papers report ``average_accuracy`` -- the *unweighted* mean over question
  types -- alongside overall accuracy, because the test splits are dominated by
  presence questions and overall accuracy hides a model that cannot count. Both
  are emitted; the report shows them side by side.
* A missing, empty or errored prediction is **wrong**, never skipped. Dropping
  the samples a model failed on inflates accuracy exactly where the model is
  weakest, which is the one failure mode a baseline harness must not have.
* ``numeric_tolerance`` is off by default. Counting is where RS VQA models
  actually fail, so an off-by-one credit is a real scoring decision that has to
  be switched on deliberately and reported next to the number.
* ``accuracy_lenient`` is emitted alongside, and exists to protect the one number
  this project actually claims: the before/after adaptation delta. Exact match
  scores ``"There are 2 buildings."`` against gold ``"2"`` as **wrong**. That is
  the published convention and stays the headline -- but an *un-tuned* baseline
  answers in sentences while the tuned model is trained to answer tersely, so a
  strict-only comparison credits the fine-tune for learning the output format and
  reports it as learning the task. Quoting both separates the two. The lenient
  variant is a diagnostic, never a headline: it accepts a gold answer appearing
  as a whole-token run inside the prediction, which does over-credit a reply like
  ``"no idea"`` against gold ``"no"``.
"""

from __future__ import annotations

import math
from typing import Any

from ..contracts import Prediction, Sample, Task, UnsupportedTask
from ..text import extract_answer, normalize_answer

__all__ = ["VqaAccuracy"]


def _as_number(text: str) -> float | None:
    """Parse a *bare* numeric answer, else ``None``.

    Deliberately refuses anything that is not a single number token. RSVQA area
    classes (``"0m2 - 10m2"``) and comparison answers (``"between 10 and 100"``)
    contain digits but are not counts, and letting the tolerance path see them
    would silently mark two distinct answer classes as equal -- a scoring bug
    that looks like a win. A trailing ``%`` is tolerated because percentage
    answers are common and mean the same thing with or without the sign.
    """
    s = text.strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if not s:
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    # "nan" and "inf" parse as floats; they are not answers.
    return value if math.isfinite(value) else None


class VqaAccuracy:
    """Streaming exact-match accuracy with per-question-type breakdown.

    ``slice_by`` names the ``sample.meta`` key that defines a question type
    (``None`` collapses everything into one ``all`` bucket). Samples missing the
    key land in ``all``, matching :attr:`Sample.question_type`, so a dataset that
    does not label its questions still produces a coherent report.
    """

    name = "vqa_accuracy"
    tasks: tuple[Task, ...] = ("vqa", "change_vqa")

    def __init__(
        self,
        *,
        slice_by: str | None = "question_type",
        numeric_tolerance: float = 0.0,
        strict: bool = False,
    ) -> None:
        if numeric_tolerance < 0:
            raise ValueError("numeric_tolerance must be >= 0")
        self.slice_by = slice_by
        self.numeric_tolerance = float(numeric_tolerance)
        self.strict = bool(strict)
        self.reset()

    # -- state --------------------------------------------------------------- #

    def reset(self) -> None:
        self._n = 0
        self._correct = 0
        self._answered = 0
        self._correct_lenient = 0
        self._n_by_slice: dict[str, int] = {}
        self._correct_by_slice: dict[str, int] = {}
        self._lenient_by_slice: dict[str, int] = {}

    # -- scoring ------------------------------------------------------------- #

    def _hypothesis(self, prediction: Prediction) -> str:
        """The comparable form of a prediction; empty string when unscoreable.

        An errored prediction counts as no answer even if it carries text:
        whatever a predictor emitted before failing is not evidence of skill.
        """
        if not prediction.ok:
            return ""
        text = prediction.text
        if text is None:
            return ""
        if not isinstance(text, str):  # a predictor that returned a number or label
            text = str(text)
        if not self.strict:
            text = extract_answer(text)
        return normalize_answer(text)

    def _match(self, hypothesis: str, sample: Sample) -> bool:
        if not hypothesis:
            return False
        golds = [g for g in (normalize_answer(a) for a in sample.all_answers) if g]
        if hypothesis in golds:
            return True
        if self.numeric_tolerance > 0:
            value = _as_number(hypothesis)
            if value is not None:
                return any(
                    gold is not None and abs(value - gold) <= self.numeric_tolerance
                    for gold in (_as_number(g) for g in golds)
                )
        return False

    def _match_lenient(self, hypothesis: str, sample: Sample) -> bool:
        """Exact match, or a gold answer present as a whole-token run.

        Token-level containment, never substring: gold ``"2"`` must not be found
        inside ``"12 buildings"``, which is the failure that would make the
        lenient number worthless. Anything the strict path accepts is accepted
        here too, so ``accuracy_lenient >= accuracy`` always holds.

        One extra guard is needed because :func:`normalize_answer` drops the
        decimal point: ``"0.2 hectares"`` normalises to tokens
        ``["0", "2", "hectares"]``, so a bare token match would credit it against
        gold ``"2"``. A purely numeric gold therefore only counts when the tokens
        flanking the match are not themselves numeric -- which still accepts the
        legitimate case of a numeric gold that was itself split the same way.
        """
        if not hypothesis:
            return False
        if self._match(hypothesis, sample):
            return True
        hyp_tokens = hypothesis.split()
        for gold in (normalize_answer(a) for a in sample.all_answers):
            gold_tokens = gold.split()
            if not gold_tokens or len(gold_tokens) > len(hyp_tokens):
                continue
            span = len(gold_tokens)
            numeric_gold = all(_as_number(t) is not None for t in gold_tokens)
            for i in range(len(hyp_tokens) - span + 1):
                if hyp_tokens[i : i + span] != gold_tokens:
                    continue
                if numeric_gold and self._flanked_by_a_number(hyp_tokens, i, span):
                    continue
                return True
        return False

    @staticmethod
    def _flanked_by_a_number(tokens: list[str], start: int, span: int) -> bool:
        """True when the matched run sits directly beside another numeric token."""
        before = tokens[start - 1] if start > 0 else None
        after = tokens[start + span] if start + span < len(tokens) else None
        return any(t is not None and _as_number(t) is not None for t in (before, after))

    def score(self, sample: Sample, prediction: Prediction) -> float:
        """``1.0`` / ``0.0`` for one pair, without touching the accumulators.

        Stateless on purpose: the harness records per-sample correctness in its
        ``per_sample`` rows and must be able to ask for it without perturbing the
        run-level counters.
        """
        return 1.0 if self._match(self._hypothesis(prediction), sample) else 0.0

    def _slice(self, sample: Sample) -> str:
        if not self.slice_by:
            return "all"
        value = sample.meta.get(self.slice_by)
        if value is None:
            return "all"
        # '/' is the metric-key breakdown separator, so a raw type containing one
        # would produce an unsplittable key like "accuracy/rural/urban".
        return "_".join(str(value).split()).replace("/", "_") or "all"

    def update(self, sample: Sample, prediction: Prediction) -> None:
        if sample.task not in self.tasks:
            raise UnsupportedTask(
                f"{self.name} scores {self.tasks}, not {sample.task!r} "
                f"(sample {sample.sample_id})"
            )
        hypothesis = self._hypothesis(prediction)
        key = self._slice(sample)
        self._n += 1
        self._n_by_slice[key] = self._n_by_slice.get(key, 0) + 1
        if hypothesis:
            self._answered += 1
        if self._match(hypothesis, sample):
            self._correct += 1
            self._correct_by_slice[key] = self._correct_by_slice.get(key, 0) + 1
        if self._match_lenient(hypothesis, sample):
            self._correct_lenient += 1
            self._lenient_by_slice[key] = self._lenient_by_slice.get(key, 0) + 1

    # -- results ------------------------------------------------------------- #

    def compute(self) -> dict[str, float]:
        """``accuracy``, ``average_accuracy``, ``accuracy_lenient``, ``n``,
        ``answered_pct`` and, per question type, ``accuracy/<type>``,
        ``accuracy_lenient/<type>`` and ``n/<type>``.

        ``answered_pct`` counts predictions that survive normalisation to a
        non-empty string. That is stricter than "text was returned" on purpose:
        a reply of ``"???"`` is a refusal, and reading answered_pct as coverage
        is only safe if refusals are excluded from it. Every unanswered sample
        still counts in ``n`` and against ``accuracy``.
        """
        n = self._n
        per_slice = {
            key: self._correct_by_slice.get(key, 0) / count
            for key, count in sorted(self._n_by_slice.items())
        }
        out: dict[str, float] = {
            "accuracy": (self._correct / n) if n else 0.0,
            "average_accuracy": (sum(per_slice.values()) / len(per_slice)) if per_slice else 0.0,
            "n": float(n),
            "answered_pct": (100.0 * self._answered / n) if n else 0.0,
        }
        out["accuracy_lenient"] = (self._correct_lenient / n) if n else 0.0
        for key, accuracy in per_slice.items():
            out[f"accuracy/{key}"] = accuracy
        for key, count in sorted(self._n_by_slice.items()):
            out[f"accuracy_lenient/{key}"] = self._lenient_by_slice.get(key, 0) / count
        for key in per_slice:
            out[f"n/{key}"] = float(self._n_by_slice[key])
        return out

    def describe(self) -> dict[str, Any]:
        """Configuration for the run record -- a score is meaningless without it."""
        return {
            "name": self.name,
            "tasks": list(self.tasks),
            "slice_by": self.slice_by,
            "numeric_tolerance": self.numeric_tolerance,
            "strict": self.strict,
        }

    def __repr__(self) -> str:
        return (
            f"VqaAccuracy(slice_by={self.slice_by!r}, "
            f"numeric_tolerance={self.numeric_tolerance!r}, strict={self.strict!r}, "
            f"n={self._n})"
        )
