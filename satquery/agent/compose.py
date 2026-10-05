"""Step 5: assemble the answer, tie every claim to a number, and score confidence.

This is where invariant 04 is enforced rather than requested. The text tool has
produced a sentence; the deterministic tools have produced a dict of measured
quantities. The composer walks the sentence clause by clause and asks, of each
one, *which measurement backs this*:

* **cited** -- the clause makes a quantitative claim and every number in it
  matches an evidence value within tolerance. It survives with its citation
  attached: "approximately 14%" -> ``change_map.changed_area_pct = 14.2``.
* **dropped** -- the clause makes a quantitative claim and a number in it matches
  *nothing*. This is the hallucination case, and it is deleted from the answer,
  with the offending value recorded in the trace. A model that invents "3.2 km
  of new road" does not get to keep the sentence.
* **qualitative** -- the clause makes no numeric claim at all ("a residential
  area with buildings arranged along a road"). It is kept and *flagged*, and it
  lowers the confidence.

That third category is the honest part of the design. Dropping every clause
without a number would make captioning impossible -- a caption is qualitative by
definition -- so the rule cannot be "cite or die" without deleting a mandatory
task. What it can be, and is: a claim with a number must be backed by that
number, unbacked numbers are removed rather than softened, and the ratio of
cited to qualitative clauses is reported and feeds the confidence figure. A
reader can see exactly how much of an answer is measurement and how much is
description.

Confidence is composed, never asserted by a model:

    confidence = backend_confidence
               x (0.5 + 0.5 x cited_fraction)      how much is measured
               x validation_factor                  how sure we are of the input
               - 0.1 per dropped clause             evidence of confabulation

and is clipped to ``[0.05, 0.95]``: this system does not claim certainty, and it
does not claim a measurement it made is worthless either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Citation",
    "Clause",
    "Composition",
    "compose",
    "REL_TOLERANCE",
    "ABS_TOLERANCE",
]

#: A quoted number counts as matching an evidence value within 5% of it -- the
#: slack a model needs to write "approximately 14%" for 14.2 without being
#: accused of making it up.
REL_TOLERANCE = 0.05

#: Absolute floor on that tolerance, for values near zero where 5% is nothing.
ABS_TOLERANCE = 0.5

#: Numbers this small are counts and years and enumerations rather than
#: measurements ("in 2 regions"); requiring an evidence match for every "1" in a
#: sentence would drop half of ordinary English.
_TRIVIAL_MAX = 3.0

#: A number plus the unit written next to it. The unit is what stops "3.0%"
#: from being credited to ``bands = 3``: a percentage can only be backed by a
#: percentage, and a dB figure only by a dB measurement.
_NUMBER_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(%|dB|db|px|m)?")

#: Which evidence keys can back a claim carrying a given unit. Empty tuple means
#: "no restriction".
_UNIT_KEYS: dict[str, tuple[str, ...]] = {
    "%": ("pct", "percent", "fraction"),
    "db": ("db", "backscatter", "threshold"),
    "px": ("px", "width", "height", "region"),
}

#: Structural facts about the file rather than measurements of the scene. They
#: stay in the evidence namespace (a caption may legitimately cite the image
#: size) but they lose every tie, so a band count never outranks a real
#: measurement that happens to be a similar number.
_STRUCTURAL_KEYS = frozenset({"bands", "width", "height", "n_valid", "num_regions"})
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+")
_WORD_RE = re.compile(r"[a-z][a-z-]+")


@dataclass(frozen=True, slots=True)
class Citation:
    """One claim tied to one measured value."""

    claim: str
    key: str
    value: Any
    tool: str

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "key": self.key, "value": self.value, "tool": self.tool}


@dataclass(frozen=True, slots=True)
class Clause:
    """A fragment of the answer plus the verdict on it."""

    text: str
    status: str  # cited | qualitative | dropped
    citations: tuple[Citation, ...] = ()
    unmatched: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "citations": [c.to_dict() for c in self.citations],
            "unmatched": list(self.unmatched),
        }


@dataclass(frozen=True, slots=True)
class Composition:
    """The finished answer and everything needed to defend it."""

    answer: str
    confidence: float
    clauses: tuple[Clause, ...] = ()
    citations: tuple[Citation, ...] = ()
    dropped: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    refused: bool = False

    @property
    def cited_fraction(self) -> float:
        kept = [c for c in self.clauses if c.status != "dropped"]
        if not kept:
            return 0.0
        return sum(1 for c in kept if c.status == "cited") / len(kept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": round(float(self.confidence), 3),
            "refused": self.refused,
            "cited_fraction": round(self.cited_fraction, 3),
            "clauses": [c.to_dict() for c in self.clauses],
            "citations": [c.to_dict() for c in self.citations],
            "dropped": list(self.dropped),
            "evidence": dict(self.evidence),
            "warnings": list(self.warnings),
        }


def compose(
    query: str,
    results: Sequence[Any],
    *,
    validation_warnings: Sequence[str] = (),
    modality_confidence: float = 1.0,
    withheld: Sequence[tuple[str, str]] = (),
) -> Composition:
    """Build the answer from the tool results. ``results`` are :class:`ToolResult`.

    The answer text comes from the *last successful text tool*, because a plan
    that runs ``indices`` then ``vqa`` means the VQA answer is the answer and the
    index numbers are its support. If no text tool succeeded, the composer writes
    the refusal itself out of whatever was measured -- which is why a query with
    no VLM available still produces something defensible rather than an error.
    """
    evidence, provenance = _merge_evidence(results)
    text, backend_confidence, backend, refused, refusal_reason = _primary_answer(results)

    if not text:
        return _refusal(
            query, evidence, provenance, results, refusal_reason, validation_warnings, withheld
        )

    clauses = tuple(_check_clause(clause, evidence, provenance) for clause in _split(text))
    kept = [c for c in clauses if c.status != "dropped"]
    dropped = tuple(c.text for c in clauses if c.status == "dropped")
    answer = " ".join(c.text for c in kept).strip()

    citations = _unique(c for clause in kept for c in clause.citations)
    warnings = list(validation_warnings)
    if dropped:
        warnings.append(
            f"{len(dropped)} clause(s) were removed from the answer because they quoted numbers "
            "no tool produced"
        )
    if not answer:
        answer = (
            "Every clause of the generated answer quoted an unsupported number, so nothing "
            "survived the citation check."
        )
        refused = True

    cited_fraction = (
        0.0 if not kept else sum(1 for c in kept if c.status == "cited") / len(kept)
    )
    confidence = _confidence(
        backend_confidence,
        cited_fraction=cited_fraction,
        n_dropped=len(dropped),
        modality_confidence=modality_confidence,
        n_warnings=len(validation_warnings),
    )
    return Composition(
        answer=answer,
        confidence=confidence,
        clauses=clauses,
        citations=citations,
        dropped=dropped,
        evidence=evidence,
        warnings=tuple(warnings),
        refused=refused,
    )


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def _merge_evidence(results: Sequence[Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Flatten every tool's evidence into one namespace, remembering the source.

    Later tools win on a key collision, which is the right precedence: a
    ``change_map`` that measured ``changed_area_pct`` after an ``indices`` call
    is the more specific measurement. The provenance map keeps the losing tool's
    name out of the citation, so a citation never points at the wrong tool.
    """
    evidence: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for result in results:
        if not getattr(result, "ok", False):
            continue
        for source in (getattr(result, "evidence", {}), getattr(result, "output", {})):
            for key, value in dict(source).items():
                if not _is_evidence(key, value):
                    continue
                evidence[key] = value
                provenance[key] = result.name
    return evidence, provenance


#: Output keys that carry prose rather than measurement. Letting the answer text
#: itself into the evidence namespace makes every clause trivially "cited" by
#: itself, which is the one bug that would render the whole citation check
#: decorative.
_PROSE_KEYS = frozenset(
    {"answer", "reason", "note", "rationale", "phrase", "question", "detail",
     "backend_reason", "band_assumption", "vegetation_index", "threshold_method",
     "index_source", "task", "indices_available", "image_id", "optical_image", "sar_image"}
)

#: Longest string that can be a citable label ("north-east", "NDVI"). Anything
#: longer is a sentence.
_MAX_LABEL_LEN = 24


def _is_evidence(key: str, value: Any) -> bool:
    if isinstance(value, bool) or key in _PROSE_KEYS or key.startswith("_"):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and 3 < len(value) <= _MAX_LABEL_LEN


def _primary_answer(results: Sequence[Any]) -> tuple[str, float, str, bool, str]:
    """``(text, confidence, backend, refused, reason)`` from the last text tool."""
    text_tools = ("change_describe", "vqa", "caption", "ground")
    chosen = None
    for result in results:
        if result.name in text_tools and getattr(result, "ok", False):
            if str(result.output.get("answer") or "").strip():
                chosen = result
    if chosen is None:
        reasons = [
            str(r.output.get("reason") or r.error)
            for r in results
            if r.name in text_tools and (r.error or r.output.get("reason"))
        ]
        return ("", 0.0, "", True, reasons[-1] if reasons else "")
    confidence = chosen.output.get("confidence")
    return (
        str(chosen.output["answer"]).strip(),
        0.5 if confidence is None else float(confidence),
        chosen.backend,
        False,
        "",
    )


# --------------------------------------------------------------------------- #
# Clause checking
# --------------------------------------------------------------------------- #


def _split(text: str) -> list[str]:
    """Sentence-ish clauses. Splitting on ``.`` and ``;`` only, never on commas.

    A comma split would tear "14%, concentrated in the north-east" into a number
    with no subject and a location with no number, and the second half would then
    look unsupported. Sentences and semicolons are the units a reader would
    actually judge separately.
    """
    return [part.strip() for part in _CLAUSE_SPLIT_RE.split(text.strip()) if part.strip()]


def _check_clause(
    clause: str, evidence: Mapping[str, Any], provenance: Mapping[str, str]
) -> Clause:
    numbers = [(n, unit) for n, unit in _NUMBER_RE.findall(clause) if _significant(n)]
    citations: list[Citation] = []
    unmatched: list[str] = []

    for token, unit in numbers:
        match = _match_number(float(token), evidence, unit=unit)
        if match is None:
            unmatched.append(f"{token}{unit}")
        else:
            key, value = match
            claim = f"{token}{unit}" if unit else token
            citations.append(Citation(claim, key, value, provenance.get(key, "unknown")))

    for key, value in evidence.items():
        if isinstance(value, str) and len(value) > 3 and value.lower() in clause.lower():
            citations.append(Citation(value, key, value, provenance.get(key, "unknown")))

    if unmatched:
        return Clause(clause, "dropped", tuple(citations), tuple(unmatched))
    if citations:
        return Clause(clause, "cited", tuple(citations))
    return Clause(clause, "qualitative")


def _unique(citations: Iterable[Citation]) -> tuple[Citation, ...]:
    """One citation per (claim, key): the same number in two clauses is one fact."""
    seen: set[tuple[str, str]] = set()
    out: list[Citation] = []
    for citation in citations:
        marker = (citation.claim, citation.key)
        if marker not in seen:
            seen.add(marker)
            out.append(citation)
    return tuple(out)


def _significant(token: str) -> bool:
    """Is this number a claim, or incidental?

    Small integers are counts, list indices and years-in-words; requiring an
    evidence match for "two images" would drop a clause that is simply restating
    the input. Anything above the trivial ceiling, or with a decimal point, is
    treated as a measurement and has to be backed.
    """
    try:
        value = float(token)
    except ValueError:  # pragma: no cover - the regex only matches numbers
        return False
    if "." in token:
        return True
    return abs(value) > _TRIVIAL_MAX


def _match_number(
    value: float, evidence: Mapping[str, Any], *, unit: str = ""
) -> tuple[str, Any] | None:
    """The evidence key that backs a quoted number, or None if nothing does.

    Three filters, in order of how much they matter:

    1. **the unit**, when the clause wrote one. "3.0%" may only be backed by a
       percentage; without this, a band count of 3 wins the tie on distance and
       the citation points at a fact about the file rather than the scene.
    2. **the tolerance**, so "approximately 14%" backs 14.2 but not 19.
    3. **structural keys lose ties**, for the same reason as (1) in the cases
       where no unit was written.
    """
    wanted = _UNIT_KEYS.get(unit.lower().strip(), ())
    best: tuple[str, Any] | None = None
    best_rank = (float("inf"), 1)
    for key, candidate in evidence.items():
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        if wanted and not any(marker in key.lower() for marker in wanted):
            continue
        tolerance = max(abs(float(candidate)) * REL_TOLERANCE, ABS_TOLERANCE)
        delta = abs(float(candidate) - value)
        if delta > tolerance:
            continue
        rank = (delta, 1 if key in _STRUCTURAL_KEYS else 0)
        if rank < best_rank:
            best, best_rank = (key, candidate), rank
    return best


# --------------------------------------------------------------------------- #
# Confidence and refusal
# --------------------------------------------------------------------------- #


def _confidence(
    backend_confidence: float,
    *,
    cited_fraction: float,
    n_dropped: int,
    modality_confidence: float,
    n_warnings: int,
) -> float:
    base = max(0.0, min(1.0, float(backend_confidence) or 0.5))
    measured = 0.5 + 0.5 * float(cited_fraction)
    validation = 0.85 + 0.15 * max(0.0, min(1.0, float(modality_confidence)))
    penalty = min(0.3, 0.1 * int(n_dropped)) + min(0.1, 0.02 * int(n_warnings))
    return round(max(0.05, min(0.95, base * measured * validation - penalty)), 3)


def _sentence(text: str) -> str:
    """Capitalise and terminate, so a backend's reason reads as an answer."""
    clean = text.strip()
    if not clean:
        return ""
    clean = clean[0].upper() + clean[1:]
    return clean if clean.endswith((".", "!", "?")) else clean + "."


def _refusal(
    query: str,
    evidence: Mapping[str, Any],
    provenance: Mapping[str, str],
    results: Sequence[Any],
    reason: str,
    validation_warnings: Sequence[str],
    withheld: Sequence[tuple[str, str]] = (),
) -> Composition:
    """No text tool answered. Say what *was* measured and why the rest is missing.

    A refusal that lists the measurements it does have is worth far more than an
    error: the user learns the water percentage even though nobody could tell
    them how many boats are in it, and the assessor sees the boundary of the
    system rather than a crash.
    """
    measured = [
        f"{key} = {value}"
        for key, value in evidence.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and key not in ("width", "height", "bands")
    ]
    lines = [_sentence(reason) if reason else "No component could answer this question."]
    if measured:
        lines.append("Measured for this input: " + "; ".join(measured[:12]) + ".")
    else:
        lines.append("No deterministic measurement was available for this input either.")
    failures = [f"{r.name}: {r.error}" for r in results if not getattr(r, "ok", True) and r.error]
    if failures:
        lines.append("Tool failures: " + "; ".join(failures) + ".")
    if withheld:
        # The most useful thing a refusal can say is which capability was
        # unavailable *for this input* and what would make it available.
        lines.append(
            "Tools withheld for this input: "
            + "; ".join(f"{name} ({why})" for name, why in withheld)
            + "."
        )
    return Composition(
        answer=" ".join(lines),
        confidence=0.05,
        clauses=(),
        citations=tuple(
            Citation(str(value), key, value, provenance.get(key, "unknown"))
            for key, value in list(evidence.items())[:12]
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        evidence=dict(evidence),
        warnings=tuple(validation_warnings),
        refused=True,
    )
