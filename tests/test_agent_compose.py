"""Tests for the composer: citation, dropping, and how confidence is built.

The one behaviour worth defending above all others is that an *unbacked number
is deleted*. A model that writes "3.2 km of new road" when no tool measured a
length must not have that sentence survive, and the deletion must be visible in
the trace rather than silent.
"""

from __future__ import annotations


from satquery.agent.compose import ABS_TOLERANCE, REL_TOLERANCE, compose
from satquery.agent.tools import ToolResult


def text_result(answer: str, *, name="vqa", confidence=0.8, evidence=None, ok=True) -> ToolResult:
    return ToolResult(
        name=name,
        ok=ok,
        output={"answer": answer, "confidence": confidence},
        evidence=dict(evidence or {}),
        backend="test-backend",
        backend_kind="deterministic",
    )


def evidence_result(evidence: dict, *, name="change_map") -> ToolResult:
    return ToolResult(name=name, ok=True, output=dict(evidence), evidence=dict(evidence),
                      backend="cva-otsu", backend_kind="deterministic")


# --------------------------------------------------------------------------- #
# Citation
# --------------------------------------------------------------------------- #


def test_a_supported_number_is_cited_with_its_tool_and_value():
    result = compose(
        "what changed?",
        [
            evidence_result({"changed_area_pct": 14.2, "centroid_quadrant": "north-east"}),
            text_result(
                "Built-up area increased by approximately 14% in the north-east quadrant.",
                name="change_describe",
            ),
        ],
    )

    keys = {c.key for c in result.citations}
    assert "changed_area_pct" in keys and "centroid_quadrant" in keys
    assert all(c.tool == "change_map" for c in result.citations if c.key == "changed_area_pct")
    assert result.clauses[0].status == "cited"


def test_a_number_no_tool_produced_deletes_its_clause():
    result = compose(
        "what changed?",
        [
            evidence_result({"changed_area_pct": 14.2}),
            text_result(
                "About 14% of the scene changed. The new road is 3.2 km long.",
                name="change_describe",
            ),
        ],
    )

    assert "3.2 km" not in result.answer
    assert "14%" in result.answer
    assert result.dropped and "3.2" in result.dropped[0]
    assert any("removed from the answer" in w for w in result.warnings)


def test_the_tolerance_admits_rounding_but_not_invention():
    evidence = evidence_result({"changed_area_pct": 14.2})

    rounded = compose("q", [evidence, text_result("About 14.0% changed.")])
    invented = compose("q", [evidence, text_result("About 19.0% changed.")])

    assert not rounded.dropped
    assert invented.dropped
    assert REL_TOLERANCE < 0.2 and ABS_TOLERANCE < 1.0  # the constants stay tight


def test_a_percentage_cannot_be_backed_by_a_band_count():
    """The unit check: "3.0%" must not cite ``bands = 3``."""
    result = compose(
        "q",
        [
            evidence_result({"bands": 3, "largest_region_pct": 3.05}),
            text_result("The largest region covers 3.0%."),
        ],
    )

    assert [c.key for c in result.citations] == ["largest_region_pct"]


def test_small_integers_are_not_treated_as_claims():
    """"in 2 regions" restates the input; demanding evidence would delete it."""
    result = compose(
        "q",
        [evidence_result({"changed_area_pct": 9.0}), text_result("Change appears in 2 places.")],
    )

    assert not result.dropped


def test_a_qualitative_clause_survives_but_is_flagged():
    result = compose("q", [text_result("A residential area beside a river.")])

    assert result.answer.startswith("A residential area")
    assert [c.status for c in result.clauses] == ["qualitative"]
    assert result.cited_fraction == 0.0


def test_the_answer_text_cannot_cite_itself():
    """The evidence namespace must not absorb the prose it is checking."""
    result = compose("q", [text_result("The scene is mostly farmland.")])

    assert all(c.key != "answer" for c in result.citations)


def test_a_string_label_is_citable_when_it_is_short_and_measured():
    result = compose(
        "q",
        [
            evidence_result({"centroid_quadrant": "south-west"}),
            text_result("The change sits in the south-west of the scene."),
        ],
    )

    assert any(c.key == "centroid_quadrant" for c in result.citations)


def test_citations_are_deduplicated():
    result = compose(
        "q",
        [
            evidence_result({"water_pct": 12.0}),
            text_result("Water covers 12.0%. Water covers 12.0% of the valid pixels."),
        ],
    )

    assert len(result.citations) == len({(c.claim, c.key) for c in result.citations})


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


def test_confidence_rises_with_the_measured_share_of_the_answer():
    cited = compose(
        "q", [evidence_result({"water_pct": 12.0}), text_result("Water covers 12.0%.")]
    )
    qualitative = compose("q", [text_result("It looks like a coastal town.")])

    assert cited.confidence > qualitative.confidence


def test_a_dropped_clause_costs_confidence():
    evidence = evidence_result({"water_pct": 12.0})
    clean = compose("q", [evidence, text_result("Water covers 12.0%.")])
    dirty = compose("q", [evidence, text_result("Water covers 12.0%. The port handles 91 ships.")])

    assert dirty.confidence < clean.confidence


def test_uncertain_modality_lowers_confidence():
    args = ("q", [evidence_result({"water_pct": 12.0}), text_result("Water covers 12.0%.")])
    sure = compose(*args, modality_confidence=1.0)
    unsure = compose(*args, modality_confidence=0.3)

    assert unsure.confidence < sure.confidence


def test_confidence_is_never_zero_or_one():
    perfect = compose(
        "q",
        [evidence_result({"water_pct": 12.0}), text_result("Water covers 12.0%.", confidence=1.0)],
    )
    assert 0.05 <= perfect.confidence <= 0.95


# --------------------------------------------------------------------------- #
# Refusal
# --------------------------------------------------------------------------- #


def test_with_no_text_tool_the_refusal_still_reports_the_measurements():
    result = compose("how many boats?", [evidence_result({"water_pct": 12.0, "vegetation_pct": 40.0})])

    assert result.refused
    assert "water_pct = 12.0" in result.answer
    assert result.confidence == 0.05


def test_a_refusal_names_the_tools_that_were_withheld():
    result = compose(
        "what changed?",
        [evidence_result({"water_pct": 12.0})],
        withheld=[("change_map", "no image pair was registered for this query")],
    )

    assert "change_map" in result.answer
    assert "no image pair" in result.answer


def test_a_failed_tool_is_reported_in_the_refusal():
    failed = ToolResult(name="vqa", ok=False, error="OOM", output={})
    result = compose("q", [failed])

    assert result.refused
    assert "vqa: OOM" in result.answer


def test_an_answer_whose_every_clause_is_unsupported_becomes_a_refusal():
    result = compose(
        "q", [evidence_result({"water_pct": 12.0}), text_result("The port handles 91 ships.")]
    )

    assert result.refused
    assert "citation check" in result.answer


def test_the_last_successful_text_tool_is_the_answer():
    """``indices`` then ``vqa`` means the VQA answer is the answer."""
    result = compose(
        "q",
        [
            text_result("index summary", name="caption"),
            evidence_result({"water_pct": 12.0}, name="indices"),
            text_result("Water covers 12.0%.", name="vqa"),
        ],
    )

    assert result.answer == "Water covers 12.0%."


def test_composition_serialises():
    import json

    result = compose("q", [evidence_result({"water_pct": 12.0}), text_result("Water covers 12.0%.")])
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["cited_fraction"] == 1.0
