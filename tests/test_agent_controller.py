"""End-to-end tests for the loop, and for the invariants it is supposed to hold.

These run the real controller over synthetic scenes with no weights, no network
and no API key -- which is exactly the configuration the demo falls back to, so
the tests and the fallback path exercise each other.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from satquery.agent.controller import Controller
from satquery.agent.router import Plan, PlannedCall, Router, RouterError


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No cloud router and no VLM: the laptop configuration, made explicit."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("SATQUERY_ROUTER", "rules")
    monkeypatch.setenv("SATQUERY_VLM", "evidence")


@pytest.fixture
def before(session, scene_file):
    return session.add_image(scene_file("before.png", size=128))


@pytest.fixture
def after(session, scene_file):
    return session.add_image(scene_file("after.png", size=128, shift=16))


@pytest.fixture
def sar(session, tmp_path, write_raster):
    g = np.random.default_rng(3)
    array = g.gamma(4.0, 0.03, size=(2, 128, 128)).astype(np.float32)
    array[:, :32, :32] *= 0.02
    array[:, 80:110, 10:40] *= 15.0
    return session.add_image(write_raster(tmp_path / "s1_vv.tif", array), modality="sar")


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_a_single_image_question_is_answered_and_cited(session, before):
    result = Controller(session).answer(
        "How much of the scene is vegetation?", image_ids=[before.image_id]
    )

    assert not result.refused
    assert "%" in result.answer
    assert result.citations and result.citations[0]["tool"]
    assert result.intent == "vqa"
    assert 0.05 <= result.confidence <= 0.95


def test_a_bitemporal_query_runs_map_then_describe(session, before, after):
    result = Controller(session).answer(
        "What changed between these dates?", image_ids=[before.image_id, after.image_id]
    )

    tools = [t["tool"] for t in result.tools]
    assert tools.index("change_map") < tools.index("change_describe")
    assert result.pair_id is not None
    assert any(k.startswith("change_map.") for k in result.artifacts)


def test_two_images_without_a_declared_pair_still_form_one(session, before, after):
    """The GUI should not have to make a second call before asking a question."""
    result = Controller(session).answer("Describe the change.", image_ids=[before.image_id, after.image_id])

    assert result.pair_id is not None
    assert session.pair(result.pair_id).pair_type == "bitemporal"


def test_a_crossmodal_pair_reaches_fusion_and_a_sentence(session, before, sar):
    pair = session.add_pair(before.image_id, sar.image_id)

    result = Controller(session).answer("What land cover is present?", pair_id=pair.pair_id)

    tools = [t["tool"] for t in result.tools]
    assert "fusion" in tools
    assert tools[-1] in ("vqa", "caption")  # a specialist never addresses the user
    assert not result.refused


def test_an_unanswerable_question_refuses_with_what_was_measured(session, before):
    result = Controller(session).answer("How many boats are in the harbour?", image_ids=[before.image_id])

    assert result.refused
    assert "counting objects" in result.answer
    assert "Measured for this input" in result.answer
    assert result.confidence == 0.05


def test_an_unreadable_image_stops_the_query_with_the_reason(session, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    record = session.add_image(broken)

    result = Controller(session).answer("Describe this.", image_ids=[record.image_id])

    assert result.refused and not record.ok
    assert "cannot be answered" in result.answer
    assert result.menu == {"offered": [], "withheld": [], "context": ""}


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


class SpyRouter(Router):
    """Records what it was handed, so the test can assert on it."""

    name = "spy"

    def __init__(self) -> None:
        self.seen: list[tuple[str, Any]] = []

    def plan(self, query, menu):
        self.seen.append((query, menu))
        return Plan(intent="caption", calls=(PlannedCall("caption", {}),), router=self.name)


def test_the_router_receives_no_pixels_and_no_paths(session, before):
    """Invariant 01, checked on everything the router is actually given."""
    spy = SpyRouter()

    Controller(session, router=spy).answer("Describe this.", image_ids=[before.image_id])

    query, menu = spy.seen[0]
    assert before.path not in menu.context_summary
    assert before.path not in query
    for schema in menu.schemas():
        assert "image_id" not in json.dumps(schema)


def test_the_executor_injects_the_ids_the_router_never_saw(session, before):
    spy = SpyRouter()

    result = Controller(session, router=spy).answer("Describe this.", image_ids=[before.image_id])

    tool_entry = [e for e in result.trace["entries"] if e["step"] == "tool"][0]
    assert tool_entry["params"]["image_id"] == before.image_id


def test_withheld_tools_are_recorded_before_the_router_runs(session, before):
    """Invariant 02: the absence of a capability is a trace entry, not a silence."""
    result = Controller(session).answer("What changed?", image_ids=[before.image_id])

    withheld = [e for e in result.trace["entries"] if e["status"] == "withheld"]
    names = {e["name"] for e in withheld}
    assert {"change_map", "change_describe", "fusion"} <= names
    assert all(e["detail"] for e in withheld)
    route_index = next(i for i, e in enumerate(result.trace["entries"]) if e["step"] == "route")
    assert all(result.trace["entries"].index(e) < route_index for e in withheld)


def test_deterministic_evidence_precedes_the_text_tool(session, before):
    """Invariant 03: the numbers exist before the sentence that cites them."""
    result = Controller(session).answer("How much vegetation?", image_ids=[before.image_id])

    tools = [t["tool"] for t in result.tools]
    assert tools[0] == "indices"
    assert tools[-1] in ("vqa", "caption")


def test_only_a_text_tool_produces_the_answer(session, before, sar):
    """Invariant 04: a specialist emits numbers; one component speaks."""
    pair = session.add_pair(before.image_id, sar.image_id)

    result = Controller(session).answer("What is here?", pair_id=pair.pair_id)

    speaking = [t for t in result.tools if t["output"].get("answer")]
    assert {t["tool"] for t in speaking} <= {"vqa", "caption", "ground", "change_describe"}


# --------------------------------------------------------------------------- #
# The trace
# --------------------------------------------------------------------------- #


def test_the_trace_covers_every_stage_and_serialises(session, before, after):
    result = Controller(session).answer("What changed?", image_ids=[before.image_id, after.image_id])

    steps = [e["step"] for e in result.trace["entries"]]
    assert {"validate", "interpret", "route", "tool", "compose"} <= set(steps)
    json.dumps(result.to_dict())  # the API returns this verbatim
    assert result.trace["total_ms"] >= 0


def test_the_backends_that_served_are_named_in_the_result(session, before):
    result = Controller(session).answer("Describe this.", image_ids=[before.image_id])

    assert result.backends["text"]["backend"] == "evidence-rules"
    assert result.backends["router"]["backend"] == "rules"
    assert result.backends["change_map"]["reason"]


def test_a_failing_tool_does_not_abort_the_query(session, before, monkeypatch):
    """A partial answer with an honest gap beats a 500."""
    import satquery.agent.tools as tools_module

    real = tools_module._HANDLERS["indices"]

    def explode(params, ctx):
        raise RuntimeError("simulated tool failure")

    monkeypatch.setitem(tools_module._HANDLERS, "indices", explode)
    try:
        result = Controller(session).answer("How much vegetation?", image_ids=[before.image_id])
    finally:
        monkeypatch.setitem(tools_module._HANDLERS, "indices", real)

    failed = [t for t in result.tools if not t["ok"]]
    assert failed and "simulated tool failure" in failed[0]["error"]
    assert result.answer  # something was still returned


def test_a_router_failure_falls_back_and_says_so(session, before):
    class Broken(Router):
        name = "broken"

        def plan(self, query, menu):
            raise RouterError("429 rate limited")

    from satquery.agent.router import RuleRouter

    result = Controller(session, router=Broken(), fallback_router=RuleRouter()).answer(
        "Describe this.", image_ids=[before.image_id]
    )

    assert result.plan["fallback_reason"].startswith("429")
    assert result.plan["router"] == "rules"
    assert not result.refused


def test_a_router_failure_with_no_fallback_degrades_rather_than_raises(session, before):
    class Broken(Router):
        name = "broken"

        def plan(self, query, menu):
            raise RouterError("no network")

    result = Controller(session, router=Broken()).answer("Describe this.", image_ids=[before.image_id])

    assert result.refused
    assert [e for e in result.trace["entries"] if e["step"] == "route"][0]["status"] == "error"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_the_query_is_saved_and_reloadable(session, before):
    result = Controller(session).answer("Describe this.", image_ids=[before.image_id])

    stored = session.load_query(result.query_id)

    assert stored["result"]["answer"] == result.answer
    assert stored["trace"]["n_steps"] == result.trace["n_steps"]
    assert result.query_id in session.query_ids


def test_a_reopened_session_still_knows_its_images(session, before, tmp_config):
    from satquery.agent.session import SessionStore

    reopened = SessionStore(config=tmp_config).get(session.session_id)

    assert [r.image_id for r in reopened.images] == [before.image_id]
    assert reopened.image(before.image_id).ok
