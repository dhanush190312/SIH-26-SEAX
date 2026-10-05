"""Tests for routing and plan normalisation.

Two properties matter more than routing accuracy, and both are enforced rather
than prompted:

* the request body sent to Mistral contains no imagery, no array and no file
  path (invariant 01), and
* whatever plan comes back is repaired before it runs -- withheld tools dropped,
  ``change_map`` inserted before ``change_describe``, ``indices`` inserted before
  any text tool (invariant 03).

The second is what makes a cloud model safe to route with: it can propose a bad
plan, and the pipeline still cannot execute an unsupported one.
"""

from __future__ import annotations

import json

import pytest

from satquery.agent.router import (
    MistralRouter,
    Plan,
    PlannedCall,
    RouterError,
    RuleRouter,
    build_router,
)
from satquery.agent.tools import TOOLS, ToolMenu, build_menu


def menu_with(*names: str) -> ToolMenu:
    offered = tuple(TOOLS[n] for n in names)
    withheld = tuple((n, f"{n} is not supported by this input") for n in TOOLS if n not in names)
    return ToolMenu(offered=offered, withheld=withheld, context_summary="img_1: 512x512, optical")


SINGLE = ("vqa", "caption", "ground", "indices")
BITEMPORAL = SINGLE + ("change_map", "change_describe")
CROSSMODAL = SINGLE + ("fusion",)


# --------------------------------------------------------------------------- #
# The offline router
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query,expected_last,intent",
    [
        ("Describe this image.", "caption", "caption"),
        ("What is in this image?", "caption", "caption"),
        ("Where is the water?", "ground", "ground"),
        ("Find the storage tanks.", "ground", "ground"),
        ("Is there a bridge over the river?", "vqa", "vqa"),
        ("How many aircraft are on the apron?", "vqa", "vqa"),
    ],
)
def test_single_image_routing(query, expected_last, intent):
    plan = RuleRouter().plan(query, menu_with(*SINGLE))

    assert plan.tools[-1] == expected_last
    assert plan.intent == intent


def test_a_measurable_question_gets_the_deterministic_tool_first():
    plan = RuleRouter().plan("How much vegetation is there?", menu_with(*SINGLE))

    assert plan.tools[0] == "indices"
    assert plan.tools[-1] == "vqa"


def test_a_bitemporal_pair_maps_the_change_before_describing_it():
    plan = RuleRouter().plan("What changed between these dates?", menu_with(*BITEMPORAL))

    assert plan.tools.index("change_map") < plan.tools.index("change_describe")
    assert plan.intent == "change_vqa"
    describe = [c for c in plan.calls if c.tool == "change_describe"][0]
    assert describe.params["question"]


def test_a_change_description_request_carries_no_question():
    plan = RuleRouter().plan("Describe the differences.", menu_with(*BITEMPORAL))

    describe = [c for c in plan.calls if c.tool == "change_describe"][0]
    assert not describe.params.get("question")
    assert plan.intent == "change_caption"


def test_a_single_image_request_on_a_pair_still_routes_to_one_image():
    """"describe this image" over a registered pair is not a change question."""
    plan = RuleRouter().plan("Describe this image.", menu_with(*BITEMPORAL))

    assert "change_map" not in plan.tools
    assert plan.tools[-1] == "caption"


def test_a_crossmodal_pair_fuses_and_then_speaks():
    """Fusion emits class percentages; something has to turn them into an answer."""
    plan = RuleRouter().plan("What land cover types are present?", menu_with(*CROSSMODAL))

    assert "fusion" in plan.tools
    assert plan.tools[-1] in ("vqa", "caption")
    assert plan.intent == "fusion"


def test_an_empty_menu_produces_an_unsupported_plan():
    plan = RuleRouter().plan("What changed?", ToolMenu(offered=(), withheld=()))

    assert plan.calls == ()
    assert plan.intent == "unsupported"


def test_routing_is_deterministic():
    first = RuleRouter().plan("How much water is there?", menu_with(*SINGLE))
    second = RuleRouter().plan("How much water is there?", menu_with(*SINGLE))
    assert first.tools == second.tools


# --------------------------------------------------------------------------- #
# Plan normalisation
# --------------------------------------------------------------------------- #


def test_a_call_to_a_withheld_tool_is_dropped_with_the_reason():
    plan = Plan(
        intent="fusion", calls=(PlannedCall("fusion", {}),), router="test"
    ).normalized(menu_with(*SINGLE))

    assert "fusion" not in plan.tools
    assert any("dropped 'fusion'" in note for note in plan.notes)


def test_a_hallucinated_tool_name_is_dropped():
    plan = Plan(
        intent="vqa", calls=(PlannedCall("enhance", {}), PlannedCall("vqa", {"question": "q"})),
        router="test",
    ).normalized(menu_with(*SINGLE))

    assert plan.tools[-1] == "vqa"
    assert any("unknown tool" in note for note in plan.notes)


def test_change_map_is_inserted_before_a_lone_change_describe():
    plan = Plan(
        intent="change_caption", calls=(PlannedCall("change_describe", {}),), router="test"
    ).normalized(menu_with(*BITEMPORAL))

    assert plan.tools.index("change_map") < plan.tools.index("change_describe")
    assert any("inserted change_map" in note for note in plan.notes)


def test_indices_is_inserted_before_a_text_tool():
    plan = Plan(
        intent="vqa", calls=(PlannedCall("vqa", {"question": "q"}),), router="test"
    ).normalized(menu_with(*SINGLE))

    assert plan.tools[0] == "indices"
    assert any("indices first" in note for note in plan.notes)


def test_a_text_tool_is_appended_after_a_lone_fusion_call():
    plan = Plan(
        intent="fusion", calls=(PlannedCall("fusion", {}),), router="test"
    ).normalized(menu_with(*CROSSMODAL), "What is here?")

    assert plan.tools[-1] in ("vqa", "caption")
    assert any("fusion emits numbers" in c.reason for c in plan.calls)


def test_a_plan_that_only_measures_gets_a_voice():
    """Observed from mistral-small-latest: it planned change_map and stopped.

    change_map emits a mask and geometry, so the query measured the scene
    correctly and then answered nothing -- which reads as a failure even though
    the measurement succeeded. Invariant 04 says one component speaks; this is
    the guarantee that one of them is always in the plan.
    """
    plan = Plan(
        intent="unsupported", calls=(PlannedCall("change_map", {}),), router="mistral"
    ).normalized(menu_with(*BITEMPORAL), "What changed between these two dates?")

    assert plan.tools[-1] == "change_describe"
    assert plan.calls[-1].params["question"] == "What changed between these two dates?"
    assert plan.intent == "change_vqa"  # relabelled from the repaired plan
    assert any("reach an answer" in note for note in plan.notes)


def test_a_measure_only_plan_without_a_question_gets_a_description():
    plan = Plan(
        intent="unsupported", calls=(PlannedCall("change_map", {}),), router="mistral"
    ).normalized(menu_with(*BITEMPORAL), "changes please")

    assert plan.tools[-1] == "change_describe"
    assert not plan.calls[-1].params.get("question")


def test_a_lone_indices_plan_also_gets_a_voice():
    plan = Plan(
        intent="unsupported", calls=(PlannedCall("indices", {}),), router="mistral"
    ).normalized(menu_with(*SINGLE), "How much water is there?")

    assert plan.tools[-1] in ("vqa", "caption")


def test_a_plan_that_already_speaks_is_left_alone():
    plan = Plan(
        intent="change_caption",
        calls=(PlannedCall("change_map", {}), PlannedCall("change_describe", {})),
        router="mistral",
    ).normalized(menu_with(*BITEMPORAL), "describe the change")

    assert plan.tools == ("indices", "change_map", "change_describe")
    assert sum(1 for t in plan.tools if t == "change_describe") == 1


def test_normalisation_does_not_duplicate_what_is_already_there():
    plan = Plan(
        intent="change_caption",
        calls=(PlannedCall("indices", {}), PlannedCall("change_map", {}), PlannedCall("change_describe", {})),
        router="test",
    ).normalized(menu_with(*BITEMPORAL))

    assert plan.tools == ("indices", "change_map", "change_describe")
    assert plan.notes == ()


# --------------------------------------------------------------------------- #
# The cloud router
# --------------------------------------------------------------------------- #


def test_no_api_key_is_an_error_at_construction(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RouterError, match="MISTRAL_API_KEY"):
        MistralRouter()


def test_the_system_prompt_demands_a_speaking_tool():
    """The prompt asks; normalisation guarantees. Both, because either alone is weak."""
    from satquery.agent.router import SYSTEM_PROMPT

    assert "must END with a tool that produces text" in SYSTEM_PROMPT


def test_the_request_body_contains_no_imagery(monkeypatch, session, scene_file):
    """Invariant 01, asserted on the bytes that would actually leave the machine.

    Built from a *real* session so the check is against real ids and real paths
    rather than a hand-written menu that could not have leaked them anyway. The
    system prompt does mention the words "image_id" -- it tells the model not to
    send one -- so the assertion is about values, not vocabulary.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    record = session.add_image(scene_file("scene.png", size=64))
    menu = build_menu([record], None)

    body = MistralRouter().request_body("What is in this image?", menu)
    serialised = json.dumps(body)

    assert record.path not in serialised
    assert ".png" not in serialised and ".tif" not in serialised
    assert "base64" not in serialised
    # The one place an id may appear is the human-readable validation summary,
    # which is text about the image, not the image.
    assert serialised.count(record.image_id) == 1
    assert record.image_id in body["messages"][1]["content"]
    assert body["tool_choice"] == "any"  # a plan, never an answer
    assert {t["function"]["name"] for t in body["tools"]} == set(menu.names)


def test_a_tool_call_response_becomes_a_plan(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    router = MistralRouter()
    response = {
        "model": "mistral-small-latest",
        "usage": {"total_tokens": 120},
        "choices": [
            {
                "message": {
                    "content": "measuring first",
                    "tool_calls": [
                        {"function": {"name": "change_map", "arguments": "{}"}},
                        {
                            "function": {
                                "name": "change_describe",
                                "arguments": '{"question": "what changed?", "image_id": "img_9"}',
                            }
                        },
                    ],
                }
            }
        ],
    }
    monkeypatch.setattr(router, "_post", lambda body: response)

    plan = router.plan("What changed?", menu_with(*BITEMPORAL))

    # Normalisation prepends indices; the router's own two calls keep their order.
    assert plan.tools == ("indices", "change_map", "change_describe")
    assert plan.intent == "change_vqa"
    # An id the router invented is stripped: the executor owns session state.
    assert "image_id" not in [k for c in plan.calls for k in c.params]
    assert plan.raw["usage"]["total_tokens"] == 120


def test_prose_instead_of_a_tool_call_is_a_router_error(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    router = MistralRouter()
    monkeypatch.setattr(
        router, "_post", lambda body: {"choices": [{"message": {"content": "It looks like a city."}}]}
    )

    with pytest.raises(RouterError, match="no tool call"):
        router.plan("what is this?", menu_with(*SINGLE))


def test_unparseable_arguments_do_not_lose_the_call(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    router = MistralRouter()
    monkeypatch.setattr(
        router,
        "_post",
        lambda body: {
            "choices": [{"message": {"tool_calls": [{"function": {"name": "caption", "arguments": "{oops"}}]}}]
        },
    )

    plan = router.plan("describe", menu_with(*SINGLE))

    assert "caption" in plan.tools


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_without_a_key_the_offline_router_is_primary_and_alone(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("SATQUERY_ROUTER", raising=False)

    primary, fallback, reason = build_router()

    assert primary.name == "rules" and fallback is None
    assert "MISTRAL_API_KEY is not set" in reason


def test_with_a_key_mistral_leads_and_the_rules_catch_it(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.delenv("SATQUERY_ROUTER", raising=False)

    primary, fallback, reason = build_router()

    assert primary.name == "mistral" and fallback.name == "rules"


def test_offline_can_be_forced_even_with_a_key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("SATQUERY_ROUTER", "rules")

    primary, fallback, _reason = build_router()

    assert primary.name == "rules" and fallback is None
