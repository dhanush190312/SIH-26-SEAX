"""Tests for the cloud vision backend.

Every test here mocks the HTTP call: the suite must run offline, in seconds, and
without spending anyone's quota. What is being tested is the contract around the
call -- what goes up, what comes back, and what happens when it fails -- not
Mistral's answers.

The property that matters most is the last one: this backend is the only
component in the system that sends imagery off the machine, so it must never be
reachable from ``auto``.
"""

from __future__ import annotations

import base64
import json

import pytest

from satquery.agent.backends import EvidenceVLM, MistralVLM, select_vlm
from satquery.contracts import ImageRef, SatQueryError


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")
    monkeypatch.delenv("SATQUERY_VLM_MODEL", raising=False)


@pytest.fixture
def ref(scene_file):
    return ImageRef(image_id="i1", path=str(scene_file("scene.png", size=96)), modality="optical")


def reply(content: str, **extra):
    return {
        "model": "mistral-small-latest",
        "usage": {"total_tokens": 42},
        "choices": [{"message": {"content": content}}],
        **extra,
    }


def backend_returning(monkeypatch, content, capture=None):
    vlm = MistralVLM()

    def fake_post(body):
        if capture is not None:
            capture.append(body)
        return reply(content) if isinstance(content, str) else content

    monkeypatch.setattr(vlm, "_post", fake_post)
    return vlm


# --------------------------------------------------------------------------- #
# What goes up
# --------------------------------------------------------------------------- #


def test_the_image_is_sent_as_a_png_data_uri(monkeypatch, ref):
    sent = []
    vlm = backend_returning(monkeypatch, "a description", sent)

    vlm.answer("caption", [ref])

    content = sent[0]["messages"][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"].startswith("data:image/png;base64,")
    payload = images[0]["image_url"].split(",", 1)[1]
    assert base64.b64decode(payload)[:8] == b"\x89PNG\r\n\x1a\n"  # a real PNG


def test_two_images_go_up_as_two_parts(monkeypatch, ref):
    sent = []
    vlm = backend_returning(monkeypatch, "the change", sent)

    answer = vlm.answer("change_caption", [ref, ref])

    parts = [p for p in sent[0]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(parts) == 2
    assert answer.meta["images_sent"] == 2


def test_the_measurements_travel_with_the_image(monkeypatch, ref):
    """Invariant 03 across the network boundary: evidence arrives before the answer."""
    sent = []
    vlm = backend_returning(monkeypatch, "answer", sent)

    vlm.answer("caption", [ref], evidence={"vegetation_pct": 62.5, "water_pct": 3.0})

    text = [p for p in sent[0]["messages"][1]["content"] if p["type"] == "text"][0]["text"]
    assert "vegetation_pct: 62.5" in text
    assert "do not make it" in text  # the do-not-invent instruction survives


def test_the_prompt_is_the_same_one_the_served_qwen_uses(monkeypatch, ref):
    """A difference in answers should be a difference in models, not in wording."""
    from satquery.models.qwen_vl import PROMPTS, SYSTEM_PROMPT

    sent = []
    vlm = backend_returning(monkeypatch, "answer", sent)

    vlm.answer("vqa", [ref], question="how many ships?")

    assert sent[0]["messages"][0]["content"] == SYSTEM_PROMPT
    text = [p for p in sent[0]["messages"][1]["content"] if p["type"] == "text"][0]["text"]
    assert text.startswith(PROMPTS["vqa"].split("{")[0])
    assert "how many ships?" in text
    assert sent[0]["temperature"] == 0.0


def test_the_api_key_is_not_in_the_request_body(monkeypatch, ref):
    """It belongs in the Authorization header and nowhere a trace could capture."""
    sent = []
    vlm = backend_returning(monkeypatch, "answer", sent)

    vlm.answer("caption", [ref])

    assert "test-key-not-real" not in json.dumps(sent[0])


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #


def test_a_chatty_vqa_answer_is_shortened(monkeypatch, ref):
    vlm = backend_returning(monkeypatch, "The answer is: 3 buildings.")

    answer = vlm.answer("vqa", [ref], question="how many buildings?")

    assert answer.text == "3 buildings"
    assert not answer.refused


def test_a_caption_is_returned_verbatim(monkeypatch, ref):
    vlm = backend_returning(monkeypatch, "A river runs through farmland beside a town.")

    answer = vlm.answer("caption", [ref])

    assert answer.text.startswith("A river runs")


def test_grounding_parses_boxes_and_answers_in_words(monkeypatch, ref):
    decode = '```json\n[{"bbox_2d": [100, 200, 300, 400], "label": "reservoir"}]\n```'
    vlm = backend_returning(monkeypatch, decode)

    answer = vlm.answer("ground", [ref], question="the reservoir")

    assert len(answer.boxes) == 1
    assert answer.boxes[0].label == "reservoir"
    assert answer.text == "1 region(s) matching 'the reservoir'."   # not raw JSON
    assert any("stripped code fence" in n for n in answer.meta["box_notes"])
    assert any("raw decode" in n for n in answer.meta["box_notes"])


def test_grounding_with_no_match_says_so(monkeypatch, ref):
    vlm = backend_returning(monkeypatch, "[]")

    answer = vlm.answer("ground", [ref], question="an aircraft carrier")

    assert answer.boxes == ()
    assert "No region matching" in answer.text


def test_confidence_is_fixed_and_labelled_as_such(monkeypatch, ref):
    """The API returns no token probabilities; pretending otherwise is a lie."""
    vlm = backend_returning(monkeypatch, "an answer")

    answer = vlm.answer("caption", [ref])

    assert answer.confidence == MistralVLM.API_CONFIDENCE
    assert "no token probabilities" in answer.meta["confidence_source"]


# --------------------------------------------------------------------------- #
# When it fails
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("failure", [
    SatQueryError("mistral-small-latest returned HTTP 429: rate limited"),
    SatQueryError("mistral-small-latest is unreachable: timed out"),
])
def test_an_api_failure_refuses_rather_than_raising(monkeypatch, ref, failure):
    """A cloud outage degrades the answer; it must not abort the query loop."""
    vlm = MistralVLM()

    def explode(_body):
        raise failure

    monkeypatch.setattr(vlm, "_post", explode)

    answer = vlm.answer("caption", [ref])

    assert answer.refused and answer.text == ""
    assert "HTTP 429" in answer.meta["reason"] or "unreachable" in answer.meta["reason"]


def test_an_unexpected_response_shape_refuses(monkeypatch, ref):
    vlm = backend_returning(monkeypatch, {"choices": []})
    assert vlm.answer("caption", [ref]).refused


def test_an_empty_answer_refuses(monkeypatch, ref):
    vlm = backend_returning(monkeypatch, "   ")
    assert vlm.answer("caption", [ref]).refused


def test_an_unreadable_image_refuses_before_any_upload(monkeypatch, tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    sent = []
    vlm = backend_returning(monkeypatch, "unused", sent)

    answer = vlm.answer("caption", [ImageRef(image_id="x", path=str(broken))])

    assert answer.refused
    assert sent == []  # nothing left the machine


def test_a_task_with_no_prompt_refuses(monkeypatch, ref):
    vlm = backend_returning(monkeypatch, "unused")
    assert vlm.answer("change_map", [ref]).refused


# --------------------------------------------------------------------------- #
# Selection: the property that keeps imagery local by default
# --------------------------------------------------------------------------- #


def test_auto_never_selects_the_cloud_backend(monkeypatch):
    """Uploading imagery is an explicit decision, never an inherited default."""
    monkeypatch.setenv("SATQUERY_VLM", "auto")

    backend, _reason = select_vlm()

    assert not isinstance(backend, MistralVLM)


def test_explicit_selection_works_and_says_what_it_costs(monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM", "mistral")

    backend, reason = select_vlm()

    assert isinstance(backend, MistralVLM)
    assert "imagery is sent off this machine" in reason


def test_without_a_key_it_falls_back_instead_of_failing(monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM", "mistral")
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    backend, reason = select_vlm()

    assert isinstance(backend, EvidenceVLM)
    assert "MISTRAL_API_KEY" in reason


def test_the_model_is_configurable(monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM_MODEL", "mistral-medium-latest")
    assert MistralVLM().model == "mistral-medium-latest"
    assert "mistral-medium-latest" in MistralVLM().name


def test_describe_declares_that_it_uploads(monkeypatch):
    """Anything reporting backends must be able to show this without special-casing."""
    assert MistralVLM().describe()["sends_imagery"] is True
