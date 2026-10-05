"""Tests for the Qwen2.5-VL and EarthDial predictors.

No weights, no network, no GPU. The point of these tests is that everything
*around* the forward pass -- prompt construction, two-image message assembly,
box parsing, answer extraction, the missing-extra error, VRAM release -- is
verified on the laptop, because a five-hour Colab run is a terrible place to
discover that grounding output was silently unparseable.

The model/processor pair is faked. The fakes speak the same protocol the real
ones do (``apply_chat_template`` -> ``__call__`` -> ``generate`` ->
``batch_decode``, and InternVL's ``model.chat``), and they record what they were
handed so the assertions can be about the *call*, not just the return value.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from satquery.contracts import Prediction, Sample, SatQueryError, UnsupportedTask
from satquery.models import earthdial as ed
from satquery.models import qwen_vl as qv
from satquery.models.earthdial import EarthDialPredictor
from satquery.models.qwen_vl import QwenVLPredictor
from satquery.registry import build_predictor

WEIGHTS = pytest.mark.skipif(
    not os.environ.get("SATQUERY_WEIGHT_TESTS"),
    reason="downloads multi-GB weights; set SATQUERY_WEIGHT_TESTS=1 on a GPU box",
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeBatch(dict):
    """Stand-in for transformers' BatchFeature: a dict that can be moved."""

    moved_to: Any = None

    def to(self, device: Any) -> "_FakeBatch":
        self.moved_to = device
        return self


class _FakeProcessor:
    def __init__(self, reply: str, prompt_tokens: int = 7) -> None:
        self.reply = reply
        self.prompt_tokens = prompt_tokens
        self.messages: Any = None
        self.images: Any = None
        self.text: Any = None
        self.call_kwargs: dict[str, Any] = {}
        self.decoded: Any = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "<chat>"

    def __call__(self, text=None, images=None, **kwargs):
        self.text = text
        self.images = images
        self.call_kwargs = kwargs
        return _FakeBatch(
            input_ids=np.arange(self.prompt_tokens, dtype=np.int64)[None, :],
            pixel_values=np.zeros((len(images or []), 3, 4, 4), dtype=np.float32),
        )

    def batch_decode(self, ids, **kwargs):
        self.decoded = np.asarray(ids)
        return [self.reply]


class _FakeModel:
    device = "cpu"

    def __init__(self, prompt_tokens: int = 7, new_tokens: int = 5) -> None:
        self.prompt_tokens = prompt_tokens
        self.new_tokens = new_tokens
        self.generate_kwargs: dict[str, Any] = {}

    def eval(self) -> "_FakeModel":
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        total = self.prompt_tokens + self.new_tokens
        return SimpleNamespace(sequences=np.arange(total, dtype=np.int64)[None, :], scores=None)


class _FakeChatModel:
    """InternVL-style: one ``chat`` method returning a string."""

    device = "cpu"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def eval(self) -> "_FakeChatModel":
        return self

    def chat(self, tokenizer, pixel_values, question, generation_config, **kwargs):
        self.calls.append(
            {
                "tokenizer": tokenizer,
                "pixel_values": pixel_values,
                "question": question,
                "generation_config": generation_config,
                "kwargs": kwargs,
            }
        )
        return self.reply


def _fake_prepare_image(ref, *, max_side=768, modality=None):
    """Stand-in for prepare_image: no disk, no sibling modules, fixed geometry."""
    return f"PIL:{ref.image_id}", {
        "image_id": ref.image_id,
        "path": ref.path,
        "modality": modality or "optical",
        "width": 512,
        "height": 256,
        "steps": ("read_raster(fake)", "percentile_stretch(2,98)", f"resize_max_side({max_side})"),
    }


@pytest.fixture
def fake_qwen(monkeypatch):
    """Build a QwenVLPredictor whose loader and image reader are fakes."""

    def _make(reply: str, **kwargs: Any):
        processor = _FakeProcessor(reply)
        model = _FakeModel(prompt_tokens=processor.prompt_tokens)
        predictor = QwenVLPredictor(load_in_4bit=False, device="cpu", **kwargs)
        calls = {"n": 0}

        def _load():
            calls["n"] += 1
            return processor, model

        monkeypatch.setattr(predictor, "_load", _load)
        monkeypatch.setattr(qv, "prepare_image", _fake_prepare_image)
        return predictor, processor, model, calls

    return _make


@pytest.fixture
def fake_earthdial(monkeypatch):
    def _make(reply: str, **kwargs: Any):
        model = _FakeChatModel(reply)
        tokenizer = object()
        predictor = EarthDialPredictor(load_in_4bit=False, device="cpu", **kwargs)
        monkeypatch.setattr(predictor, "_load", lambda: (tokenizer, model))
        monkeypatch.setattr(qv, "prepare_image", _fake_prepare_image)
        # torch stays None, so build_pixel_values returns numpy and the fake
        # never needs a GPU.
        monkeypatch.setattr(ed, "_to_tile", lambda image, input_size=448: np.zeros(
            (3, input_size, input_size), dtype=np.float32
        ))
        return predictor, model

    return _make


# --------------------------------------------------------------------------- #
# Defaults: the T4 contract
# --------------------------------------------------------------------------- #


def test_defaults_are_t4_safe():
    p = QwenVLPredictor(device="cpu")
    assert p.model_id == "Qwen/Qwen2.5-VL-7B-Instruct", "the 3B is a non-commercial licence"
    assert p.dtype == "float16"
    assert p.attn_implementation == "sdpa"
    assert p.load_in_4bit is True
    assert p.max_side == 768
    assert p.loaded is False
    assert p.describe()["model_id"] == p.model_id


def test_earthdial_defaults():
    p = EarthDialPredictor(device="cpu")
    assert p.model_id == "akshaydudhane/EarthDial_4B_RGB"
    assert p.dtype == "float16"
    assert p.attn_implementation == "sdpa"


def test_registry_names_resolve_without_loading_weights():
    for name in ("qwen2_5_vl", "earthdial"):
        predictor = build_predictor(name, device="cpu")
        assert predictor.name == name
        assert predictor.loaded is False


def test_supports_only_generative_tasks():
    p = QwenVLPredictor(device="cpu")
    for task in ("vqa", "caption", "ground", "change_vqa", "change_caption"):
        assert p.supports(task)
    for task in ("change_map", "fusion_multilabel"):
        assert not p.supports(task)


def test_max_new_tokens_floor_per_task():
    p = QwenVLPredictor(device="cpu")  # default 64
    assert p.max_new_tokens_for("vqa") == 64
    assert p.max_new_tokens_for("caption") == 192, "a detailed caption does not fit in 64 tokens"
    assert p.max_new_tokens_for("ground") == 256
    explicit = QwenVLPredictor(device="cpu", max_new_tokens=512)
    assert explicit.max_new_tokens_for("vqa") == 512, "an explicit budget is never lowered"


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def test_prompts_cover_every_supported_task():
    for prompts, supported in (
        (qv.PROMPTS, QwenVLPredictor.supported),
        (ed.PROMPTS, EarthDialPredictor.supported),
    ):
        assert set(prompts) == set(supported)


def test_build_prompt_inserts_question(make_sample):
    sample = make_sample("s1", "vqa", question="how many aircraft are there?")
    prompt = qv.build_prompt(sample)
    assert "how many aircraft are there?" in prompt
    assert "{question}" not in prompt


def test_build_prompt_caption_has_no_placeholder(make_sample):
    prompt = qv.build_prompt(make_sample("s1", "caption"))
    assert "{" not in prompt and "}" not in prompt
    assert "satellite image" in prompt


def test_ground_prompt_demands_the_scale_the_parser_uses(make_sample):
    prompt = qv.build_prompt(make_sample("s1", "ground", question="the large building"))
    assert "0-1000" in prompt, "prompt and parse_boxes(scale=1000) are one contract"
    assert "bbox_2d" in prompt
    assert "the large building" in prompt
    assert QwenVLPredictor.coord_scale == 1000.0


def test_change_prompts_order_the_pair(make_sample):
    for task in ("change_vqa", "change_caption"):
        prompt = qv.build_prompt(make_sample("s1", task))
        lowered = prompt.lower()
        assert "first" in lowered and "second" in lowered
        assert "earlier" in lowered and "later" in lowered


def test_build_prompt_unknown_task_raises(make_sample):
    sample = make_sample("s1", "change_map", meta={"mask_path": "m.png"})
    with pytest.raises(UnsupportedTask):
        qv.build_prompt(sample)


def test_build_prompt_appends_deterministic_tool_context(make_sample):
    sample = make_sample("s1", "vqa", meta={"tool_context": "NDWI water = 12.4%"})
    prompt = qv.build_prompt(sample)
    assert "NDWI water = 12.4%" in prompt
    assert prompt.index("NDWI") > prompt.index("Question:")


def test_earthdial_prompts_are_not_qwens():
    assert ed.PROMPTS != qv.PROMPTS, "separate instruction tuning, separate prompts"


# --------------------------------------------------------------------------- #
# Message / question assembly
# --------------------------------------------------------------------------- #


def test_build_messages_single_image():
    messages = qv.build_messages("what is this?", 1)
    assert [m["role"] for m in messages] == ["system", "user"]
    content = messages[-1]["content"]
    assert content == [{"type": "image"}, {"type": "text", "text": "what is this?"}]


def test_build_messages_pair_uses_one_user_turn():
    messages = qv.build_messages("what changed?", 2)
    user_turns = [m for m in messages if m["role"] == "user"]
    assert len(user_turns) == 1, "a split pair loses the earlier/later referent"
    content = user_turns[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]


def test_build_messages_without_system():
    messages = qv.build_messages("x", 1, system=None)
    assert [m["role"] for m in messages] == ["user"]


def test_build_messages_rejects_zero_images():
    with pytest.raises(ValueError):
        qv.build_messages("x", 0)


def test_earthdial_question_placeholders():
    assert ed.build_question("describe", 1) == "<image>\ndescribe"
    paired = ed.build_question("what changed?", 2)
    assert paired.count("<image>") == 2
    assert "Image-1:" in paired and "Image-2:" in paired
    assert paired.endswith("what changed?")


# --------------------------------------------------------------------------- #
# Box parsing
# --------------------------------------------------------------------------- #


def test_parse_boxes_qwen_json():
    text = '[{"bbox_2d": [100, 200, 300, 400], "label": "ship"}]'
    boxes, notes = qv.parse_boxes(text, width=1000, height=500)
    assert len(boxes) == 1
    assert boxes[0].as_xyxy() == (100.0, 100.0, 300.0, 200.0)
    assert boxes[0].label == "ship"
    assert notes == ()


def test_parse_boxes_reads_score_and_multiple_objects():
    text = (
        '[{"bbox_2d": [0, 0, 500, 500], "label": "a", "score": 0.9},'
        ' {"bbox": [500, 500, 1000, 1000], "name": "b", "confidence": 0.4}]'
    )
    boxes, _ = qv.parse_boxes(text, width=100, height=100)
    assert [b.label for b in boxes] == ["a", "b"]
    assert [b.score for b in boxes] == [0.9, 0.4]
    assert boxes[1].as_xyxy() == (50.0, 50.0, 100.0, 100.0)


def test_parse_boxes_strips_code_fence():
    text = '```json\n[{"bbox_2d": [0, 0, 1000, 1000]}]\n```'
    boxes, notes = qv.parse_boxes(text, width=64, height=64)
    assert boxes[0].as_xyxy() == (0.0, 0.0, 64.0, 64.0)
    assert "stripped code fence" in notes


def test_parse_boxes_internvl_nested_box_tag():
    text = "<ref>the runway</ref><box>[[10, 20, 30, 40], [50, 60, 70, 80]]</box>"
    boxes, _ = qv.parse_boxes(text, width=1000, height=1000)
    assert [b.as_xyxy() for b in boxes] == [(10.0, 20.0, 30.0, 40.0), (50.0, 60.0, 70.0, 80.0)]


def test_parse_boxes_qwen2_point_pair_form():
    text = "<|box_start|>(100,150),(300,350)<|box_end|>"
    boxes, _ = qv.parse_boxes(text, width=1000, height=1000)
    assert boxes[0].as_xyxy() == (100.0, 150.0, 300.0, 350.0)


def test_parse_boxes_bare_quad_fallback():
    boxes, _ = qv.parse_boxes("The building is at [10, 10, 110, 110].", width=1000, height=1000)
    assert boxes[0].as_xyxy() == (10.0, 10.0, 110.0, 110.0)


def test_parse_boxes_malformed_yields_no_boxes_and_a_note():
    for text in ("I cannot find any ship in this image.", "{not json at all", "[]"):
        boxes, notes = qv.parse_boxes(text, width=64, height=64)
        assert boxes == ()
        assert notes, f"malformed decode {text!r} must explain itself"


def test_parse_boxes_empty_decode():
    boxes, notes = qv.parse_boxes("", width=64, height=64)
    assert boxes == () and notes == ("empty decode",)
    boxes, notes = qv.parse_boxes(None, width=64, height=64)
    assert boxes == () and notes == ("empty decode",)


def test_parse_boxes_clips_out_of_range_coordinates():
    boxes, notes = qv.parse_boxes(
        '[{"bbox_2d": [-50, 500, 1400, 1200]}]', width=100, height=100
    )
    assert boxes[0].as_xyxy() == (0.0, 50.0, 100.0, 100.0)
    assert any("clipped" in n for n in notes)


def test_parse_boxes_reorders_reversed_coordinates():
    boxes, notes = qv.parse_boxes('[{"bbox_2d": [800, 900, 200, 100]}]', width=1000, height=1000)
    assert boxes[0].as_xyxy() == (200.0, 100.0, 800.0, 900.0)
    assert any("reordered" in n for n in notes)


def test_parse_boxes_flags_suspected_0_1_normalisation():
    boxes, notes = qv.parse_boxes('[{"bbox_2d": [0.1, 0.1, 0.5, 0.5]}]', width=1000, height=1000)
    assert len(boxes) == 1, "still returned, never silently rescaled"
    assert any("0-1 normalised" in n for n in notes)


def test_parse_boxes_never_raises_on_hostile_input():
    for text in ("[[[[[", '{"bbox_2d": "not numbers"}', "(1,2),(3", "1000" * 200):
        boxes, notes = qv.parse_boxes(text, width=32, height=32)
        assert isinstance(boxes, tuple) and isinstance(notes, tuple)


# --------------------------------------------------------------------------- #
# predict()
# --------------------------------------------------------------------------- #


def test_predict_vqa_extracts_the_short_answer(fake_qwen, make_sample):
    predictor, processor, model, _ = fake_qwen("The answer is: two houses.")
    prediction = predictor.predict(make_sample("s1", "vqa", question="how many houses?"))

    assert isinstance(prediction, Prediction)
    assert prediction.text == "two houses"
    assert prediction.model == "qwen2_5_vl"
    assert prediction.sample_id == "s1"
    assert prediction.latency_ms is not None and prediction.latency_ms >= 0.0
    assert prediction.confidence is None, "no scores from this fake, so no invented number"
    assert prediction.evidence["confidence_source"] == "unavailable"
    assert prediction.evidence["raw"] == "The answer is: two houses."
    assert "how many houses?" in prediction.evidence["prompt"]
    assert prediction.evidence["preprocess"][0][0].startswith("read_raster")
    assert prediction.evidence["model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 64


def test_predict_trims_the_prompt_before_decoding(fake_qwen, make_sample):
    predictor, processor, model, _ = fake_qwen("yes")
    predictor.predict(make_sample("s1", "vqa"))
    assert processor.decoded.shape == (1, model.new_tokens), "prompt tokens must be trimmed"


def test_predict_caption_keeps_the_full_decode(fake_qwen, make_sample):
    text = "A dense residential block crossed by two roads.  "
    predictor, processor, model, _ = fake_qwen(text)
    prediction = predictor.predict(make_sample("s1", "caption"))
    assert prediction.text == text.strip()
    assert model.generate_kwargs["max_new_tokens"] == 192


def test_predict_ground_parses_boxes_against_the_preprocessed_size(fake_qwen, make_sample):
    predictor, _, _, _ = fake_qwen('[{"bbox_2d": [0, 0, 500, 1000], "label": "hangar"}]')
    prediction = predictor.predict(make_sample("s1", "ground", question="the hangar"))
    # _fake_prepare_image reports 512x256, so 0-1000 maps onto that, not the
    # dataset's original raster size.
    assert prediction.boxes[0].as_xyxy() == (0.0, 0.0, 256.0, 256.0)
    assert prediction.boxes[0].label == "hangar"
    assert prediction.evidence["notes"] == []


def test_predict_ground_malformed_output_is_a_note_not_an_exception(fake_qwen, make_sample):
    predictor, _, _, _ = fake_qwen("There is no hangar in this scene.")
    prediction = predictor.predict(make_sample("s1", "ground", question="the hangar"))
    assert prediction.boxes == ()
    assert prediction.ok
    assert prediction.evidence["notes"]


def test_predict_change_vqa_passes_both_images_in_one_message(fake_qwen, make_sample):
    predictor, processor, _, _ = fake_qwen("yes")
    sample = make_sample("s1", "change_vqa", question="did buildings appear?")
    prediction = predictor.predict(sample)

    assert len(processor.images) == 2
    user = [m for m in processor.messages if m["role"] == "user"]
    assert len(user) == 1
    assert [c["type"] for c in user[0]["content"]] == ["image", "image", "text"]
    assert prediction.text == "yes"
    assert len(prediction.evidence["images"]) == 2
    assert len(prediction.evidence["preprocess"]) == 2


def test_predict_records_the_messages_actually_sent(fake_qwen, make_sample):
    predictor, _, _, _ = fake_qwen("yes")
    prediction = predictor.predict(make_sample("s1", "vqa"))
    generation = prediction.evidence["generation"]
    assert generation["n_images"] == 1
    assert generation["messages"][-1]["role"] == "user"
    assert generation["prompt_tokens"] == 7


def test_predict_rejects_unsupported_task(fake_qwen, make_sample):
    predictor, _, _, _ = fake_qwen("x")
    sample = make_sample("s1", "fusion_multilabel")
    with pytest.raises(UnsupportedTask):
        predictor.predict(sample)


def test_predict_empty_answer_is_noted(fake_qwen, make_sample):
    predictor, _, _, _ = fake_qwen("   ")
    prediction = predictor.predict(make_sample("s1", "vqa"))
    assert prediction.text == ""
    assert prediction.evidence["notes"] == ["answer extraction produced an empty string"]


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_model_loads_lazily_and_only_once(fake_qwen, make_sample):
    predictor, _, _, calls = fake_qwen("yes")
    assert calls["n"] == 0, "constructing a predictor must not load weights"
    predictor.predict(make_sample("s1", "vqa"))
    predictor.predict(make_sample("s2", "vqa"))
    assert calls["n"] == 1
    assert predictor.loaded


def test_close_releases_state_and_allows_reload(fake_qwen, make_sample):
    predictor, _, _, calls = fake_qwen("yes")
    predictor.predict(make_sample("s1", "vqa"))
    predictor.close()
    assert predictor.loaded is False
    assert predictor._processor is None
    predictor.predict(make_sample("s2", "vqa"))
    assert calls["n"] == 2


def test_context_manager_closes(fake_qwen, make_sample):
    predictor, _, _, _ = fake_qwen("yes")
    with predictor as p:
        p.predict(make_sample("s1", "vqa"))
        assert p.loaded
    assert predictor.loaded is False


def test_close_before_load_is_a_noop():
    QwenVLPredictor(device="cpu").close()


# --------------------------------------------------------------------------- #
# Missing optional dependencies
# --------------------------------------------------------------------------- #


def _break_import(monkeypatch, module, *broken: str):
    def _fake_import(name: str):
        if name in broken:
            raise ImportError(f"No module named {name!r}")
        return __import__(name)

    monkeypatch.setattr(module, "_import_module", _fake_import)


@pytest.mark.parametrize("missing", ["torch", "transformers"])
def test_missing_vlm_extra_gives_an_actionable_error(monkeypatch, make_sample, missing):
    _break_import(monkeypatch, qv, missing)
    predictor = QwenVLPredictor(device="cpu")
    with pytest.raises(SatQueryError) as excinfo:
        predictor.predict(make_sample("s1", "vqa"))
    message = str(excinfo.value)
    assert missing in message
    assert "satquery[vlm]" in message
    assert "pip install" in message


def test_earthdial_missing_extra_gives_the_same_error(monkeypatch, make_sample):
    _break_import(monkeypatch, qv, "torch")
    with pytest.raises(SatQueryError) as excinfo:
        EarthDialPredictor(device="cpu").predict(make_sample("s1", "vqa"))
    assert "satquery[vlm]" in str(excinfo.value)


def test_missing_bitsandbytes_degrades_to_full_precision(monkeypatch):
    _break_import(monkeypatch, qv, "bitsandbytes")
    predictor = QwenVLPredictor(device="cpu", load_in_4bit=True)
    assert predictor._quantization_config(SimpleNamespace(float16="fp16"), None) is None


def test_unknown_dtype_is_rejected_before_a_download():
    predictor = QwenVLPredictor(device="cpu", dtype="float9")
    with pytest.raises(SatQueryError, match="unknown torch dtype"):
        predictor._resolve_dtype(SimpleNamespace(float16="fp16"))


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


def test_mean_token_probability_without_scores_is_none():
    outputs = SimpleNamespace(sequences=np.zeros((1, 3)), scores=None)
    assert qv.mean_token_probability(object(), outputs) is None


def test_mean_token_probability_from_transition_scores():
    torch = pytest.importorskip("torch")

    class _Model:
        def compute_transition_scores(self, sequences, scores, normalize_logits=True):
            # log probabilities of 0.5 and 0.25
            return torch.log(torch.tensor([[0.5, 0.25]]))

    outputs = SimpleNamespace(sequences=torch.zeros(1, 3, dtype=torch.long), scores=(1, 2))
    got = qv.mean_token_probability(_Model(), outputs)
    assert got == pytest.approx(0.375, abs=1e-6)


def test_mean_token_probability_ignores_masked_infinities():
    torch = pytest.importorskip("torch")

    class _Model:
        def compute_transition_scores(self, sequences, scores, normalize_logits=True):
            return torch.tensor([[float("-inf"), 0.0]])

    outputs = SimpleNamespace(sequences=torch.zeros(1, 2, dtype=torch.long), scores=(1,))
    assert qv.mean_token_probability(_Model(), outputs) == pytest.approx(1.0)


def test_mean_token_probability_swallows_model_errors():
    class _Model:
        def compute_transition_scores(self, *a, **k):
            raise RuntimeError("quantised head has no scores")

    outputs = SimpleNamespace(sequences=np.zeros((1, 2)), scores=(1,))
    assert qv.mean_token_probability(_Model(), outputs) is None


# --------------------------------------------------------------------------- #
# Image preparation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bands,modality,expected",
    [
        (3, "optical", (0, 1, 2)),
        (13, "optical", (0, 1, 2)),
        (2, "sar", (0, 1, 0)),
        (1, "sar", (0, 0, 0)),
    ],
)
def test_band_selection(bands, modality, expected):
    assert qv._band_selection(bands, modality) == expected


def test_band_selection_override_wins():
    assert qv._band_selection(13, "optical", (3, 2, 1)) == (3, 2, 1)


def test_band_selection_rejects_bad_override():
    with pytest.raises(SatQueryError):
        qv._band_selection(3, "optical", (0, 1, 9))
    with pytest.raises(SatQueryError):
        qv._band_selection(3, "optical", (0, 1))


def test_array_to_pil_round_trips():
    pytest.importorskip("PIL")
    array = np.zeros((3, 4, 6), dtype=np.uint8)
    array[0] = 255
    image = qv._array_to_pil(array)
    assert image.size == (6, 4)  # PIL is (width, height)
    assert np.asarray(image)[0, 0].tolist() == [255, 0, 0]


def test_prepare_image_records_the_optical_pipeline(tmp_path, write_png, make_image_ref):
    pytest.importorskip(
        "satquery.io.raster", reason="satquery.io.raster is written by another Phase 1 module"
    )
    pytest.importorskip("satquery.preprocess.optical")
    pytest.importorskip("satquery.preprocess.tiling")

    array = np.linspace(0.0, 1.0, 3 * 900 * 900, dtype=np.float32).reshape(3, 900, 900)
    path = write_png(tmp_path / "scene.png", array)
    ref = make_image_ref("i1", str(path), modality="optical")

    image, info = qv.prepare_image(ref, max_side=768)
    assert info["modality"] == "optical"
    assert max(info["width"], info["height"]) <= 768
    assert image.size == (info["width"], info["height"])
    steps = " ".join(info["steps"])
    assert "percentile_stretch" in steps and "resize_max_side" in steps and "to_uint8" in steps


def test_prepare_image_uses_the_sar_pipeline(tmp_path, write_png, make_image_ref, sar_array):
    pytest.importorskip("satquery.io.raster")
    pytest.importorskip("satquery.preprocess.sar")

    path = write_png(tmp_path / "sar.png", np.clip(sar_array(bands=1), 0, 1))
    ref = make_image_ref("s1", str(path), modality="sar")
    _, info = qv.prepare_image(ref, max_side=256)
    steps = " ".join(info["steps"])
    assert "linear_to_db" in steps and "db_to_uint8" in steps


def test_prepare_image_reports_a_broken_install(monkeypatch, make_image_ref):
    _break_import(monkeypatch, qv, "satquery.io.raster")
    with pytest.raises(SatQueryError, match="satquery.io.raster"):
        qv.prepare_image(make_image_ref("i1", "nope.png"))


# --------------------------------------------------------------------------- #
# EarthDial specifics
# --------------------------------------------------------------------------- #


def test_earthdial_predict_uses_the_chat_api(fake_earthdial, make_sample):
    predictor, model = fake_earthdial("Answer: 3")
    prediction = predictor.predict(make_sample("s1", "vqa", question="how many planes?"))

    assert prediction.text == "3"
    assert prediction.model == "earthdial"
    assert prediction.confidence is None
    call = model.calls[0]
    assert call["question"].startswith("<image>\n")
    assert call["generation_config"] == {"max_new_tokens": 64, "do_sample": False}
    assert np.asarray(call["pixel_values"]).shape == (1, 3, 448, 448)
    generation = prediction.evidence["generation"]
    assert generation["api"] == "internvl_chat"
    assert "no token scores" in generation["confidence_unavailable"]


def test_earthdial_pair_passes_num_patches_list(fake_earthdial, make_sample):
    predictor, model = fake_earthdial("more buildings appeared")
    predictor.predict(make_sample("s1", "change_caption"))
    call = model.calls[0]
    assert call["kwargs"]["num_patches_list"] == [1, 1]
    assert np.asarray(call["pixel_values"]).shape == (2, 3, 448, 448)
    assert call["question"].count("<image>") == 2


def test_earthdial_tolerates_a_tuple_response(fake_earthdial, make_sample):
    predictor, model = fake_earthdial("x")
    model.reply = ("a harbour with ships", ["history"])
    prediction = predictor.predict(make_sample("s1", "caption"))
    assert prediction.text == "a harbour with ships"


def test_earthdial_ground_output_parses(fake_earthdial, make_sample):
    predictor, _ = fake_earthdial("<ref>ship</ref><box>[[0, 0, 1000, 1000]]</box>")
    prediction = predictor.predict(make_sample("s1", "ground", question="the ship"))
    assert prediction.boxes[0].as_xyxy() == (0.0, 0.0, 512.0, 256.0)


def test_earthdial_without_chat_method_raises(monkeypatch, make_sample):
    predictor = EarthDialPredictor(device="cpu")
    monkeypatch.setattr(predictor, "_load", lambda: (object(), object()))
    monkeypatch.setattr(qv, "prepare_image", _fake_prepare_image)
    with pytest.raises(SatQueryError, match="InternVL-style chat"):
        predictor.predict(make_sample("s1", "vqa"))


def test_build_pixel_values_normalises_with_imagenet_statistics():
    pytest.importorskip("PIL")
    from PIL import Image

    white = Image.new("RGB", (16, 16), (255, 255, 255))
    stacked = ed.build_pixel_values([white], input_size=32)
    assert stacked.shape == (1, 3, 32, 32)
    expected = (1.0 - ed.IMAGENET_MEAN[0]) / ed.IMAGENET_STD[0]
    assert stacked[0, 0, 0, 0] == pytest.approx(expected, abs=1e-5)


def test_build_pixel_values_wraps_in_torch_when_available():
    pytest.importorskip("PIL")
    torch = pytest.importorskip("torch")
    from PIL import Image

    image = Image.new("RGB", (8, 8), (10, 20, 30))
    tensor = ed.build_pixel_values([image, image], input_size=16, torch=torch)
    assert tensor.shape == (2, 3, 16, 16)
    assert tensor.dtype == torch.float32


def test_build_pixel_values_rejects_no_images():
    with pytest.raises(ValueError):
        ed.build_pixel_values([])


# --------------------------------------------------------------------------- #
# Real weights (opt-in only)
# --------------------------------------------------------------------------- #


@WEIGHTS
def test_real_qwen_answers_a_question(make_sample):  # pragma: no cover - GPU only
    predictor = QwenVLPredictor()
    try:
        prediction = predictor.predict(make_sample("s1", "vqa"))
        assert prediction.text
    finally:
        predictor.close()


@WEIGHTS
def test_real_earthdial_answers_a_question(make_sample):  # pragma: no cover - GPU only
    predictor = EarthDialPredictor()
    try:
        prediction = predictor.predict(make_sample("s1", "vqa"))
        assert prediction.text
    finally:
        predictor.close()
