"""Qwen2.5-VL predictor, plus the VLM plumbing EarthDial reuses.

Why this exact checkpoint: ``Qwen/Qwen2.5-VL-7B-Instruct`` is Apache-2.0, while
the 3B ships under the Qwen *Research* licence (non-commercial). Silently
falling back to the 3B because it fits more comfortably in VRAM would make the
whole project unlicensable, so nothing here ever substitutes it -- pass
``model_id`` explicitly if you want it and accept the licence.

Every default in the load path is dictated by the free-tier GPU. A T4 is SM75:
bf16 needs SM80 and FlashAttention-2 needs SM80, so fp16 + sdpa are not a
preference but the only combination that runs. transformers will happily pick
bf16 when it thinks the hardware supports it, which is why the dtype and
attention defaults come from :data:`satquery.config.T4_SAFE_DTYPE` and
:data:`satquery.config.T4_SAFE_ATTENTION` instead of from the library.

Prompts live in the module-level :data:`PROMPTS` table, one entry per task,
because prompt text is simultaneously a tuning surface and an auditable
artefact: a benchmark number is only reproducible if the instruction that
produced it can be diffed between runs, and prompts scattered through f-strings
cannot be. The same reason drives :attr:`Prediction.evidence` carrying the raw
decode alongside the prompt -- the composer's rule downstream is that every
claim cites a tool output, so the tool has to emit one.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import SatQueryConfig, T4_SAFE_ATTENTION, T4_SAFE_DTYPE, get_config
from ..contracts import (
    Box,
    ImageRef,
    Prediction,
    Sample,
    SatQueryError,
    Task,
    UnsupportedTask,
)
from ..log import get_logger
from ..text import extract_answer

__all__ = [
    "PROMPTS",
    "SYSTEM_PROMPT",
    "TASK_MAX_NEW_TOKENS",
    "Generation",
    "VLPredictorBase",
    "QwenVLPredictor",
    "build_prompt",
    "build_messages",
    "parse_boxes",
    "prepare_image",
    "mean_token_probability",
]

LOG = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Prompts. Tuning surface: edit here, nowhere else.
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are a remote-sensing image analyst. You look at satellite and aerial "
    "imagery from a top-down view, where scale is coarse and objects are small. "
    "Answer only from what is visible in the given image or images."
)

#: task -> instruction template. ``{question}`` is filled from the sample; a
#: template without the placeholder is used verbatim (captioning has no
#: question). The grounding template *demands* 0-1000 coordinates because
#: :func:`parse_boxes` reads them with ``scale=1000`` -- prompt and parser are
#: one contract, and Qwen2.5-VL otherwise emits absolute pixels of its own
#: internally resized image, which the caller cannot reconstruct.
PROMPTS: dict[str, str] = {
    "vqa": (
        "Look at the satellite image and answer the question.\n"
        "Reply with the shortest possible answer: a number, a single word, or "
        "yes/no. Do not explain and do not repeat the question.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "caption": (
        "Describe this satellite image in one or two sentences. Cover the "
        "dominant land cover, the man-made structures that are present and how "
        "they are arranged relative to each other. State only what is visible; "
        "do not guess a place name, a date or a sensor."
    ),
    "ground": (
        "Find every object in the satellite image that matches the referring "
        "expression.\n"
        "Reply with a JSON list and nothing else. One entry per object, each of "
        'the form {{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}}, '
        "where the coordinates are integers normalised to the range 0-1000 "
        "along the image width (x) and height (y), with x0 < x1 and y0 < y1. "
        "Reply with [] if no object matches.\n"
        "Referring expression: {question}"
    ),
    "change_vqa": (
        "You are given two satellite images of the same place. The FIRST image "
        "is the earlier date and the SECOND image is the later date.\n"
        "Answer the question about what changed between them with the shortest "
        "possible answer: a number, a single word, or yes/no. Do not explain.\n"
        "Question: {question}\n"
        "Answer:"
    ),
    "change_caption": (
        "You are given two satellite images of the same place. The FIRST image "
        "is the earlier date and the SECOND image is the later date.\n"
        "Describe in one or two sentences what changed between them -- "
        "buildings, roads, vegetation, water, bare ground. Mention where in the "
        "scene the change is. If nothing changed, say that nothing changed."
    ),
}

#: Floor on the decode budget per task, applied as ``max(max_new_tokens, ...)``.
#: 64 tokens is right for a VQA answer and wrong for a VRSBench detailed
#: caption, which runs to ~80 words; truncating those would depress BLEU/CIDEr
#: for a reason that has nothing to do with the model. An explicit caller asking
#: for more still wins, so this only ever raises the floor.
TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "vqa": 64,
    "change_vqa": 64,
    "caption": 192,
    "change_caption": 192,
    "ground": 256,
}

_VLM_HINT = (
    'install the GPU extra: pip install "satquery[vlm]" '
    "(torch, transformers, accelerate, bitsandbytes)"
)
_PKG_HINT = (
    "it ships with satquery, so an import failure here means a broken install "
    'or a missing image library (pip install "satquery[data]")'
)


# --------------------------------------------------------------------------- #
# Optional imports. One seam, so a missing extra is reported the same way
# everywhere and tests can simulate it without touching sys.modules.
# --------------------------------------------------------------------------- #


def _import_module(name: str) -> Any:
    """Import by name. The single seam for every optional dependency."""
    return importlib.import_module(name)


def _require_module(name: str, *, who: str, hint: str) -> Any:
    try:
        return _import_module(name)
    except ImportError as exc:
        raise SatQueryError(
            f"{who} requires the {name!r} package, which is not importable "
            f"({exc}); {hint}"
        ) from exc


# --------------------------------------------------------------------------- #
# Prompt / message construction
# --------------------------------------------------------------------------- #


def build_prompt(sample: Sample, *, prompts: Mapping[str, str] = PROMPTS) -> str:
    """Render the instruction for one sample.

    ``sample.meta['tool_context']`` is appended verbatim when present. That is
    the architectural hinge of the whole system: deterministic tools (NDVI,
    backscatter statistics, mask areas) feed the VLM and never the reverse,
    because a VLM cannot read speckle but can explain a number it was handed.
    """
    template = prompts.get(sample.task)
    if template is None:
        raise UnsupportedTask(
            f"no prompt for task {sample.task!r}; known: {', '.join(sorted(prompts))}"
        )
    prompt = template.format(question=(sample.question or "").strip())
    context = sample.meta.get("tool_context")
    if context:
        prompt = (
            f"{prompt}\n\nMeasurements from deterministic tools (these numbers are "
            f"correct, use them): {context}"
        )
    return prompt


def build_messages(
    prompt: str,
    n_images: int,
    *,
    system: str | None = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Qwen chat-template messages: all images in ONE user turn.

    The two-image tasks are the reason this is a function rather than a literal.
    Splitting the pair across two user turns makes the model read the second
    image as a follow-up question instead of the later date, and "what changed"
    loses its referent. Image content blocks carry no payload: the processor
    substitutes the vision placeholders and takes the pixels separately.
    """
    if n_images < 1:
        raise ValueError("at least one image is required")
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt})
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append({"role": "user", "content": content})
    return messages


# --------------------------------------------------------------------------- #
# Grounding output -> Box
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.DOTALL)
_JSON_SPAN_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)
_PAIR_RE = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)
_QUAD_RE = re.compile(
    r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]"
)
_BBOX_KEYS = ("bbox_2d", "bbox", "box", "bounding_box", "boxes", "rect")
_LABEL_KEYS = ("label", "name", "category", "class", "ref")
_SCORE_KEYS = ("score", "confidence", "prob")
_CONTAINER_KEYS = ("objects", "boxes", "results", "result", "detections", "data")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quads(value: Any) -> list[tuple[float, float, float, float]]:
    """Every 4-number group reachable inside ``value``.

    Covers the three shapes models actually emit: ``[x0,y0,x1,y1]``,
    ``[[x0,y0,x1,y1], ...]`` (InternVL's ``<box>[[...]]</box>``) and
    ``[[x0,y0],[x1,y1]]`` (Qwen2-VL's point-pair form).
    """
    if not isinstance(value, (list, tuple)):
        return []
    if len(value) == 4 and all(_is_number(v) for v in value):
        return [(float(value[0]), float(value[1]), float(value[2]), float(value[3]))]
    if (
        len(value) == 2
        and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in value)
        and all(_is_number(c) for v in value for c in v)
    ):
        (x0, y0), (x1, y1) = value
        return [(float(x0), float(y0), float(x1), float(y1))]
    out: list[tuple[float, float, float, float]] = []
    for sub in value:
        out.extend(_quads(sub))
    return out


def _entries(parsed: Any) -> list[tuple[Any, str | None, float | None]]:
    """Flatten parsed JSON into ``(coords, label, score)`` candidates."""
    if isinstance(parsed, dict):
        for key in _CONTAINER_KEYS:
            inner = parsed.get(key)
            if isinstance(inner, list):
                parsed = inner
                break
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    out: list[tuple[Any, str | None, float | None]] = []
    for entry in parsed:
        if isinstance(entry, dict):
            coords: Any = None
            for key in _BBOX_KEYS:
                if key in entry:
                    coords = entry[key]
                    break
            label = next((str(entry[k]) for k in _LABEL_KEYS if entry.get(k)), None)
            raw_score = next((entry[k] for k in _SCORE_KEYS if _is_number(entry.get(k))), None)
            score = float(raw_score) if raw_score is not None else None
            if coords is not None:
                out.append((coords, label, score))
        else:
            out.append((entry, None, None))
    return out


def parse_boxes(
    text: str | None,
    *,
    width: int,
    height: int,
    scale: float = 1000.0,
    label: str | None = None,
) -> tuple[tuple[Box, ...], tuple[str, ...]]:
    """Parse a grounding decode into pixel boxes. Never raises.

    Returns ``(boxes, notes)``. A model that answers "I cannot see any ship"
    yields zero boxes and a note; that has to score as a miss, not abort a
    five-hour benchmark run, so every failure mode here becomes a note that ends
    up in the prediction's evidence.

    ``scale`` is the model's coordinate range (1000 for Qwen and InternVL, 1.0
    for 0-1 normalised output). Coordinates are ordered before conversion and
    clipped to the image afterwards: out-of-range output is common, and a
    reversed box would trip :class:`~satquery.contracts.Box`'s own validation.
    """
    notes: list[str] = []
    raw = (text or "").strip()
    if not raw:
        return (), ("empty decode",)

    body = raw
    fenced = _FENCE_RE.search(raw)
    if fenced:
        body = fenced.group(1).strip()
        notes.append("stripped code fence")

    found: list[tuple[tuple[float, float, float, float], str | None, float | None]] = []
    span = _JSON_SPAN_RE.search(body)
    if span:
        try:
            parsed: Any = json.loads(span.group(0))
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None:
            for coords, entry_label, score in _entries(parsed):
                for quad in _quads(coords):
                    found.append((quad, entry_label, score))

    if not found:
        for match in _PAIR_RE.finditer(body):
            x0, y0, x1, y1 = (float(g) for g in match.groups())
            found.append(((x0, y0, x1, y1), None, None))
    if not found:
        for match in _QUAD_RE.finditer(body):
            x0, y0, x1, y1 = (float(g) for g in match.groups())
            found.append(((x0, y0, x1, y1), None, None))
    if not found:
        notes.append("no box coordinates found in decode")
        return (), tuple(notes)

    limit = scale * 1.001
    out_of_range = False
    reordered = False
    boxes: list[Box] = []
    for (x0, y0, x1, y1), entry_label, score in found:
        if max(x0, y0, x1, y1) > limit or min(x0, y0, x1, y1) < 0.0:
            out_of_range = True
        if x1 < x0 or y1 < y0:
            reordered = True
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
        box = Box.from_normalized(
            x0,
            y0,
            x1,
            y1,
            width=width,
            height=height,
            scale=scale,
            label=entry_label or label,
            score=score,
        )
        boxes.append(box.clip(width, height))

    if out_of_range:
        notes.append(f"coordinates outside 0-{scale:g}; clipped to the image")
    if reordered:
        notes.append("reversed coordinates; reordered")
    if scale > 1.0 and all(max(quad) <= 1.0 for quad, _, _ in found):
        # Every coordinate is <= 1: the decode was almost certainly 0-1
        # normalised. Flag it, never silently rescale -- guessing here invents a
        # number that the report would then cite as evidence.
        notes.append(f"coordinates look 0-1 normalised, not 0-{scale:g}")
    return tuple(boxes), tuple(notes)


# --------------------------------------------------------------------------- #
# Image preparation
# --------------------------------------------------------------------------- #


def _band_selection(
    bands: int, modality: str, override: Sequence[int] | None = None
) -> tuple[int, int, int]:
    """Three band indices to render as RGB.

    ``override`` comes from ``ImageRef.meta['rgb_bands']`` and is the only way to
    get a true-colour composite out of a 13-band Sentinel-2 stack, whose RGB is
    bands 3,2,1 rather than 0,1,2. Without it the first three bands are used and
    the assumption is recorded in the evidence: the hidden Cartosat/RISAT eval
    set will not follow Sentinel's ordering, so inferring from the band count
    would be a guess dressed up as a rule.
    """
    if override is not None:
        sel = tuple(int(i) for i in override)
        if len(sel) != 3 or any(i < 0 or i >= bands for i in sel):
            raise SatQueryError(
                f"rgb_bands {tuple(override)!r} is not three valid indices for a "
                f"{bands}-band image"
            )
        return sel  # type: ignore[return-value]
    if bands >= 3:
        return (0, 1, 2)
    if bands == 2:
        # SAR VV/VH: R=VV, G=VH, B=VV again. The dual-pol false colour normally
        # puts the VV/VH ratio in blue, but a ratio of dB values is meaningless,
        # so repeat a real channel instead of fabricating one.
        return (0, 1, 0)
    return (0, 0, 0)


def _array_to_pil(array: np.ndarray) -> Any:
    """``(3, h, w)`` uint8 -> PIL RGB image."""
    pil = _require_module("PIL.Image", who="VLM image preparation", hint=_PKG_HINT)
    hwc = np.ascontiguousarray(np.transpose(np.asarray(array, dtype=np.uint8), (1, 2, 0)))
    return pil.fromarray(hwc, mode="RGB")


def prepare_image(
    ref: ImageRef, *, max_side: int = 768, modality: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Read one :class:`ImageRef` and render it the way the VLM was trained.

    Returns ``(pil_image, info)``; ``info`` records the exact pipeline so the
    evidence block can cite it. Two modality-specific paths, because handing raw
    SAR power to a VLM produces a black square: optical goes through a 2-98
    percentile stretch, SAR through linear->dB and the [-25, 0] dB clip.
    Resizing happens on the float representation with quantisation last, so the
    stretch is never applied twice at two resolutions.

    768 px is the cap the LoRA run trains at, so inference has to match it or the
    adapter sees a distribution it never saw.
    """
    raster = _require_module("satquery.io.raster", who="VLM image preparation", hint=_PKG_HINT)
    pre = _require_module("satquery.preprocess", who="VLM image preparation", hint=_PKG_HINT)

    image = raster.read_raster(ref.path)
    array = np.asarray(image.array, dtype=np.float32)
    steps: list[str] = [f"read_raster(bands={array.shape[0]}, shape={tuple(array.shape[1:])})"]

    declared = modality or (ref.modality if ref.modality in ("optical", "sar") else None)
    if declared is None:
        guess = _require_module(
            "satquery.io.modality", who="VLM image preparation", hint=_PKG_HINT
        ).infer_modality(image, path=ref.path)
        resolved = str(guess.modality)
        steps.append(f"infer_modality={resolved}@{guess.confidence:.2f}")
    else:
        resolved = str(declared)
        steps.append(f"modality={resolved}(declared)")

    if resolved == "sar":
        array = np.asarray(pre.linear_to_db(array), dtype=np.float32)
        steps.append("linear_to_db")
        array = np.asarray(pre.resize_max_side(array, max_side=max_side), dtype=np.float32)
        steps.append(f"resize_max_side({max_side})->{tuple(array.shape[1:])}")
        array = np.asarray(pre.db_to_uint8(array), dtype=np.uint8)
        steps.append("db_to_uint8(clip=(-25,0))")
    else:
        array = np.asarray(pre.percentile_stretch(array, low=2.0, high=98.0), dtype=np.float32)
        steps.append("percentile_stretch(2,98)")
        array = np.asarray(pre.resize_max_side(array, max_side=max_side), dtype=np.float32)
        steps.append(f"resize_max_side({max_side})->{tuple(array.shape[1:])}")
        array = np.asarray(pre.to_uint8(array), dtype=np.uint8)
        steps.append("to_uint8")

    selection = _band_selection(int(array.shape[0]), resolved, ref.meta.get("rgb_bands"))
    array = array[list(selection)]
    steps.append(f"rgb_bands={selection}")
    pil_image = _array_to_pil(array)

    info = {
        "image_id": ref.image_id,
        "path": ref.path,
        "modality": resolved,
        "width": int(array.shape[2]),
        "height": int(array.shape[1]),
        "steps": tuple(steps),
    }
    return pil_image, info


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Generation:
    """One decode: the text, a confidence when the model gives one, audit meta."""

    text: str
    confidence: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


def mean_token_probability(model: Any, outputs: Any) -> float | None:
    """Mean probability of the sampled tokens, or ``None``.

    This is the only confidence a plain ``generate`` call can honestly produce.
    It needs ``output_scores=True`` *and* ``compute_transition_scores``, and a
    quantised model wrapped in custom remote code may offer neither -- in which
    case the answer is ``None``, never a plausible-looking constant. The composer
    shows confidence to a judge; inventing one is the unforgivable bug here.
    """
    scores = getattr(outputs, "scores", None)
    sequences = getattr(outputs, "sequences", None)
    if not scores or sequences is None:
        return None
    try:
        transition = model.compute_transition_scores(sequences, scores, normalize_logits=True)
        # Filter on the log-prob side: a -inf transition score marks a position
        # with no distribution at all (padding, a masked beam), not a token the
        # model considered impossible. Exponentiating first would fold those in
        # as p=0 and understate the confidence.
        finite = transition[transition.isfinite()]
        if finite.numel() == 0:
            return None
        return float(finite.exp().mean())
    except Exception as exc:  # pragma: no cover - depends on the loaded model
        LOG.debug("no token probabilities available: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #


class VLPredictorBase:
    """Shared lifecycle for the generative predictors.

    Subclasses supply :meth:`_load` (processor/tokenizer + model) and
    :meth:`_generate` (one decode) and inherit the parts that must not diverge
    between adapters: T4-safe defaults, a single lazy load, task dispatch, answer
    extraction, box parsing, evidence assembly and VRAM release.
    """

    name = "vl"
    supported: tuple[Task, ...] = ("vqa", "caption", "ground", "change_vqa", "change_caption")
    default_model_id = ""
    prompts: Mapping[str, str] = PROMPTS
    system_prompt: str | None = SYSTEM_PROMPT
    #: Coordinate range the grounding decode is interpreted in. Qwen2.5-VL and
    #: the InternVL derivatives are both prompted for 0-1000 here.
    coord_scale = 1000.0

    def __init__(
        self,
        model_id: str | None = None,
        *,
        adapter: str | None = None,
        load_in_4bit: bool = True,
        dtype: str | None = None,
        attn_implementation: str | None = None,
        max_new_tokens: int = 64,
        max_side: int | None = None,
        device: str | None = None,
        config: SatQueryConfig | None = None,
    ) -> None:
        self.config = config or get_config()
        self.model_id = model_id or self.default_model_id
        self.adapter = adapter
        self.load_in_4bit = bool(load_in_4bit)
        self.dtype = dtype or T4_SAFE_DTYPE
        self.attn_implementation = attn_implementation or T4_SAFE_ATTENTION
        self.max_new_tokens = int(max_new_tokens)
        self.max_side = int(max_side or self.config.max_side)
        self.device = device or self.config.resolve_device()
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._load_count = 0

        if self.dtype != T4_SAFE_DTYPE:
            LOG.warning(
                "dtype=%s requested; the free-tier T4 (SM75) supports only %s and will "
                "fail or silently upcast",
                self.dtype,
                T4_SAFE_DTYPE,
            )
        if self.attn_implementation != T4_SAFE_ATTENTION:
            LOG.warning(
                "attn_implementation=%s requested; the free-tier T4 supports only %s "
                "(FlashAttention-2 needs SM80)",
                self.attn_implementation,
                T4_SAFE_ATTENTION,
            )

    # -- introspection ------------------------------------------------------- #

    def supports(self, task: Task) -> bool:
        return task in self.supported

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        """Row for the execution trace: what actually ran, not what was asked for."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "adapter": self.adapter,
            "tasks": list(self.supported),
            "load_in_4bit": self.load_in_4bit,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": self.max_new_tokens,
            "max_side": self.max_side,
            "device": self.device,
            "loaded": self.loaded,
        }

    def max_new_tokens_for(self, task: Task) -> int:
        return max(self.max_new_tokens, TASK_MAX_NEW_TOKENS.get(str(task), 0))

    # -- lifecycle ----------------------------------------------------------- #

    def _load(self) -> tuple[Any, Any]:  # pragma: no cover - subclass hook
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        """Load exactly once, on first use.

        Constructing a predictor has to stay free: the CLI builds every
        registered predictor to report capabilities, and must not pull 16 GB of
        weights to do it.
        """
        if self._model is None:
            self._processor, self._model = self._load()
            self._load_count += 1

    def close(self) -> None:
        """Drop the model and give the VRAM back.

        Worth being explicit about: a 4-bit 7B leaves ~7 GB resident, which is
        exactly the difference between the next model in a benchmark matrix
        loading and OOM-ing inside the same T4 session.
        """
        self._model = None
        self._processor = None
        self._torch = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None:  # only if something already imported it
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:  # pragma: no cover - driver dependent
                LOG.debug("cuda cache not released: %s", exc)

    def __enter__(self) -> "VLPredictorBase":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _inference_ctx(self) -> Any:
        torch = self._torch
        return contextlib.nullcontext() if torch is None else torch.inference_mode()

    # -- prediction ---------------------------------------------------------- #

    def _generate(  # pragma: no cover - subclass hook
        self, prompt: str, images: Sequence[Any], *, task: Task
    ) -> Generation:
        raise NotImplementedError

    def _prepare_images(self, sample: Sample) -> tuple[list[Any], list[dict[str, Any]]]:
        images: list[Any] = []
        infos: list[dict[str, Any]] = []
        for ref in sample.images:
            image, info = prepare_image(ref, max_side=self.max_side)
            images.append(image)
            infos.append(info)
        return images, infos

    def predict(self, sample: Sample) -> Prediction:
        if not self.supports(sample.task):
            raise UnsupportedTask(
                f"{self.name} does not serve task {sample.task!r}; "
                f"supported: {', '.join(self.supported)}"
            )
        started = time.perf_counter()
        self._ensure_loaded()
        prompt = build_prompt(sample, prompts=self.prompts)
        images, infos = self._prepare_images(sample)
        generation = self._generate(prompt, images, task=sample.task)

        notes: list[str] = []
        text: str | None = None
        boxes: tuple[Box, ...] = ()
        raw = generation.text

        if sample.task in ("vqa", "change_vqa"):
            text = extract_answer(raw)
            if not text:
                notes.append("answer extraction produced an empty string")
        elif sample.task in ("caption", "change_caption"):
            text = raw.strip()
            if not text:
                notes.append("empty caption")
        elif sample.task == "ground":
            width = int(infos[0]["width"]) if infos else 0
            height = int(infos[0]["height"]) if infos else 0
            boxes, parse_notes = parse_boxes(
                raw, width=width, height=height, scale=self.coord_scale
            )
            notes.extend(parse_notes)
            text = raw.strip() or None

        evidence: dict[str, Any] = {
            "model_id": self.model_id,
            "adapter": self.adapter,
            "task": sample.task,
            "prompt": prompt,
            "raw": raw,
            "images": [{k: v for k, v in info.items() if k != "steps"} for info in infos],
            "preprocess": [list(info.get("steps", ())) for info in infos],
            "confidence_source": (
                "mean_token_probability" if generation.confidence is not None else "unavailable"
            ),
            "notes": notes,
        }
        if generation.meta:
            evidence["generation"] = dict(generation.meta)

        return Prediction(
            sample_id=sample.sample_id,
            task=sample.task,
            text=text,
            boxes=boxes,
            confidence=generation.confidence,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            evidence=evidence,
            model=self.name,
        )

    # -- shared load helpers ------------------------------------------------- #

    def _resolve_dtype(self, torch: Any) -> Any:
        dtype = getattr(torch, self.dtype, None)
        if dtype is None:
            raise SatQueryError(
                f"unknown torch dtype {self.dtype!r}; expected e.g. {T4_SAFE_DTYPE!r}"
            )
        return dtype

    def _quantization_config(self, torch: Any, transformers: Any) -> Any | None:
        """NF4 double-quantised config, or ``None`` when bitsandbytes is absent.

        A missing bitsandbytes degrades to a full-precision load rather than
        failing: on a box with enough VRAM that is the right behaviour, and the
        warning says what it costs.
        """
        if not self.load_in_4bit:
            return None
        try:
            _import_module("bitsandbytes")
        except ImportError:
            LOG.warning(
                "bitsandbytes is not installed; loading %s in %s without 4-bit "
                "quantisation (a 7B then needs ~16 GB, not ~7 GB)",
                self.model_id,
                self.dtype,
            )
            return None
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Compute dtype stays fp16 on a T4; bf16 here is the classic
            # "works on an A100, NaNs on Colab" bug.
            bnb_4bit_compute_dtype=self._resolve_dtype(torch),
        )

    def _attach_adapter(self, model: Any) -> Any:
        peft = _require_module(
            "peft", who=f"{self.name} adapter loading", hint="pip install peft"
        )
        LOG.info("attaching adapter %s", self.adapter)
        return peft.PeftModel.from_pretrained(model, self.adapter)


class QwenVLPredictor(VLPredictorBase):
    """Qwen2.5-VL-7B-Instruct: the core VLM slot (VQA, caption, grounding).

    The 4-bit default is what makes the 7B fit a free T4 -- roughly 7 GB instead
    of 16 -- at a small accuracy cost; pass ``load_in_4bit=False`` on real
    hardware. ``adapter`` points at the QLoRA adapter from the Phase 2 run, so
    one class produces both halves of the before/after adaptation table.
    """

    name = "qwen2_5_vl"
    supported: tuple[Task, ...] = ("vqa", "caption", "ground", "change_vqa", "change_caption")
    default_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    def _load(self) -> tuple[Any, Any]:
        torch = _require_module("torch", who=f"{self.name} inference", hint=_VLM_HINT)
        transformers = _require_module(
            "transformers", who=f"{self.name} inference", hint=_VLM_HINT
        )
        self._torch = torch

        kwargs: dict[str, Any] = {
            # `torch_dtype`, not the newer `dtype` alias: it is accepted by every
            # transformers release the vlm extra allows (>=4.49).
            "torch_dtype": self._resolve_dtype(torch),
            "attn_implementation": self.attn_implementation,
            "low_cpu_mem_usage": True,
        }
        quant = self._quantization_config(torch, transformers)
        if quant is not None:
            kwargs["quantization_config"] = quant
            # bitsandbytes places layers during from_pretrained; calling .to()
            # on a quantised model afterwards raises.
            kwargs["device_map"] = "auto"

        model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None) or getattr(
            transformers, "AutoModelForImageTextToText", None
        )
        if model_cls is None:  # pragma: no cover - very old transformers
            raise SatQueryError(
                "transformers is too old for Qwen2.5-VL (needs >=4.49); " + _VLM_HINT
            )
        LOG.info(
            "loading %s (4bit=%s, dtype=%s, attn=%s)",
            self.model_id,
            quant is not None,
            self.dtype,
            self.attn_implementation,
        )
        model = model_cls.from_pretrained(self.model_id, **kwargs)
        if quant is None and self.device not in ("", "auto"):
            model = model.to(self.device)
        # max_pixels caps the visual token budget: a 768 px tile is ~750 tokens,
        # and the processor's 1280*28*28 default would spend KV-cache on
        # resolution that resize_max_side already threw away.
        processor = transformers.AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=self.max_side * self.max_side,
        )
        if self.adapter:
            model = self._attach_adapter(model)
        if hasattr(model, "eval"):
            model.eval()
        return processor, model

    def _generate(self, prompt: str, images: Sequence[Any], *, task: Task) -> Generation:
        processor, model = self._processor, self._model
        messages = build_messages(prompt, len(images), system=self.system_prompt)
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_text], images=list(images), padding=True, return_tensors="pt"
        )
        device = getattr(model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        prompt_len = 0
        try:
            prompt_len = int(inputs["input_ids"].shape[1])
        except (KeyError, TypeError, IndexError, AttributeError):
            LOG.debug("no input_ids shape available; decoding the full sequence")

        with self._inference_ctx():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens_for(task),
                do_sample=False,  # greedy: a benchmark number must be reproducible
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = getattr(outputs, "sequences", outputs)
        trimmed = sequences[:, prompt_len:] if prompt_len else sequences
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = (decoded[0] if decoded else "").strip()
        return Generation(
            text=text,
            confidence=mean_token_probability(model, outputs),
            meta={
                "messages": messages,
                "n_images": len(images),
                "max_new_tokens": self.max_new_tokens_for(task),
                "prompt_tokens": prompt_len,
            },
        )
