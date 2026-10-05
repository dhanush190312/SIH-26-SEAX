"""What actually runs behind each tool, and what runs when the weights are absent.

Three slots are served (approach doc §3.1): the tuned Qwen owns every sentence,
Change-Agent MCI owns the change mask, TerraMind owns optical-SAR fusion. None of
those weights fit on a laptop, and the prototype has to answer a question on a
laptop, so each slot has two implementations:

===================  ==============================  ==============================
slot                 model backend                   deterministic backend
===================  ==============================  ==============================
text (5 tasks)       Qwen2.5-VL-7B (+LoRA later)     :class:`EvidenceVLM`
change map           Change-Agent MCI                :class:`CvaChangeMap`
optical-SAR fusion   TerraMind-1.0-base + head       :class:`RuleFusion`
===================  ==============================  ==============================

The deterministic backends are **not** mocks. They compute real quantities --
change vector analysis with an Otsu cut, index thresholds, SAR backscatter rules
-- and they refuse the questions they cannot answer instead of inventing a
sentence. That is the difference that matters: a demo that says "I measured 14.2%
of the scene changed, here is the mask, and I cannot tell you what the new
buildings are used for" is honest, and swapping the tuned Qwen in later only adds
prose on top of the same numbers.

Every backend reports its own ``name`` and ``kind`` (``model`` or
``deterministic``) into the trace, so no answer can be read without knowing which
one produced it. Selection is by environment variable, defaulting to ``auto``:

``SATQUERY_VLM``              ``auto`` | a predictor name from the registry | ``evidence``
``SATQUERY_CHANGE_BACKEND``   ``auto`` | ``mci`` | ``cva``
``SATQUERY_FUSION_BACKEND``   ``auto`` | ``terramind`` | ``rules``

``auto`` picks the model backend only when its dependencies import *and* a GPU is
visible -- a 4-bit 7B on laptop CPU is not a slow demo, it is a hung one -- and
records the reason it fell back.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import SatQueryConfig, get_config
from ..contracts import Box, ImageRef, Prediction, Sample, SatQueryError, Task
from ..log import get_logger


__all__ = [
    "VlmAnswer",
    "ChangeResult",
    "FusionResult",
    "EvidenceVLM",
    "PredictorVLM",
    "MistralVLM",
    "CvaChangeMap",
    "MciChangeMap",
    "RuleFusion",
    "TerraMindFusion",
    "select_vlm",
    "select_change_backend",
    "select_fusion_backend",
    "backend_status",
    "otsu_threshold",
    "noise_floor",
    "label_components",
    "FUSION_LABELS",
]

_LOG = get_logger(__name__)

#: The land-cover vocabulary :class:`RuleFusion` can assert. A deliberately
#: coarse subset of BigEarthNet's 19 classes: these five are the ones two
#: sensors and a handful of thresholds can actually separate. The TerraMind head
#: replaces this with the full nomenclature once it is trained -- and the
#: multi-label metric already scores against whatever vocabulary it is given.
FUSION_LABELS: tuple[str, ...] = (
    "water_bodies",
    "urban_fabric",
    "forest_and_vegetation",
    "agricultural_land",
    "bare_or_sparse_vegetation",
)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class VlmAnswer:
    """One text answer, with the boxes and provenance that came with it."""

    text: str
    confidence: float | None = None
    boxes: tuple[Box, ...] = ()
    backend: str = ""
    kind: str = ""
    refused: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": None if self.confidence is None else round(float(self.confidence), 3),
            "boxes": [
                {"bbox": list(b.as_xyxy()), "label": b.label, "score": b.score} for b in self.boxes
            ],
            "backend": self.backend,
            "kind": self.kind,
            "refused": self.refused,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class ChangeResult:
    """A change mask plus the statistics an answer is allowed to quote."""

    mask: np.ndarray
    stats: Mapping[str, Any]
    backend: str = ""
    kind: str = ""
    regions: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"stats": dict(self.stats), "backend": self.backend, "kind": self.kind,
                "regions": [dict(r) for r in self.regions]}


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Multi-label land cover from an optical-SAR pair, with per-class evidence."""

    labels: tuple[str, ...]
    label_scores: Mapping[str, float]
    area_pct: Mapping[str, float]
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    rationale: Mapping[str, str] = field(default_factory=dict)
    backend: str = ""
    kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "label_scores": {k: round(float(v), 3) for k, v in self.label_scores.items()},
            "area_pct": {k: round(float(v), 2) for k, v in self.area_pct.items()},
            "rationale": dict(self.rationale),
            "backend": self.backend,
            "kind": self.kind,
        }


# --------------------------------------------------------------------------- #
# Text: the only component allowed to produce a user-facing sentence
# --------------------------------------------------------------------------- #


class PredictorVLM:
    """The served VLM slot, backed by a registry predictor (Qwen2.5-VL + LoRA).

    Evidence is injected into the prompt rather than appended to the answer:
    the deterministic numbers arrive *before* generation, which is invariant 03
    and the reason a SAR question gets a defensible answer at all. The prompt
    templates live in :mod:`satquery.models.qwen_vl`; this class copies them,
    appends the evidence block for the duration of one call and restores them
    afterwards, so two concurrent tasks cannot inherit each other's context.
    """

    kind = "model"

    def __init__(self, predictor_name: str = "qwen2_5_vl", **kwargs: Any) -> None:
        from ..registry import build_predictor

        self.predictor_name = predictor_name
        self.predictor = build_predictor(predictor_name, **kwargs)
        self.name = getattr(self.predictor, "name", predictor_name)

    def describe(self) -> dict[str, Any]:
        describe = getattr(self.predictor, "describe", None)
        return {"name": self.name, "kind": self.kind, **(describe() if describe else {})}

    def answer(
        self,
        task: Task,
        images: Sequence[ImageRef],
        *,
        question: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        sample_id: str = "query",
    ) -> VlmAnswer:
        sample = Sample(
            sample_id=sample_id,
            task=task,
            images=tuple(images),
            question=question,
            dataset="session",
            split="live",
        )
        original = getattr(self.predictor, "prompts", None)
        if evidence and original is not None:
            block = format_evidence_block(evidence)
            prompts = dict(original)
            if task in prompts:
                prompts[task] = f"{prompts[task]}\n\n{block}"
                self.predictor.prompts = prompts
        try:
            prediction: Prediction = self.predictor.predict(sample)
        finally:
            if original is not None:
                self.predictor.prompts = original

        if not prediction.ok:
            return VlmAnswer(
                text="",
                confidence=0.0,
                backend=self.name,
                kind=self.kind,
                refused=True,
                meta={"error": prediction.error},
            )
        return VlmAnswer(
            text=prediction.text or "",
            confidence=prediction.confidence,
            boxes=tuple(prediction.boxes),
            backend=self.name,
            kind=self.kind,
            meta=dict(prediction.evidence),
        )

    def close(self) -> None:
        close = getattr(self.predictor, "close", None)
        if close:
            close()


class MistralVLM:
    """The text slot served by Mistral's vision API. No GPU, real answers.

    This is the pragmatic middle of the three text backends: it answers the
    questions :class:`EvidenceVLM` has to refuse -- counting, land use, layout,
    free-form grounding -- without the 16 GB of weights :class:`PredictorVLM`
    needs. It is *not* a substitute for the Phase 3 fine-tune: a general-purpose
    VLM is exactly what the challenge disqualifies on its own, and the RS
    adaptation that gets scored comes from the QLoRA run, not from here.

    **It sends imagery off the machine.** That is the one thing the other two
    backends never do, and it is why this backend is never selected by ``auto``
    -- it requires ``SATQUERY_VLM=mistral`` explicitly. Invariant 01 is untouched
    (the *router* still receives no pixels), but "no imagery leaves the machine"
    stops being true of the system as a whole, and on a free tier that carries a
    data-training opt-in that is a decision an operator has to make deliberately
    rather than inherit from a default.

    Three things are reused rather than reinvented, so this backend and the
    served Qwen stay comparable:

    * the task prompts in :mod:`satquery.models.qwen_vl` -- the same wording, so
      a difference in answers is a difference in models;
    * :func:`satquery.models.qwen_vl.parse_boxes`, because the grounding prompt
      demands 0-1000 coordinates and that parser already reads them;
    * the display transform in :mod:`satquery.agent.render`, so the model is sent
      exactly the rendering the GUI shows -- a SAR scene goes as dB, not as raw
      power, and the preview a user sees is the evidence the answer came from.

    Confidence is *not* model-derived: the API returns no token probabilities, so
    a fixed :data:`API_CONFIDENCE` is reported and the composer's citation ratio
    does the real work. The evidence dict records that, rather than passing a
    made-up number off as a measurement.
    """

    kind = "model"

    #: What a cloud answer is worth before the composer checks its citations.
    #: Not a probability -- an admission that the API gives us nothing better.
    API_CONFIDENCE = 0.7

    #: Longest side sent. 768 matches the LoRA training size, keeps a PNG under
    #: ~400 KB of base64, and is the same cap the GUI preview uses.
    MAX_SIDE = 768

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 90.0,
        max_side: int | None = None,
    ) -> None:
        self.model = model or os.environ.get("SATQUERY_VLM_MODEL", "mistral-small-latest")
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.endpoint = endpoint or os.environ.get(
            "SATQUERY_VLM_ENDPOINT", "https://api.mistral.ai/v1/chat/completions"
        )
        self.timeout = float(timeout)
        self.max_side = int(max_side or self.MAX_SIDE)
        self.name = f"mistral-vision:{self.model}"
        if not self.api_key:
            raise SatQueryError(
                "SATQUERY_VLM=mistral needs MISTRAL_API_KEY (put it in .env; see .env.example)"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "max_side": self.max_side,
            "sends_imagery": True,
            "tasks": list(_TEXT_TASKS),
        }

    # -- the call ------------------------------------------------------------ #

    def answer(
        self,
        task: Task,
        images: Sequence[ImageRef],
        *,
        question: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        sample_id: str = "query",
    ) -> VlmAnswer:
        from ..models.qwen_vl import PROMPTS, SYSTEM_PROMPT

        template = PROMPTS.get(str(task))
        if template is None:
            return self._refuse(f"task {task!r} has no prompt for this backend")
        instruction = template.format(question=question or "") if "{question}" in template else template
        if evidence:
            instruction = f"{instruction}\n\n{format_evidence_block(evidence)}"

        try:
            rendered = [self._render(ref) for ref in images]
        except Exception as exc:
            return self._refuse(f"could not render the imagery: {type(exc).__name__}: {exc}")

        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for data_uri, _size in rendered:
            content.append({"type": "image_url", "image_url": data_uri})

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        try:
            raw = self._post(body)
            text = str(raw["choices"][0]["message"]["content"] or "").strip()
        except SatQueryError as exc:
            return self._refuse(str(exc))
        except (KeyError, IndexError, TypeError) as exc:
            return self._refuse(f"unexpected response shape from {self.model}: {exc}")

        if not text:
            return self._refuse(f"{self.model} returned an empty answer")

        boxes: tuple[Box, ...] = ()
        box_notes: tuple[str, ...] = ()
        if task == "ground":
            width, height = rendered[0][1]
            boxes, box_notes = self._parse_boxes(text, width, height)
            # The decode is a JSON list; a user reads a sentence. The raw text
            # stays in meta so the trace shows what the model actually emitted.
            phrase = (question or "the phrase").strip()
            box_notes = box_notes + (f"raw decode: {text[:200]}",)
            text = (
                f"{len(boxes)} region(s) matching {phrase!r}."
                if boxes
                else f"No region matching {phrase!r} was located."
            )

        if task in ("vqa", "change_vqa"):
            from ..text import extract_answer

            text = extract_answer(text) or text

        usage = raw.get("usage") or {}
        return VlmAnswer(
            text=text,
            confidence=self.API_CONFIDENCE,
            boxes=boxes,
            backend=self.name,
            kind=self.kind,
            meta={
                "model": raw.get("model", self.model),
                "tokens": usage.get("total_tokens"),
                "images_sent": len(rendered),
                "confidence_source": "fixed: the API returns no token probabilities",
                **({"box_notes": list(box_notes)} if box_notes else {}),
            },
        )

    # -- internals ------------------------------------------------------------ #

    def _render(self, ref: ImageRef) -> tuple[str, tuple[int, int]]:
        """``(data-uri, (width, height))`` of the image as the GUI displays it."""
        import base64
        import io

        from .render import display_array
        from .validate import load_image

        array = display_array(
            load_image(ref.path), modality=ref.modality or "optical", max_side=self.max_side
        )
        from PIL import Image

        hwc = np.ascontiguousarray(np.transpose(np.asarray(array, dtype=np.uint8), (1, 2, 0)))
        image = Image.fromarray(hwc)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return (f"data:image/png;base64,{encoded}", (int(array.shape[2]), int(array.shape[1])))

    def _parse_boxes(
        self, text: str, width: int, height: int
    ) -> tuple[tuple[Box, ...], tuple[str, ...]]:
        """Boxes plus the parser's own notes, which belong in the trace.

        ``parse_boxes`` never raises -- every failure mode (empty decode, a
        stripped code fence, coordinates that were 0-1 rather than 0-1000) comes
        back as a note, and those notes are exactly what explains an empty
        overlay to whoever is looking at it.
        """
        from ..models.qwen_vl import parse_boxes

        boxes, notes = parse_boxes(text, width=width, height=height, scale=1000.0)
        if notes:
            _LOG.debug("%s box notes: %s", self.name, "; ".join(notes))
        return boxes, notes

    def _post(self, body: Mapping[str, Any]) -> dict[str, Any]:
        import json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise SatQueryError(f"{self.model} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SatQueryError(f"{self.model} is unreachable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SatQueryError(f"{self.model} returned a non-JSON body: {exc}") from exc

    def _refuse(self, reason: str) -> VlmAnswer:
        """A cloud failure degrades the answer; it never raises into the loop."""
        _LOG.warning("%s: %s", self.name, reason)
        return VlmAnswer(
            text="",
            confidence=0.0,
            backend=self.name,
            kind=self.kind,
            refused=True,
            meta={"reason": reason},
        )


class EvidenceVLM:
    """The no-weights text backend: says what the measurements say, and no more.

    It answers by *reading the evidence dict* the deterministic tools produced --
    index means, water percentages, backscatter statistics, change-mask geometry
    -- and phrasing them. Three consequences, all intentional:

    * it cannot count objects, name a land use, or read a referring expression
      that is not a land-cover concept, and it says so rather than guessing;
    * every sentence it emits is already citable, so the composer's
      drop-unsupported-clauses pass is a no-op on its output -- which makes that
      pass testable without a GPU;
    * the phrasing is templated, so the same scene produces the same words. A
      demo that reworded itself between runs would make the trace useless as a
      receipt.

    This is the backend the prototype runs on today. The tuned Qwen replaces it
    in the same slot, consuming the same evidence dict.
    """

    name = "evidence-rules"
    kind = "deterministic"

    #: Concept -> the words that name it in a question, and the evidence keys
    #: that measure it. Order matters only for reporting.
    CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
        "water": {
            "words": ("water", "lake", "river", "sea", "ocean", "pond", "reservoir", "flood", "wet"),
            "keys": ("water_pct", "ndwi_positive_pct"),
        },
        "vegetation": {
            "words": ("vegetation", "vegetated", "forest", "tree", "trees", "green", "crop",
                      "crops", "farmland", "grass", "plantation"),
            "keys": ("vegetation_pct", "ndvi_mean", "vari_mean"),
        },
        "built_up": {
            "words": ("building", "buildings", "built", "urban", "city", "town", "house",
                      "houses", "residential", "industrial", "road", "roads", "settlement"),
            "keys": ("built_up_pct", "ndbi_mean"),
        },
        "bare": {
            "words": ("bare", "soil", "sand", "desert", "barren", "rock"),
            "keys": ("bare_pct",),
        },
        "change": {
            "words": ("change", "changed", "difference", "differ", "new", "appeared",
                      "disappeared", "grew", "expanded"),
            "keys": ("changed_area_pct", "num_regions"),
        },
    }

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "tasks": list(_TEXT_TASKS)}

    def answer(
        self,
        task: Task,
        images: Sequence[ImageRef],
        *,
        question: str | None = None,
        evidence: Mapping[str, Any] | None = None,
        sample_id: str = "query",
    ) -> VlmAnswer:
        ev = dict(evidence or {})
        if task == "caption":
            return self._caption(images, ev)
        if task == "change_caption":
            return self._change_caption(ev)
        if task in ("vqa", "change_vqa"):
            return self._vqa(question or "", ev, change=task == "change_vqa")
        if task == "ground":
            return self._ground(question or "", ev)
        return self._refuse(f"task {task!r} has no deterministic formulation")

    # -- per task ------------------------------------------------------------ #

    def _caption(self, images: Sequence[ImageRef], ev: Mapping[str, Any]) -> VlmAnswer:
        ref = images[0] if images else None
        modality = (ref.modality if ref else "unknown") or "unknown"
        size = f"{ev.get('width', '?')}x{ev.get('height', '?')} px"
        article = "An" if modality[:1] in "aeiou" else "A"
        parts = [f"{article} {modality} scene of {size}."]
        fused = self._fusion_summary(ev)
        if fused:
            parts.append(fused)
            parts.append(
                "The classes come from thresholded indices and backscatter rules over both "
                "sensors, not from a trained classifier."
            )
            return self._answer(" ".join(parts), ev, confidence=0.55)
        cover = self._cover_clauses(ev)
        if cover:
            parts.append("Measured cover: " + "; ".join(cover) + ".")
        else:
            parts.append(
                "No land-cover index could be computed from the available bands, so there is "
                "nothing measured to describe."
            )
        if "backscatter_mean" in ev:
            parts.append(
                f"Mean backscatter is {ev['backscatter_mean']:.1f} dB "
                f"(p5 {ev.get('backscatter_p5', float('nan')):.1f}, "
                f"p95 {ev.get('backscatter_p95', float('nan')):.1f})."
            )
        parts.append(
            "This description is derived from measurements only -- no vision-language model "
            "was available to describe layout or object function."
        )
        return self._answer(" ".join(parts), ev, confidence=0.55)

    def _change_caption(self, ev: Mapping[str, Any]) -> VlmAnswer:
        if "changed_area_pct" not in ev:
            return self._refuse("no change map was produced, so there is nothing to describe")
        pct = float(ev["changed_area_pct"])
        regions = int(ev.get("num_regions", 0))
        where = ev.get("centroid_quadrant")
        if pct < 0.5:
            text = (
                f"Essentially nothing changed between the two dates: {pct:.1f}% of pixels differ, "
                "which is at the level of registration and illumination noise."
            )
            return self._answer(text, ev, confidence=0.6)
        sentence = f"{pct:.1f}% of the scene changed between the two dates"
        if regions:
            sentence += f", in {regions} distinct region{'s' if regions != 1 else ''}"
        if where:
            sentence += f", concentrated in the {str(where).replace('_', '-')} quadrant"
        sentence += "."
        extra = []
        if "largest_region_pct" in ev:
            extra.append(f"the largest single region covers {float(ev['largest_region_pct']):.1f}%")
        if "brightness_delta" in ev:
            direction = "brighter" if float(ev["brightness_delta"]) > 0 else "darker"
            extra.append(f"the changed area is {direction} in the later image")
        if "ndvi_delta" in ev:
            delta = float(ev["ndvi_delta"])
            extra.append(
                f"mean vegetation index {'rose' if delta > 0 else 'fell'} by {abs(delta):.2f}"
            )
        if extra:
            sentence += " " + ("; ".join(extra)).capitalize() + "."
        sentence += (
            " The change was detected by differencing the two images, so it locates change "
            "without identifying what was built or removed."
        )
        return self._answer(sentence, ev, confidence=0.6)

    def _vqa(self, question: str, ev: Mapping[str, Any], *, change: bool) -> VlmAnswer:
        concepts = self._concepts_in(question)
        lowered = question.lower()
        if change and "changed_area_pct" in ev:
            pct = float(ev["changed_area_pct"])
            if _asks_yes_no(lowered):
                answer = "yes" if pct >= 1.0 else "no"
                return self._answer(
                    f"{answer} -- {pct:.1f}% of the scene differs between the two dates.",
                    ev,
                    confidence=0.6,
                )
            if _asks_how_much(lowered):
                return self._answer(f"{pct:.1f}% of the scene changed.", ev, confidence=0.65)
            # "what changed", "describe the difference": an open question about
            # the change wants the description, not a single statistic.
            return self._change_caption(ev)

        if _asks_count(lowered):
            # Checked before any concept match: "how many buildings" must not be
            # answered with "built-up cover measures 5.0%". An area is not a
            # count, and quietly substituting one for the other is the exact
            # failure this backend exists to avoid.
            return self._refuse(
                "counting objects needs the vision-language model; the deterministic tools "
                "measure area and index statistics, not instances"
            )

        for concept in concepts:
            measured = self._measure(concept, ev)
            if measured is None:
                continue
            label, value, unit = measured
            if _asks_yes_no(lowered):
                present = value >= 1.0 if unit == "%" else value > 0.0
                return self._answer(
                    f"{'yes' if present else 'no'} -- {label} measures {value:.1f}{unit}.",
                    ev,
                    confidence=0.6,
                )
            if _asks_how_much(lowered):
                return self._answer(f"{label} measures {value:.1f}{unit}.", ev, confidence=0.65)
            return self._answer(f"{label} measures {value:.1f}{unit}.", ev, confidence=0.5)

        fused = self._fusion_summary(ev)
        if fused:
            return self._answer(fused, ev, confidence=0.55)

        measurable = self._measurable(ev)
        return self._refuse(
            "no vision-language model is loaded, and this question is not answerable from the "
            "measurements available for this input"
            + (f" ({measurable})" if measurable else " (nothing measurable was produced)")
        )

    def _ground(self, phrase: str, ev: Mapping[str, Any]) -> VlmAnswer:
        boxes = ev.get("concept_boxes") or {}
        for concept in self._concepts_in(phrase):
            found = boxes.get(concept)
            if found:
                return VlmAnswer(
                    text=f"{len(found)} region(s) matching {concept.replace('_', ' ')}.",
                    confidence=0.5,
                    boxes=tuple(found),
                    backend=self.name,
                    kind=self.kind,
                    meta={"grounded_on": concept, "method": "index threshold + connected components"},
                )
        locatable = ", ".join(sorted(k.replace("_", " ") for k, v in boxes.items() if v))
        return self._refuse(
            "referring expressions are grounded by the vision-language model. Without it, only "
            "thresholded land-cover regions can be located, and for this image those are: "
            + (locatable if locatable else "none -- its bands support no land-cover index")
        )

    # -- helpers -------------------------------------------------------------- #

    def _concepts_in(self, text: str) -> list[str]:
        """Which measurable concepts a question mentions.

        Matching is on whole words over a punctuation-stripped string, because
        "is this vegetation?" and "vegetation," are the shapes questions actually
        arrive in, and a substring match would find "water" inside "watershed".
        """
        words = set(_WORD_RE.findall(text.lower()))
        return [
            name
            for name, spec in self.CONCEPTS.items()
            if any(w.strip("_") in words for w in spec["words"])
        ]

    def _measure(self, concept: str, ev: Mapping[str, Any]) -> tuple[str, float, str] | None:
        for key in self.CONCEPTS[concept]["keys"]:
            if key in ev and isinstance(ev[key], (int, float)):
                unit = "%" if key.endswith("_pct") else ""
                return (key.replace("_", " ").replace(" pct", " cover"), float(ev[key]), unit)
        return None

    def _measurable(self, ev: Mapping[str, Any]) -> str:
        """The concepts this evidence *can* speak to, in the user's words."""
        names = []
        for concept, spec in self.CONCEPTS.items():
            if any(key in ev for key in spec["keys"]):
                names.append(concept.replace("_", " "))
        return ", ".join(names)

    def _fusion_summary(self, ev: Mapping[str, Any]) -> str:
        """Phrase the fusion classes, largest first, when a fusion tool has run.

        Detected by the presence of ``<class>_pct`` keys for the fusion
        vocabulary rather than by a flag, so it works whether the fusion result
        arrived from this plan or from an earlier tool in it.
        """
        found = [
            (label, float(ev[f"{label}_pct"]))
            for label in FUSION_LABELS
            if isinstance(ev.get(f"{label}_pct"), (int, float))
        ]
        found = [(label, pct) for label, pct in found if pct > 0]
        if not found:
            return ""
        found.sort(key=lambda pair: pair[1], reverse=True)
        listed = ", ".join(f"{label.replace('_', ' ')} {pct:.1f}%" for label, pct in found)
        lead = "Fusing the optical and SAR images gives"
        if "backscatter_mean" in ev:
            lead += f" (mean backscatter {float(ev['backscatter_mean']):.1f} dB)"
        return f"{lead}: {listed} of the scene."

    def _cover_clauses(self, ev: Mapping[str, Any]) -> list[str]:
        clauses = []
        for key, label in (
            ("water_pct", "water"),
            ("vegetation_pct", "vegetation"),
            ("built_up_pct", "built-up or bare"),
        ):
            if key in ev:
                clauses.append(f"{label} {float(ev[key]):.1f}% of valid pixels")
        for key, label in (("ndvi_mean", "NDVI"), ("vari_mean", "VARI"), ("ndwi_mean", "NDWI")):
            if key in ev:
                clauses.append(f"mean {label} {float(ev[key]):.2f}")
        return clauses

    def _answer(self, text: str, ev: Mapping[str, Any], *, confidence: float) -> VlmAnswer:
        return VlmAnswer(text=text, confidence=confidence, backend=self.name, kind=self.kind)

    def _refuse(self, reason: str) -> VlmAnswer:
        return VlmAnswer(
            text="",
            confidence=0.0,
            backend=self.name,
            kind=self.kind,
            refused=True,
            meta={"reason": reason},
        )


_TEXT_TASKS: tuple[str, ...] = ("vqa", "caption", "ground", "change_vqa", "change_caption")

#: Words, not substrings: "water" must not match inside "watershed".
_WORD_RE = re.compile(r"[a-z][a-z-]*")


def _asks_yes_no(q: str) -> bool:
    return q.strip().startswith(("is ", "are ", "does ", "do ", "can ", "was ", "were ", "has ", "have "))


def _asks_how_much(q: str) -> bool:
    return any(w in q for w in ("how much", "what percentage", "what proportion", "what fraction",
                                "how large", "what area", "percentage", "proportion"))


def _asks_count(q: str) -> bool:
    return "how many" in q or q.strip().startswith("count ")


def format_evidence_block(evidence: Mapping[str, Any]) -> str:
    """Render the deterministic measurements as prompt context.

    Deliberately flat ``key: value`` lines rather than JSON: instruction-tuned
    VLMs follow a short labelled list far more reliably than a nested object, and
    the whole point is that these numbers end up *in* the answer.
    """
    lines = ["Measured evidence from deterministic tools (cite these numbers, do not invent others):"]
    for key, value in evidence.items():
        if key.startswith("_") or isinstance(value, (dict, list, tuple)) or value is None:
            continue
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.3g}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("If the evidence does not support a claim, do not make it.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Change map
# --------------------------------------------------------------------------- #


class MciChangeMap:
    """Slot for Change-Agent MCI (~230M, Apache-2.0, HF ``lcybuaa/Change-Agent``).

    Not wired up: MCI ships as a research repo rather than a ``transformers``
    architecture, so loading it means vendoring its model code, and that is
    Phase 3 work with a T4 attached. The class exists so the slot is visible in
    ``satquery doctor`` and so selecting it fails with an explanation instead of
    an ``ImportError`` three frames deep.
    """

    name = "change-agent-mci"
    kind = "model"

    @staticmethod
    def available() -> tuple[bool, str]:
        return (
            False,
            "Change-Agent MCI is not vendored yet (Phase 3); the deterministic CVA backend "
            "produces the mask and statistics in the meantime",
        )

    def detect(self, *_args: Any, **_kwargs: Any) -> ChangeResult:  # pragma: no cover - guarded
        raise SatQueryError(self.available()[1])


class CvaChangeMap:
    """Change vector analysis with an Otsu cut: the deterministic change detector.

    The pipeline is the classical one, chosen because every step is inspectable:

    1. render both images through the *same* display transform, so an optical
       pair is compared in stretched reflectance and a SAR pair in dB -- never
       raw power, where a 3 dB change at the dark end swamps everything else;
    2. match the second image's brightness to the first (mean/std normalisation)
       so a sun-angle or gain difference does not read as change across the whole
       scene. This is the single most important step and the one a naive
       difference gets wrong;
    3. take the per-pixel L2 norm of the band difference -- the change *vector*
       magnitude -- rather than a single-band difference;
    4. threshold it at ``max(Otsu, noise floor)``. Otsu alone is a trap here:
       it always returns a cut, so an *unchanged* pair -- a unimodal histogram of
       pure residual noise -- gets split down the middle and reported as ~50%
       change. The noise floor (median + 3 robust sigmas of the magnitude) is
       what makes "nothing changed" an available answer;
    5. drop regions below ``min_region_px`` as speckle, then measure.

    What it cannot do is say *what* changed, and it does not pretend to: the
    result carries geometry and arithmetic, and the sentence is somebody else's
    job (invariant 04).
    """

    name = "cva-otsu"
    kind = "deterministic"

    def __init__(self, *, min_region_px: int = 16, sensitivity: float = 1.0) -> None:
        self.min_region_px = int(min_region_px)
        self.sensitivity = float(sensitivity)

    @staticmethod
    def available() -> tuple[bool, str]:
        return (True, "numpy only")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "min_region_px": self.min_region_px,
            "sensitivity": self.sensitivity,
        }

    def detect(self, before: np.ndarray, after: np.ndarray) -> ChangeResult:
        """``before``/``after`` are ``(3, h, w)`` uint8 display renderings on one grid."""
        a = np.asarray(before, dtype=np.float32)
        b = np.asarray(after, dtype=np.float32)
        if a.shape != b.shape:
            raise SatQueryError(
                f"change detection needs one grid, got {a.shape} against {b.shape}; "
                "resample the pair first"
            )
        b_matched, gain, offset = _match_brightness(b, a)
        magnitude = np.sqrt(np.sum((b_matched - a) ** 2, axis=0) / a.shape[0])

        otsu = otsu_threshold(magnitude)
        floor = noise_floor(magnitude)
        threshold = max(otsu, floor) * self.sensitivity
        mask = magnitude > threshold
        labels, n_labels = label_components(mask)
        regions = _region_stats(labels, n_labels, min_px=self.min_region_px)
        if regions:
            keep = {int(r["label"]) for r in regions}
            mask = np.isin(labels, list(keep))
        else:
            mask = np.zeros_like(mask)

        total = float(mask.size)
        changed_pct = float(100.0 * np.count_nonzero(mask) / total)
        stats: dict[str, Any] = {
            "changed_area_pct": round(changed_pct, 2),
            "num_regions": len(regions),
            "largest_region_pct": round(regions[0]["area_pct"], 2) if regions else 0.0,
            "threshold": round(float(threshold), 3),
            "threshold_otsu": round(float(otsu), 3),
            "threshold_noise_floor": round(float(floor), 3),
            "threshold_method": "max(Otsu, median + 3 robust sigma) on change-vector magnitude",
            "brightness_gain": round(float(gain), 3),
            "brightness_offset": round(float(offset), 2),
            "min_region_px": self.min_region_px,
        }
        if regions:
            stats["centroid_quadrant"] = _quadrant(
                regions[0]["centroid_y"], regions[0]["centroid_x"], mask.shape
            )
            stats["brightness_delta"] = round(
                float(np.mean(b_matched[:, mask]) - np.mean(a[:, mask])), 2
            )
        return ChangeResult(
            mask=mask,
            stats=stats,
            backend=self.name,
            kind=self.kind,
            regions=tuple(regions[:10]),
        )


# --------------------------------------------------------------------------- #
# Optical-SAR fusion
# --------------------------------------------------------------------------- #


class TerraMindFusion:
    """Slot for TerraMind-1.0-base (IBM/ESA, Apache-2.0) plus a trained head.

    Not wired up: TerraMind arrives through TerraTorch and needs a task head
    trained on BigEarthNet-MM, which is Phase 3. Selecting it explicitly reports
    that rather than importing something that is not installed.
    """

    name = "terramind-1.0-base"
    kind = "model"

    @staticmethod
    def available() -> tuple[bool, str]:
        try:  # terratorch is not in any extra yet; check anyway so the message is accurate
            import terratorch  # type: ignore # noqa: F401
        except ImportError:
            return (
                False,
                "TerraTorch is not installed and no fusion head has been trained yet (Phase 3); "
                "the rule-based fusion backend runs instead",
            )
        return (False, "TerraTorch is present but no BigEarthNet-MM head has been trained yet")

    def classify(self, *_args: Any, **_kwargs: Any) -> FusionResult:  # pragma: no cover - guarded
        raise SatQueryError(self.available()[1])


class RuleFusion:
    """Optical indices plus SAR backscatter, combined by stated rules.

    The rules are the textbook ones, and each carries the sentence that explains
    it, because the composer has to be able to justify a label:

    * **water** -- low SAR backscatter (a smooth surface reflects the pulse away)
      *and* positive NDWI where an optical NIR band exists. Requiring both is
      what rejects the classic false positives: radar shadow is dark but not wet,
      and building shade lifts NDWI but is not smooth.
    * **urban fabric** -- bright, rough SAR return (double-bounce off walls) with
      low vegetation index. This is the one place SAR genuinely outperforms
      optical, and it is the reason fusion is worth the plumbing.
    * **forest / vegetation** -- high NDVI (or VARI on RGB-only input), with mid
      SAR backscatter from volume scattering.
    * **agricultural land** -- moderate vegetation index over a *smooth* radar
      surface, which is what separates a field from a canopy.
    * **bare or sparse vegetation** -- low vegetation index and low roughness.

    Every label comes with the percentage of the scene it covers and the rule
    text; a class under ``min_area_pct`` is scored but not asserted, because a
    0.3% detection is noise in a 512 px tile.
    """

    name = "fusion-rules-v1"
    kind = "deterministic"

    def __init__(self, *, min_area_pct: float = 2.0) -> None:
        self.min_area_pct = float(min_area_pct)

    @staticmethod
    def available() -> tuple[bool, str]:
        return (True, "numpy only")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "min_area_pct": self.min_area_pct,
                "labels": list(FUSION_LABELS)}

    def classify(
        self,
        *,
        optical_evidence: Mapping[str, Any],
        sar_evidence: Mapping[str, Any],
        masks: Mapping[str, np.ndarray] | None = None,
    ) -> FusionResult:
        masks = dict(masks or {})
        veg = masks.get("vegetation")
        water_optical = masks.get("water_optical")
        water_sar = masks.get("water_sar")
        rough = masks.get("rough")

        out_masks: dict[str, np.ndarray] = {}
        rationale: dict[str, str] = {}
        area: dict[str, float] = {}
        scores: dict[str, float] = {}

        water = _combine(water_sar, water_optical, mode="and")
        if water is None:
            water = water_sar if water_sar is not None else water_optical
            if water is not None:
                rationale["water_bodies"] = (
                    "one sensor only: "
                    + ("SAR backscatter below the water threshold" if water_sar is not None
                       else "positive NDWI")
                )
        else:
            rationale["water_bodies"] = (
                "SAR backscatter below the water threshold AND positive NDWI -- both sensors "
                "agree, which rejects radar shadow and building shade"
            )
        _record("water_bodies", water, out_masks, area, scores)

        if rough is not None and veg is not None:
            urban = rough & ~veg & ~(water if water is not None else np.zeros_like(rough))
            rationale["urban_fabric"] = (
                "bright, rough SAR return (double-bounce off vertical structures) with a low "
                "vegetation index"
            )
            _record("urban_fabric", urban, out_masks, area, scores)
        elif rough is not None:
            rationale["urban_fabric"] = "bright, rough SAR return; no optical band to exclude canopy"
            _record("urban_fabric", rough, out_masks, area, scores)

        if veg is not None:
            dense = veg if rough is None else (veg & rough)
            sparse = veg if rough is None else (veg & ~rough)
            rationale["forest_and_vegetation"] = (
                "high vegetation index with volume scattering in the radar return"
                if rough is not None else "high vegetation index"
            )
            rationale["agricultural_land"] = (
                "moderate vegetation index over a smooth radar surface -- a field rather than a "
                "canopy" if rough is not None else "vegetation index in the crop range"
            )
            _record("forest_and_vegetation", dense, out_masks, area, scores)
            _record("agricultural_land", sparse, out_masks, area, scores)

            bare = ~veg
            if water is not None:
                bare = bare & ~water
            if rough is not None:
                bare = bare & ~rough
            rationale["bare_or_sparse_vegetation"] = "low vegetation index, smooth and unbuilt"
            _record("bare_or_sparse_vegetation", bare, out_masks, area, scores)

        labels = tuple(
            name for name in FUSION_LABELS if area.get(name, 0.0) >= self.min_area_pct
        )
        return FusionResult(
            labels=labels,
            label_scores={k: scores.get(k, 0.0) for k in FUSION_LABELS},
            area_pct={k: area.get(k, 0.0) for k in FUSION_LABELS},
            masks=out_masks,
            rationale=rationale,
            backend=self.name,
            kind=self.kind,
        )


def _record(
    name: str,
    mask: np.ndarray | None,
    masks: dict[str, np.ndarray],
    area: dict[str, float],
    scores: dict[str, float],
) -> None:
    if mask is None:
        return
    pct = float(100.0 * np.count_nonzero(mask) / max(1, mask.size))
    masks[name] = mask
    area[name] = round(pct, 2)
    # Score is the covered fraction, squashed so a 20% class does not read as
    # 0.2 confidence: presence is about detection, not dominance.
    scores[name] = round(float(min(1.0, (pct / 100.0) ** 0.5 * 1.6)), 3)


def _combine(a: np.ndarray | None, b: np.ndarray | None, *, mode: str) -> np.ndarray | None:
    if a is None or b is None:
        return None
    return (a & b) if mode == "and" else (a | b)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def select_vlm(config: SatQueryConfig | None = None) -> tuple[Any, str]:
    """``(backend, reason)`` for the text slot. Never raises for a missing model.

    ``auto`` requires both the ``vlm`` extra and a visible GPU. Loading a 4-bit
    7B onto laptop CPU technically works and takes minutes per token, which in a
    live demo is indistinguishable from a hang -- so the rule is explicit rather
    than best-effort.
    """
    cfg = config or get_config()
    choice = os.environ.get("SATQUERY_VLM", "auto").strip().lower()

    if choice in ("evidence", "none", "off"):
        return EvidenceVLM(), "SATQUERY_VLM=evidence: deterministic text backend requested"
    if choice in ("mistral", "mistral-vision", "cloud"):
        # Never reachable from "auto": this backend uploads imagery, and that is
        # a decision an operator makes explicitly, not one they inherit.
        try:
            backend = MistralVLM()
            return backend, (
                f"SATQUERY_VLM=mistral: serving {backend.model} over the API -- "
                "imagery is sent off this machine"
            )
        except SatQueryError as exc:
            return EvidenceVLM(), f"{exc}; fell back to evidence rules"
    if choice not in ("auto", ""):
        try:
            return PredictorVLM(choice), f"SATQUERY_VLM={choice}: requested explicitly"
        except Exception as exc:
            _LOG.warning("requested VLM %r could not be built: %s", choice, exc)
            return EvidenceVLM(), f"{choice} could not be built ({exc}); fell back to evidence rules"

    missing = [m for m in ("torch", "transformers") if not _importable(m)]
    if missing:
        return EvidenceVLM(), (
            f"{' and '.join(missing)} not installed, so no VLM can be served "
            '(pip install "satquery[vlm]")'
        )
    if cfg.resolve_device() != "cuda":
        return EvidenceVLM(), (
            "no CUDA device visible; a 4-bit 7B on CPU is minutes per answer, so the "
            "deterministic backend serves instead (set SATQUERY_VLM=qwen2_5_vl to force it)"
        )
    try:
        return PredictorVLM("qwen2_5_vl"), "auto: torch + CUDA available, serving Qwen2.5-VL"
    except Exception as exc:  # pragma: no cover - GPU box only
        return EvidenceVLM(), f"Qwen2.5-VL could not be built ({exc}); fell back to evidence rules"


def select_change_backend() -> tuple[Any, str]:
    choice = os.environ.get("SATQUERY_CHANGE_BACKEND", "auto").strip().lower()
    if choice == "mci":
        ok, why = MciChangeMap.available()
        if not ok:
            raise SatQueryError(f"SATQUERY_CHANGE_BACKEND=mci: {why}")
        return MciChangeMap(), "requested explicitly"  # pragma: no cover - unreachable today
    if choice in ("cva", "rules"):
        return CvaChangeMap(), "SATQUERY_CHANGE_BACKEND=cva: requested explicitly"
    ok, why = MciChangeMap.available()
    return (MciChangeMap(), "auto: MCI available") if ok else (CvaChangeMap(), f"auto: {why}")


def select_fusion_backend() -> tuple[Any, str]:
    choice = os.environ.get("SATQUERY_FUSION_BACKEND", "auto").strip().lower()
    if choice == "terramind":
        ok, why = TerraMindFusion.available()
        if not ok:
            raise SatQueryError(f"SATQUERY_FUSION_BACKEND=terramind: {why}")
        return TerraMindFusion(), "requested explicitly"  # pragma: no cover - unreachable today
    if choice in ("rules", "rule"):
        return RuleFusion(), "SATQUERY_FUSION_BACKEND=rules: requested explicitly"
    ok, why = TerraMindFusion.available()
    return (TerraMindFusion(), "auto: TerraMind available") if ok else (RuleFusion(), f"auto: {why}")


def backend_status(config: SatQueryConfig | None = None) -> dict[str, Any]:
    """What would serve each slot right now, and why. Rendered by ``satquery doctor``."""
    vlm, vlm_why = select_vlm(config)
    change, change_why = select_change_backend()
    fusion, fusion_why = select_fusion_backend()
    return {
        "text": {"backend": vlm.name, "kind": vlm.kind, "reason": vlm_why},
        "change_map": {"backend": change.name, "kind": change.kind, "reason": change_why},
        "fusion": {"backend": fusion.name, "kind": fusion.kind, "reason": fusion_why},
    }


def _importable(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Numeric helpers, shared by the deterministic backends
# --------------------------------------------------------------------------- #


def otsu_threshold(values: np.ndarray, *, bins: int = 256) -> float:
    """Otsu's between-class-variance threshold over the finite values.

    Chosen over a fixed cut because the next sensor has a different dynamic
    range, and over a percentile because the fraction of a scene that changed is
    exactly the quantity being measured -- fixing it at "the top 5%" would make
    the answer a restatement of the assumption.

    A degenerate histogram (one non-empty bin) returns a threshold above the
    maximum, i.e. "nothing changed", which is the right failure: an unchanging
    pair must not be forced to produce 50% change.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("inf")
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return float(hi + 1.0)
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return float(hi + 1.0)
    p = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(centres[int(np.argmax(sigma_b))])


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling. ``(labels, count)``, labels start at 1.

    scipy does this in C when it is installed; the numpy fallback is a two-pass
    union-find so the change tools keep working on a bare install. Both give the
    same partition -- only the label *numbering* may differ, so never compare
    label ids across environments, only region geometry.
    """
    m = np.asarray(mask, dtype=bool)
    try:
        from scipy import ndimage  # optional extra

        labels, count = ndimage.label(m)
        return np.asarray(labels), int(count)
    except ImportError:
        pass

    labels = np.zeros(m.shape, dtype=np.int32)
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    rows, cols = m.shape
    for y in range(rows):
        row = m[y]
        if not row.any():
            continue
        for x in np.flatnonzero(row):
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up and left:
                labels[y, x] = min(up, left)
                union(int(up), int(left))
            elif up or left:
                labels[y, x] = up or left
            else:
                parent.append(len(parent))
                labels[y, x] = len(parent) - 1

    remap: dict[int, int] = {0: 0}
    out = np.zeros_like(labels)
    nonzero = np.flatnonzero(labels.reshape(-1))
    flat = labels.reshape(-1)
    out_flat = out.reshape(-1)
    for i in nonzero:
        root = find(int(flat[i]))
        if root not in remap:
            remap[root] = len(remap)
        out_flat[i] = remap[root]
    return out, len(remap) - 1


def _region_stats(
    labels: np.ndarray, n_labels: int, *, min_px: int
) -> list[dict[str, Any]]:
    """Per-region area and centroid, largest first, regions under ``min_px`` dropped."""
    if n_labels <= 0:
        return []
    flat = labels.reshape(-1)
    counts = np.bincount(flat, minlength=n_labels + 1)
    total = float(flat.size)
    ys, xs = np.indices(labels.shape)
    sum_y = np.bincount(flat, weights=ys.reshape(-1), minlength=n_labels + 1)
    sum_x = np.bincount(flat, weights=xs.reshape(-1), minlength=n_labels + 1)
    out: list[dict[str, Any]] = []
    for label in range(1, n_labels + 1):
        area = int(counts[label])
        if area < min_px:
            continue
        out.append(
            {
                "label": label,
                "area_px": area,
                "area_pct": round(100.0 * area / total, 3),
                "centroid_y": float(sum_y[label] / area),
                "centroid_x": float(sum_x[label] / area),
            }
        )
    out.sort(key=lambda r: r["area_px"], reverse=True)
    return out


def noise_floor(values: np.ndarray, *, k: float = 3.0) -> float:
    """``median + k * robust sigma``, the level below which a difference is noise.

    Robust statistics, not mean and standard deviation: the changed pixels are
    precisely the outliers, so a non-robust spread estimate is inflated by the
    very signal it is supposed to be a floor under. ``1.4826 * MAD`` is the
    standard consistent estimator of sigma for Gaussian data.

    ``k = 3`` puts roughly 0.1% of unchanged pixels above the floor by chance,
    and those survivors are scattered single pixels that the minimum-region
    filter then removes.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("inf")
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return median + float(k) * 1.4826 * mad


def _match_brightness(
    source: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Linearly rescale ``source`` onto ``reference``'s radiometry, robustly.

    Without this, a bi-temporal pair taken at different sun angles differs
    *everywhere*, the threshold lands in the middle of that global offset, and
    the change map is the whole scene.

    Median and MAD rather than mean and standard deviation, for the same reason
    the noise floor uses them: a genuine change region is an outlier population,
    and matching on non-robust statistics lets it drag the correction -- a bright
    new development would scale the whole later image *down*, smearing false
    change across the unchanged remainder.

    The gain and offset are returned so the trace shows how much correction was
    applied; a gain far from 1.0 means the two images are not the same sensor.
    """
    src_median = float(np.median(source))
    ref_median = float(np.median(reference))
    src_scale = float(np.median(np.abs(source - src_median)))
    ref_scale = float(np.median(np.abs(reference - ref_median)))
    if src_scale <= 1e-6:
        return source, 1.0, 0.0
    gain = ref_scale / src_scale
    offset = ref_median - gain * src_median
    return source * gain + offset, gain, offset


def _quadrant(y: float, x: float, shape: tuple[int, int]) -> str:
    """Compass name for a point, or ``central`` when it sits near the middle."""
    height, width = shape
    ny, nx = y / max(1, height), x / max(1, width)
    vertical = "north" if ny < 0.4 else ("south" if ny > 0.6 else "")
    horizontal = "west" if nx < 0.4 else ("east" if nx > 0.6 else "")
    if not vertical and not horizontal:
        return "central"
    return "-".join(part for part in (vertical, horizontal) if part)
