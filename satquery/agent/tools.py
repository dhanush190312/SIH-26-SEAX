"""The seven tools: what they are, when they are offered, and how they run.

This module is where invariant 02 lives. Every tool exists in the codebase for
every request; what changes per request is how many of them are *described to the
router at all*. :func:`build_menu` takes the validator's report and returns the
offered set plus, for each withheld tool, the reason -- and both halves go into
the trace. The router cannot call a tool it was never told about, so misrouting
the obvious cases is structurally impossible rather than merely unlikely.

The second thing this module enforces is invariant 03: **deterministic tools run
first and their numbers go into the prompt**. Every text tool computes
:func:`scene_evidence` before it calls the VLM -- index means, water and
vegetation percentages, backscatter statistics, mask geometry -- and passes it as
context. That ordering is what makes a SAR answer defensible (a VLM cannot read
speckle; it can explain a -19.4 dB median) and what makes the composer's
citation check possible at all.

Tool inputs are split in two on purpose. The *router* supplies semantic
parameters (the question, the index name, the detail level); the *executor*
injects ``image_id`` and ``pair_id`` from the session. That split is invariant
01: the router is a text-only planner that never receives, and cannot request, a
pixel.

Every tool returns a :class:`ToolResult` with the same shape -- output for the
answer, evidence for the citations, artifacts for the GUI, and the backend that
produced it -- so the composer and the report do not branch per tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..config import SatQueryConfig, get_config
from ..contracts import Box, SatQueryError, Task
from ..log import get_logger
from ..preprocess.bands import INDEX_REQUIREMENTS, available_indices, resolve_band_roles
from ..preprocess.indices import index_stats, ndbi, ndvi, ndwi, vari
from ..preprocess.sar import WATER_DB_THRESHOLD, backscatter_stats, linear_to_db, sar_water_mask
from .backends import (
    ChangeResult,
    FusionResult,
    VlmAnswer,
    label_components,
    select_change_backend,
    select_fusion_backend,
    select_vlm,
)
from .render import display_array, draw_boxes, overlay_mask, save_png
from .session import ImageRecord, PairRecord, Session
from .validate import load_image

__all__ = [
    "ToolSpec",
    "ToolResult",
    "ToolContext",
    "ToolMenu",
    "TOOLS",
    "build_menu",
    "run_tool",
    "scene_evidence",
    "THRESHOLDS",
]

_LOG = get_logger(__name__)

#: Every cut a deterministic tool applies, in one place, because each one is a
#: number an answer will quote and a reviewer will challenge. They are rules of
#: thumb from the literature, not learned parameters -- which is exactly why they
#: are named, exported and echoed into the evidence dict rather than inlined.
THRESHOLDS: dict[str, float] = {
    #: NDVI above this is vegetation. 0.3 is the conventional "green" cut; 0.2
    #: catches sparse/stressed cover and also catches bright soil, so the
    #: stricter value is used and the mean is reported next to it.
    "ndvi_vegetation": 0.3,
    #: McFeeters NDWI: positive is water. The threshold is 0 by construction.
    "ndwi_water": 0.0,
    #: NDBI above zero is built-up *or* bare soil -- it does not separate them,
    #: so the label this feeds is deliberately "built-up or bare".
    "ndbi_built_up": 0.0,
    #: VARI is an RGB proxy with no physical zero; 0.05 is an empirical
    #: greenness cut and is reported as a proxy wherever it is used.
    "vari_vegetation": 0.05,
    #: Sentinel-1 VV open-water rule of thumb, shared with preprocess.sar.
    "sar_water_db": WATER_DB_THRESHOLD,
    #: Above this, the radar return is bright and rough: the double-bounce
    #: signature of walls and vertical structure.
    "sar_rough_db": -8.0,
}

#: Smallest region, in pixels, that any deterministic tool will report as an
#: object or a change. Below this it is speckle at benchmark tile resolution.
MIN_REGION_PX = 24


# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool: its contract, its requirements and what serves it.

    ``requires`` is a set of tokens the menu builder checks against the session's
    input configuration:

    ``image``               at least one validated image
    ``pair:bitemporal``     a valid pair of two same-modality images
    ``pair:crossmodal``     a valid optical + SAR pair
    ``index``               an image whose bands support at least one index

    ``parameters`` is the JSON-schema property block the router fills in. Note
    what is absent from every one of them: ``image_id`` and ``pair_id``. Those
    are injected by the executor.
    """

    name: str
    endpoint: str
    summary: str
    description: str
    requires: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    required_params: tuple[str, ...] = ()
    task: Task | None = None
    slot: str = "deterministic"

    def schema(self) -> dict[str, Any]:
        """OpenAI/Mistral function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": dict(self.parameters),
                    "required": list(self.required_params),
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "summary": self.summary,
            "requires": list(self.requires),
            "parameters": dict(self.parameters),
            "required": list(self.required_params),
            "task": self.task,
            "slot": self.slot,
        }


TOOLS: dict[str, ToolSpec] = {
    "vqa": ToolSpec(
        name="vqa",
        endpoint="/tools/v1/vqa",
        summary="Answer a question about one image",
        description=(
            "Answer a natural-language question about a single satellite image. Use for "
            "presence, counting, land-cover and comparison questions about one scene. Returns a "
            "short answer with a confidence."
        ),
        requires=("image",),
        parameters={
            "question": {
                "type": "string",
                "description": "The question to answer, in plain English.",
            }
        },
        required_params=("question",),
        task="vqa",
        slot="text",
    ),
    "caption": ToolSpec(
        name="caption",
        endpoint="/tools/v1/caption",
        summary="Describe one image",
        description=(
            "Describe a single satellite image: dominant land cover, man-made structures and "
            "their arrangement. Use when the user asks what is in the image rather than a "
            "specific question about it."
        ),
        requires=("image",),
        parameters={
            "detail": {
                "type": "string",
                "enum": ["brief", "detailed"],
                "description": "brief is one sentence, detailed is two to three.",
            }
        },
        task="caption",
        slot="text",
    ),
    "ground": ToolSpec(
        name="ground",
        endpoint="/tools/v1/ground",
        summary="Locate objects matching a phrase",
        description=(
            "Locate every object in one image matching a referring expression (\"the large "
            "building near the river\") and return bounding boxes with scores plus an overlay "
            "image. Use when the user asks where something is."
        ),
        requires=("image",),
        parameters={
            "phrase": {
                "type": "string",
                "description": "What to find, e.g. 'the storage tanks' or 'water'.",
            }
        },
        required_params=("phrase",),
        task="ground",
        slot="text",
    ),
    "indices": ToolSpec(
        name="indices",
        endpoint="/tools/v1/indices",
        summary="Compute a spectral index or backscatter statistics",
        description=(
            "Compute a deterministic spectral index (NDVI vegetation, NDWI water, NDBI built-up, "
            "VARI for RGB-only imagery) or SAR backscatter statistics over one image, and return "
            "its mean, the percentage of the scene above the threshold, and a map. Call this "
            "BEFORE answering any question about vegetation, water or area: it produces the "
            "numbers the answer must cite."
        ),
        requires=("image", "index"),
        parameters={
            "index": {
                "type": "string",
                "enum": list(INDEX_REQUIREMENTS),
                "description": "Which index to compute.",
            }
        },
        required_params=("index",),
        slot="deterministic",
    ),
    "change_map": ToolSpec(
        name="change_map",
        endpoint="/tools/v1/change/map",
        summary="Detect what changed between two dates",
        description=(
            "Compare two co-registered images of the same place taken at different dates and "
            "return a change mask, the changed area as a percentage, the number of changed "
            "regions and where they are. Produces geometry only, no description. Call this "
            "before change_describe."
        ),
        requires=("pair:bitemporal",),
        parameters={},
        task="change_map",
        slot="change",
    ),
    "change_describe": ToolSpec(
        name="change_describe",
        endpoint="/tools/v1/change/describe",
        summary="Describe or answer a question about the change between two dates",
        description=(
            "Describe in words what changed between two dates, or answer a specific question "
            "about the change. Omit the question for a description; supply it for change-VQA. "
            "Uses the change_map statistics as evidence, so call change_map first."
        ),
        requires=("pair:bitemporal",),
        parameters={
            "question": {
                "type": "string",
                "description": "Optional question about the change; omit for a description.",
            }
        },
        task="change_caption",
        slot="text",
    ),
    "fusion": ToolSpec(
        name="fusion",
        endpoint="/tools/v1/fusion",
        summary="Combine an optical and a SAR image into land-cover labels",
        description=(
            "Fuse a co-registered optical + SAR pair into multi-label land cover (water, urban "
            "fabric, vegetation, agriculture, bare ground) with per-class area percentages. Use "
            "when the two images are from different sensors rather than different dates -- SAR "
            "sees water and built-up structure that optical alone confuses with shadow."
        ),
        requires=("pair:crossmodal",),
        parameters={
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of classes to report.",
            }
        },
        task="fusion_multilabel",
        slot="fusion",
    ),
}


# --------------------------------------------------------------------------- #
# Menu
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolMenu:
    """What the router is allowed to see, and what it is not -- with reasons."""

    offered: tuple[ToolSpec, ...]
    withheld: tuple[tuple[str, str], ...]
    context_summary: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.offered)

    def has(self, name: str) -> bool:
        return name in self.names

    def spec(self, name: str) -> ToolSpec:
        for spec in self.offered:
            if spec.name == name:
                return spec
        reason = dict(self.withheld).get(name)
        if reason:
            raise SatQueryError(f"tool {name!r} was withheld for this input: {reason}")
        raise SatQueryError(f"unknown tool {name!r}")

    def schemas(self) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self.offered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "offered": [spec.name for spec in self.offered],
            "withheld": [{"tool": name, "reason": reason} for name, reason in self.withheld],
            "context": self.context_summary,
        }


def build_menu(
    images: Sequence[ImageRecord],
    pair: PairRecord | None = None,
) -> ToolMenu:
    """Prune the tool set to what this input configuration can actually support.

    The reasons are written for a human reading the trace, not for a log grep:
    "no SAR image in the pair" is the sentence an assessor should see next to a
    withheld fusion tool.
    """
    usable = [r for r in images if r.ok]
    offered: list[ToolSpec] = []
    withheld: list[tuple[str, str]] = []

    index_capable = any(_index_options(r) for r in usable)
    pair_type = pair.pair_type if (pair and pair.ok) else "none"

    for spec in TOOLS.values():
        missing = _unmet(spec.requires, usable, pair, pair_type, index_capable)
        if missing:
            withheld.append((spec.name, missing))
        else:
            offered.append(spec)

    return ToolMenu(
        offered=tuple(offered),
        withheld=tuple(withheld),
        context_summary=_context_summary(usable, pair),
    )


def _unmet(
    requires: Iterable[str],
    images: Sequence[ImageRecord],
    pair: PairRecord | None,
    pair_type: str,
    index_capable: bool,
) -> str:
    for token in requires:
        if token == "image" and not images:
            return "no image passed validation"
        if token == "index" and not index_capable:
            return (
                "no spectral index is computable from these bands (an index needs at least "
                "two named bands; a single-band scene supports none)"
            )
        if token.startswith("pair:"):
            wanted = token.split(":", 1)[1]
            if pair is None:
                return "no image pair was registered for this query"
            if not pair.ok:
                reasons = "; ".join(pair.report.check.reasons) or "the pair check failed"
                return f"the pair is not usable: {reasons}"
            if pair_type != wanted:
                if wanted == "crossmodal":
                    return (
                        f"fusion needs one optical and one SAR image; this pair is "
                        f"{pair.report.check.modality_a} + {pair.report.check.modality_b}"
                    )
                return (
                    f"change tools need two images of the same place at different dates; this "
                    f"pair is {pair_type} ({pair.report.check.modality_a} + "
                    f"{pair.report.check.modality_b})"
                )
    return ""


def _context_summary(images: Sequence[ImageRecord], pair: PairRecord | None) -> str:
    """One paragraph of text -- the *only* thing about the pixels the router sees."""
    lines = []
    for record in images:
        summary = record.report.summary
        lines.append(
            f"{record.image_id}: {summary.get('width')}x{summary.get('height')} px, "
            f"{summary.get('bands')} bands, {record.modality} "
            f"({record.report.modality_source}, confidence "
            f"{record.report.modality_confidence:.2f})"
            + (f", CRS {summary.get('crs')}" if summary.get("crs") else ", no CRS")
        )
    if pair is not None:
        check = pair.report.check
        lines.append(
            f"pair {pair.pair_id}: {check.pair_type}, "
            f"{'usable' if check.ok else 'NOT usable'}, co-registered={check.co_registered}"
            + (
                f", footprint overlap {check.footprint_overlap:.0%}"
                if check.footprint_overlap is not None
                else ""
            )
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One tool call's outcome, in the shape the composer and the report expect."""

    name: str
    ok: bool
    output: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    backend: str = ""
    backend_kind: str = ""
    duration_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "ok": self.ok,
            "output": dict(self.output),
            "evidence": {k: v for k, v in self.evidence.items() if not k.startswith("_")},
            "artifacts": dict(self.artifacts),
            "backend": self.backend,
            "backend_kind": self.backend_kind,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


class ToolContext:
    """Everything a tool call needs, and the per-query caches that keep it quick.

    One query typically runs two or three tools over the same one or two images.
    Recomputing the display rendering and the index statistics for each of them
    would triple the latency of the demo for no gain, so both are memoised here
    -- per query, never across queries, because a re-uploaded image must be
    re-measured.
    """

    def __init__(
        self,
        session: Session,
        *,
        query_id: str,
        images: Sequence[ImageRecord],
        pair: PairRecord | None = None,
        config: SatQueryConfig | None = None,
    ) -> None:
        self.session = session
        self.query_id = query_id
        self.images = list(images)
        self.pair = pair
        self.config = config or get_config()
        self._display: dict[str, np.ndarray] = {}
        self._evidence: dict[str, dict[str, Any]] = {}
        #: Numeric evidence produced by tools already run in this plan. A text
        #: tool sees it on top of its own scene measurements, which is how the
        #: fusion labels and the change statistics reach the sentence that cites
        #: them -- the specialists emit numbers, one component speaks.
        self.accumulated: dict[str, Any] = {}
        self._change: ChangeResult | None = None
        self._vlm: Any | None = None
        self._vlm_reason: str = ""

    # -- lazily-built collaborators ------------------------------------------ #

    @property
    def vlm(self) -> Any:
        if self._vlm is None:
            self._vlm, self._vlm_reason = select_vlm(self.config)
        return self._vlm

    @property
    def vlm_reason(self) -> str:
        _ = self.vlm
        return self._vlm_reason

    def image(self, image_id: str | None = None) -> ImageRecord:
        if image_id is None:
            if not self.images:
                raise SatQueryError("this query has no image registered")
            return self.images[0]
        for record in self.images:
            if record.image_id == image_id:
                return record
        return self.session.image(image_id)

    def display(self, record: ImageRecord) -> np.ndarray:
        """``(3, h, w)`` uint8 of one image, rendered exactly as the VLM sees it."""
        if record.image_id not in self._display:
            self._display[record.image_id] = display_array(
                load_image(record.path),
                modality=record.modality,
                max_side=self.config.max_side,
            )
        return self._display[record.image_id]

    def evidence(self, record: ImageRecord) -> dict[str, Any]:
        if record.image_id not in self._evidence:
            self._evidence[record.image_id] = scene_evidence(record, max_side=self.config.max_side)
        return self._evidence[record.image_id]

    def accumulate(self, result: "ToolResult") -> None:
        """Fold a finished tool's measurements into the evidence later tools see."""
        for key, value in dict(result.evidence).items():
            if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                self.accumulated[key] = value

    def evidence_for_text(self, record: ImageRecord | None = None) -> dict[str, Any]:
        """Scene measurements plus whatever earlier tools in this plan produced."""
        scene = self.evidence(record) if record is not None else {}
        return {**self.accumulated, **_promptable(scene)}

    def artifact(self, name: str) -> Path:
        return self.session.artifact_path(self.query_id, name)

    def artifact_url(self, path: str | Path) -> str:
        return f"/files/{self.session.session_id}/{self.query_id}/{Path(path).name}"


def run_tool(name: str, params: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Execute one tool. Never raises: a failure is a result with ``ok=False``.

    The controller keeps going after a failed tool because a partial answer with
    an honest gap ("the change mask could not be computed: the pair is not on one
    grid") beats a 500. The trace carries the exception text.
    """
    started = perf_counter()
    handler = _HANDLERS.get(name)
    if handler is None:
        return ToolResult(name=name, ok=False, error=f"unknown tool {name!r}")
    try:
        result = handler(dict(params), ctx)
    except Exception as exc:  # deliberately broad: one tool must not kill a query
        _LOG.warning("tool %s failed: %s", name, exc, exc_info=True)
        return ToolResult(
            name=name,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=(perf_counter() - started) * 1000.0,
        )
    if result.ok:
        ctx.accumulate(result)
    return _with_duration(result, (perf_counter() - started) * 1000.0)


def _with_duration(result: ToolResult, ms: float) -> ToolResult:
    return ToolResult(
        name=result.name,
        ok=result.ok,
        output=result.output,
        evidence=result.evidence,
        artifacts=result.artifacts,
        backend=result.backend,
        backend_kind=result.backend_kind,
        duration_ms=ms,
        error=result.error,
    )


# --------------------------------------------------------------------------- #
# Deterministic evidence
# --------------------------------------------------------------------------- #


def scene_evidence(record: ImageRecord, *, max_side: int = 768) -> dict[str, Any]:
    """Measure one image. This is the dict that goes into the VLM's prompt.

    Flat, numeric and citable: every key is a quantity the composer can match a
    clause against. Keys prefixed with ``_`` are working data (masks, boxes) that
    the tools use and the trace drops.

    Which quantities are available depends entirely on the bands present, and the
    dict says so -- ``index_source`` names the band preset that was assumed and
    ``indices_available`` lists what could be computed. An RGB tile gets VARI and
    a note that it is a proxy; a Sentinel-2 stack gets NDVI, NDWI and NDBI; a SAR
    image gets backscatter statistics and nothing optical at all.
    """
    image = load_image(record.path)
    roles = resolve_band_roles(
        image.bands,
        modality=record.modality,
        override=record.report.summary.get("band_roles") or None,
    )
    from ..preprocess.tiling import resize_max_side

    array = resize_max_side(np.asarray(image.array, dtype=np.float32), max_side=max_side)
    height, width = int(array.shape[1]), int(array.shape[2])

    ev: dict[str, Any] = {
        "image_id": record.image_id,
        "modality": record.modality,
        "width": width,
        "height": height,
        "bands": image.bands,
        "index_source": roles.preset,
        "indices_available": list(available_indices(roles)),
    }
    if roles.note:
        ev["band_assumption"] = roles.note
    masks: dict[str, np.ndarray] = {}

    def plane(role: str) -> np.ndarray | None:
        idx = roles.get(role)
        return None if idx is None else array[idx]

    if record.modality == "sar":
        db = linear_to_db(array) if _looks_linear(array) else array
        stats = backscatter_stats(db)
        ev.update(
            {
                "backscatter_mean": round(stats["mean"], 2),
                "backscatter_p5": round(stats["p5"], 2),
                "backscatter_p50": round(stats["p50"], 2),
                "backscatter_p95": round(stats["p95"], 2),
                "water_pct": round(stats["water_pct"], 2),
                "water_threshold_db": stats["water_threshold"],
            }
        )
        water = sar_water_mask(db, threshold=THRESHOLDS["sar_water_db"])
        rough = _reduce_db(db) > THRESHOLDS["sar_rough_db"]
        masks["water"] = water
        masks["rough"] = rough
        ev["rough_surface_pct"] = round(100.0 * float(np.count_nonzero(rough)) / rough.size, 2)
    else:
        nir, red, green, blue, swir = (
            plane("nir"), plane("red"), plane("green"), plane("blue"), plane("swir1")
        )
        if nir is not None and red is not None:
            index = ndvi(nir, red)
            stats = index_stats(index, threshold=THRESHOLDS["ndvi_vegetation"], name="ndvi")
            ev["ndvi_mean"] = round(stats["mean"], 3)
            ev["vegetation_pct"] = round(stats["positive_pct"], 2)
            ev["vegetation_threshold"] = THRESHOLDS["ndvi_vegetation"]
            ev["vegetation_index"] = "NDVI"
            masks["vegetation"] = np.nan_to_num(index) > THRESHOLDS["ndvi_vegetation"]
        elif green is not None and red is not None and blue is not None:
            index = vari(green, red, blue)
            stats = index_stats(index, threshold=THRESHOLDS["vari_vegetation"], name="vari")
            ev["vari_mean"] = round(stats["mean"], 3)
            ev["vegetation_pct"] = round(stats["positive_pct"], 2)
            ev["vegetation_threshold"] = THRESHOLDS["vari_vegetation"]
            ev["vegetation_index"] = "VARI (RGB proxy -- no NIR band, so not NDVI)"
            masks["vegetation"] = np.nan_to_num(index) > THRESHOLDS["vari_vegetation"]

        if green is not None and nir is not None:
            index = ndwi(green, nir)
            stats = index_stats(index, threshold=THRESHOLDS["ndwi_water"], name="ndwi")
            ev["ndwi_mean"] = round(stats["mean"], 3)
            ev["water_pct"] = round(stats["positive_pct"], 2)
            ev["water_threshold"] = THRESHOLDS["ndwi_water"]
            masks["water"] = np.nan_to_num(index) > THRESHOLDS["ndwi_water"]

        if swir is not None and nir is not None:
            index = ndbi(swir, nir)
            stats = index_stats(index, threshold=THRESHOLDS["ndbi_built_up"], name="ndbi")
            ev["ndbi_mean"] = round(stats["mean"], 3)
            ev["built_up_pct"] = round(stats["positive_pct"], 2)
            ev["built_up_threshold"] = THRESHOLDS["ndbi_built_up"]
            masks["built_up"] = np.nan_to_num(index) > THRESHOLDS["ndbi_built_up"]

        if "vegetation" in masks:
            bare = ~masks["vegetation"]
            if "water" in masks:
                bare = bare & ~masks["water"]
            masks["bare"] = bare
            ev["bare_pct"] = round(100.0 * float(np.count_nonzero(bare)) / bare.size, 2)

    ev["_masks"] = masks
    ev["concept_boxes"] = {
        concept: _boxes_from_mask(mask, label=concept)
        for concept, mask in masks.items()
        if concept in ("water", "vegetation", "built_up", "bare")
    }
    return ev


def _looks_linear(array: np.ndarray) -> bool:
    """Linear power or already dB? Same test the SAR module warns on."""
    finite = array[np.isfinite(array)]
    return bool(finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 10.0)


def _reduce_db(db: np.ndarray) -> np.ndarray:
    return db if db.ndim == 2 else np.nanmean(db, axis=0)


def _boxes_from_mask(mask: np.ndarray, *, label: str, limit: int = 8) -> list[Box]:
    """Bounding boxes of the largest connected components of a boolean mask.

    The score is the *fill ratio* -- how much of the box the region actually
    occupies. It is not a model confidence and is not presented as one: a compact
    lake scores near 1.0, a sprawling diagonal river scores 0.3, and that is
    information the reader should have.
    """
    labels, count = label_components(mask)
    if count <= 0:
        return []
    boxes: list[tuple[int, Box]] = []
    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == label_id)
        area = int(ys.size)
        if area < MIN_REGION_PX:
            continue
        x0, x1 = float(xs.min()), float(xs.max() + 1)
        y0, y1 = float(ys.min()), float(ys.max() + 1)
        fill = area / max(1.0, (x1 - x0) * (y1 - y0))
        boxes.append((area, Box(x0, y0, x1, y1, label, round(float(fill), 3))))
    boxes.sort(key=lambda pair: pair[0], reverse=True)
    return [box for _area, box in boxes[:limit]]


def _index_options(record: ImageRecord) -> tuple[str, ...]:
    """Which indices this image's bands allow, without reading its pixels."""
    summary = record.report.summary
    roles = resolve_band_roles(
        int(summary.get("bands") or 0),
        modality=record.modality,
        override=summary.get("band_roles") or None,
    )
    return available_indices(roles)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def _tool_indices(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    record = ctx.image(params.get("image_id"))
    ev = ctx.evidence(record)
    wanted = str(params.get("index") or "").upper() or _default_index(ev)
    if wanted not in INDEX_REQUIREMENTS:
        return ToolResult(
            name="indices",
            ok=False,
            error=f"unknown index {wanted!r}; known: {', '.join(INDEX_REQUIREMENTS)}",
        )

    key_map = {
        "NDVI": ("ndvi_mean", "vegetation_pct", "vegetation"),
        "VARI": ("vari_mean", "vegetation_pct", "vegetation"),
        "NDWI": ("ndwi_mean", "water_pct", "water"),
        "NDBI": ("ndbi_mean", "built_up_pct", "built_up"),
        "SAR_BACKSCATTER": ("backscatter_mean", "water_pct", "water"),
    }
    mean_key, pct_key, mask_key = key_map[wanted]
    if mean_key not in ev:
        return ToolResult(
            name="indices",
            ok=False,
            error=(
                f"{wanted} needs bands {', '.join(INDEX_REQUIREMENTS[wanted])}, which this "
                f"{ev['bands']}-band {ev['modality']} image does not provide "
                f"(available: {', '.join(ev['indices_available']) or 'none'})"
            ),
        )

    artifacts: dict[str, str] = {}
    mask = ev["_masks"].get(mask_key)
    if mask is not None:
        overlay = overlay_mask(ctx.display(record), mask, alpha=0.4)
        path = save_png(overlay, ctx.artifact(f"{record.image_id}_{wanted.lower()}.png"))
        artifacts["map_url"] = ctx.artifact_url(path)

    output = {
        "index": wanted,
        "mean": ev[mean_key],
        "area_pct": ev.get(pct_key),
        "threshold": ev.get(f"{mask_key}_threshold", ev.get("water_threshold_db")),
        "band_source": ev["index_source"],
        **({"note": ev["vegetation_index"]} if wanted == "VARI" else {}),
    }
    return ToolResult(
        name="indices",
        ok=True,
        output=output,
        evidence={k: v for k, v in ev.items() if not k.startswith("_") and k != "concept_boxes"},
        artifacts=artifacts,
        backend="numpy",
        backend_kind="deterministic",
    )


def _default_index(ev: Mapping[str, Any]) -> str:
    options = list(ev.get("indices_available") or ())
    return options[0] if options else "NDVI"


def _tool_vqa(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    record = ctx.image(params.get("image_id"))
    question = str(params.get("question") or "").strip()
    if not question:
        return ToolResult(name="vqa", ok=False, error="vqa needs a question")
    ev = ctx.evidence_for_text(record)
    answer = ctx.vlm.answer(
        "vqa", [record.as_ref()], question=question, evidence=ev, sample_id=ctx.query_id
    )
    return _text_result("vqa", answer, ev, ctx)


def _tool_caption(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    record = ctx.image(params.get("image_id"))
    ev = ctx.evidence_for_text(record)
    answer = ctx.vlm.answer(
        "caption", [record.as_ref()], evidence=ev, sample_id=ctx.query_id
    )
    return _text_result("caption", answer, ev, ctx, extra={"detail": params.get("detail", "brief")})


def _tool_ground(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    record = ctx.image(params.get("image_id"))
    phrase = str(params.get("phrase") or "").strip()
    if not phrase:
        return ToolResult(name="ground", ok=False, error="ground needs a phrase to locate")
    scene = ctx.evidence(record)
    ev = ctx.evidence_for_text(record)
    answer = ctx.vlm.answer(
        "ground",
        [record.as_ref()],
        question=phrase,
        evidence={**ev, "concept_boxes": scene.get("concept_boxes", {})},
        sample_id=ctx.query_id,
    )
    artifacts: dict[str, str] = {}
    if answer.boxes:
        overlay = draw_boxes(ctx.display(record), answer.boxes)
        path = save_png(overlay, ctx.artifact(f"{record.image_id}_ground.png"))
        artifacts["overlay_url"] = ctx.artifact_url(path)
    output = {
        "phrase": phrase,
        "boxes": [
            {"bbox": [round(v, 1) for v in b.as_xyxy()], "label": b.label, "score": b.score}
            for b in answer.boxes
        ],
        "n_boxes": len(answer.boxes),
        "answer": answer.text,
    }
    result = _text_result("ground", answer, ev, ctx, extra=output)
    return ToolResult(
        name="ground",
        ok=result.ok,
        output=result.output,
        evidence=result.evidence,
        artifacts={**result.artifacts, **artifacts},
        backend=result.backend,
        backend_kind=result.backend_kind,
        error=result.error,
    )


def _tool_change_map(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pair = _require_pair(ctx)
    before, after = _pair_displays(ctx, pair)
    backend, reason = select_change_backend()
    result: ChangeResult = backend.detect(before, after)
    ctx._change = result

    overlay = overlay_mask(after, result.mask)
    overlay_path = save_png(overlay, ctx.artifact(f"{pair.pair_id}_change_overlay.png"))
    mask_path = save_png(
        (result.mask.astype(np.uint8) * 255), ctx.artifact(f"{pair.pair_id}_change_mask.png")
    )
    stats = dict(result.stats)
    stats.setdefault("pair_id", pair.pair_id)
    return ToolResult(
        name="change_map",
        ok=True,
        output={**stats, "backend_reason": reason},
        evidence=stats,
        artifacts={
            "mask_url": ctx.artifact_url(mask_path),
            "overlay_url": ctx.artifact_url(overlay_path),
        },
        backend=result.backend,
        backend_kind=result.kind,
    )


def _tool_change_describe(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pair = _require_pair(ctx)
    if ctx._change is None:
        # The router is supposed to call change_map first; if it did not, run it
        # rather than describing a change nobody measured. Invariant 03 is not
        # optional because a plan was sloppy.
        _tool_change_map({}, ctx)
    change_stats = dict(ctx._change.stats) if ctx._change else {}

    a, b = ctx.session.image(pair.image_a), ctx.session.image(pair.image_b)
    ev: dict[str, Any] = {**ctx.accumulated, **change_stats, "_pair": pair.pair_id}
    for label, record in (("before", a), ("after", b)):
        scene = ctx.evidence(record)
        for key in ("vegetation_pct", "water_pct", "built_up_pct"):
            if key in scene:
                ev[f"{label}_{key}"] = scene[key]
    if "before_vegetation_pct" in ev and "after_vegetation_pct" in ev:
        ev["vegetation_pct_delta"] = round(
            float(ev["after_vegetation_pct"]) - float(ev["before_vegetation_pct"]), 2
        )

    question = str(params.get("question") or "").strip()
    # An *open* question about the change wants the description prompt, not the
    # short-answer one. "What changed between these dates?" is a request for a
    # paragraph; sending it through the change-VQA template -- which instructs
    # "the shortest possible answer: a number, a single word, or yes/no" -- gets
    # back "Vegetation loss", which is accurate, thin, and cites nothing.
    # Closed questions ("is there", "how much", "how many") still get the terse
    # template, because a short answer is what those actually want.
    task: Task = "change_vqa" if question and not _is_open_question(question) else "change_caption"
    answer = ctx.vlm.answer(
        task,
        [a.as_ref(), b.as_ref()],
        question=question or None,
        evidence=_promptable(ev),
        sample_id=ctx.query_id,
    )
    return _text_result("change_describe", answer, ev, ctx, extra={"task": task})


def _tool_fusion(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pair = _require_pair(ctx)
    a, b = ctx.session.image(pair.image_a), ctx.session.image(pair.image_b)
    optical, sar = (a, b) if a.modality == "optical" else (b, a)
    if optical.modality != "optical" or sar.modality != "sar":
        return ToolResult(
            name="fusion",
            ok=False,
            error=f"fusion needs one optical and one SAR image, got {a.modality} + {b.modality}",
        )

    optical_ev, sar_ev = ctx.evidence(optical), ctx.evidence(sar)
    optical_masks, sar_masks = optical_ev["_masks"], sar_ev["_masks"]
    grid = ctx.display(optical).shape[1:]
    masks = {
        "vegetation": _to_grid(optical_masks.get("vegetation"), grid),
        "water_optical": _to_grid(optical_masks.get("water"), grid),
        "water_sar": _to_grid(sar_masks.get("water"), grid),
        "rough": _to_grid(sar_masks.get("rough"), grid),
    }
    backend, reason = select_fusion_backend()
    result: FusionResult = backend.classify(
        optical_evidence=optical_ev, sar_evidence=sar_ev,
        masks={k: v for k, v in masks.items() if v is not None},
    )

    targets = [str(t) for t in (params.get("targets") or ())]
    labels = tuple(l for l in result.labels if not targets or l in targets)

    artifacts: dict[str, str] = {}
    for name, mask in result.masks.items():
        if name in labels:
            path = save_png(
                overlay_mask(ctx.display(optical), mask),
                ctx.artifact(f"{pair.pair_id}_fusion_{name}.png"),
            )
            artifacts[f"{name}_url"] = ctx.artifact_url(path)

    evidence = {
        **{f"{k}_pct": v for k, v in result.area_pct.items() if v > 0},
        "optical_image": optical.image_id,
        "sar_image": sar.image_id,
        "backscatter_mean": sar_ev.get("backscatter_mean"),
    }
    return ToolResult(
        name="fusion",
        ok=True,
        output={
            "labels": list(labels),
            "label_scores": {k: v for k, v in result.label_scores.items()},
            "area_pct": dict(result.area_pct),
            "rationale": {k: v for k, v in result.rationale.items() if k in labels},
            "backend_reason": reason,
        },
        evidence={k: v for k, v in evidence.items() if v is not None},
        artifacts=artifacts,
        backend=result.backend,
        backend_kind=result.kind,
    )


_HANDLERS = {
    "vqa": _tool_vqa,
    "caption": _tool_caption,
    "ground": _tool_ground,
    "indices": _tool_indices,
    "change_map": _tool_change_map,
    "change_describe": _tool_change_describe,
    "fusion": _tool_fusion,
}


# --------------------------------------------------------------------------- #
# Shared handler helpers
# --------------------------------------------------------------------------- #


def _text_result(
    name: str,
    answer: VlmAnswer,
    evidence: Mapping[str, Any],
    ctx: ToolContext,
    *,
    extra: Mapping[str, Any] | None = None,
) -> ToolResult:
    output: dict[str, Any] = {
        "answer": answer.text,
        "confidence": answer.confidence,
        **dict(extra or {}),
    }
    if answer.refused:
        output["refused"] = True
        output["reason"] = answer.meta.get("reason") or answer.meta.get("error") or "no answer"
    return ToolResult(
        name=name,
        ok=not answer.refused,
        output=output,
        evidence={k: v for k, v in evidence.items() if not k.startswith("_") and k != "concept_boxes"},
        backend=answer.backend,
        backend_kind=answer.kind,
        error=None if not answer.refused else str(output.get("reason")),
    )


#: Openers that ask for description rather than a value. Everything else --
#: "is", "are", "how many", "how much", "does" -- is a closed question.
_OPEN_OPENERS: tuple[str, ...] = (
    "what", "describe", "tell me", "explain", "summarise", "summarize", "give me",
    "which changes", "what kind", "what sort",
)


def _is_open_question(text: str) -> bool:
    """Does this question want a paragraph or a value?

    Deliberately crude and deliberately conservative: only a leading word decides,
    so "what is the water percentage" is read as open and gets a description that
    still contains the number, while "how much water" stays terse. Getting it
    wrong in this direction costs verbosity; getting it wrong the other way loses
    the evidence-citing paragraph that the whole pipeline exists to produce.
    """
    lowered = text.strip().lower()
    return any(lowered.startswith(opener) for opener in _OPEN_OPENERS)


def _promptable(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Strip working data before evidence reaches a prompt or a trace."""
    return {k: v for k, v in evidence.items() if not k.startswith("_")}


def _require_pair(ctx: ToolContext) -> PairRecord:
    if ctx.pair is None:
        raise SatQueryError("this tool needs an image pair, and none was registered")
    if not ctx.pair.ok:
        raise SatQueryError(
            "the registered pair failed validation: " + "; ".join(ctx.pair.report.check.reasons)
        )
    return ctx.pair


def _pair_displays(ctx: ToolContext, pair: PairRecord) -> tuple[np.ndarray, np.ndarray]:
    """Both images of a pair, rendered and forced onto one grid."""
    a = ctx.session.image(pair.image_a)
    b = ctx.session.image(pair.image_b)
    before, after = ctx.display(a), ctx.display(b)
    if before.shape != after.shape:
        from ..preprocess.tiling import resample_to

        after = resample_to(after, height=before.shape[1], width=before.shape[2], order=1)
    return before, np.asarray(after, dtype=np.uint8)


def _to_grid(mask: np.ndarray | None, grid: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.shape == tuple(grid):
        return mask
    from ..preprocess.tiling import resample_to

    return resample_to(mask.astype(np.uint8), height=grid[0], width=grid[1], order=0).astype(bool)
