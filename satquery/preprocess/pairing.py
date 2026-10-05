"""Can these two images be used as a pair -- and as which kind of pair?

This is the check that makes invariant 02 possible. The validator runs it before
the router is invoked, and its verdict decides which tools exist in the schema
the router receives: change tools need a *bi-temporal* pair, fusion needs a
*cross-modal* one, and a pair that fails outright gets neither. A refusal here
is a good outcome -- "footprints overlap 12%, so the change tools were withheld"
is a defensible answer; a confident change description of two different towns is
not.

The hard part is that half the inputs have no georeferencing at all. Benchmark
pairs (LEVIR-CD, CDVQA) are 8-bit PNG tiles already cut to the same grid, so
demanding a CRS would refuse the entire evaluation set. So there are two modes,
and the check says which one it used:

* **geo mode** -- both images carry a CRS and a transform. Footprint overlap,
  GSD ratio and co-registration are measured in ground units, which is the only
  way to catch two scenes of different places that happen to be the same size.
* **grid mode** -- no georeferencing. The pixel grid is all there is, so
  identical shape is taken as "co-registered" and the trace records that the
  claim rests on the dataset's own promise rather than on coordinates.

What this module does *not* do is reproject. Two rasters in different CRSs are
reported as needing reprojection and the pair is refused; doing it silently
would mean resampling a scene inside a validation step, which is both slow and
exactly the sort of invisible transform that makes a wrong answer untraceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from ..log import get_logger
from .tiling import resample_to

__all__ = [
    "PairCheck",
    "PairType",
    "check_pair",
    "resample_to_match",
    "MIN_FOOTPRINT_OVERLAP",
    "MAX_GSD_RATIO",
]

_LOG = get_logger(__name__)

PairType = Literal["bitemporal", "crossmodal", "none"]

#: Intersection-over-union of the two footprints, below which the pair is
#: refused. 0.5 is loose on purpose: bi-temporal acquisitions of the same AOI
#: routinely differ by a few percent at the edges, and the number is reported in
#: the trace either way, so a reviewer sees 0.98 versus 0.52 rather than a bare
#: pass.
MIN_FOOTPRINT_OVERLAP = 0.5

#: How far apart the two ground sample distances may be before resampling is
#: called for. 1.5x covers S1 (10 m) against S2 (10 m/20 m) without complaint;
#: Cartosat (~1 m) against RISAT (~3 m) trips it and gets resampled explicitly.
MAX_GSD_RATIO = 1.5


@dataclass(frozen=True, slots=True)
class PairCheck:
    """The verdict, with every number the trace has to show.

    ``ok`` means "these two images are the same place at a comparable scale", not
    "this pair supports the task the user asked for" -- that second question is
    :attr:`pair_type` plus the caller's own requirement. A bi-temporal pair is
    ``ok`` and still useless for fusion.
    """

    ok: bool
    pair_type: PairType
    mode: Literal["geo", "grid"]
    co_registered: bool
    resample_needed: bool
    reasons: tuple[str, ...] = ()
    same_crs: bool | None = None
    footprint_overlap: float | None = None
    overlap_of_smaller: float | None = None
    gsd_ratio: float | None = None
    shape_a: tuple[int, int] | None = None
    shape_b: tuple[int, int] | None = None
    modality_a: str = "unknown"
    modality_b: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def supports(self, task: str) -> bool:
        """Whether this pair can back a given task. The tool menu is built from this."""
        if not self.ok:
            return False
        if task in ("change_vqa", "change_caption", "change_map"):
            return self.pair_type == "bitemporal"
        if task == "fusion_multilabel":
            return self.pair_type == "crossmodal"
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "pair_type": self.pair_type,
            "mode": self.mode,
            "co_registered": self.co_registered,
            "resample_needed": self.resample_needed,
            "reasons": list(self.reasons),
            "same_crs": self.same_crs,
            "footprint_overlap": _round(self.footprint_overlap),
            "overlap_of_smaller": _round(self.overlap_of_smaller),
            "gsd_ratio": _round(self.gsd_ratio),
            "shape_a": list(self.shape_a) if self.shape_a else None,
            "shape_b": list(self.shape_b) if self.shape_b else None,
            "modality_a": self.modality_a,
            "modality_b": self.modality_b,
            **({"meta": dict(self.meta)} if self.meta else {}),
        }


def check_pair(
    a: Any,
    b: Any,
    *,
    modality_a: str = "unknown",
    modality_b: str = "unknown",
    min_overlap: float = MIN_FOOTPRINT_OVERLAP,
    max_gsd_ratio: float = MAX_GSD_RATIO,
) -> PairCheck:
    """Decide whether two :class:`~satquery.io.raster.RasterImage` form a usable pair.

    ``modality_a``/``modality_b`` come from the validator (declared or inferred);
    they are what separates a bi-temporal pair from a cross-modal one, and they
    are the caller's business rather than this function's, because a *declared*
    modality must outrank anything guessable from pixels here.
    """
    shape_a = (int(a.height), int(a.width))
    shape_b = (int(b.height), int(b.width))
    pair_type = _pair_type(modality_a, modality_b)
    reasons: list[str] = []
    resample_needed = shape_a != shape_b

    geo = bool(getattr(a, "georeferenced", False) and getattr(b, "georeferenced", False))
    if not geo:
        missing = [
            name
            for name, img in (("A", a), ("B", b))
            if not getattr(img, "georeferenced", False)
        ]
        subject = (
            f"image {missing[0]} carries" if len(missing) == 1 else f"images {' and '.join(missing)} carry"
        )
        reasons.append(
            f"{subject} no CRS/transform, so the pair was checked on the pixel grid alone -- "
            "co-registration is the dataset's claim, not a measurement"
        )
        co_registered = shape_a == shape_b
        if not co_registered:
            reasons.append(
                f"pixel grids differ ({shape_a[1]}x{shape_a[0]} against "
                f"{shape_b[1]}x{shape_b[0]}); resample before any per-pixel comparison"
            )
        return PairCheck(
            ok=co_registered,
            pair_type=pair_type if co_registered else "none",
            mode="grid",
            co_registered=co_registered,
            resample_needed=resample_needed,
            reasons=tuple(reasons),
            shape_a=shape_a,
            shape_b=shape_b,
            modality_a=modality_a,
            modality_b=modality_b,
        )

    same_crs = _same_crs(a.crs, b.crs)
    if not same_crs:
        reasons.append(
            f"different CRS ({a.crs} against {b.crs}); reproject to a common CRS first -- "
            "this check does not reproject silently"
        )
        return PairCheck(
            ok=False,
            pair_type="none",
            mode="geo",
            co_registered=False,
            resample_needed=True,
            reasons=tuple(reasons),
            same_crs=False,
            shape_a=shape_a,
            shape_b=shape_b,
            modality_a=modality_a,
            modality_b=modality_b,
        )

    iou, of_smaller = _overlap(a.bounds, b.bounds)
    gsd_ratio = _gsd_ratio(a.gsd, b.gsd)
    ok = iou >= float(min_overlap)
    if not ok:
        reasons.append(
            f"footprints overlap {iou:.0%} (threshold {min_overlap:.0%}) -- these are not the "
            "same area"
        )
    else:
        reasons.append(f"footprints overlap {iou:.0%} in {a.crs}")

    if gsd_ratio is not None and gsd_ratio > float(max_gsd_ratio):
        resample_needed = True
        reasons.append(
            f"ground sample distances differ by {gsd_ratio:.2f}x "
            f"({_fmt_gsd(a.gsd)} against {_fmt_gsd(b.gsd)}); resample to the finer grid"
        )
    if getattr(a, "rotated", False) or getattr(b, "rotated", False):
        reasons.append("a transform has rotation terms, so GSD and overlap are approximate")

    # Co-registration is stricter than overlap: the grids must line up well
    # enough that pixel (i, j) means the same ground in both, which is what a
    # change mask assumes.
    co_registered = bool(ok and iou >= 0.95 and not resample_needed)
    if ok and not co_registered:
        reasons.append(
            "pair is usable but not pixel-aligned; per-pixel products (change masks) need "
            "resampling onto a common grid first"
        )

    return PairCheck(
        ok=ok,
        pair_type=pair_type if ok else "none",
        mode="geo",
        co_registered=co_registered,
        resample_needed=resample_needed,
        reasons=tuple(reasons),
        same_crs=True,
        footprint_overlap=iou,
        overlap_of_smaller=of_smaller,
        gsd_ratio=gsd_ratio,
        shape_a=shape_a,
        shape_b=shape_b,
        modality_a=modality_a,
        modality_b=modality_b,
    )


def resample_to_match(array: np.ndarray, reference: Any, *, order: int = 1) -> np.ndarray:
    """Put ``array`` on ``reference``'s pixel grid.

    ``reference`` is a :class:`~satquery.io.raster.RasterImage` or any object
    with ``height``/``width`` (a ``(h, w)`` tuple works too). Use ``order=0`` for
    masks and class maps.

    This is a *grid* alignment, not a reprojection: it assumes the two rasters
    already cover the same ground, which :func:`check_pair` is what establishes.
    Calling it on a pair whose footprints do not overlap produces a perfectly
    shaped, entirely meaningless array -- so call it after the check, never
    instead of it.
    """
    if isinstance(reference, (tuple, list)) and len(reference) == 2:
        height, width = int(reference[0]), int(reference[1])
    else:
        height, width = int(reference.height), int(reference.width)
    return resample_to(np.asarray(array), height=height, width=width, order=order)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _pair_type(modality_a: str, modality_b: str) -> PairType:
    pair = {str(modality_a), str(modality_b)}
    if pair == {"optical", "sar"}:
        return "crossmodal"
    if pair <= {"optical"} or pair <= {"sar"}:
        return "bitemporal"
    return "none"


def _same_crs(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if str(a).strip().upper() == str(b).strip().upper():
        return True
    try:  # pyproj ships with the raster extra and knows EPSG:4326 == WGS 84
        from pyproj import CRS  # type: ignore

        return bool(CRS.from_user_input(a) == CRS.from_user_input(b))
    except Exception:
        return False


def _overlap(
    bounds_a: tuple[float, float, float, float] | None,
    bounds_b: tuple[float, float, float, float] | None,
) -> tuple[float, float]:
    """``(iou, intersection_over_smaller)`` of two envelopes.

    Both are reported because they answer different questions: IoU catches "one
    scene is a small crop of the other", intersection-over-smaller says how much
    of the smaller image is actually covered, which is what a user cropping a
    subscene cares about.
    """
    if bounds_a is None or bounds_b is None:
        return (0.0, 0.0)
    ax0, ay0, ax1, ay1 = bounds_a
    bx0, by0, bx1, by1 = bounds_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (
        0.0 if union <= 0 else float(inter / union),
        0.0 if smaller <= 0 else float(inter / smaller),
    )


def _gsd_ratio(gsd_a: tuple[float, float] | None, gsd_b: tuple[float, float] | None) -> float | None:
    if not gsd_a or not gsd_b:
        return None
    mean_a = (abs(gsd_a[0]) + abs(gsd_a[1])) / 2.0
    mean_b = (abs(gsd_b[0]) + abs(gsd_b[1])) / 2.0
    if mean_a <= 0 or mean_b <= 0:
        return None
    return float(max(mean_a, mean_b) / min(mean_a, mean_b))


def _fmt_gsd(gsd: tuple[float, float] | None) -> str:
    return "unknown" if not gsd else f"{gsd[0]:.3g}x{gsd[1]:.3g}"


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)
