"""Step 1 of the loop: what did the user actually upload?

Nothing downstream is allowed to guess. The validator opens each raster, records
what it is, decides the modality (or says it could not), and -- for two images --
runs the pair check. Its report is what prunes the tool menu, so a wrong or
missing verdict here does not produce a bad answer, it produces a *withheld
tool* and a reason in the trace. That asymmetry is the point.

The checks are the ones that have actually broken remote-sensing pipelines, not
a generic file-upload sanity list:

* **Unreadable or truncated file** -- refuse with the reader's own error rather
  than handing a half-decoded array to a model.
* **Modality** -- optical and SAR need different display transforms; sending SAR
  through a percentile stretch yields a picture whose brightness means nothing.
  Declared beats inferred, and the provenance is recorded either way.
* **Constant / empty scene** -- an all-nodata or all-zero tile answers every
  question with confident nonsense. Warn loudly; do not refuse, because a
  legitimately uniform ocean tile exists.
* **Nodata coverage** -- a scene that is 60% fill will produce index means and
  backscatter statistics dominated by the fill value. Reported as a percentage
  so the composer can qualify the number instead of quoting it flat.
* **Size** -- a thumbnail cannot support a grounding answer in pixel
  coordinates, and a 20,000 px scene has to be tiled rather than squashed to
  768 px. Both are warnings with the numbers attached.

Reads are cached by (path, mtime, size): one query touches the same raster from
the validator, two or three tools and the preview renderer, and re-decoding a
scene four times is the difference between a snappy demo and a slow one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contracts import ImageRef, Modality
from ..io.modality import infer_modality
from ..io.raster import RasterError, RasterImage, read_raster
from ..log import get_logger
from ..preprocess.pairing import PairCheck, check_pair

__all__ = [
    "ImageReport",
    "PairReport",
    "validate_image",
    "validate_pair",
    "load_image",
    "MIN_USABLE_SIDE",
    "LARGE_SCENE_SIDE",
    "MAX_NODATA_FRACTION",
]

_LOG = get_logger(__name__)

#: Below this, grounding boxes and change masks are not worth returning.
MIN_USABLE_SIDE = 32

#: Above this, the scene should be tiled at inference rather than resized to the
#: VLM's 768 px input -- squashing 10,000 px into 768 destroys every small object.
LARGE_SCENE_SIDE = 2048

#: Nodata coverage above which every statistic gets a health warning.
MAX_NODATA_FRACTION = 0.3


@lru_cache(maxsize=4)
def _load_cached(path: str, _mtime: float, _size: int) -> RasterImage:
    return read_raster(path)


def load_image(path: str | Path) -> RasterImage:
    """Read a raster, reusing the decode within a query.

    Keyed on modification time and size as well as the path, so editing a file
    in place between two queries is picked up rather than served stale. Only a
    handful of scenes are kept; this is a per-query working set, not a store.
    """
    p = Path(path)
    try:
        stat = p.stat()
    except OSError:
        # Missing file: let read_raster raise the error that names the problem
        # (including the .npy sidecar case, which stat() on the .tif would miss).
        return read_raster(p)
    return _load_cached(str(p), stat.st_mtime, stat.st_size)


@dataclass(frozen=True, slots=True)
class ImageReport:
    """Everything known about one uploaded image before any model sees it."""

    image_id: str
    path: str
    ok: bool
    modality: Modality = "unknown"
    modality_source: str = "inferred"
    modality_confidence: float = 0.0
    modality_reasons: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    stats: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return Path(self.path).name

    def as_ref(self) -> ImageRef:
        """The :class:`~satquery.contracts.ImageRef` the predictors consume."""
        return ImageRef(
            image_id=self.image_id,
            path=self.path,
            modality=self.modality,
            bands=self.summary.get("bands"),
            height=self.summary.get("height"),
            width=self.summary.get("width"),
            meta={
                "modality_source": self.modality_source,
                "modality_confidence": round(float(self.modality_confidence), 3),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path": self.path,
            "name": self.name,
            "ok": self.ok,
            "modality": self.modality,
            "modality_source": self.modality_source,
            "modality_confidence": round(float(self.modality_confidence), 3),
            "modality_reasons": list(self.modality_reasons),
            "summary": dict(self.summary),
            "stats": dict(self.stats),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PairReport:
    """Two images plus the verdict on using them together."""

    pair_id: str
    image_a: str
    image_b: str
    check: PairCheck

    @property
    def ok(self) -> bool:
        return self.check.ok

    @property
    def pair_type(self) -> str:
        return self.check.pair_type

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "image_a": self.image_a,
            "image_b": self.image_b,
            **self.check.to_dict(),
        }


def validate_image(
    path: str | Path,
    *,
    image_id: str,
    declared_modality: str | None = None,
) -> ImageReport:
    """Open one raster and report what it is. Never raises for a bad input.

    A file that cannot be read comes back as ``ok=False`` with the reader's
    message in ``errors`` -- the caller turns that into a 400 and a trace entry,
    which is a better answer than a stack trace.
    """
    p = Path(path)
    try:
        image = load_image(p)
    except (RasterError, OSError, ValueError) as exc:
        return ImageReport(
            image_id=image_id,
            path=str(p),
            ok=False,
            errors=(f"{type(exc).__name__}: {exc}",),
        )

    guess = infer_modality(image, path=p, declared=declared_modality)
    warnings: list[str] = []
    errors: list[str] = []

    if not guess.certain:
        warnings.append(
            f"modality is uncertain ({guess.modality}, confidence {guess.confidence:.2f}); "
            "declare it explicitly if the display transform looks wrong"
        )

    stats = _scene_stats(image)
    if image.height < MIN_USABLE_SIDE or image.width < MIN_USABLE_SIDE:
        errors.append(
            f"image is {image.width}x{image.height} px; below {MIN_USABLE_SIDE} px there is "
            "nothing to ground a box or a mask in"
        )
    if max(image.height, image.width) > LARGE_SCENE_SIDE:
        warnings.append(
            f"scene is {image.width}x{image.height} px; it will be tiled at 512 px rather than "
            "resized whole, so expect per-tile evidence"
        )
    if stats["nodata_fraction"] > MAX_NODATA_FRACTION:
        warnings.append(
            f"{stats['nodata_fraction']:.0%} of pixels are nodata; every index mean and "
            "backscatter statistic is computed over the remainder only"
        )
    if stats["constant"]:
        warnings.append(
            "every valid pixel has the same value -- this scene carries no information, and "
            "any confident answer about it would be fabricated"
        )

    return ImageReport(
        image_id=image_id,
        path=str(p),
        ok=not errors,
        modality=guess.modality,
        modality_source="declared" if declared_modality in ("optical", "sar") else "inferred",
        modality_confidence=guess.confidence,
        modality_reasons=guess.reasons,
        summary=image.describe(),
        stats=stats,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_pair(a: ImageReport, b: ImageReport, *, pair_id: str) -> PairReport:
    """Run the geometric pair check on two already-validated images.

    Ordering matters for bi-temporal work: ``a`` is treated as the earlier date
    throughout the stack (the change prompts say "the FIRST image is the earlier
    date"). The caller is responsible for that ordering -- there is no reliable
    acquisition timestamp in a PNG.
    """
    if not (a.ok and b.ok):
        broken = ", ".join(r.image_id for r in (a, b) if not r.ok)
        return PairReport(
            pair_id=pair_id,
            image_a=a.image_id,
            image_b=b.image_id,
            check=PairCheck(
                ok=False,
                pair_type="none",
                mode="grid",
                co_registered=False,
                resample_needed=False,
                reasons=(f"image {broken} failed validation, so no pair was formed",),
                modality_a=a.modality,
                modality_b=b.modality,
            ),
        )
    image_a, image_b = load_image(a.path), load_image(b.path)
    check = check_pair(
        image_a,
        image_b,
        modality_a=a.modality,
        modality_b=b.modality,
    )
    return PairReport(pair_id=pair_id, image_a=a.image_id, image_b=b.image_id, check=check)


# --------------------------------------------------------------------------- #
# Scene statistics
# --------------------------------------------------------------------------- #


def _scene_stats(image: RasterImage) -> dict[str, Any]:
    """Cheap health numbers over a strided subsample.

    ``nodata_fraction`` counts NaN plus, when the file declares one, the nodata
    value. It deliberately does not treat 0 as nodata: zero is a legitimate
    reflectance and a legitimate SAR power, and guessing otherwise would silently
    delete a third of a coastal scene.
    """
    array = image.array
    flat = array.reshape(-1)
    step = max(1, int(flat.size // 200_000))
    sample = flat[::step].astype(np.float64, copy=False)
    n = int(sample.size)
    if n == 0:
        return {"n_sampled": 0, "nodata_fraction": 1.0, "constant": True}

    invalid = ~np.isfinite(sample)
    if image.nodata is not None:
        invalid |= np.isclose(sample, float(image.nodata), equal_nan=True)
    valid = sample[~invalid]
    constant = bool(valid.size == 0 or float(valid.max() - valid.min()) == 0.0)
    return {
        "n_sampled": n,
        "nodata_fraction": float(np.count_nonzero(invalid) / n),
        "constant": constant,
        "min": None if valid.size == 0 else float(valid.min()),
        "max": None if valid.size == 0 else float(valid.max()),
        "mean": None if valid.size == 0 else float(valid.mean()),
    }
