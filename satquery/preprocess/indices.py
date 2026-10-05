"""Spectral indices and the numeric evidence a composed answer has to cite.

These are the deterministic tools in the "deterministic tools feed the VLMs,
never the reverse" rule. A VLM asked whether a field is vegetated guesses from
colour; NDVI over the raw reflectance is arithmetic. So the pipeline computes
the index, summarises it with :func:`index_stats`, and hands the numbers to the
model as structured context -- which is also what makes the composer's "every
claim cites a tool output" check enforceable rather than aspirational.

Three conventions that are easy to get wrong, so they are stated once here:

* **Raw values only.** Indices are ratios, and a per-band percentile stretch is
  a per-band affine map, so it does not commute with a normalised difference.
  NDVI of stretched bands is a different, meaningless quantity that still lands
  in ``[-1, 1]`` and still looks like a vegetation map. Pass the reflectance or
  DN straight off the reader. The one thing that *is* safe is a shared linear
  factor on both inputs (DN vs reflectance scaling cancels).
* **Argument order carries the sign.** ``ndvi(nir, red)`` and ``ndvi(red, nir)``
  both run and both produce a plausible-looking map with the sign flipped,
  which turns "dense vegetation" into "open water". The parameter names, not
  the position, are the contract -- call them with keywords when the bands come
  out of a dict.
* **Undefined is NaN, not zero.** Where the two bands sum to (near) zero the
  index has no value. Returning ``0.0`` there would assert "bare soil / no
  vegetation" for what is really a nodata border or a black shadow, and that
  number would end up quoted in an answer. NaN propagates instead, and every
  statistic in this module ignores it.
"""

from __future__ import annotations

import numpy as np

from ..contracts import SatQueryError
from ..log import get_logger

__all__ = [
    "normalized_difference",
    "ndvi",
    "ndwi",
    "ndbi",
    "vari",
    "index_stats",
]

log = get_logger(__name__)


def _as_plane(array: np.ndarray, argname: str) -> np.ndarray:
    """Coerce one band to a float32 2-D plane with invalid pixels as NaN.

    A leading singleton axis is accepted and dropped: band slices arrive both as
    ``(h, w)`` from ``RasterImage.band`` and as ``(1, h, w)`` from anything that
    kept the band-first convention, and rejecting one of those spellings would
    only push a reshape into every caller. A genuine multi-band stack is
    refused -- broadcasting band 0 of one stack against 3 bands of another is a
    bug, not a feature.
    """
    a = np.asarray(array)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise SatQueryError(
            f"{argname} must be a single band -- (h, w) or (1, h, w) -- got shape {a.shape}"
        )
    x = a.astype(np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan  # +/-inf as well as pre-existing NaN
    return x


def normalized_difference(
    high: np.ndarray,
    low: np.ndarray,
    *,
    eps: float = 1e-6,
    name: str = "index",
) -> np.ndarray:
    """``(high - low) / (high + low)``, the shape every index here takes.

    ``high`` is the band that makes the index *positive* (NIR for NDVI, green
    for NDWI, SWIR for NDBI), which is the only way to remember the argument
    order without a lookup table.

    ``eps`` is not added to the denominator -- that would quietly bias every
    small-signal pixel and still divide by something meaningless. It is the
    smallest denominator magnitude worth dividing by: at or below it the pixel
    becomes NaN. Output is clipped to ``[-1, 1]``, which is exact arithmetic for
    non-negative reflectance and a guard for the negative values that L2A
    atmospheric correction genuinely produces over dark water.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    a = _as_plane(high, "high")
    b = _as_plane(low, "low")
    if a.shape != b.shape:
        raise SatQueryError(
            f"{name}: band shapes differ, {a.shape} vs {b.shape}; resample to a common "
            "grid first (satquery.preprocess.pairing.resample_to_match)"
        )
    den = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(den) <= np.float32(eps), np.float32(np.nan), (a - b) / den)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def ndvi(nir: np.ndarray, red: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Normalised Difference Vegetation Index, ``(NIR - Red) / (NIR + Red)``.

    Sentinel-2 L2A: NIR = B08 (band index 7), Red = B04 (index 3) in the
    12-band ordering documented in :mod:`satquery.preprocess.optical`. On a
    4-band RGBN product: NIR = 3, Red = 0.

    Rough reading, for the prompt context rather than for classification:
    ``> 0.5`` dense canopy, ``0.2-0.5`` sparse or stressed vegetation and
    crops, ``0.0-0.2`` bare soil and built-up, ``< 0`` water, snow, cloud.
    """
    return normalized_difference(nir, red, eps=eps, name="ndvi")


def ndwi(green: np.ndarray, nir: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """McFeeters NDWI, ``(Green - NIR) / (Green + NIR)`` -- open water.

    This is the water-body formulation (green/NIR), not Gao's vegetation-liquid
    NDWI (NIR/SWIR); both are called "NDWI" in the literature and they answer
    different questions. Positive values are water. Green = B03 (index 2),
    NIR = B08 (index 7) on Sentinel-2 L2A.

    ``> 0`` is the usual water cut, though deep building shade also lifts it,
    which is why SAR backscatter is the better water evidence whenever a radar
    image is in the pair.
    """
    return normalized_difference(green, nir, eps=eps, name="ndwi")


def ndbi(swir: np.ndarray, nir: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Normalised Difference Built-up Index, ``(SWIR - NIR) / (SWIR + NIR)``.

    SWIR = B11 (index 10), NIR = B08 (index 7) on Sentinel-2 L2A; B11 rather
    than B12 is the standard pairing.

    Positive over impervious surfaces and bare soil alike -- NDBI on its own
    does not separate a rooftop from a dry field, so cite it next to NDVI
    (built-up is high NDBI *and* low NDVI) instead of calling it a building
    detector. It needs a SWIR band, so it is unavailable for RGB/RGBN products
    and for Cartosat-2S MX; callers must check the band count before offering
    it as a tool.
    """
    return normalized_difference(swir, nir, eps=eps, name="ndbi")


def vari(green: np.ndarray, red: np.ndarray, blue: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Visible Atmospherically Resistant Index, ``(G - R) / (G + R - B)``.

    The RGB-only greenness index (Gitelson et al., 2002). It exists here for one
    reason: most of the benchmark corpus and most user uploads are three-band
    tiles with no NIR, and the honest options are "no vegetation evidence at
    all" or a published index that works without NIR. It is *not* a substitute
    for NDVI and must never be reported as one -- it saturates over dense
    canopy and is sensitive to shadow, so it supports "vegetated / not" rather
    than a biomass claim.

    The denominator can pass through zero (unlike a normalised difference, it is
    not a sum of non-negative terms), which is why the guard is on ``|G + R - B|``
    and those pixels come back NaN rather than as a spike of +-1.
    """
    g = _as_plane(green, "green")
    r = _as_plane(red, "red")
    b = _as_plane(blue, "blue")
    if not (g.shape == r.shape == b.shape):
        raise SatQueryError(
            f"vari: band shapes differ (green {g.shape}, red {r.shape}, blue {b.shape})"
        )
    den = g + r - b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(np.abs(den) <= np.float32(eps), np.float32(np.nan), (g - r) / den)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def index_stats(
    index_array: np.ndarray,
    *,
    threshold: float = 0.0,
    name: str = "index",
) -> dict[str, float]:
    """Summarise an index map into the numbers an answer is allowed to quote.

    Keys are deliberately unprefixed and identical for every index, so a trace
    nests them (``{"ndvi": index_stats(v, name="ndvi")}``) rather than parsing
    key names; ``name`` only labels log lines and error messages.

    Returns:
        ``mean``, ``std`` (population, ddof=0), ``min``, ``max`` over the valid
        pixels; ``positive_pct`` -- percentage of valid pixels **strictly
        greater than** ``threshold``, 0-100, because "share of the scene
        affected" is the quantity that gets cited; ``threshold`` echoed back so
        the trace records which cut produced that percentage; ``valid_pct`` and
        ``n_valid`` so a reader can see how much of the scene the mean covers.

    Every value is a plain ``float`` and never NaN: these dicts are serialised
    into ``runs/<id>/result.json`` and NaN is not valid JSON. An all-invalid map
    therefore reports zeros with ``valid_pct == 0.0`` -- check that field before
    citing the mean, because 0.0 NDVI over nothing is not bare soil.
    """
    x = np.asarray(index_array).astype(np.float32, copy=False).reshape(-1)
    total = int(x.size)
    valid = x[np.isfinite(x)]
    n_valid = int(valid.size)

    if n_valid == 0:
        log.warning(
            "%s: no valid pixels in a %d-pixel index map; stats reported as zeros", name, total
        )
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_pct": 0.0,
            "threshold": float(threshold),
            "valid_pct": 0.0,
            "n_valid": 0.0,
        }

    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "positive_pct": float(np.count_nonzero(valid > np.float32(threshold)) / n_valid * 100.0),
        "threshold": float(threshold),
        "valid_pct": float(n_valid / total * 100.0),
        "n_valid": float(n_valid),
    }
