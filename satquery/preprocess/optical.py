"""Optical display preparation: percentile stretch, band composites, uint8.

Why this module exists: the hidden evaluation set is Cartosat-2S optical (plus
RISAT SAR), while everything we tune on is Sentinel-2 or aerial RGB. Those
arrive with different bit depths (11-bit DN, 12-bit DN, 16-bit scaled
reflectance, float 0-1), different band counts and different nodata sentinels.
A VLM sees none of that -- it sees three 8-bit channels. :func:`percentile_stretch`
is the one function that makes an 11-bit Cartosat scene and a Sentinel-2 L2A
reflectance patch reach the model looking the same, which is what lets a model
adapted on one work on the other.

Two gotchas, stated once here rather than in every caller:

* A stretch is a *display* transform, per-band affine by default, so it does not
  commute with a normalized difference. NDVI computed on stretched bands is not
  NDVI. :mod:`satquery.preprocess.indices` always takes raw reflectance/DN.
* Band indices are the highest-risk constants in the repo. A wrong triple fails
  *silently*: the composite still looks like a satellite image, so nothing
  downstream can tell that what the model was told is "vegetation" is really
  SWIR. Hence every entry in :data:`BAND_PRESETS` states the on-disk band order
  it assumes, and :func:`select_composite` refuses out-of-range indices loudly
  instead of wrapping negatives or truncating.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..contracts import SatQueryError
from ..log import get_logger

__all__ = [
    "BAND_PRESETS",
    "S2_L2A_BAND_NAMES",
    "percentile_stretch",
    "select_composite",
    "to_uint8",
]

log = get_logger(__name__)

#: A 2-98% span narrower than this is treated as a constant band. Rescaling it
#: would turn float rounding noise into fake texture, which reads as real
#: structure to a VLM -- worse than a flat grey patch.
_MIN_SPAN = 1e-12

#: Band names of the standard Sentinel-2 **L2A** 12-band stack, in the order
#: BigEarthNet-S2 and most L2A patch exports (TerraMind's and CROMA's included)
#: write them. L2A has no B10: the cirrus band is consumed by the atmospheric
#: correction and dropped, so the stack is 12 bands, not 13, and every index
#: past B09 sits one lower than an L1C band table would suggest. That off-by-one
#: is the most common way to end up with a wrong-but-plausible composite.
S2_L2A_BAND_NAMES: tuple[str, ...] = (
    "B01",  # 0   443 nm  coastal aerosol, 60 m
    "B02",  # 1   490 nm  blue, 10 m
    "B03",  # 2   560 nm  green, 10 m
    "B04",  # 3   665 nm  red, 10 m
    "B05",  # 4   705 nm  red edge 1, 20 m
    "B06",  # 5   740 nm  red edge 2, 20 m
    "B07",  # 6   783 nm  red edge 3, 20 m
    "B08",  # 7   842 nm  NIR, 10 m
    "B8A",  # 8   865 nm  narrow NIR, 20 m
    "B09",  # 9   945 nm  water vapour, 60 m
    "B11",  # 10 1610 nm  SWIR 1, 20 m
    "B12",  # 11 2190 nm  SWIR 2, 20 m
)

#: Named band selections for :func:`select_composite`, keyed by the product
#: whose on-disk band order they assume. Length 3 selects a composite; length 1
#: means "single band, replicated to grey". Presets are a convenience, never a
#: substitute for checking the band count of an unfamiliar product.
BAND_PRESETS: dict[str, tuple[int, ...]] = {
    # --- already-RGB imagery: the PNG/JPEG benchmark path (VRSBench,
    # DIOR-RSVG, LEVIR-CC/MCI) where band 0 really is red.
    "rgb": (0, 1, 2),
    "bgr": (2, 1, 0),  # BGR-ordered arrays (OpenCV-written PNGs) read as RGB
    "gray": (0,),  # panchromatic, e.g. Cartosat-2S PAN -> replicated to grey
    # --- Sentinel-2 L2A, 12-band stack in S2_L2A_BAND_NAMES order.
    "s2_rgb": (3, 2, 1),  # B04/B03/B02  natural colour
    "s2_false_color": (7, 3, 2),  # B08/B04/B03  colour-infrared, vegetation red
    "s2_swir": (11, 8, 3),  # B12/B8A/B04  burn scars, bare rock, moisture
    "s2_agriculture": (10, 7, 1),  # B11/B08/B02  crop vigour
    "s2_urban": (11, 10, 3),  # B12/B11/B04  built-up against bare soil
    # --- BigEarthNet-MM. Its S2 patch is the same 12-band L2A ordering, so the
    # indices coincide with the s2_* presets; kept as separate keys so an
    # adapter can name the dataset it means instead of relying on that identity.
    "bigearthnet_rgb": (3, 2, 1),
    "bigearthnet_false_color": (7, 3, 2),
    # --- 4-band red,green,blue,NIR products (NAIP, many pan-sharpened MX
    # deliveries). Natural colour is just "rgb"; this is the CIR pairing.
    "rgbn_false_color": (3, 0, 1),
    # VERIFY: Cartosat-2S MX band order recalled as blue,green,red,NIR (4 bands).
    # No real product has been opened. Confirm against the hidden eval imagery
    # before trusting either of these two.
    "cartosat_mx_rgb": (2, 1, 0),
    "cartosat_mx_false_color": (3, 2, 1),
    # VERIFY: LISS-III (Resourcesat-2/2A) ships green,red,NIR,SWIR. There is no
    # blue band, so natural colour is *impossible* for LISS-III -- asking for it
    # is a data error, not a preset we can supply.
    "liss3_false_color": (2, 1, 0),
}


# --------------------------------------------------------------------------- #
# Stretch
# --------------------------------------------------------------------------- #


def _to_float32_nan(array: np.ndarray, nodata: float | None) -> np.ndarray:
    """Copy to float32 with every invalid pixel unified into NaN.

    The sentinel dies here and nowhere else: downstream code only has to know
    about NaN, so a forgotten ``-9999`` cannot survive as a real value and drag
    a percentile with it -- which is how a nodata-bordered scene ends up
    rendered as one flat bright rectangle.
    """
    x = np.asarray(array).astype(np.float32, copy=True)
    if nodata is not None and not np.isnan(nodata):
        x[x == np.float32(nodata)] = np.nan
    x[~np.isfinite(x)] = np.nan  # +/-inf too
    return x


def _limits(values: np.ndarray, low: float, high: float) -> tuple[float, float] | None:
    """Percentile anchors over the valid pixels, or ``None`` if there are none.

    Masks explicitly instead of calling ``np.nanpercentile``, which warns on an
    all-NaN slice. An all-nodata band is normal at a scene edge, not an event
    worth a stderr line in the middle of a benchmark run.
    """
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return None
    lo, hi = np.percentile(valid, (low, high))
    return float(lo), float(hi)


def _apply_limits(
    plane: np.ndarray,
    limits: tuple[float, float] | None,
    out_lo: float,
    out_hi: float,
) -> np.ndarray:
    mid = 0.5 * (out_lo + out_hi)
    nan_mask = np.isnan(plane)
    if limits is None or (limits[1] - limits[0]) <= _MIN_SPAN:
        # Constant (or entirely invalid) band: mid-range, not 0. A flat band
        # carries no information, and mapping it to the bottom of the range
        # would render a uniformly bright surface as black -- a visible claim
        # about the scene that the data does not support. Mid-grey makes the
        # "no contrast here" case obvious to a human reading the preview.
        out = np.full(plane.shape, np.float32(mid), dtype=np.float32)
    else:
        lo, hi = limits
        scale = (out_hi - out_lo) / (hi - lo)
        out = (plane - np.float32(lo)) * np.float32(scale) + np.float32(out_lo)
        np.clip(out, min(out_lo, out_hi), max(out_lo, out_hi), out=out)
    return np.where(nan_mask, np.float32(np.nan), out).astype(np.float32)


def percentile_stretch(
    array: np.ndarray,
    *,
    low: float = 2.0,
    high: float = 98.0,
    per_band: bool = True,
    nodata: float | None = None,
    out_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Linear contrast stretch between the ``low`` and ``high`` percentiles.

    The sensor-agnostic normaliser. Percentiles rather than min/max because a
    single saturated pixel -- a specular return, a hot pixel, a nodata border
    that survived -- collapses a min/max stretch into near-uniform grey, and
    remote-sensing scenes are full of them. 2-98% is what QGIS and every RS
    preview pipeline default to, so a stretched preview matches what an analyst
    expects to see.

    Args:
        array: ``(band, h, w)`` or ``(h, w)``. Any numeric dtype; DN or
            reflectance -- the point of the function is not caring which.
        low, high: percentile anchors, ``0 <= low < high <= 100``.
        per_band: stretch each band independently (maximum contrast, standard
            for display) or share one pair of anchors across all bands (keeps
            relative band brightness, which is what you want when two channels
            are going to be compared by eye).
        nodata: sentinel value to exclude from the percentiles and mask out.
        out_range: target range -- ``(0, 1)`` for float, ``(0, 255)`` when the
            caller wants to skip :func:`to_uint8`'s own scaling.

    Returns:
        float32, same shape as the input. Invalid pixels (NaN, +/-inf,
        ``nodata``) come back as **NaN**, deliberately: the stretch does not get
        to decide what nodata should look like. :func:`to_uint8` renders NaN as 0.

    A band with no contrast -- constant, or entirely invalid -- maps to the
    midpoint of ``out_range``. It never returns NaN for a merely-constant band.
    """
    a = np.asarray(array)
    if a.ndim not in (2, 3):
        raise ValueError(f"percentile_stretch expects (band,h,w) or (h,w), got shape {a.shape}")
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"need 0 <= low < high <= 100, got low={low}, high={high}")
    out_lo, out_hi = float(out_range[0]), float(out_range[1])
    if out_lo == out_hi:
        raise ValueError(f"out_range must span a nonzero interval, got {out_range!r}")

    x = _to_float32_nan(a, nodata)
    stack = x if x.ndim == 3 else x[None]

    if per_band:
        limits = [_limits(stack[i], low, high) for i in range(stack.shape[0])]
    else:
        shared = _limits(stack, low, high)
        limits = [shared] * stack.shape[0]

    out = np.stack(
        [_apply_limits(stack[i], limits[i], out_lo, out_hi) for i in range(stack.shape[0])]
    )
    return out if x.ndim == 3 else out[0]


def to_uint8(array: np.ndarray) -> np.ndarray:
    """``(b,h,w)`` float in ``[0, 1]`` -> uint8 ``[0, 255]``.

    Out-of-range input is **clipped, not rescaled**: this is the last step of
    the display path, and rescaling here would hide a skipped
    :func:`percentile_stretch` -- raw 0-10000 reflectance would come out looking
    plausible while every value above 1 had been flattened into white. NaN
    renders as 0 (black), the usual nodata convention for a preview PNG.

    An input that is already uint8 is returned as a copy rather than crushed to
    ``{0, 255}``, so an accidental second call is harmless.
    """
    a = np.asarray(array)
    if a.dtype == np.uint8:
        return a.copy()
    x = a.astype(np.float32, copy=True)
    x[~np.isfinite(x)] = 0.0
    np.clip(x, 0.0, 1.0, out=x)
    return np.rint(x * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Composites
# --------------------------------------------------------------------------- #


def select_composite(
    array: np.ndarray,
    bands: Sequence[int] | None = None,
    preset: str | None = None,
) -> np.ndarray:
    """Pick three bands out of a stack: ``(b,h,w)`` -> ``(3,h,w)``.

    Pure band selection -- values and dtype are untouched, so stretching before
    or after gives the same picture (with ``per_band=True`` the selected bands
    are stretched independently either way).

    The rules, all chosen to make a wrong composite loud rather than quiet:

    * ``bands`` and ``preset`` are mutually exclusive. Passing both is a config
      bug, and picking a winner would hide it.
    * Neither given: bands ``(0, 1, 2)``. Correct for RGB benchmark imagery,
      *wrong* for a 12-band L2A patch (it would show coastal-aerosol / blue /
      green), which is why every multispectral caller names a preset.
    * A single-band array -- Cartosat PAN, one SAR channel, an index map -- is
      replicated to three identical channels, and an explicit ``bands``/
      ``preset`` is then ignored and logged. The generic preview path passes
      ``preset="rgb"`` for everything, and raising there would only force every
      caller to special-case panchromatic input.
    * Out-of-range and negative indices raise :class:`SatQueryError`. Negatives
      are rejected rather than wrapped: ``-1`` silently meaning "SWIR 2" on a
      12-band stack is exactly the failure mode this module exists to stop.
    """
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3:
        raise SatQueryError(
            f"select_composite expects (band,h,w) or (h,w), got shape {np.asarray(array).shape}"
        )
    if bands is not None and preset is not None:
        raise SatQueryError(
            f"pass bands or preset, not both (got bands={tuple(bands)!r}, preset={preset!r})"
        )

    n_bands = a.shape[0]
    if n_bands == 1:
        if bands is not None or preset is not None:
            log.debug(
                "select_composite: single-band input, replicating to grey and ignoring %s",
                f"preset={preset!r}" if preset is not None else f"bands={tuple(bands or ())!r}",
            )
        return np.repeat(a, 3, axis=0)

    if preset is not None:
        if preset not in BAND_PRESETS:
            raise SatQueryError(
                f"unknown band preset {preset!r}; known presets: "
                f"{', '.join(sorted(BAND_PRESETS))}"
            )
        idx = tuple(BAND_PRESETS[preset])
        source = f"preset {preset!r}"
    elif bands is not None:
        try:
            idx = tuple(int(b) for b in bands)
        except (TypeError, ValueError) as exc:
            raise SatQueryError(f"bands must be integers, got {bands!r}") from exc
        source = f"bands {idx!r}"
    else:
        idx = (0, 1, 2)
        source = "default first-three-bands selection"

    if len(idx) == 1:
        idx = idx * 3
    if len(idx) != 3:
        raise SatQueryError(f"{source} must name 1 or 3 bands, got {len(idx)}: {idx!r}")

    bad = [i for i in idx if i < 0 or i >= n_bands]
    if bad:
        hint = ""
        if preset is not None and n_bands in (3, 4):
            hint = (
                f" A {n_bands}-band array is probably already RGB-ordered:"
                " try preset='rgb' or an explicit bands=(...)."
            )
        raise SatQueryError(
            f"{source} needs band indices {idx} but the array has {n_bands} band(s)"
            f" (valid 0..{n_bands - 1}); out of range: {bad}.{hint}"
        )
    return a[list(idx)]
