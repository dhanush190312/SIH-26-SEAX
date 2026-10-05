"""SAR preprocessing: linear power -> dB -> 8-bit, plus the numeric evidence.

This module serves two consumers that want opposite things, which is why both
paths live here and must stay in sync:

* The VLM gets a *picture*: dB-scaled, clipped to ``[-25, 0]`` dB, stretched to
  8-bit, optionally despeckled. A VLM cannot read speckle -- raw linear power is
  a near-black image with a handful of blown-out corner reflectors -- so the
  display transform is not cosmetic, it is what makes the input legible at all.
* The composer gets *numbers*: :func:`backscatter_stats` and
  :func:`sar_water_mask` are the deterministic tools whose output goes into the
  prompt as structured context. The project rule is that the model explains
  measurements it is handed rather than measuring off pixels, and this is the
  SAR half of that rule.

Everything here takes **linear power** (sigma0/gamma0, as SNAP writes it and as
BigEarthNet S1 patches ship), never amplitude and never dB. Amplitude products
need ``20*log10``: square them before calling :func:`linear_to_db`, or the
dynamic range comes out half of what it should be, the ``[-25, 0]`` clip lands
in the wrong place, and every backscatter number is quietly wrong by 2x.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..log import get_logger

__all__ = [
    "DEFAULT_DB_CLIP",
    "WATER_DB_THRESHOLD",
    "linear_to_db",
    "db_to_uint8",
    "lee_filter",
    "sar_water_mask",
    "backscatter_stats",
]

_LOG = get_logger(__name__)

#: Locked project convention for SAR display scaling (see the approach doc,
#: preprocessing section). It has to match between training-data generation and
#: inference: change it on one side only and the fine-tuned model sees a
#: different brightness distribution than the one it learned on.
DEFAULT_DB_CLIP: tuple[float, float] = (-25.0, 0.0)

#: Sentinel-1 VV rule of thumb for open water. Sensor- and incidence-angle
#: dependent, hence a keyword argument everywhere rather than a hardcode.
WATER_DB_THRESHOLD: float = -18.0


def linear_to_db(array: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Convert linear power to decibels: ``10*log10(x)``, floored at ``eps``.

    Real SAR products contain exact zeros (masked borders, nodata fill) and
    occasionally small negatives (badly calibrated or interpolated pixels), so
    an unguarded ``log10`` returns ``-inf``/``nan`` and poisons every downstream
    mean. ``eps`` is a *power* floor, not an output floor: the default maps to
    -80 dB, far below anything a sensor records, so those pixels end up
    obviously-dark instead of infinite, and one bad pixel does not kill a scene.

    NaN is propagated deliberately. Nodata that already arrives as NaN stays
    NaN so :func:`backscatter_stats` can exclude it and the report can tell
    "no data" apart from "very dark water".
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps!r}")
    a = np.asarray(array, dtype=np.float32)
    floored = np.maximum(a, np.float32(eps))  # NaN survives np.maximum
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(floored)
    return np.asarray(db, dtype=np.float32)


def db_to_uint8(array: np.ndarray, *, clip: tuple[float, float] = DEFAULT_DB_CLIP) -> np.ndarray:
    """Clip a dB array to ``clip`` and linearly stretch it to ``0..255``.

    Fixed clip limits rather than a percentile stretch, unlike the optical path:
    dB is an absolute physical scale, so a fixed window makes brightness mean
    the same thing in every scene and lets the model learn "dark = smooth".
    A per-scene stretch would rescale a calm lake into mid-grey and destroy
    exactly the cue we need.

    NaN maps to 0 (black) because an 8-bit PNG has nowhere else to put nodata;
    keep the dB array around if you need to distinguish the two.
    """
    lo, hi = float(clip[0]), float(clip[1])
    if not hi > lo:
        raise ValueError(f"clip must be (low, high) with high > low, got {clip!r}")
    a = np.asarray(array, dtype=np.float32)
    scaled = (np.clip(a, lo, hi) - lo) / (hi - lo)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def lee_filter(array: np.ndarray, *, size: int = 5) -> np.ndarray:
    """Adaptive (Lee, 1980) speckle filter over a ``size x size`` window.

    Speckle is multiplicative, so a plain box blur both destroys edges and
    biases bright targets down. Lee interpolates between the local mean and the
    raw pixel by how much of the local variance looks like signal rather than
    noise: homogeneous water is smoothed hard, a ship or a corner reflector
    keeps its amplitude.

    Applied per band -- filtering across polarisations would mix VV into VH,
    which are different physics, not different samples of the same quantity.

    Two gotchas worth stating:

    * Filter for the *display* path, and compute :func:`backscatter_stats` on
      the **unfiltered** dB array. Despeckling shrinks the standard deviation
      and the p5-p95 spread, which are two of the numbers we report as evidence.
    * NaN spreads over a whole window (the local means are box sums). The
      global noise-variance estimate uses ``nanvar`` so nodata does not wipe out
      the entire band, but fill or crop nodata first if you care about edges.
    """
    if int(size) != size or size < 1:
        raise ValueError(f"size must be an integer >= 1, got {size!r}")
    size = int(size)
    a = np.asarray(array)
    if a.ndim not in (2, 3):
        raise ValueError(f"expected (h, w) or (band, h, w), got shape {a.shape}")
    # float64 throughout: the local variance is computed as E[x^2] - E[x]^2,
    # which cancels catastrophically in float32 once the mean is much larger
    # than the variance (dB arrays around -20 do exactly that).
    x = a.astype(np.float64, copy=False)
    squeeze = x.ndim == 2
    if squeeze:
        x = x[None, ...]

    mean = _uniform_filter(x, size)
    mean_sq = _uniform_filter(x * x, size)
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    with warnings.catch_warnings():  # all-NaN band -> NaN noise estimate
        warnings.simplefilter("ignore", RuntimeWarning)
        noise_var = np.nanvar(x, axis=(-2, -1), keepdims=True)
    denom = local_var + noise_var
    weights = np.divide(local_var, denom, out=np.zeros_like(local_var), where=denom > 0)
    out = mean + weights * (x - mean)
    if squeeze:
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def sar_water_mask(db_array: np.ndarray, *, threshold: float = WATER_DB_THRESHOLD) -> np.ndarray:
    """Boolean ``(h, w)`` mask of low-backscatter pixels: smooth surfaces.

    Smooth water reflects the pulse away from the sensor, so it is the darkest
    thing in a SAR scene -- which is why a single threshold works at all. It is
    also why this is *evidence*, not truth: radar shadow, dry sand, tarmac and
    calm-water look-alikes all fall below the threshold, and a roughened lake
    under wind does not. The composer must cite "backscatter below -18 dB",
    with the threshold, rather than claiming "water".

    Multi-band input (VV+VH) is reduced by the per-pixel mean dB. That is a
    display convention rather than radiometry, but water is dark in every
    polarisation, so the mean is the robust reduction; pass a single band if you
    want strictly VV behaviour.

    NaN is never water: the comparison is False, so nodata cannot inflate an
    area percentage.
    """
    a = _as_2d_db(db_array)
    _warn_if_linear(a)
    with np.errstate(invalid="ignore"):
        mask = a < float(threshold)
    return np.asarray(mask, dtype=bool)


def backscatter_stats(
    db_array: np.ndarray, *, water_threshold: float = WATER_DB_THRESHOLD
) -> dict[str, float]:
    """Backscatter evidence in dB, ready to be cited in an answer.

    Keys: ``mean``, ``std`` (population, ddof=0), ``min``, ``max``, ``p5``,
    ``p50``, ``p95``, ``water_pct``, ``water_threshold``, ``n_valid``.

    Percentiles, not min/max, carry the story: one corner reflector sits 30 dB
    above its neighbours, so ``max`` describes that pixel and nothing else.
    Cite p5/p50/p95 and keep min/max for range sanity checks.

    ``water_pct`` is a percentage of the *valid* (non-NaN) pixels, not of the
    array, so a half-nodata scene reports the fraction of the area actually
    imaged; ``n_valid`` is returned alongside so a reader can see the
    denominator. An all-nodata array returns NaN for the distribution stats
    rather than a fabricated 0.0 -- the composer must be able to tell "no
    measurement" from "0 dB".
    """
    a = _as_2d_db(db_array)
    valid = np.isfinite(a)
    finite = a[valid]
    n_valid = int(finite.size)
    mask = sar_water_mask(a, threshold=water_threshold)
    if n_valid == 0:
        nan = float("nan")
        return {
            "mean": nan,
            "std": nan,
            "min": nan,
            "max": nan,
            "p5": nan,
            "p50": nan,
            "p95": nan,
            "water_pct": 0.0,
            "water_threshold": float(water_threshold),
            "n_valid": 0.0,
        }
    p5, p50, p95 = (float(v) for v in np.percentile(finite, (5.0, 50.0, 95.0)))
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "water_pct": float(100.0 * np.count_nonzero(mask & valid) / n_valid),
        "water_threshold": float(water_threshold),
        "n_valid": float(n_valid),
    }


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _as_2d_db(db_array: np.ndarray) -> np.ndarray:
    """Reduce ``(band, h, w)`` to ``(h, w)`` by mean dB; pass ``(h, w)`` through."""
    a = np.asarray(db_array, dtype=np.float32)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[0] == 1:
            return a[0]
        with warnings.catch_warnings():  # a pixel that is nodata in every band
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.asarray(np.nanmean(a, axis=0), dtype=np.float32)
    raise ValueError(f"expected (h, w) or (band, h, w), got shape {a.shape}")


def _warn_if_linear(a: np.ndarray) -> None:
    """Catch the classic mistake of handing linear power to a dB function.

    A dB array of a land scene is essentially always negative somewhere; linear
    power is non-negative and small. Cheap check, and it turns a silently wrong
    water percentage into a log line.
    """
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if lo >= 0.0 and hi <= 10.0:
        _LOG.warning(
            "input looks like linear power (range %.4g..%.4g, no negative values); "
            "these functions expect dB -- call linear_to_db first",
            lo,
            hi,
        )


def _uniform_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Box mean over the last two axes of a ``(band, h, w)`` array.

    scipy is an optional extra, so there is a numpy fallback -- and the two
    paths are numerically identical, not merely similar: the fallback uses
    scipy's edge convention (``reflect``, which numpy calls ``symmetric``) and
    its even-size origin (the extra sample goes on the leading side). Otherwise
    a laptop baseline and a Colab baseline would disagree for no reason a
    reviewer could see.
    """
    try:
        from scipy.ndimage import uniform_filter  # optional extra
    except ImportError:
        return _box_filter(x, size)
    # size 1 on the band axis: never filter across polarisations.
    return np.asarray(uniform_filter(x, size=(1, size, size), mode="reflect"))


def _box_filter(x: np.ndarray, size: int) -> np.ndarray:
    """Separable box mean via cumulative sums: O(1) per pixel, numpy only."""
    lead = size // 2
    trail = size - 1 - lead
    padded = np.pad(x, ((0, 0), (lead, trail), (lead, trail)), mode="symmetric")
    summed = _box_sum(_box_sum(padded, size, axis=1), size, axis=2)
    return summed / float(size * size)


def _box_sum(x: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Sliding window sum of length ``size`` along ``axis`` (shrinks it by size-1)."""
    cumulative = np.cumsum(x, axis=axis, dtype=np.float64)
    zero = np.zeros_like(np.take(cumulative, (0,), axis=axis))
    cumulative = np.concatenate((zero, cumulative), axis=axis)
    n = cumulative.shape[axis]
    upper = [slice(None)] * cumulative.ndim
    lower = [slice(None)] * cumulative.ndim
    upper[axis] = slice(size, n)
    lower[axis] = slice(0, n - size)
    return cumulative[tuple(upper)] - cumulative[tuple(lower)]
