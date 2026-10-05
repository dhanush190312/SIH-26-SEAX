"""Tests for the spectral indices and their statistics.

The arithmetic is one line per index, so these tests spend their effort on the
things that actually go wrong downstream: the sign convention (a flipped
argument order turns water into dense canopy and still looks like a map), the
"undefined is NaN, never 0.0" rule (a 0.0 NDVI over a nodata border would be
cited in an answer as bare soil), and the JSON-safety of the dicts that
:func:`index_stats` emits into ``runs/<id>/result.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from satquery.contracts import SatQueryError
from satquery.preprocess.indices import (
    index_stats,
    ndbi,
    ndvi,
    ndwi,
    normalized_difference,
)
from satquery.preprocess.optical import percentile_stretch

# A 2x2 patch covering the four cases an index has to survive: a strong
# positive, an exact zero, a strong negative, and an undefined pixel where both
# bands are zero (a nodata border, or the inside of a hard shadow).
NIR = np.array([[0.5, 0.4], [0.1, 0.0]], dtype=np.float32)
RED = np.array([[0.1, 0.4], [0.5, 0.0]], dtype=np.float32)

#: (0.5-0.1)/0.6, (0.4-0.4)/0.8, (0.1-0.5)/0.6, undefined.
NDVI_EXPECTED = [2.0 / 3.0, 0.0, -2.0 / 3.0]


# --------------------------------------------------------------------------- #
# The three indices: hand-computed values, range, sign
# --------------------------------------------------------------------------- #


def test_ndvi_matches_hand_computed_values():
    out = ndvi(NIR, RED)

    assert out.dtype == np.float32
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(NDVI_EXPECTED[0], abs=1e-6)
    assert out[0, 1] == pytest.approx(NDVI_EXPECTED[1], abs=1e-6)
    assert out[1, 0] == pytest.approx(NDVI_EXPECTED[2], abs=1e-6)
    assert np.isnan(out[1, 1])


def test_ndwi_matches_hand_computed_values():
    """McFeeters green/NIR: positive over open water."""
    green = np.array([[0.3, 0.1]], dtype=np.float32)
    nir = np.array([[0.1, 0.3]], dtype=np.float32)

    out = ndwi(green, nir)

    assert out[0, 0] == pytest.approx(0.5, abs=1e-6)  # (0.3-0.1)/0.4
    assert out[0, 1] == pytest.approx(-0.5, abs=1e-6)


def test_ndbi_matches_hand_computed_values():
    swir = np.array([[0.4]], dtype=np.float32)
    nir = np.array([[0.2]], dtype=np.float32)

    out = ndbi(swir, nir)

    assert out[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-6)  # (0.4-0.2)/0.6


@pytest.mark.parametrize("fn", [ndvi, ndwi, ndbi])
def test_indices_land_inside_the_closed_unit_interval(fn, optical_array):
    """On real-shaped reflectance, every defined pixel is in [-1, 1]."""
    stack = optical_array(bands=12, height=16, width=16)

    out = fn(stack[7], stack[3])

    defined = out[np.isfinite(out)]
    assert defined.size == out.size  # nothing undefined in this fixture
    assert defined.min() >= -1.0 and defined.max() <= 1.0


@pytest.mark.parametrize("fn", [ndvi, ndwi, ndbi])
def test_swapping_the_arguments_flips_the_sign(fn):
    """The failure this module's docstring warns about, made checkable.

    ``ndvi(red, nir)`` runs happily and produces a plausible map of exactly the
    wrong thing, so the only defence is that callers pass keywords -- these
    tests pin that the sign really is the whole difference.
    """
    forward = fn(NIR, RED)
    reverse = fn(RED, NIR)

    both_defined = np.isfinite(forward) & np.isfinite(reverse)
    assert both_defined.sum() == 3
    assert np.allclose(forward[both_defined], -reverse[both_defined], atol=1e-6)


def test_a_shared_linear_factor_cancels_out():
    """DN vs reflectance scaling is safe: the index is a ratio.

    This is why adapters may hand raw 0-10000 L2A DN straight to these
    functions without dividing by 10000 first.
    """
    reflectance = ndvi(NIR, RED)
    dn = ndvi(NIR * 10_000.0, RED * 10_000.0)

    defined = np.isfinite(reflectance)
    assert np.allclose(reflectance[defined], dn[defined], atol=1e-5)
    assert np.isnan(dn[~defined]).all()


def test_a_percentile_stretch_does_not_commute_with_the_index():
    """Guard the "raw values only" rule with the number it protects.

    A per-band stretch is per-band affine, so it changes the ratio. The point
    is not that one value differs by a hair: NDVI of stretched bands is a
    different quantity that still lands in [-1, 1] and still renders as a
    credible vegetation map, so nothing downstream can notice the mistake.
    """
    # Different spans per band, which is the normal case -- a shared scale
    # factor would cancel (see the DN test above) and prove nothing.
    raw = np.stack(
        [
            np.array([[0.3, 0.4, 0.5]], dtype=np.float32),  # NIR
            np.array([[0.1, 0.2, 0.4]], dtype=np.float32),  # Red
        ]
    )
    stretched = percentile_stretch(raw, low=0.0, high=100.0)

    from_raw = ndvi(raw[0], raw[1])
    from_stretched = ndvi(stretched[0], stretched[1])

    defined = np.isfinite(from_raw) & np.isfinite(from_stretched)
    assert defined.any()
    assert not np.allclose(from_raw[defined], from_stretched[defined], atol=0.05)
    # ...and yet the wrong map is still a perfectly well-formed index map.
    assert from_stretched[defined].min() >= -1.0 and from_stretched[defined].max() <= 1.0


def test_negative_reflectance_is_clipped_rather_than_exploding():
    """L2A atmospheric correction genuinely returns negatives over dark water.

    ``(-0.05 - 0.01) / (-0.05 + 0.01) == 1.5`` -- arithmetically fine, outside
    the range every NDVI threshold in the literature assumes, so it is clipped.
    """
    out = ndvi(np.array([[-0.05]], dtype=np.float32), np.array([[0.01]], dtype=np.float32))

    assert out[0, 0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Invalid input: NaN, nodata, near-zero denominators
# --------------------------------------------------------------------------- #


def test_nan_in_either_band_propagates_to_nan():
    nir = np.array([[0.5, np.nan], [0.5, 0.5]], dtype=np.float32)
    red = np.array([[0.1, 0.1], [np.nan, 0.1]], dtype=np.float32)

    out = ndvi(nir, red)

    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])
    assert np.isfinite(out[0, 0]) and np.isfinite(out[1, 1])


def test_infinities_are_treated_as_missing_data():
    """Consistent with percentile_stretch: non-finite means "no value"."""
    out = ndvi(
        np.array([[np.inf, -np.inf, 0.5]], dtype=np.float32),
        np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
    )

    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert np.isfinite(out[0, 2])


def test_zero_sum_pixels_are_nan_not_zero():
    """The single most consequential choice in the module.

    0.0 asserts "bare soil"; NaN asserts nothing, and index_stats drops it.
    """
    out = ndvi(np.zeros((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32))

    assert np.isnan(out).all()
    assert not (out == 0.0).any()


def test_eps_sets_the_smallest_denominator_worth_dividing_by():
    nir = np.array([[1e-9]], dtype=np.float32)
    red = np.array([[3e-9]], dtype=np.float32)

    assert np.isnan(ndvi(nir, red)[0, 0])  # default eps=1e-6 rejects it
    assert ndvi(nir, red, eps=1e-12)[0, 0] == pytest.approx(-0.5, abs=1e-5)


def test_negative_denominators_are_judged_on_magnitude():
    """Both bands negative gives a large-magnitude negative sum: still valid."""
    out = normalized_difference(
        np.array([[-0.3]], dtype=np.float32), np.array([[-0.1]], dtype=np.float32)
    )

    assert out[0, 0] == pytest.approx(0.5, abs=1e-6)  # (-0.2)/(-0.4)


def test_no_runtime_warning_is_emitted_for_undefined_pixels(recwarn):
    """A nodata border is normal, not something to spam a benchmark log with."""
    ndvi(np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32))

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --------------------------------------------------------------------------- #
# Input shapes and errors
# --------------------------------------------------------------------------- #


def test_a_leading_singleton_band_axis_is_accepted():
    """Band slices arrive both as (h,w) and as (1,h,w); both must work."""
    flat = ndvi(NIR, RED)
    banded = ndvi(NIR[None], RED[None])

    assert banded.shape == (2, 2)
    assert np.allclose(flat, banded, equal_nan=True)


def test_a_multi_band_stack_is_refused():
    """Broadcasting band 0 against three bands is a bug, not a convenience."""
    with pytest.raises(SatQueryError):
        ndvi(np.zeros((3, 4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.float32))


def test_mismatched_shapes_point_at_the_resampler():
    with pytest.raises(SatQueryError) as exc:
        ndvi(np.zeros((4, 4), dtype=np.float32), np.zeros((4, 5), dtype=np.float32))

    assert "resample" in str(exc.value)


def test_non_positive_eps_is_a_programming_error():
    with pytest.raises(ValueError):
        ndvi(NIR, RED, eps=0.0)


def test_integer_dn_input_needs_no_cast_at_the_call_site():
    nir = np.array([[2000, 1000]], dtype=np.uint16)
    red = np.array([[1000, 3000]], dtype=np.uint16)

    out = ndvi(nir, red)

    assert out.dtype == np.float32
    assert out[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert out[0, 1] == pytest.approx(-0.5, abs=1e-6)


def test_indices_do_not_mutate_their_inputs():
    nir = np.array([[0.5, np.inf]], dtype=np.float32)
    red = np.array([[0.1, 0.1]], dtype=np.float32)
    nir_before, red_before = nir.copy(), red.copy()

    ndvi(nir, red)

    assert np.array_equal(nir, nir_before, equal_nan=True)
    assert np.array_equal(red, red_before)


# --------------------------------------------------------------------------- #
# index_stats -- the evidence dict
# --------------------------------------------------------------------------- #

EXPECTED_KEYS = {
    "mean",
    "std",
    "min",
    "max",
    "positive_pct",
    "threshold",
    "valid_pct",
    "n_valid",
}


def test_index_stats_reports_the_documented_keys_and_hand_computed_values():
    # valid = [-1.0, 0.0, 0.5]; mean = -1/6, population std = sqrt(7/18).
    array = np.array([[-1.0, 0.0], [0.5, np.nan]], dtype=np.float32)

    stats = index_stats(array, name="ndvi")

    assert set(stats) == EXPECTED_KEYS
    assert stats["mean"] == pytest.approx(-1.0 / 6.0, abs=1e-6)
    assert stats["std"] == pytest.approx(np.sqrt(7.0 / 18.0), abs=1e-6)
    assert stats["min"] == pytest.approx(-1.0)
    assert stats["max"] == pytest.approx(0.5)
    assert stats["positive_pct"] == pytest.approx(100.0 / 3.0, abs=1e-4)
    assert stats["threshold"] == 0.0
    assert stats["valid_pct"] == pytest.approx(75.0)
    assert stats["n_valid"] == 3.0


def test_index_stats_values_are_plain_floats():
    """These dicts are json.dump-ed into the run result; np.float32 is not."""
    stats = index_stats(np.array([[0.25, -0.25]], dtype=np.float32))

    assert all(type(v) is float for v in stats.values())


def test_index_stats_positive_pct_is_strictly_above_the_threshold():
    """A pixel sitting exactly on the cut is not "affected"."""
    at_threshold = index_stats(np.full((10, 10), 0.3, dtype=np.float32), threshold=0.3)
    above = index_stats(np.full((10, 10), 0.3001, dtype=np.float32), threshold=0.3)

    assert at_threshold["positive_pct"] == 0.0
    assert above["positive_pct"] == 100.0


def test_index_stats_threshold_is_echoed_so_the_trace_records_the_cut():
    array = np.array([[0.0, 0.1, 0.3, 0.6]], dtype=np.float32)

    loose = index_stats(array, threshold=0.0)
    strict = index_stats(array, threshold=0.2)

    assert loose["positive_pct"] == pytest.approx(75.0)
    assert loose["threshold"] == 0.0
    assert strict["positive_pct"] == pytest.approx(50.0)
    assert strict["threshold"] == pytest.approx(0.2)
    # Same map, different cut: only the threshold key explains the difference.
    assert loose["mean"] == strict["mean"]


def test_index_stats_ignores_invalid_pixels_in_every_statistic():
    clean = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    holed = np.array([[0.2, 0.4, 0.6, np.nan, np.inf, -np.inf]], dtype=np.float32)

    a, b = index_stats(clean), index_stats(holed)

    for key in ("mean", "std", "min", "max", "positive_pct"):
        assert a[key] == pytest.approx(b[key]), key
    assert a["valid_pct"] == 100.0
    assert b["valid_pct"] == pytest.approx(50.0)


def test_index_stats_on_an_all_invalid_map_returns_zeros_not_nan():
    """NaN is not valid JSON, so an empty scene reports zeros plus valid_pct=0.

    valid_pct is the field that stops a caller citing that 0.0 as bare soil.
    """
    stats = index_stats(np.full((4, 4), np.nan, dtype=np.float32), name="ndwi")

    assert set(stats) == EXPECTED_KEYS
    assert stats["valid_pct"] == 0.0
    assert stats["n_valid"] == 0.0
    assert not any(np.isnan(v) for v in stats.values())
    assert json.loads(json.dumps(stats))["mean"] == 0.0


def test_index_stats_is_json_serialisable_for_a_normal_map(optical_array):
    stack = optical_array(bands=12, height=8, width=8)

    stats = index_stats(ndvi(stack[7], stack[3]), name="ndvi")

    round_tripped = json.loads(json.dumps(stats))
    assert round_tripped == pytest.approx(stats)


def test_index_stats_flattens_any_shape():
    plane = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    assert index_stats(plane) == index_stats(plane[None]) == index_stats(plane.reshape(-1))


def test_index_stats_is_deterministic():
    array = np.linspace(-1.0, 1.0, 101, dtype=np.float32)

    assert index_stats(array) == index_stats(array.copy())


def test_index_stats_of_a_uniform_map_has_zero_spread():
    stats = index_stats(np.full((5, 5), 0.42, dtype=np.float32))

    assert stats["std"] == 0.0
    assert stats["min"] == stats["max"] == pytest.approx(0.42)
    assert stats["positive_pct"] == 100.0


# --------------------------------------------------------------------------- #
# The path the change/fusion tools actually walk
# --------------------------------------------------------------------------- #


def test_water_scene_reads_as_water_in_both_ndvi_and_ndwi():
    """End to end on a two-region synthetic scene, the way a tool would run it.

    Water is dark in NIR and bright-ish in green; vegetation is the reverse.
    The assertion is on the *sign agreement* of the two indices, which is what
    an answer would actually cite ("NDWI positive over 50% of the scene, NDVI
    negative there") rather than on any single pixel value.
    """
    green = np.full((8, 8), 0.10, dtype=np.float32)
    red = np.full((8, 8), 0.08, dtype=np.float32)
    nir = np.full((8, 8), 0.35, dtype=np.float32)
    green[:4], red[:4], nir[:4] = 0.09, 0.05, 0.02  # top half of the scene: water

    veg = index_stats(ndvi(nir, red), name="ndvi")
    water = index_stats(ndwi(green, nir), name="ndwi")

    assert water["positive_pct"] == pytest.approx(50.0)
    assert veg["positive_pct"] == pytest.approx(50.0)
    assert veg["min"] < 0.0 < veg["max"]
