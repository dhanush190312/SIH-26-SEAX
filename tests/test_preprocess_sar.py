"""Tests for the SAR path: the picture the VLM sees and the numbers it cites.

Two properties are worth defending here, and neither is arithmetic:

* the display transform is *absolute* -- -25 dB is always black and 0 dB always
  white, in every scene -- because the fine-tuned model learns brightness as a
  physical quantity, so a per-scene stretch would move the goalposts;
* the evidence dict is honest -- nodata excluded from the denominator, no
  fabricated 0.0 for an empty scene, and no silent NaN leaking into a number the
  composer is required to quote.

The scipy/numpy branch of :func:`lee_filter` is exercised both ways: a laptop
baseline and a Colab baseline must agree bit for bit, otherwise "before/after
adaptation" deltas pick up an environment difference.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from satquery.preprocess.sar import (
    DEFAULT_DB_CLIP,
    WATER_DB_THRESHOLD,
    backscatter_stats,
    db_to_uint8,
    lee_filter,
    linear_to_db,
    sar_water_mask,
)

HAS_SCIPY = True
try:  # the fallback exists precisely because this may fail
    import scipy.ndimage  # noqa: F401
except ImportError:  # pragma: no cover - scipy is installed on the dev laptop
    HAS_SCIPY = False


@pytest.fixture
def sar_db(sar_array) -> np.ndarray:
    """The synthetic fixture in dB: 2 bands, dark patch in the top-left third."""
    return linear_to_db(sar_array(bands=2, height=64, width=64))


# --------------------------------------------------------------------------- #
# linear_to_db
# --------------------------------------------------------------------------- #


def test_linear_to_db_known_values():
    linear = np.array([[1.0, 0.1, 0.01, 0.001]], dtype=np.float32)

    db = linear_to_db(linear)

    assert db.dtype == np.float32
    np.testing.assert_allclose(db, [[0.0, -10.0, -20.0, -30.0]], atol=1e-4)


def test_linear_to_db_floors_zero_and_negative_power():
    # Masked borders arrive as exact 0 and mis-calibrated pixels as small
    # negatives; both must land far below any real backscatter, not at -inf.
    db = linear_to_db(np.array([0.0, -0.5, 1e-12], dtype=np.float32), eps=1e-8)

    assert np.all(np.isfinite(db))
    np.testing.assert_allclose(db, [-80.0, -80.0, -80.0], atol=1e-4)


def test_linear_to_db_propagates_nan_and_rejects_bad_eps():
    db = linear_to_db(np.array([np.nan, 1.0], dtype=np.float32))

    assert np.isnan(db[0]) and db[1] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        linear_to_db(np.ones((2, 2), dtype=np.float32), eps=0.0)


def test_linear_to_db_puts_the_fixture_in_sensor_range(sar_array):
    db = linear_to_db(sar_array(bands=2, height=64, width=64))

    assert db.shape == (2, 64, 64)
    # A land scene of gamma-distributed power around 0.2 sits near -7 dB; the
    # water patch drops ~17 dB below it. Anything outside this window means the
    # fixture is no longer linear power (or someone squared it).
    assert -40.0 < float(db.min()) < -18.0
    assert -12.0 < float(db.max()) < 5.0


# --------------------------------------------------------------------------- #
# db_to_uint8
# --------------------------------------------------------------------------- #


def test_db_to_uint8_anchors_the_locked_clip():
    assert DEFAULT_DB_CLIP == (-25.0, 0.0)
    db = np.array([[-25.0, -12.5, 0.0]], dtype=np.float32)

    out = db_to_uint8(db)

    assert out.dtype == np.uint8
    assert out.shape == db.shape
    np.testing.assert_array_equal(out, [[0, 128, 255]])


def test_db_to_uint8_saturates_outside_the_window_and_zeroes_nan():
    db = np.array([-80.0, -25.001, 0.001, 12.0, np.nan], dtype=np.float32)

    out = db_to_uint8(db)

    np.testing.assert_array_equal(out, [0, 0, 255, 255, 0])


def test_db_to_uint8_honours_a_custom_clip_and_rejects_a_bad_one():
    db = np.array([[-30.0, -20.0, -10.0]], dtype=np.float32)

    out = db_to_uint8(db, clip=(-30.0, -10.0))

    np.testing.assert_array_equal(out, [[0, 128, 255]])
    with pytest.raises(ValueError):
        db_to_uint8(db, clip=(0.0, -25.0))


def test_db_to_uint8_is_scene_independent(sar_db):
    # The same physical brightness must map to the same byte whether or not the
    # rest of the scene is bright: that is the whole reason for a fixed clip.
    bright = sar_db.copy()
    bright[:, 40:, 40:] = 0.0

    a = db_to_uint8(sar_db)[0, 5, 5]
    b = db_to_uint8(bright)[0, 5, 5]

    assert a == b  # same dB in, same byte out, whatever surrounds it
    # And the clip floor is absolute, not a percentile of this scene: a pixel
    # below -25 dB is black in the dark scene and in the bright one alike.
    floored = sar_db.copy()
    floored[0, 5, 5] = -40.0
    bright_floored = bright.copy()
    bright_floored[0, 5, 5] = -40.0
    assert db_to_uint8(floored)[0, 5, 5] == db_to_uint8(bright_floored)[0, 5, 5] == 0


# --------------------------------------------------------------------------- #
# lee_filter
# --------------------------------------------------------------------------- #


def test_lee_filter_preserves_shape_and_dtype(sar_db):
    assert lee_filter(sar_db).shape == (2, 64, 64)
    assert lee_filter(sar_db).dtype == np.float32
    assert lee_filter(sar_db[0]).shape == (64, 64)  # 2-D in, 2-D out


def test_lee_filter_smooths_homogeneous_speckle_without_moving_the_mean(rng):
    field = rng.normal(-20.0, 2.0, size=(64, 64)).astype(np.float32)

    out = lee_filter(field, size=5)

    assert out.std() < 0.7 * field.std()
    assert float(out.mean()) == pytest.approx(float(field.mean()), abs=0.05)


def test_lee_filter_leaves_a_constant_region_untouched():
    # Zero local variance means "all noise", so the estimate collapses to the
    # local mean -- which equals the constant. Any drift here is a padding bug.
    flat = np.full((16, 16), -13.5, dtype=np.float32)

    np.testing.assert_allclose(lee_filter(flat), flat, atol=1e-5)


def test_lee_filter_keeps_a_point_target():
    # A ship or corner reflector must survive: a plain 5x5 box mean would spread
    # it to 1/25 of its amplitude, which is the failure Lee exists to avoid.
    scene = np.zeros((32, 32), dtype=np.float32)
    scene[16, 16] = 1.0

    out = lee_filter(scene, size=5)

    assert out[16, 16] > 0.9
    assert out[16, 17] < 0.05


def test_lee_filter_does_not_mix_polarisations():
    field = np.linspace(-25.0, -5.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    stack = np.stack([field, np.full((64, 64), 3.0, dtype=np.float32)])

    out = lee_filter(stack)

    np.testing.assert_allclose(out[1], 3.0, atol=1e-5)
    np.testing.assert_allclose(out[0], lee_filter(field), atol=0)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed; only one branch exists")
def test_lee_filter_numpy_fallback_matches_scipy(monkeypatch, sar_db):
    with_scipy = lee_filter(sar_db, size=5)

    # None in sys.modules makes ``from scipy.ndimage import ...`` raise
    # ImportError, which is exactly the laptop-without-scipy situation.
    monkeypatch.setitem(sys.modules, "scipy.ndimage", None)
    fallback = lee_filter(sar_db, size=5)

    assert fallback.shape == with_scipy.shape
    np.testing.assert_array_equal(fallback, with_scipy)


@pytest.mark.parametrize("size", [0, -3, 2.5])
def test_lee_filter_rejects_a_bad_window(size):
    with pytest.raises(ValueError):
        lee_filter(np.zeros((8, 8), dtype=np.float32), size=size)


def test_lee_filter_rejects_wrong_rank():
    with pytest.raises(ValueError):
        lee_filter(np.zeros((2, 2, 8, 8), dtype=np.float32))


# --------------------------------------------------------------------------- #
# sar_water_mask
# --------------------------------------------------------------------------- #


def test_water_mask_finds_the_dark_patch(sar_db):
    mask = sar_water_mask(sar_db)

    assert mask.dtype == np.bool_
    assert mask.shape == (64, 64)  # bands reduced away
    # The fixture's water is the top-left third; everything else is land.
    assert mask[:21, :21].all()
    assert mask[25:, 25:].mean() < 0.01


def test_water_mask_threshold_is_a_knob():
    db = np.array([[-30.0, -18.0, -17.9, -5.0]], dtype=np.float32)

    default = sar_water_mask(db)
    strict = sar_water_mask(db, threshold=-25.0)

    assert WATER_DB_THRESHOLD == -18.0
    np.testing.assert_array_equal(default, [[True, False, False, False]])
    np.testing.assert_array_equal(strict, [[True, False, False, False]])
    np.testing.assert_array_equal(sar_water_mask(db, threshold=-6.0), [[True, True, True, False]])
    np.testing.assert_array_equal(sar_water_mask(db, threshold=-4.0), [[True, True, True, True]])


def test_water_mask_never_counts_nodata_as_water():
    db = np.array([[np.nan, -30.0, -5.0]], dtype=np.float32)

    np.testing.assert_array_equal(sar_water_mask(db), [[False, True, False]])


def test_water_mask_reduces_bands_by_mean_db():
    # VV dark + VH bright averages above the threshold: a pixel dark in one
    # polarisation only is not called water.
    stack = np.stack(
        [np.full((4, 4), -30.0, dtype=np.float32), np.full((4, 4), -4.0, dtype=np.float32)]
    )

    assert not sar_water_mask(stack).any()
    assert sar_water_mask(stack[:1]).all()


# --------------------------------------------------------------------------- #
# backscatter_stats
# --------------------------------------------------------------------------- #


def test_backscatter_stats_reports_the_documented_keys(sar_db):
    stats = backscatter_stats(sar_db)

    assert set(stats) == {
        "mean", "std", "min", "max", "p5", "p50", "p95",
        "water_pct", "water_threshold", "n_valid",
    }
    assert all(isinstance(v, float) for v in stats.values())  # JSON-serialisable


def test_backscatter_stats_matches_numpy_on_the_reduced_plane(sar_db):
    plane = sar_db.mean(axis=0)

    stats = backscatter_stats(sar_db)

    assert stats["mean"] == pytest.approx(float(plane.mean()), abs=1e-4)
    assert stats["std"] == pytest.approx(float(plane.std()), abs=1e-4)
    assert stats["p50"] == pytest.approx(float(np.percentile(plane, 50)), abs=1e-4)
    assert stats["min"] <= stats["p5"] <= stats["p50"] <= stats["p95"] <= stats["max"]
    assert stats["n_valid"] == 64.0 * 64.0


def test_backscatter_stats_finds_the_water_patch(sar_db):
    stats = backscatter_stats(sar_db)

    # The fixture's dark patch is (64//3)^2 = 441 of 4096 pixels = 10.8%.
    assert stats["water_pct"] == pytest.approx(10.8, abs=1.5)
    assert stats["water_threshold"] == -18.0
    # The patch pulls p5 far below the median: that spread is the evidence.
    assert stats["p5"] < -18.0 < stats["p50"]


def test_backscatter_stats_excludes_nodata_from_the_denominator():
    db = np.full((10, 10), -5.0, dtype=np.float32)
    db[:5, :] = np.nan
    db[5:8, :] = -30.0  # 30 water pixels of 50 valid

    stats = backscatter_stats(db)

    assert stats["n_valid"] == 50.0
    assert stats["water_pct"] == pytest.approx(60.0)
    assert stats["mean"] == pytest.approx(-20.0)


def test_backscatter_stats_on_an_empty_scene_says_nan_not_zero():
    stats = backscatter_stats(np.full((4, 4), np.nan, dtype=np.float32))

    assert stats["n_valid"] == 0.0
    assert stats["water_pct"] == 0.0
    for key in ("mean", "std", "min", "max", "p5", "p50", "p95"):
        assert np.isnan(stats[key]), f"{key} fabricated a value for a no-data scene"


def test_backscatter_stats_is_deterministic(sar_db):
    assert backscatter_stats(sar_db) == backscatter_stats(sar_db)


def test_stats_are_computed_on_unfiltered_data_by_the_caller(sar_db):
    # Not a property of the function but of how it must be used: despeckling
    # shrinks std and the p5-p95 spread, two of the numbers we report. The test
    # pins the size of that bias so nobody "cleans up" the pipeline by filtering
    # before measuring.
    raw = backscatter_stats(sar_db)
    filtered = backscatter_stats(lee_filter(sar_db))

    assert filtered["std"] < raw["std"]
    assert (filtered["p95"] - filtered["p5"]) < (raw["p95"] - raw["p5"])


def test_linear_input_is_flagged_rather_than_silently_scored(sar_array, capture_logs):
    records = capture_logs("satquery.preprocess.sar")
    linear = sar_array(bands=1, height=16, width=16)

    stats = backscatter_stats(linear)

    assert any("linear power" in r.getMessage() for r in records)
    assert stats["water_pct"] == 0.0  # every linear value is above -18
