"""Tests for the sensor-agnostic optical display path.

The point of these is not that the arithmetic is right -- a linear stretch is
three lines -- but that the *failure modes* stay loud: a nodata sentinel must
not drag a percentile, a flat band must not become NaN or a black rectangle,
and a band preset must never quietly pick the wrong triple.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.contracts import SatQueryError
from satquery.preprocess.optical import (
    BAND_PRESETS,
    S2_L2A_BAND_NAMES,
    percentile_stretch,
    select_composite,
    to_uint8,
)


def _band_indexed(bands: int = 12, height: int = 4, width: int = 5) -> np.ndarray:
    """Stack whose band ``i`` is filled with the constant ``i``.

    Makes a composite self-identifying: the value of an output channel *is* the
    source band index, so a preset can be asserted against without hand-reading
    slices.
    """
    return np.stack(
        [np.full((height, width), float(i), dtype=np.float32) for i in range(bands)]
    )


# --------------------------------------------------------------------------- #
# percentile_stretch
# --------------------------------------------------------------------------- #


def test_stretch_maps_into_out_range(optical_array):
    out = percentile_stretch(optical_array(bands=4, height=32, width=32))

    assert out.dtype == np.float32
    assert out.shape == (4, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # 2-98% anchors saturate both tails, so the extremes are actually reached.
    assert out.min() == pytest.approx(0.0, abs=1e-6)
    assert out.max() == pytest.approx(1.0, abs=1e-6)


def test_stretch_honours_out_range_and_2d_input():
    plane = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)

    out = percentile_stretch(plane, low=0.0, high=100.0, out_range=(0.0, 255.0))

    assert out.shape == (10, 10)  # 2-D in, 2-D out; no band axis invented
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(255.0)


def test_stretch_is_robust_to_a_single_outlier():
    """The whole reason for percentiles rather than min/max.

    One saturated pixel must not collapse the rest of the scene into the bottom
    of the range; the untouched pixels must land where they landed before.
    """
    base = np.linspace(0.1, 0.4, 2500, dtype=np.float32).reshape(1, 50, 50)
    spiked = base.copy()
    spiked[0, 0, 0] = 10_000.0

    clean = percentile_stretch(base)
    dirty = percentile_stretch(spiked)

    interior = (slice(None), slice(1, None), slice(None))
    assert np.allclose(clean[interior], dirty[interior], atol=2e-3)
    assert dirty[0, 0, 0] == pytest.approx(1.0)  # the outlier itself clips


def test_stretch_min_max_would_have_collapsed_under_the_same_outlier():
    """Guard the claim above: low=0/high=100 really is the bad behaviour."""
    base = np.linspace(0.1, 0.4, 2500, dtype=np.float32).reshape(1, 50, 50)
    spiked = base.copy()
    spiked[0, 0, 0] = 10_000.0

    minmax = percentile_stretch(spiked, low=0.0, high=100.0)

    assert minmax[0, 1:, :].max() < 1e-3


def test_constant_band_maps_to_midrange_not_nan():
    array = np.stack(
        [
            np.full((8, 8), 0.7, dtype=np.float32),  # constant
            np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8),
        ]
    )

    out = percentile_stretch(array)

    assert np.all(np.isfinite(out[0]))
    assert np.allclose(out[0], 0.5)
    assert out[1].max() == pytest.approx(1.0)


def test_constant_band_midpoint_follows_out_range():
    out = percentile_stretch(np.full((1, 4, 4), 3.0, dtype=np.float32), out_range=(0.0, 255.0))

    assert np.allclose(out, 127.5)


def test_nan_pixels_stay_nan_and_do_not_move_the_anchors():
    clean = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(1, 10, 10)
    holed = clean.copy()
    holed[0, 5, 5] = np.nan

    out = percentile_stretch(holed)

    assert np.isnan(out[0, 5, 5])
    assert np.isfinite(out[np.isfinite(holed)]).all()
    assert out[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[0, 9, 9] == pytest.approx(1.0, abs=1e-6)


def test_nodata_sentinel_is_excluded_from_the_percentiles():
    """A -9999 border is the classic way to render a scene as one flat block."""
    data = np.linspace(0.2, 0.8, 96, dtype=np.float32)
    array = np.concatenate([np.full(4, -9999.0, dtype=np.float32), data]).reshape(1, 10, 10)

    out = percentile_stretch(array, nodata=-9999.0)

    assert np.isnan(out[0, 0, :4]).all()
    valid = out[np.isfinite(out)]
    assert valid.min() == pytest.approx(0.0, abs=1e-6)
    assert valid.max() == pytest.approx(1.0, abs=1e-6)
    # Without the sentinel declared, everything real would sit at the top.
    naive = percentile_stretch(array)
    assert naive[0, 5, 5] > 0.99


def test_infinities_are_treated_as_nodata():
    array = np.array([[[0.0, 1.0], [np.inf, -np.inf]]], dtype=np.float32)

    out = percentile_stretch(array)

    assert np.isnan(out[0, 1, 0]) and np.isnan(out[0, 1, 1])
    assert np.isfinite(out[0, 0, :]).all()


def test_all_invalid_band_does_not_warn_or_produce_garbage(recwarn):
    array = np.full((1, 4, 4), np.nan, dtype=np.float32)

    out = percentile_stretch(array)

    assert np.isnan(out).all()  # nothing valid in, nothing valid out
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_per_band_false_shares_one_pair_of_anchors():
    dim = np.full((4, 4), 0.1, dtype=np.float32)
    bright = np.full((4, 4), 0.9, dtype=np.float32)
    array = np.stack([dim, bright])

    shared = percentile_stretch(array, per_band=False)
    independent = percentile_stretch(array, per_band=True)

    # Shared anchors preserve which band is brighter...
    assert shared[0].mean() < shared[1].mean()
    # ...while per-band treats each flat band as contrast-free.
    assert np.allclose(independent, 0.5)


def test_stretch_does_not_mutate_its_input():
    array = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    before = array.copy()

    percentile_stretch(array, nodata=1.0)

    assert np.array_equal(array, before)


def test_stretch_accepts_integer_dn_input():
    """11-bit Cartosat-style DN must not need a cast at the call site."""
    dn = np.arange(0, 2048, 8, dtype=np.uint16).reshape(1, 16, 16)

    out = percentile_stretch(dn)

    assert out.dtype == np.float32
    assert out.min() == pytest.approx(0.0, abs=1e-6)
    assert out.max() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low": 98.0, "high": 2.0},
        {"low": -1.0, "high": 50.0},
        {"low": 0.0, "high": 101.0},
        {"out_range": (1.0, 1.0)},
    ],
)
def test_stretch_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        percentile_stretch(np.zeros((1, 4, 4), dtype=np.float32), **kwargs)


def test_stretch_rejects_4d_input():
    with pytest.raises(ValueError):
        percentile_stretch(np.zeros((2, 3, 4, 5), dtype=np.float32))


# --------------------------------------------------------------------------- #
# to_uint8
# --------------------------------------------------------------------------- #


def test_to_uint8_scales_and_rounds():
    array = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

    out = to_uint8(array)

    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 128, 255]]]


def test_to_uint8_clips_rather_than_rescales():
    """Rescaling here would hide a skipped stretch; clipping makes it visible."""
    array = np.array([[[-0.5, 0.25, 4.0, 10_000.0]]], dtype=np.float32)

    out = to_uint8(array)

    assert out.tolist() == [[[0, 64, 255, 255]]]


def test_to_uint8_renders_every_non_finite_value_as_black():
    """Including +inf: non-finite means "no data", not "very bright".

    Consistent with :func:`percentile_stretch`, which folds infinities into NaN
    rather than clipping them to the top of the range.
    """
    out = to_uint8(np.array([[[np.nan, np.inf, -np.inf, 1.0]]], dtype=np.float32))

    assert out.tolist() == [[[0, 0, 0, 255]]]


def test_to_uint8_is_idempotent_on_uint8():
    already = np.array([[[0, 7, 200]]], dtype=np.uint8)

    out = to_uint8(already)

    assert out.dtype == np.uint8
    assert np.array_equal(out, already)
    assert out is not already


# --------------------------------------------------------------------------- #
# Band presets and composites
# --------------------------------------------------------------------------- #


def test_required_presets_exist_with_documented_indices():
    """These four constants are the highest-risk numbers in the module."""
    assert BAND_PRESETS["rgb"] == (0, 1, 2)
    assert BAND_PRESETS["s2_rgb"] == (3, 2, 1)  # B04, B03, B02
    assert BAND_PRESETS["s2_false_color"] == (7, 3, 2)  # B08, B04, B03
    assert BAND_PRESETS["bigearthnet_rgb"] == BAND_PRESETS["s2_rgb"]


def test_s2_presets_agree_with_the_band_name_table():
    """The preset tuples and the 12-band L2A name table must not drift apart."""
    assert len(S2_L2A_BAND_NAMES) == 12
    assert "B10" not in S2_L2A_BAND_NAMES  # L2A drops the cirrus band
    names = list(S2_L2A_BAND_NAMES)

    assert [names[i] for i in BAND_PRESETS["s2_rgb"]] == ["B04", "B03", "B02"]
    assert [names[i] for i in BAND_PRESETS["s2_false_color"]] == ["B08", "B04", "B03"]
    assert [names[i] for i in BAND_PRESETS["s2_swir"]] == ["B12", "B8A", "B04"]


def test_every_preset_is_one_or_three_bands_and_non_negative():
    for name, idx in BAND_PRESETS.items():
        assert len(idx) in (1, 3), f"{name} selects {len(idx)} bands"
        assert all(isinstance(i, int) and i >= 0 for i in idx), name


def test_select_composite_uses_the_preset_indices():
    stack = _band_indexed(12)

    out = select_composite(stack, preset="s2_false_color")

    assert out.shape == (3, 4, 5)
    assert [float(out[c, 0, 0]) for c in range(3)] == [7.0, 3.0, 2.0]


def test_select_composite_explicit_bands_win_over_the_default():
    stack = _band_indexed(6)

    out = select_composite(stack, bands=(5, 0, 2))

    assert [float(out[c, 0, 0]) for c in range(3)] == [5.0, 0.0, 2.0]


def test_select_composite_defaults_to_first_three_bands():
    out = select_composite(_band_indexed(8))

    assert [float(out[c, 0, 0]) for c in range(3)] == [0.0, 1.0, 2.0]


def test_select_composite_does_not_copy_values():
    stack = _band_indexed(4)

    out = select_composite(stack, preset="rgb")

    assert out.dtype == stack.dtype
    assert np.array_equal(out, stack[:3])


def test_single_band_input_is_replicated_to_three_channels():
    pan = np.arange(16, dtype=np.float32).reshape(1, 4, 4)

    out = select_composite(pan, preset="s2_rgb")  # preset ignored, not an error

    assert out.shape == (3, 4, 4)
    assert np.array_equal(out[0], out[1]) and np.array_equal(out[1], out[2])
    assert np.array_equal(out[0], pan[0])


def test_two_dimensional_input_is_treated_as_one_band():
    out = select_composite(np.arange(9, dtype=np.float32).reshape(3, 3))

    assert out.shape == (3, 3, 3)


def test_gray_preset_replicates_a_single_named_band():
    out = select_composite(_band_indexed(4), preset="gray")

    assert out.shape == (3, 4, 5)
    assert np.allclose(out, 0.0)


def test_out_of_range_index_raises_with_the_band_count_in_the_message():
    stack = _band_indexed(4)

    with pytest.raises(SatQueryError) as exc:
        select_composite(stack, preset="s2_swir")  # wants band 11

    msg = str(exc.value)
    assert "11" in msg and "4 band" in msg
    assert "rgb" in msg  # the actionable hint for an already-RGB stack


def test_negative_index_is_rejected_not_wrapped():
    with pytest.raises(SatQueryError):
        select_composite(_band_indexed(12), bands=(-1, 0, 1))


def test_unknown_preset_lists_the_known_ones():
    with pytest.raises(SatQueryError) as exc:
        select_composite(_band_indexed(12), preset="sentinel2_rgb")

    assert "s2_rgb" in str(exc.value)


def test_bands_and_preset_together_is_a_config_error():
    with pytest.raises(SatQueryError):
        select_composite(_band_indexed(12), bands=(0, 1, 2), preset="s2_rgb")


def test_wrong_band_count_is_rejected():
    with pytest.raises(SatQueryError):
        select_composite(_band_indexed(12), bands=(0, 1))


# --------------------------------------------------------------------------- #
# The path an adapter actually walks
# --------------------------------------------------------------------------- #


def test_full_display_path_from_multispectral_dn_to_uint8_rgb(optical_array):
    """stretch -> composite -> uint8, the preview endpoint's whole job."""
    stack = (optical_array(bands=12, height=24, width=24) * 10_000).astype(np.uint16)

    rgb = to_uint8(percentile_stretch(select_composite(stack, preset="s2_rgb")))

    assert rgb.shape == (3, 24, 24)
    assert rgb.dtype == np.uint8
    assert rgb.min() == 0 and rgb.max() == 255


def test_stretch_before_or_after_selection_agrees_per_band():
    """Documented property: per-band stretching commutes with band selection."""
    stack = np.stack(
        [np.linspace(i, i + 3, 64, dtype=np.float32).reshape(8, 8) for i in range(6)]
    )

    a = select_composite(percentile_stretch(stack), preset="rgb")
    b = percentile_stretch(select_composite(stack, preset="rgb"))

    assert np.allclose(a, b)
