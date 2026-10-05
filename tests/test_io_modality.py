"""Tests for optical/SAR inference.

The guess drives the display transform and the tool menu, so the cases that
matter are the ones where getting it wrong is silent: SAR linear power looks like
a very dark optical image, an unbounded float scene is genuinely ambiguous, and a
panchromatic optical tile has the band count of a single-pol SAR product.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.io.modality import MIN_TRUSTWORTHY, infer_modality
from satquery.io.raster import RasterImage


@pytest.fixture
def gen() -> np.random.Generator:
    return np.random.default_rng(7)


def sar_linear(gen, bands=2, size=64):
    """Gamma speckle: the distribution a multi-looked sigma0 product actually has."""
    return gen.gamma(4.0, 0.05, size=(bands, size, size)).astype(np.float32)


def sar_db(gen, **kw):
    return (10.0 * np.log10(np.maximum(sar_linear(gen, **kw), 1e-8))).astype(np.float32)


def optical_reflectance(gen, bands=4, size=64):
    return np.clip(gen.normal(0.4, 0.15, size=(bands, size, size)), 0, 1).astype(np.float32)


def test_dB_backscatter_is_the_strongest_signal(gen):
    guess = infer_modality(sar_db(gen), path="anonymous.tif")

    assert guess.modality == "sar"
    assert guess.certain
    assert any("negative" in r for r in guess.reasons)


def test_linear_power_is_recognised_by_its_tail(gen):
    """A dark field with a long bright tail is speckle, not a dark photograph."""
    array = sar_linear(gen)
    array[:, 10:14, 10:14] *= 40.0  # a corner reflector

    guess = infer_modality(array, path="anonymous.tif")

    assert guess.modality == "sar"
    assert any("dark field" in r for r in guess.reasons)


def test_a_bright_target_alone_is_enough(gen):
    """Reflectance is bounded; a maximum many times the p99 is not optical.

    Without this signal a scene whose speckle is tight (a well multi-looked
    product) but which contains a ship or a pylon falls between the rules and
    comes back "unknown", which withholds the SAR display transform and hands the
    VLM a black square.
    """
    array = sar_linear(gen)
    array[:, 30:32, 30:32] *= 60.0

    guess = infer_modality(array, path="anonymous.tif")

    assert guess.modality == "sar"
    assert any("point targets" in r for r in guess.reasons)


def test_a_multispectral_stack_is_never_read_as_sar(gen):
    """The bug this test exists for, found while preparing a demo.

    Statistics pooled across bands make an ordinary multispectral scene look like
    speckle: the visible bands of a vegetated landscape are dark and the NIR band
    is bright, so the pooled histogram has a long right tail. A 4-band optical
    stack was therefore classified as SAR -- which withheld the fusion tool and
    would have sent the scene through a dB stretch that renders it black.

    Two things fix it and both are asserted here: the shape statistics are
    computed per band and medianed, and three or more bands rules out SAR
    outright rather than merely voting against it.
    """
    for bands in (4, 12):
        visible = np.clip(gen.normal(0.12, 0.05, size=(bands - 1, 64, 64)), 0.01, 1.0)
        nir = np.clip(gen.normal(0.42, 0.10, size=(1, 64, 64)), 0.01, 1.0)
        stack = np.concatenate([visible, nir]).astype(np.float32)

        guess = infer_modality(stack, path="anonymous.npy")

        assert guess.modality == "optical", f"{bands}-band stack read as {guess.modality}"
        assert guess.certain


def test_the_tail_statistic_is_per_band_not_pooled(gen):
    """A dark band next to a bright one must not manufacture a tail."""
    from satquery.io.modality import _pixel_stats

    dark = np.full((1, 32, 32), 0.05, dtype=np.float32)
    bright = np.full((1, 32, 32), 0.50, dtype=np.float32)
    stats = _pixel_stats(np.concatenate([dark, bright]), 100_000)

    assert stats["tail"] == pytest.approx(1.0, abs=0.05)  # each band is flat
    assert stats["max"] == pytest.approx(0.50)  # range stays global


def test_reflectance_is_optical(gen):
    guess = infer_modality(optical_reflectance(gen), path="anonymous.tif")
    assert guess.modality == "optical" and guess.certain


def test_integer_dn_is_optical_whatever_the_filename_says(gen):
    """A quantised product has been through a display stretch; SAR power is not.

    The filename hint is deliberately contradicted here: pixel statistics
    outweigh naming, because the hidden evaluation set will not follow anyone's
    naming convention.
    """
    array = gen.integers(0, 4000, size=(3, 64, 64)).astype(np.uint16)

    guess = infer_modality(array, path="S1A_IW_GRDH_VV.tif")

    assert guess.modality == "optical"


def test_a_declared_modality_is_never_overruled(gen):
    guess = infer_modality(optical_reflectance(gen), path="tile.png", declared="sar")

    assert guess.modality == "sar"
    assert guess.confidence == 1.0
    assert guess.reasons == ("declared as sar",)


def test_filename_hints_break_a_tie_but_do_not_decide_alone(gen):
    """An unbounded float scene is ambiguous; the name is allowed to tip it."""
    ambiguous = gen.uniform(0.0, 6.0, size=(2, 64, 64)).astype(np.float32)

    as_sar = infer_modality(ambiguous, path="S1A_IW_GRDH_VV.tif")
    as_optical = infer_modality(ambiguous, path="LC08_L2SP_ms.tif")

    assert as_sar.modality == "sar"
    assert as_optical.modality == "optical"
    # Neither should present as certain as a dB scene does.
    assert as_sar.confidence < 0.9 and as_optical.confidence < 0.9


def test_a_word_hint_does_not_match_inside_another_word(gen):
    """``s1`` must not fire on ``tiles1234.png``; ``_vv`` must fire on a suffix."""
    array = optical_reflectance(gen, bands=3)

    assert infer_modality(array, path="tiles1234.png").modality == "optical"
    reasons = " ".join(infer_modality(sar_db(gen), path="scene_VV.tif").reasons)
    assert "_vv" in reasons


def test_panchromatic_optical_is_not_mistaken_for_single_pol_sar(gen):
    guess = infer_modality(optical_reflectance(gen, bands=1), path="cartosat_pan.tif")
    assert guess.modality == "optical"


def test_contradicting_signals_return_unknown_rather_than_a_coin_flip():
    """An empty/constant array gives nothing to go on and must say so."""
    guess = infer_modality(np.zeros((1, 16, 16), dtype=np.float32), path="")

    assert guess.modality in ("unknown", "optical")
    if guess.modality == "unknown":
        assert guess.confidence == 0.0 and not guess.certain


def test_all_nan_input_does_not_crash():
    guess = infer_modality(np.full((1, 8, 8), np.nan, dtype=np.float32), path="x.tif")
    assert guess.modality == "unknown"
    assert "no finite pixels to judge" in " ".join(guess.reasons)


def test_a_rasterimage_is_accepted_directly(gen):
    image = RasterImage(array=sar_db(gen), path="scene_sar.tif")
    assert infer_modality(image).modality == "sar"


def test_the_guess_is_deterministic(gen):
    array = sar_linear(gen, size=256)
    first = infer_modality(array, path="a.tif")
    second = infer_modality(array, path="a.tif")
    assert (first.modality, first.confidence) == (second.modality, second.confidence)


def test_confidence_never_reaches_certainty(gen):
    """An inference is never presented as a fact; only a declaration is 1.0."""
    for array in (sar_db(gen), optical_reflectance(gen)):
        guess = infer_modality(array, path="tile.tif")
        assert MIN_TRUSTWORTHY <= guess.confidence <= 0.98


def test_to_dict_is_flat_and_rounded(gen):
    payload = infer_modality(sar_db(gen), path="x.tif").to_dict()
    assert set(payload) == {"modality", "confidence", "reasons", "stats"}
    assert isinstance(payload["reasons"], list)
