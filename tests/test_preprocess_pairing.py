"""Tests for the pair check and for band-role resolution.

These two decide what the router is even offered, so the assertions are about
*refusals* as much as acceptances: two scenes of different towns must not become
a change pair, and an RGB tile must not be offered NDVI.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.io.raster import RasterImage
from satquery.preprocess.bands import (
    INDEX_REQUIREMENTS,
    available_indices,
    resolve_band_roles,
)
from satquery.preprocess.pairing import (
    MAX_GSD_RATIO,
    MIN_FOOTPRINT_OVERLAP,
    check_pair,
    resample_to_match,
)

UTM = "EPSG:32644"


def geo(
    *,
    size: int = 64,
    bands: int = 3,
    gsd: float = 10.0,
    x: float = 500_000.0,
    y: float = 2_000_000.0,
    crs: str | None = UTM,
) -> RasterImage:
    return RasterImage(
        array=np.zeros((bands, size, size), np.float32),
        crs=crs,
        transform=(gsd, 0.0, x, 0.0, -gsd, y) if crs else None,
        path="scene.tif",
    )


def plain(size: int = 64, bands: int = 3) -> RasterImage:
    """A benchmark PNG: no CRS, no transform, nothing but a pixel grid."""
    return RasterImage(array=np.zeros((bands, size, size), np.uint8), path="tile.png")


# --------------------------------------------------------------------------- #
# geo mode
# --------------------------------------------------------------------------- #


def test_the_same_footprint_is_a_co_registered_bitemporal_pair():
    check = check_pair(geo(), geo(), modality_a="optical", modality_b="optical")

    assert check.ok and check.mode == "geo"
    assert check.pair_type == "bitemporal"
    assert check.co_registered
    assert check.footprint_overlap == pytest.approx(1.0)
    assert check.supports("change_map") and not check.supports("fusion_multilabel")


def test_two_different_places_are_refused_with_the_overlap_attached():
    """The failure mode this whole check exists to prevent."""
    check = check_pair(geo(), geo(x=900_000.0), modality_a="optical", modality_b="optical")

    assert not check.ok
    assert check.pair_type == "none"
    assert check.footprint_overlap == 0.0
    assert any("not the same area" in r for r in check.reasons)
    assert not check.supports("change_map")


def test_partial_overlap_below_the_threshold_is_refused():
    # Shift by 60% of a 640 m footprint: IoU lands under the 0.5 cut.
    check = check_pair(geo(), geo(x=500_000.0 + 0.6 * 640), modality_a="optical", modality_b="optical")

    assert check.footprint_overlap is not None
    assert check.footprint_overlap < MIN_FOOTPRINT_OVERLAP
    assert not check.ok


def test_optical_plus_sar_is_crossmodal_and_only_supports_fusion():
    check = check_pair(geo(), geo(bands=2), modality_a="optical", modality_b="sar")

    assert check.ok and check.pair_type == "crossmodal"
    assert check.supports("fusion_multilabel")
    assert not check.supports("change_map")


def test_a_coarser_sensor_is_flagged_for_resampling_not_refused():
    check = check_pair(
        geo(gsd=1.0, size=640), geo(gsd=3.0, size=214), modality_a="optical", modality_b="sar"
    )

    assert check.ok
    assert check.gsd_ratio == pytest.approx(3.0)
    assert check.gsd_ratio > MAX_GSD_RATIO
    assert check.resample_needed
    assert not check.co_registered  # a per-pixel product needs one grid first
    assert any("ground sample distances differ" in r for r in check.reasons)


def test_different_crs_is_refused_rather_than_reprojected_silently():
    check = check_pair(geo(), geo(crs="EPSG:4326"), modality_a="optical", modality_b="optical")

    assert not check.ok and check.same_crs is False
    assert any("reproject" in r for r in check.reasons)


def test_a_rotated_transform_is_noted():
    rotated = RasterImage(
        array=np.zeros((3, 64, 64), np.float32),
        crs=UTM,
        transform=(10.0, 1.0, 500_000.0, 0.5, -10.0, 2_000_000.0),
    )
    check = check_pair(geo(), rotated, modality_a="optical", modality_b="optical")
    assert any("rotation" in r for r in check.reasons)


# --------------------------------------------------------------------------- #
# grid mode
# --------------------------------------------------------------------------- #


def test_benchmark_pngs_fall_back_to_the_pixel_grid_and_say_so():
    """LEVIR-CD pairs have no CRS; refusing them would refuse the eval set."""
    check = check_pair(plain(), plain(), modality_a="optical", modality_b="optical")

    assert check.ok and check.mode == "grid"
    assert check.co_registered
    assert check.footprint_overlap is None
    assert any("pixel grid alone" in r for r in check.reasons)
    assert any("dataset's claim" in r for r in check.reasons)


def test_grid_mode_refuses_mismatched_shapes():
    check = check_pair(plain(64), plain(96), modality_a="optical", modality_b="optical")

    assert not check.ok and check.resample_needed
    assert any("pixel grids differ" in r for r in check.reasons)


def test_the_message_is_grammatical_for_one_missing_crs():
    check = check_pair(geo(), plain(), modality_a="optical", modality_b="optical")
    assert any(r.startswith("image B carries no CRS") for r in check.reasons)


def test_unknown_modality_yields_no_pair_type():
    check = check_pair(geo(), geo(), modality_a="unknown", modality_b="optical")
    assert check.pair_type == "none"
    assert not check.supports("change_map") and not check.supports("fusion_multilabel")


def test_to_dict_is_flat_and_serialisable():
    import json

    payload = check_pair(geo(), geo(), modality_a="optical", modality_b="optical").to_dict()
    json.dumps(payload)
    assert payload["pair_type"] == "bitemporal" and payload["mode"] == "geo"


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #


def test_resample_to_match_uses_the_reference_grid():
    mask = np.zeros((32, 32), dtype=bool)
    mask[:16] = True

    out = resample_to_match(mask.astype(np.uint8), geo(size=64), order=0)

    assert out.shape == (64, 64)
    assert out[:32].all() and not out[32:].any()  # nearest keeps the hard edge


def test_resample_accepts_a_shape_tuple():
    assert resample_to_match(np.zeros((3, 8, 8)), (16, 20)).shape == (3, 16, 20)


# --------------------------------------------------------------------------- #
# band roles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bands,modality,preset,expected",
    [
        (3, "optical", "rgb", ("VARI",)),
        (4, "optical", "rgbn", ("NDVI", "NDWI", "VARI")),
        (12, "optical", "s2_l2a", ("NDVI", "NDWI", "NDBI", "VARI")),
        (13, "optical", "s2_l1c", ("NDVI", "NDWI", "NDBI", "VARI")),
        (2, "sar", "sar_dual_pol", ("SAR_BACKSCATTER",)),
        (1, "sar", "sar_single_pol", ("SAR_BACKSCATTER",)),
        (1, "optical", "panchromatic", ()),
    ],
)
def test_band_presets_decide_which_indices_exist(bands, modality, preset, expected):
    roles = resolve_band_roles(bands, modality=modality)
    assert roles.preset == preset
    assert available_indices(roles) == expected


def test_an_rgb_tile_is_never_offered_ndvi():
    """The substitution that would make every downstream number unfalsifiable."""
    roles = resolve_band_roles(3, modality="optical")
    assert roles.nir is None
    assert "NDVI" not in available_indices(roles)
    assert "VARI" in available_indices(roles)


def test_l1c_shifts_swir_by_one_band():
    """B10 is present in L1C and gone in L2A; everything past it moves."""
    assert resolve_band_roles(12, modality="optical").swir1 == 10
    assert resolve_band_roles(13, modality="optical").swir1 == 11


def test_an_explicit_mapping_wins_and_is_marked_declared():
    roles = resolve_band_roles(6, modality="optical", override={"red": 4, "nir": 5})

    assert (roles.red, roles.nir) == (4, 5)
    assert roles.source == "declared" and roles.preset == "explicit"
    assert "NDVI" in available_indices(roles)


def test_an_out_of_range_override_is_dropped_not_clamped():
    """A clamped band index is a wrong number that still computes."""
    roles = resolve_band_roles(3, modality="optical", override={"nir": 9})
    assert roles.nir is None


def test_every_index_declares_its_requirements():
    for name, needed in INDEX_REQUIREMENTS.items():
        assert needed, f"{name} declares no band requirement"
        roles = resolve_band_roles(12, modality="optical" if name != "SAR_BACKSCATTER" else "sar")
        assert roles.has(*needed) or name == "SAR_BACKSCATTER"
