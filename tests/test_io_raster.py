"""Tests for raster IO across the three formats it has to survive.

The interesting cases are not "does numpy round-trip"; they are the ones where a
missing optional dependency changes the code path: a GeoTIFF written without
rasterio becomes an npy sidecar pair, and every reader downstream has to open it
through the same call with the same result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from satquery.io.raster import (
    HAS_RASTERIO,
    RasterError,
    RasterImage,
    image_size,
    read_raster,
    write_geotiff,
)

TRANSFORM = (10.0, 0.0, 500_000.0, 0.0, -10.0, 2_000_000.0)


def test_write_then_read_preserves_pixels_dtype_and_geo(tmp_path):
    array = np.arange(2 * 4 * 6, dtype=np.uint16).reshape(2, 4, 6)

    path = write_geotiff(tmp_path / "scene.tif", array, crs="EPSG:32644", transform=TRANSFORM)
    image = read_raster(path)

    np.testing.assert_array_equal(image.array, array)
    assert image.array.dtype == np.uint16  # never silently promoted to float
    assert image.crs == "EPSG:32644"
    assert image.transform == pytest.approx(TRANSFORM)
    assert image.gsd == (10.0, 10.0)


def test_a_tif_path_still_opens_when_the_sidecar_holds_the_data(tmp_path):
    """The rasterio-less fallback must be invisible to callers.

    ``write_geotiff`` returns ``scene.tif.npy`` on a box without rasterio, but an
    adapter that recorded ``scene.tif`` earlier has to keep working -- otherwise
    a laptop-written fixture breaks on the machine that reads it.
    """
    array = np.zeros((1, 8, 8), dtype=np.float32)
    written = write_geotiff(tmp_path / "scene.tif", array, crs="EPSG:4326", transform=TRANSFORM)

    if HAS_RASTERIO:
        assert written.name == "scene.tif"
    else:
        assert written.name == "scene.tif.npy"
        assert (tmp_path / "scene.tif.json").exists()
    assert read_raster(tmp_path / "scene.tif").shape == (1, 8, 8)
    assert read_raster(written).shape == (1, 8, 8)


def test_bounds_are_computed_from_the_corners(tmp_path):
    path = write_geotiff(
        tmp_path / "b.tif", np.zeros((1, 10, 20), np.uint8), crs="EPSG:32644", transform=TRANSFORM
    )
    bounds = read_raster(path).bounds
    assert bounds == pytest.approx((500_000.0, 1_999_900.0, 500_200.0, 2_000_000.0))


def test_png_reads_band_first_with_no_georeferencing(tmp_path, write_png, rgb_scene):
    path = write_png(tmp_path / "tile.png", rgb_scene(size=32))

    image = read_raster(path)

    assert image.shape[0] == 3  # (band, h, w), not PIL's (h, w, c)
    assert image.crs is None and image.transform is None
    assert image.georeferenced is False
    assert image.gsd is None


def test_band_selection_uses_zero_based_indices(tmp_path):
    array = np.stack([np.full((4, 4), i, dtype=np.uint8) for i in range(5)])
    path = write_geotiff(tmp_path / "multi.tif", array)

    image = read_raster(path, bands=[0, 3])

    assert image.bands == 2
    assert image.array[0][0, 0] == 0 and image.array[1][0, 0] == 3
    with pytest.raises(RasterError, match="out of range"):
        read_raster(path, bands=[0, 99])


def test_image_size_matches_a_full_read(tmp_path, write_png, rgb_scene):
    png = write_png(tmp_path / "t.png", rgb_scene(size=48))
    tif = write_geotiff(tmp_path / "t.tif", np.zeros((2, 12, 30), np.uint8))

    assert image_size(png) == (48, 48)
    assert image_size(tif) == (30, 12)  # (width, height), the PIL convention
    assert image_size(tif) == (read_raster(tif).width, read_raster(tif).height)


def test_a_missing_file_names_itself(tmp_path):
    with pytest.raises(RasterError, match="not found"):
        read_raster(tmp_path / "nope.tif")


def test_a_corrupt_sidecar_is_an_error_not_a_silent_default(tmp_path):
    np.save(tmp_path / "x.tif.npy", np.zeros((1, 4, 4)))
    (tmp_path / "x.tif.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RasterError, match="corrupt sidecar"):
        read_raster(tmp_path / "x.tif")


def test_two_dimensional_input_is_promoted_on_write(tmp_path):
    path = write_geotiff(tmp_path / "flat.tif", np.zeros((5, 7), np.uint8))
    assert read_raster(path).shape == (1, 5, 7)


def test_boolean_masks_are_written_as_bytes(tmp_path):
    """A mask is written as uint8: no format we target has a 1-bit band type."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[1] = True

    image = read_raster(write_geotiff(tmp_path / "m.tif", mask))

    assert image.array.dtype == np.uint8
    assert image.array[0, 1].tolist() == [1, 1, 1, 1]


def test_rasterimage_rejects_the_wrong_shape():
    with pytest.raises(RasterError, match="band, height, width"):
        RasterImage(array=np.zeros((4, 4)))


def test_describe_is_json_serialisable(tmp_path):
    path = write_geotiff(
        tmp_path / "d.tif", np.zeros((1, 3, 3), np.float32), crs="EPSG:4326", transform=TRANSFORM
    )
    described = read_raster(path).describe()
    json.dumps(described)  # the trace carries this dict; it must not raise
    assert described["bands"] == 1 and described["georeferenced"] is True


def test_rotation_is_reported_rather_than_ignored(tmp_path):
    rotated = (10.0, 2.0, 500_000.0, 1.5, -10.0, 2_000_000.0)
    path = write_geotiff(tmp_path / "r.tif", np.zeros((1, 4, 4), np.uint8), crs="EPSG:32644",
                         transform=rotated)
    image = read_raster(path)
    assert image.rotated is True
    assert image.gsd == (10.0, 10.0)  # still approximated, but the caller can check
