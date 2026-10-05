"""Reading and writing rasters, with rasterio as an extra rather than a given.

Three formats reach this module and they are not equivalent:

* **GeoTIFF** -- the real input. Carries CRS, an affine transform, a nodata
  value and often more than three bands. Only rasterio reads all of that.
* **PNG / JPEG** -- "benchmark mode". Every RS-VQA benchmark ships 8-bit RGB
  tiles with no georeferencing at all, so the harness must run on a machine
  where rasterio was never installed. PIL covers this.
* **``.npy`` + JSON sidecar** -- the documented fallback :func:`write_geotiff`
  uses when rasterio is absent. It exists so the *geo* code paths (pair checks,
  GSD comparison, co-registration) stay testable on the dev laptop, where
  building rasterio against GDAL is the single most likely install failure.

Two conventions the rest of the codebase depends on:

* arrays are ``(band, height, width)``, band-first, like rasterio -- never
  ``(h, w, c)`` like PIL. The transpose happens here, once.
* the array dtype is whatever the file holds. uint16 stays uint16; scaling to
  reflectance or to 8-bit is :mod:`satquery.preprocess`'s job, and doing it here
  would apply it twice for anyone who asked for raw pixels.

:class:`RasterImage` is deliberately not a lazy handle. Everything downstream
wants the whole array (a 512-px VLM input, an NDVI map, a change mask), and a
handle that can be closed underneath a caller is a class of bug not worth buying
for a benchmark tile.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import SatQueryError
from ..log import get_logger

__all__ = [
    "HAS_RASTERIO",
    "RasterImage",
    "RasterError",
    "read_raster",
    "image_size",
    "write_geotiff",
    "SIDECAR_DRIVER",
]

_LOG = get_logger(__name__)

#: ``driver`` value on an image loaded from the npy+JSON fallback. Anything that
#: reports provenance (the trace, the validator) prints it, so a reviewer can
#: tell a real GeoTIFF read from a laptop stand-in.
SIDECAR_DRIVER = "npy-sidecar"

_PIL_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
_TIFF_SUFFIXES = frozenset({".tif", ".tiff"})


def _rasterio() -> Any | None:
    try:
        import rasterio  # type: ignore
    except ImportError:
        return None
    return rasterio


HAS_RASTERIO: bool = _rasterio() is not None


class RasterError(SatQueryError):
    """A raster could not be read or written."""


@dataclass(frozen=True, slots=True)
class RasterImage:
    """One raster in memory, plus the georeferencing if the file had any.

    ``transform`` is the six-element affine in rasterio's ``(a, b, c, d, e, f)``
    order: ``a`` is the pixel width in CRS units, ``e`` the (negative) pixel
    height, ``c``/``f`` the top-left corner. It is stored as a plain tuple rather
    than an ``Affine`` so this dataclass stays JSON-serialisable and importable
    without rasterio -- the execution trace has to carry it.

    ``crs`` is a string ("EPSG:32644"), again to keep the trace printable. None
    means the file was not georeferenced, which is normal in benchmark mode and
    is *not* an error: it only removes the pair checks that need coordinates.
    """

    array: np.ndarray
    path: str = ""
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    nodata: float | None = None
    driver: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise RasterError(
                f"RasterImage.array must be (band, height, width), got shape {self.array.shape}"
            )

    # -- shape --------------------------------------------------------------- #

    @property
    def bands(self) -> int:
        return int(self.array.shape[0])

    @property
    def height(self) -> int:
        return int(self.array.shape[1])

    @property
    def width(self) -> int:
        return int(self.array.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.bands, self.height, self.width)

    @property
    def georeferenced(self) -> bool:
        return self.transform is not None and self.crs is not None

    # -- geometry ------------------------------------------------------------ #

    @property
    def gsd(self) -> tuple[float, float] | None:
        """``(x, y)`` ground sample distance in CRS units, or None.

        Taken as ``|a|`` and ``|e|`` from the transform, which is exact for the
        north-up rasters every product we handle ships. A rotated transform
        makes it an approximation; :attr:`rotated` says when that applies so a
        caller can refuse rather than quietly compare the wrong numbers.
        """
        if self.transform is None:
            return None
        a, _b, _c, _d, e, _f = self.transform
        return (abs(float(a)), abs(float(e)))

    @property
    def rotated(self) -> bool:
        """True when the transform has non-zero rotation/skew terms."""
        if self.transform is None:
            return False
        _a, b, _c, d, _e, _f = self.transform
        return abs(float(b)) > 1e-9 or abs(float(d)) > 1e-9

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """``(left, bottom, right, top)`` in CRS units, or None if ungeoreferenced.

        Computed from all four corners rather than from ``c``/``f`` plus size, so
        a transform with rotation terms still yields the correct envelope.
        """
        if self.transform is None:
            return None
        a, b, c, d, e, f = self.transform
        xs: list[float] = []
        ys: list[float] = []
        for col, row in ((0, 0), (self.width, 0), (0, self.height), (self.width, self.height)):
            xs.append(a * col + b * row + c)
            ys.append(d * col + e * row + f)
        return (min(xs), min(ys), max(xs), max(ys))

    def band(self, index: int) -> np.ndarray:
        return np.asarray(self.array[index])

    def describe(self) -> dict[str, Any]:
        """One flat dict for the execution trace. No numpy types, all JSON-safe."""
        return {
            "path": self.path,
            "driver": self.driver,
            "bands": self.bands,
            "height": self.height,
            "width": self.width,
            "dtype": str(self.array.dtype),
            "crs": self.crs,
            "transform": list(self.transform) if self.transform else None,
            "gsd": list(self.gsd) if self.gsd else None,
            "nodata": self.nodata,
            "georeferenced": self.georeferenced,
        }


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_raster(path: str | Path, *, bands: Sequence[int] | None = None) -> RasterImage:
    """Read any supported raster to ``(band, h, w)``, georeferencing included.

    ``bands`` selects 0-based band indices (rasterio's own indices are 1-based;
    the conversion happens here so no caller has to remember which convention it
    is holding).

    Resolution order is deliberate: a real GeoTIFF read wins, the npy sidecar is
    tried next so a laptop can still exercise the geo paths, and PIL is the last
    resort for benchmark tiles. A ``.tif`` on a machine without rasterio
    therefore still opens through PIL -- with a warning, because the CRS and
    transform are silently gone and a pair check that needs them will refuse.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy" or _sidecar_for(p) is not None:
        return _read_sidecar(p, bands)
    if suffix in _TIFF_SUFFIXES:
        rio = _rasterio()
        if rio is not None:
            return _read_rasterio(rio, p, bands)
        _LOG.warning(
            "reading %s without rasterio: CRS, transform and nodata are unavailable, "
            'so geo-dependent checks will be skipped (pip install "satquery[raster]")',
            p.name,
        )
        return _read_pil(p, bands)
    if suffix in _PIL_SUFFIXES:
        return _read_pil(p, bands)

    rio = _rasterio()
    if rio is not None:  # unknown suffix: GDAL sniffs the content, so let it try
        return _read_rasterio(rio, p, bands)
    return _read_pil(p, bands)


def image_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` without decoding the pixels where the format allows.

    Used by the validator and by dataset adapters to fill in
    :class:`~satquery.contracts.ImageRef` dimensions for thousands of samples;
    decoding every JPEG to learn its size is the difference between a listing
    that takes a second and one that takes a minute.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    sidecar = _sidecar_for(p)
    if suffix == ".npy" or sidecar is not None:
        array_path = p if suffix == ".npy" else Path(str(p) + ".npy")
        shape = _npy_shape(array_path)
        if len(shape) < 2:
            raise RasterError(f"sidecar array must be 2-D or 3-D, got shape {shape} in {array_path}")
        return (int(shape[-1]), int(shape[-2]))
    if suffix in _TIFF_SUFFIXES and HAS_RASTERIO:
        rio = _rasterio()
        assert rio is not None
        with rio.open(p) as src:
            return (int(src.width), int(src.height))
    try:
        from PIL import Image  # optional extra

        with Image.open(p) as img:
            return (int(img.width), int(img.height))
    except ImportError:
        pass
    image = read_raster(p)
    return (image.width, image.height)


def _read_rasterio(rio: Any, path: Path, bands: Sequence[int] | None) -> RasterImage:
    if not path.exists():
        raise RasterError(f"raster not found: {path}")
    try:
        with rio.open(path) as src:
            indexes = None if bands is None else [int(b) + 1 for b in bands]
            array = src.read() if indexes is None else src.read(indexes)
            transform = None
            if src.transform is not None:
                # to_gdal() is (c, a, b, f, d, e); reorder to (a, b, c, d, e, f).
                c, a, b, f, d, e = (float(v) for v in src.transform.to_gdal())
                transform = (a, b, c, d, e, f)
            return RasterImage(
                array=np.asarray(array),
                path=str(path),
                crs=str(src.crs) if src.crs else None,
                transform=transform,
                nodata=None if src.nodata is None else float(src.nodata),
                driver=str(src.driver),
                meta={"count": int(src.count), "descriptions": list(src.descriptions or ())},
            )
    except RasterError:
        raise
    except Exception as exc:  # rasterio raises its own exception hierarchy
        raise RasterError(f"rasterio could not read {path}: {exc}") from exc


def _read_pil(path: Path, bands: Sequence[int] | None) -> RasterImage:
    if not path.exists():
        raise RasterError(f"raster not found: {path}")
    try:
        from PIL import Image  # optional extra
    except ImportError as exc:  # pragma: no cover - pillow ships in the data extra
        raise RasterError(
            f"reading {path.name} needs Pillow or rasterio; "
            'pip install "satquery[data]" (or "satquery[raster]" for GeoTIFF)'
        ) from exc
    try:
        with Image.open(path) as img:
            img.load()
            array = np.asarray(img)
            n_frames = int(getattr(img, "n_frames", 1))
            mode = str(img.mode)
    except Exception as exc:
        raise RasterError(f"PIL could not read {path}: {exc}") from exc

    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise RasterError(f"unsupported array shape {array.shape} from {path}")
    array = _select(array, bands, path)
    return RasterImage(
        array=np.ascontiguousarray(array),
        path=str(path),
        crs=None,
        transform=None,
        nodata=None,
        driver="PIL",
        meta={"mode": mode, "n_frames": n_frames},
    )


def _read_sidecar(path: Path, bands: Sequence[int] | None) -> RasterImage:
    """Read the ``.npy`` + JSON pair :func:`write_geotiff` writes without rasterio."""
    array_path = path if path.suffix.lower() == ".npy" else Path(str(path) + ".npy")
    if not array_path.exists():
        raise RasterError(f"raster not found: {array_path}")
    array = np.load(array_path)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise RasterError(
            f"sidecar array must be 2-D or 3-D, got shape {array.shape} in {array_path}"
        )
    array = _select(array, bands, array_path)

    meta: dict[str, Any] = {}
    json_path = array_path.with_suffix(".json")
    if json_path.exists():
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RasterError(f"corrupt sidecar {json_path}: {exc}") from exc
    transform = meta.get("transform")
    return RasterImage(
        array=np.ascontiguousarray(array),
        path=str(path),
        crs=meta.get("crs"),
        transform=tuple(float(v) for v in transform) if transform else None,  # type: ignore[arg-type]
        nodata=meta.get("nodata"),
        driver=SIDECAR_DRIVER,
        meta={k: v for k, v in meta.items() if k not in ("crs", "transform", "nodata")},
    )


def _select(array: np.ndarray, bands: Sequence[int] | None, path: Path) -> np.ndarray:
    if bands is None:
        return array
    idx = [int(b) for b in bands]
    if any(b < 0 or b >= array.shape[0] for b in idx):
        raise RasterError(f"band selection {idx} out of range for {array.shape[0]} bands in {path}")
    return array[idx]


def _sidecar_for(path: Path) -> Path | None:
    """The ``<path>.npy`` companion, when the given path itself does not exist.

    This is what lets ``write_geotiff("scene.tif", ...)`` on a rasterio-less box
    return ``scene.tif.npy`` and still have ``read_raster("scene.tif")`` work --
    an adapter can hold one path either way.
    """
    if path.suffix.lower() == ".npy":
        return path if path.exists() else None
    if path.exists():
        return None
    candidate = Path(str(path) + ".npy")
    return candidate if candidate.exists() else None


def _npy_shape(path: Path) -> tuple[int, ...]:
    if not path.exists():
        raise RasterError(f"raster not found: {path}")
    return tuple(int(v) for v in np.load(path, mmap_mode="r").shape)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    *,
    crs: str | None = None,
    transform: Sequence[float] | None = None,
    nodata: float | None = None,
    **profile: Any,
) -> Path:
    """Write ``(band, h, w)`` (or ``(h, w)``) and return the path actually written.

    With rasterio this is a GeoTIFF at ``path``. Without it, the array goes to
    ``<path>.npy`` and its georeferencing to ``<path>.json``, and *that* path is
    returned -- writing npy bytes into a file named ``.tif`` would be a lie the
    next tool to open it pays for. :func:`read_raster` accepts either, so callers
    hold whatever came back and do not branch on the environment.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None, ...]
    if a.ndim != 3:
        raise RasterError(f"expected (h, w) or (band, h, w), got shape {a.shape}")
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)

    rio = _rasterio()
    if rio is None:
        array_path = p if p.suffix.lower() == ".npy" else Path(str(p) + ".npy")
        np.save(array_path, a)
        sidecar = {
            "crs": crs,
            "transform": [float(v) for v in transform] if transform else None,
            "nodata": nodata,
            "driver": SIDECAR_DRIVER,
            "note": "written without rasterio; see satquery.io.raster",
        }
        array_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return array_path

    from rasterio.transform import Affine  # type: ignore

    affine = Affine(*[float(v) for v in transform]) if transform else Affine.identity()
    kwargs: dict[str, Any] = {
        "driver": "GTiff",
        "height": int(a.shape[1]),
        "width": int(a.shape[2]),
        "count": int(a.shape[0]),
        "dtype": a.dtype,
        "transform": affine,
    }
    if crs:
        kwargs["crs"] = crs
    if nodata is not None:
        kwargs["nodata"] = float(nodata)
    kwargs.update(profile)
    with rio.open(p, "w", **kwargs) as dst:
        dst.write(a)
    return p
