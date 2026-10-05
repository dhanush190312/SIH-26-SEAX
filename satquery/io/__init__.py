"""Raster IO and modality inference (rasterio optional, PIL fallback)."""

from .._lazy import lazy_getattr

_TABLE = {
    "HAS_RASTERIO": ".raster:HAS_RASTERIO",
    "RasterImage": ".raster:RasterImage",
    "read_raster": ".raster:read_raster",
    "image_size": ".raster:image_size",
    "write_geotiff": ".raster:write_geotiff",
    "ModalityGuess": ".modality:ModalityGuess",
    "infer_modality": ".modality:infer_modality",
}

__all__ = sorted(_TABLE)
__getattr__ = lazy_getattr(__name__, _TABLE)
