"""Sensor-agnostic preprocessing: the layer that has to survive Cartosat/RISAT."""

from .._lazy import lazy_getattr

_TABLE = {
    "BAND_PRESETS": ".optical:BAND_PRESETS",
    "percentile_stretch": ".optical:percentile_stretch",
    "select_composite": ".optical:select_composite",
    "to_uint8": ".optical:to_uint8",
    "ndvi": ".indices:ndvi",
    "ndwi": ".indices:ndwi",
    "ndbi": ".indices:ndbi",
    "vari": ".indices:vari",
    "index_stats": ".indices:index_stats",
    "linear_to_db": ".sar:linear_to_db",
    "db_to_uint8": ".sar:db_to_uint8",
    "lee_filter": ".sar:lee_filter",
    "sar_water_mask": ".sar:sar_water_mask",
    "backscatter_stats": ".sar:backscatter_stats",
    "Tile": ".tiling:Tile",
    "tile_image": ".tiling:tile_image",
    "merge_tiles": ".tiling:merge_tiles",
    "resize_max_side": ".tiling:resize_max_side",
    "resample_to": ".tiling:resample_to",
    "BandRoles": ".bands:BandRoles",
    "resolve_band_roles": ".bands:resolve_band_roles",
    "available_indices": ".bands:available_indices",
    "INDEX_REQUIREMENTS": ".bands:INDEX_REQUIREMENTS",
    "PairCheck": ".pairing:PairCheck",
    "check_pair": ".pairing:check_pair",
    "resample_to_match": ".pairing:resample_to_match",
}

__all__ = sorted(_TABLE)
__getattr__ = lazy_getattr(__name__, _TABLE)
