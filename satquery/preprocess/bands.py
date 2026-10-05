"""Which band is red? Everything deterministic depends on the answer.

NDVI is ``(NIR - Red) / (NIR + Red)`` and the formula is not the hard part --
knowing that NIR is band 7 in a Sentinel-2 L2A stack, band 3 in an RGBN
delivery, and *absent* from an RSVQA PNG is. Getting it wrong does not raise; it
produces a confident, wrong number that the composer will then dutifully cite.

So band roles are resolved once, here, with the provenance attached:

* an explicit mapping on the image (``ImageRef.meta['band_roles']``) always
  wins -- a dataset adapter or a user who knows their product outranks any
  inference;
* otherwise the band count picks a preset, and the preset says which product it
  assumed;
* and when a role is simply not present -- no NIR in an RGB tile -- it is
  ``None``, so the index that needs it is *withheld from the tool menu* rather
  than computed from a substitute band.

The last point is the one that matters. A three-band aerial tile cannot produce
NDVI. It can produce VARI, a published RGB-only greenness index, and calling
that "NDVI" would be the kind of quiet substitution that makes every downstream
number unfalsifiable. :func:`available_indices` therefore offers VARI on RGB
input and NDVI only where a real NIR band exists, and the trace records which
preset was assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..log import get_logger

__all__ = [
    "BandRoles",
    "ROLES",
    "resolve_band_roles",
    "available_indices",
    "INDEX_REQUIREMENTS",
]

_LOG = get_logger(__name__)

#: Every role name this module can fill. Optical roles are wavelengths; SAR
#: roles are polarisations, which are not interchangeable with them in any
#: formula -- an index over VV and VH is not an index, it is a ratio of two
#: different scattering mechanisms.
ROLES: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2", "vv", "vh")

#: index name -> the roles its formula needs. Consulted by
#: :func:`available_indices` and by the tool menu, so an index whose bands are
#: missing is never offered to the router.
INDEX_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "NDVI": ("nir", "red"),
    "NDWI": ("green", "nir"),
    "NDBI": ("swir1", "nir"),
    "VARI": ("green", "red", "blue"),
    "SAR_BACKSCATTER": ("vv",),
}


@dataclass(frozen=True, slots=True)
class BandRoles:
    """Role -> band index, plus what that assignment was based on."""

    blue: int | None = None
    green: int | None = None
    red: int | None = None
    nir: int | None = None
    swir1: int | None = None
    swir2: int | None = None
    vv: int | None = None
    vh: int | None = None
    preset: str = "unknown"
    source: str = "inferred"
    note: str = ""

    def get(self, role: str) -> int | None:
        return getattr(self, role, None)

    def has(self, *roles: str) -> bool:
        return all(self.get(r) is not None for r in roles)

    def to_dict(self) -> dict[str, Any]:
        mapping = {r: self.get(r) for r in ROLES if self.get(r) is not None}
        return {"roles": mapping, "preset": self.preset, "source": self.source, "note": self.note}


def resolve_band_roles(
    bands: int,
    *,
    modality: str = "optical",
    override: Mapping[str, int] | None = None,
) -> BandRoles:
    """Map role names onto band indices for a ``bands``-band image.

    ``override`` is ``{"red": 3, "nir": 7, ...}`` from an image's metadata and
    is used verbatim (out-of-range entries are dropped with a warning rather
    than silently clamped, because a clamped band index is a wrong number that
    still computes).
    """
    if override:
        kwargs: dict[str, Any] = {}
        for role, index in override.items():
            key = str(role).lower()
            if key not in ROLES:
                _LOG.warning("ignoring unknown band role %r", role)
                continue
            idx = int(index)
            if not 0 <= idx < bands:
                _LOG.warning("ignoring band role %s=%d: out of range for %d bands", key, idx, bands)
                continue
            kwargs[key] = idx
        if kwargs:
            return BandRoles(**kwargs, preset="explicit", source="declared")

    if modality == "sar":
        if bands >= 2:
            return BandRoles(vv=0, vh=1, preset="sar_dual_pol", note="assumed VV then VH")
        return BandRoles(vv=0, preset="sar_single_pol", note="single polarisation, assumed VV")

    if bands >= 12:
        # Sentinel-2 L2A ordering (see optical.S2_L2A_BAND_NAMES). A 13-band
        # stack is L1C, which keeps B10, so everything from SWIR on shifts by one.
        if bands == 13:
            return BandRoles(
                blue=1, green=2, red=3, nir=7, swir1=11, swir2=12,
                preset="s2_l1c",
                note="13 bands read as Sentinel-2 L1C (B10 present, so SWIR sits one higher)",
            )
        return BandRoles(
            blue=1, green=2, red=3, nir=7, swir1=10, swir2=11,
            preset="s2_l2a",
            note="12 bands read as Sentinel-2 L2A",
        )
    if bands == 4:
        return BandRoles(
            red=0, green=1, blue=2, nir=3,
            preset="rgbn",
            note="4 bands read as red, green, blue, NIR -- VERIFY for Cartosat-2S MX, "
            "which is recalled as blue, green, red, NIR",
        )
    if bands == 3:
        return BandRoles(
            red=0, green=1, blue=2,
            preset="rgb",
            note="3 bands read as RGB; there is no NIR band, so NDVI is not computable",
        )
    if bands == 2:
        return BandRoles(
            green=0, red=1, preset="two_band", note="2 optical bands, roles are a guess"
        )
    return BandRoles(preset="panchromatic", note="single band: no index needs fewer than two")


def available_indices(roles: BandRoles) -> tuple[str, ...]:
    """Which indices this band layout can actually produce, in citation order."""
    return tuple(name for name, needed in INDEX_REQUIREMENTS.items() if roles.has(*needed))
