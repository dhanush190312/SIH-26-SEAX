"""Ocean Intelligence Agent: SST, chlorophyll-a, SST gradient, currents, tide."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..decision.evidence import FusionResult
from ..schema import Observation
from .base import Agent


class OceanAgent(Agent):
    def __init__(self, ctx, trace, evidence):
        super().__init__("ocean", ctx, trace, evidence)

    def ocean_conditions(self, lat: float, lon: float, when: datetime) -> list[Observation]:
        iso = when.isoformat(timespec="seconds")
        return self.call("get_ocean_conditions", latitude=lat, longitude=lon, start_time=iso, end_time=iso).observations

    def pfz_support(self, fusion: FusionResult) -> dict[str, Any]:
        """PFZ-supporting ocean evidence as three explicit, transparent criteria (not a probability)."""
        th = self.ctx.cfg.thresholds
        items = []
        for var, test, thr, desc in (
            ("chlorophyll", lambda v: v >= th.chl_min_mg_m3, th.chl_min_mg_m3, "chlorophyll-a at or above the minimum"),
            ("sst_gradient", lambda v: v >= th.sst_gradient_min_c_per_10km, th.sst_gradient_min_c_per_10km, "SST front (gradient) at or above the minimum"),
            ("sst", lambda v: th.sst_min_c <= v <= th.sst_max_c, f"{th.sst_min_c:g}-{th.sst_max_c:g}", "SST inside the productive band"),
        ):
            o = fusion.observation(var)
            met = None if o is None or o.value is None else bool(test(o.value))
            items.append({"variable": var, "criterion": desc, "threshold": thr, "value": None if o is None else o.value,
                          "unit": None if o is None else o.unit, "met": met, "obs_id": None if o is None else o.obs_id,
                          "source": None if o is None else o.source_label,
                          "freshness_hours": None if o is None else o.freshness_hours})
        return {"met": sum(1 for i in items if i["met"]), "of": 3, "unknown": [i["variable"] for i in items if i["met"] is None], "items": items}
