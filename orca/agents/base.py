"""Shared agent plumbing: every tool call an agent makes is recorded as a trace step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from satquery.agent.trace import Trace

from ..tools import ToolContext, ToolResult, run_tool


def compact(res: ToolResult) -> dict[str, Any]:
    """Trace-sized digest of a tool result (full observations live in the evidence store)."""
    by_var: dict[str, list[Any]] = {}
    for o in res.observations:
        by_var.setdefault(o.variable, []).append(o.value if o.value is not None else "MISSING")
    out: dict[str, Any] = {"ok": res.ok, "n_observations": len(res.observations), "values": by_var}
    if res.note:
        out["note"] = res.note
    if res.observations:
        srcs = sorted({o.source_label for o in res.observations})
        out["sources"] = srcs
        out["quality"] = sorted({o.quality for o in res.observations})
    d = res.data
    if d is not None and hasattr(d, "to_dict"):
        dd = d.to_dict()
        out["data_keys"] = sorted(dd.keys())[:12]
        if "zones" in dd:
            out["zones"] = [z["zone_id"] for z in dd["zones"]]
        if "items" in dd:
            out["advisories"] = [a["advisory_id"] for a in dd["items"]]
            out["cyclone_status"] = dd.get("cyclone_status")
    elif isinstance(d, dict):
        out["data"] = {k: d[k] for k in list(d)[:6] if k != "routes"}
        if "routes" in d:
            out["routes"] = [{"objective": r["objective"], "feasible": r.get("feasible"), "distance_km": r.get("distance_km")} for r in d["routes"]]
    return out


@dataclass
class Agent:
    name: str
    ctx: ToolContext
    trace: Trace
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def call(self, tool: str, **params: Any) -> ToolResult:
        shown = {k: v for k, v in params.items() if k not in ("field", "zones")}
        with self.trace.step("agent", f"{self.name}.{tool}", params=shown) as st:
            res = run_tool(tool, self.ctx, **params)
            st.output = compact(res)
            st.status = "ok" if res.ok else "error"
            if not res.ok:
                st.detail = res.note
        for o in res.observations:
            self.evidence[o.obs_id] = o.to_dict()
        self.calls.append({"agent": self.name, "tool": tool, "ok": res.ok, "params": shown, "n_observations": len(res.observations),
                           "note": res.note})
        return res
