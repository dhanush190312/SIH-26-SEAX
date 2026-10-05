"""Evidence fusion: many sources in, one auditable view out.

For each (variable, location, time) the fuser

1. drops nothing silently -- missing and stale observations are *reported*;
2. keeps only ``available`` observations as candidates;
3. if two or more sources disagree by more than the variable's tolerance, emits a
   :class:`~orca.schema.Conflict` carrying every value, and selects the value from the
   highest-priority source (priority is configuration, never a hard-coded preference).

The selected value is what the decision engine consumes; the conflict travels with it to
the evidence card, the trace, and the decision's sensitivity check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..config import OrcaConfig
from ..schema import Conflict, Observation

__all__ = ["FusedVariable", "FusionResult", "fuse", "rank"]


def rank(priority: Sequence[str], family: str) -> int:
    """Lower is better. Unlisted sources rank after every listed one."""
    try:
        return list(priority).index(family)
    except ValueError:
        return len(priority)


@dataclass(frozen=True)
class FusedVariable:
    variable: str
    selected: Observation | None
    candidates: tuple[Observation, ...]
    rejected: tuple[Observation, ...]           # missing or stale
    conflict: Conflict | None = None

    @property
    def status(self) -> str:
        if self.selected is not None:
            return "conflict" if self.conflict else "ok"
        if any(o.quality == "stale" for o in self.rejected):
            return "stale"
        return "missing"


@dataclass(frozen=True)
class FusionResult:
    variables: Mapping[str, FusedVariable]
    conflicts: tuple[Conflict, ...]
    missing: tuple[str, ...]
    stale: tuple[str, ...]

    def value(self, variable: str) -> float | None:
        fv = self.variables.get(variable)
        return fv.selected.value if fv and fv.selected else None

    def observation(self, variable: str) -> Observation | None:
        fv = self.variables.get(variable)
        return fv.selected if fv else None


def fuse(observations: Iterable[Observation], cfg: OrcaConfig, *, wanted: Sequence[str] = ()) -> FusionResult:
    """Fuse one location/time slice. ``wanted`` variables with no observation at all become
    explicit ``missing`` entries (so nothing required can vanish without a trace)."""
    by_var: dict[str, list[Observation]] = {}
    for o in observations:
        by_var.setdefault(o.variable, []).append(o)
    for v in wanted:
        by_var.setdefault(v, [])

    tol_map = cfg.thresholds.conflict_tolerance
    fused: dict[str, FusedVariable] = {}
    conflicts: list[Conflict] = []
    missing: list[str] = []
    stale: list[str] = []
    for var, obs in by_var.items():
        usable = [o for o in obs if o.usable or (cfg.allow_stale and o.value is not None and o.quality == "stale")]
        rejected = tuple(o for o in obs if o not in usable)
        usable.sort(key=lambda o: (rank(cfg.source_priority, o.source_family), o.offset_km or 0.0))
        if not usable:
            fused[var] = FusedVariable(var, None, (), rejected)
            (stale if any(o.quality == "stale" for o in rejected) else missing).append(var)
            continue
        chosen = usable[0]
        conflict = None
        tol = tol_map.get(var)
        others = [o for o in usable[1:] if o.source_family != chosen.source_family and o.value is not None]
        if tol is not None and others:
            spread = max(abs(o.value - chosen.value) for o in others)  # type: ignore[operator]
            if spread > tol:
                allv = [chosen, *others]
                lo, hi = min(o.value for o in allv), max(o.value for o in allv)  # type: ignore[type-var]
                conflict = Conflict(
                    variable=var, latitude=chosen.latitude, longitude=chosen.longitude, valid_time=chosen.valid_time,
                    values=tuple({"source_family": o.source_family, "source_label": o.source_label, "value": o.value,
                                  "unit": o.unit, "obs_id": o.obs_id} for o in allv),
                    selected_source=chosen.source_family, selected_value=chosen.value,  # type: ignore[arg-type]
                    difference=round(hi - lo, 3), unit=chosen.unit, tolerance=tol, priority=tuple(cfg.source_priority))
                conflicts.append(conflict)
        fused[var] = FusedVariable(var, chosen, tuple(usable), rejected, conflict)
        if chosen.quality == "stale":  # only reachable with allow_stale
            stale.append(var)
    return FusionResult(fused, tuple(conflicts), tuple(missing), tuple(stale))
