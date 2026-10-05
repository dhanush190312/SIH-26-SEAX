"""Resolved ORCA configuration.

Reuses SatQuery's ``.env`` loader (:func:`satquery.config.load_env_file`), so a real
environment variable beats the file and no value is ever logged. Every threshold
that drives a decision lives here, in one frozen dataclass, and is copied into each
decision's output -- a decision can always say which thresholds produced it.

The thresholds are **ORCA prototype defaults, not official criteria**. Official
INCOIS/IMD warnings always take precedence over them (an advisory of severity
``warning``/``nogo`` forces HIGHER_RISK; a ``ban`` forces RESTRICTED).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from satquery.config import load_env_file, repo_root

__all__ = ["MODES", "IST", "OrcaConfig", "Thresholds", "get_config", "set_config"]

MODES = ("DEMO", "LIVE", "OFFLINE")
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

load_env_file()  # idempotent; same loader SatQuery uses


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds. Prototype defaults -- see module docstring."""

    wave_moderate_m: float = 1.5
    wave_higher_m: float = 2.5
    wind_moderate_kmh: float = 25.0
    wind_higher_kmh: float = 40.0
    precip_moderate_mmh: float = 7.5
    # PFZ ocean-support criteria (transparent rules, not probabilities)
    chl_min_mg_m3: float = 0.5
    sst_gradient_min_c_per_10km: float = 0.25
    sst_min_c: float = 26.0
    sst_max_c: float = 31.0
    # maximum age (hours since issue/observation) before data counts as stale
    max_age_hours: Mapping[str, float] = field(
        default_factory=lambda: {"weather": 12.0, "ocean": 48.0, "pfz": 30.0, "advisory": 12.0, "tide": 72.0}
    )
    # two sources "disagree" when they differ by more than this (per variable)
    conflict_tolerance: Mapping[str, float] = field(
        default_factory=lambda: {
            "wave_height": 0.3, "wind_speed": 8.0, "precipitation": 2.0,
            "sst": 0.5, "chlorophyll": 0.2, "current_speed": 0.2, "tide_height": 0.2,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["max_age_hours"] = dict(self.max_age_hours)
        d["conflict_tolerance"] = dict(self.conflict_tolerance)
        return d


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class OrcaConfig:
    """Everything a run needs. Build with :meth:`from_env`; override with ``replace``."""

    mode: str = "DEMO"
    source_priority: tuple[str, ...] = ("incois", "imd", "open_model")
    allow_stale: bool = False
    mpa_hard: bool = True                 # treat marine protected areas as hard constraints
    snap_radius_km: float = 40.0          # max distance from requested point to an observation point
    vessel_speed_kmh: float = 15.0        # small mechanised boat, for route duration only
    hazard_buffer_km: float = 15.0        # analysis area around a PFZ = its radius + this buffer
    pfz_radius_km: float = 150.0          # PFZ candidates farther than this from the departure point are not offered
    llm: str = "none"                     # none | mistral (narration only, always verified)
    llm_model: str = "mistral-small-latest"
    host: str = "127.0.0.1"
    port: int = 8100
    runs_dir: Path = field(default_factory=lambda: repo_root() / "runs" / "orca")
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    thresholds: Thresholds = field(default_factory=Thresholds)
    now: datetime | None = None           # injectable clock for tests; None = real time

    @classmethod
    def from_env(cls, **overrides: Any) -> "OrcaConfig":
        env = os.environ
        mode = env.get("ORCA_MODE", "DEMO").strip().upper()
        if mode not in MODES:
            raise ValueError(f"ORCA_MODE must be one of {MODES}, got {mode!r}")
        prio = tuple(p.strip().lower() for p in env.get("ORCA_SOURCE_PRIORITY", "incois,imd,open_model").split(",") if p.strip())
        base = cls(
            mode=mode,
            source_priority=prio,
            allow_stale=_bool(env.get("ORCA_ALLOW_STALE")),
            mpa_hard=_bool(env.get("ORCA_MPA_HARD"), True),
            snap_radius_km=float(env.get("ORCA_SNAP_KM", 40.0)),
            pfz_radius_km=float(env.get("ORCA_PFZ_RADIUS_KM", 150.0)),
            hazard_buffer_km=float(env.get("ORCA_HAZARD_BUFFER_KM", 15.0)),
            vessel_speed_kmh=float(env.get("ORCA_VESSEL_KMH", 15.0)),
            llm=env.get("ORCA_LLM", "none").strip().lower(),
            llm_model=env.get("ORCA_LLM_MODEL", "mistral-small-latest"),
            host=env.get("ORCA_HOST", "127.0.0.1"),
            port=int(env.get("ORCA_PORT", 8100)),
            runs_dir=Path(env["ORCA_RUNS_DIR"]) if env.get("ORCA_RUNS_DIR") else repo_root() / "runs" / "orca",
        )
        return replace(base, **overrides) if overrides else base

    def clock(self) -> datetime:
        """The current instant, timezone-aware, in IST."""
        return (self.now or datetime.now(timezone.utc)).astimezone(IST)

    def local_midnight(self) -> datetime:
        n = self.clock()
        return n.replace(hour=0, minute=0, second=0, microsecond=0)

    def public(self) -> dict[str, Any]:
        """Safe-to-serialise view (never includes credentials)."""
        return {
            "mode": self.mode,
            "source_priority": list(self.source_priority),
            "allow_stale": self.allow_stale,
            "mpa_hard": self.mpa_hard,
            "llm": self.llm,
            "thresholds": self.thresholds.to_dict(),
        }


_CONFIG: OrcaConfig | None = None


def get_config() -> OrcaConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = OrcaConfig.from_env()
    return _CONFIG


def set_config(cfg: OrcaConfig | None) -> None:
    global _CONFIG
    _CONFIG = cfg
