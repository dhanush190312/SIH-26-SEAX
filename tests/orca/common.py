"""Shared test helpers: repo on sys.path, a fixed clock, isolated data dirs."""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orca.config import IST, OrcaConfig  # noqa: E402
from orca.schema import Observation  # noqa: E402

NOW = datetime(2026, 9, 30, 10, 0, tzinfo=IST)
TOMORROW = datetime(2026, 10, 1, 0, 0, tzinfo=IST)


def at(hour: int, day: int = 1) -> datetime:
    return TOMORROW + timedelta(days=day - 1, hours=hour)


def make_cfg(**kw) -> OrcaConfig:
    base = OrcaConfig.from_env(mode="DEMO")
    return replace(base, now=NOW, runs_dir=Path(tempfile.mkdtemp(prefix="orca-runs-")), **kw)


def isolated_data_dir() -> Path:
    """A private copy of orca/data so cache-writing tests never touch the real cache."""
    d = Path(tempfile.mkdtemp(prefix="orca-data-"))
    shutil.copytree(ROOT / "orca" / "data" / "demo", d / "demo")
    (d / "cache").mkdir()
    return d


def mk_obs(variable: str, value, *, family="incois", unit="m", quality="available", lat=10.765, lon=79.9, when=None, uid=None) -> Observation:
    when = when or at(6)
    return Observation(
        obs_id=uid or f"t:{family}:{variable}", variable=variable, value=value, unit=unit, latitude=lat, longitude=lon, time_kind="forecast",
        observed_at=None, forecast_for=when.isoformat(), issued_at=NOW.isoformat(), retrieved_at=NOW.isoformat(), source=f"t-{family}",
        source_family=family, source_label=f"{family.upper()} (test)", quality=quality if value is not None else "missing",
        freshness_hours=1.0, data_mode="DEMO", offset_km=1.0)
