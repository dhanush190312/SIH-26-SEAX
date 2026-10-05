# ORCA data sources — what each mode really provides

| Capability | DEMO | OFFLINE | LIVE |
|---|---|---|---|
| Wave height, wind, precipitation, lightning | synthetic, 2 sources (`incois`-labelled, `open_model`-labelled) on a 14-point grid + squall sub-grid, hourly for 48 h from local midnight | last cache snapshot | **Open-Meteo Marine + Forecast API** (free, no key) — *open model* forecast only; `open_model` family |
| SST, chlorophyll, SST gradient, currents, tide | synthetic | cache | **unavailable** (no documented public API used) |
| PFZ advisory | synthetic, valid today + tomorrow, 4 zones (one deliberately inside a geofence) | cache | **unavailable** unless you place an official file at `orca/data/imports/pfz.json` |
| Marine advisories / cyclone status | synthetic | cache | **unavailable** unless `orca/data/imports/advisories.json` |
| Geofences, coastline | illustrative shapes | illustrative | illustrative unless `orca/data/geofences_official.json` |

## Rules the code follows
1. **No silent substitution.** If LIVE can't fetch something, it is reported unavailable; it is never filled from DEMO. (`tests/orca/test_providers.py::test_live_failure_is_reported_and_never_filled_from_demo`)
2. **DEMO is never presented as real.** Every observation carries `data_mode`; every source label says "synthetic"; the UI banner is amber.
3. **OFFLINE is labelled with its origin and age**: *"OFFLINE / CACHED DEMO DATA (synthetic) - last updated …"*; it goes `stale` by the same thresholds as live data.
4. **Three clocks, never merged:** `observed_at` (sensor), `forecast_for` (prediction target), `issued_at` (forecast run) — plus `retrieved_at` (ORCA fetched it). Freshness is measured from `issued_at`/`observed_at`.
5. **Coverage ≠ clear.** Outside a dataset's coverage box the answer is `INSUFFICIENT DATA`, not "no hazard".

## Freshness limits (prototype defaults, `Thresholds.max_age_hours`)
weather 12 h · ocean 48 h · PFZ 30 h · advisory 12 h · tide 72 h. Older ⇒ `stale` ⇒ excluded (unless `ORCA_ALLOW_STALE=true`, still flagged).

## Import file format (for official data)
Same JSON schema as `orca/data/demo/pfz.json` / `advisories.json` / `geofences.json`, with **absolute ISO timestamps** (`issued_at`, `valid_from`, `valid_to`, `valid_dates`) instead of the demo's relative hours. Place in `orca/data/imports/` (git-ignored). A geofence file may carry `meta.coverage = [south, west, north, east]`.

## Thresholds are not official
Wave ≥1.5 m / ≥2.5 m, wind ≥25 / ≥40 km/h, rain ≥7.5 mm/h, PFZ support (chlorophyll ≥0.5 mg/m³, SST gradient ≥0.25 °C/10 km, SST 26–31 °C) are **ORCA prototype defaults**. An official `warning`/`nogo` advisory forces HIGHER RISK, a `ban` forces RESTRICTED, regardless of these numbers. Replace with agency criteria before any real use.
