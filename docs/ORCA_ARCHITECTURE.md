# ORCA architecture

## 1. Three engines, three rules
| Engine | Code | Rule |
|---|---|---|
| **Fact Engine** | `data/providers.py`, `schema.py`, `decision/evidence.py`, agents+tools | Only `Observation` objects carry marine values; each has source, valid time, issue time, retrieval time, freshness, quality, mode. Missing/stale are *reported*, never filled. |
| **Decision Engine** | `decision/risk.py`, `geofence.py`, `route.py` | Pure functions of (fused evidence, advisories, geofence status, thresholds). States only, with `Reason` records (`rule_id`, variable, value, unit, threshold, comparison, `obs_id`). No LLM import anywhere in `decision/`. |
| **Explanation Engine** | `explain.py`, `i18n.py`, `cards.py` | Templates fill slots from decision/evidence objects; verifier rejects unbacked numbers, `%`, assurance wording; optional LLM may only rephrase and is gated. |

## 2. States and precedence
`RESTRICTED` (hard geofence, on land, official ban) > `HIGHER_RISK` (wave ≥ 2.5 m, wind ≥ 40 km/h, lightning, official warning/cyclone warning) >
`INSUFFICIENT_DATA` (a required input missing/stale/uncovered) > `MODERATE_CONDITIONS` > `LOWER_RISK`.
Required for a site decision: wave height, wind speed, advisory feed (covering the point), cyclone status, geofence coverage.
Proven danger is never hidden by missing data; missing data never produces `LOWER_RISK`.

## 3. Source conflicts
`fuse()` keeps only usable observations, sorts by configured priority (`ORCA_SOURCE_PRIORITY`), and raises a `Conflict` when another *family* differs by more than the per-variable tolerance.
`decide_site()` then re-runs itself with each alternative value: if the state would change, the decision carries `SOURCE_SENSITIVE` and the answer says so.

## 4. Geofences as hard constraints (defence in depth)
1. PFZ zones are screened before any weather is read (centre inside, or radius overlapping, a binding fence ⇒ `RESTRICTED`; `geofence_override:*` trace entry).
2. The route grid search never uses a blocked node or edge.
3. Every finished route is re-verified (`ConstraintSet.verify_path`); a violating route is dropped and never shown; result is written to the trace as `route_geofence_check:*`.
4. Marine protected areas bind by default (`ORCA_MPA_HARD`), port approaches are advisory only.

## 5. Routes
Three objectives, criteria printed on each: `shortest_distance`; `lower_exposure` (distance × (1 + 8·e²)); `pfz_oriented` (distance × (1 + 2·e²), to the best-supported legal PFZ). `e = max(wave/2.5 m, wind/40 km/h)`, interpolated (IDW) from selected observations at departure time.
If two routes are the same path the answer says so rather than pretending they differ. A route whose max exposure ≥ 1.0 gets an explicit warning. No route is ever labelled "safest".

## 6. The trace
Each query → `Trace` with steps `input, data, interpret, plan, validate, agent, fuse, decide, rank, constraint, verify, explain, compose`.
`WHY THIS RESULT?` is assembled from those entries and the stored decision, in eight stages (question → intent → required evidence → agents/tools → observations → rules & constraints → decision → explanation). `GET /api/orca/trace/{id}` returns the raw trace; results persist under `runs/orca/`.

## 7. API
`POST /api/orca/query` `{text, session_id?, mode?, lang?, location?{lat,lon}}` · `/route/check` `{waypoints[[lat,lon],…], time?}` · `/ocean` `/weather` `{latitude, longitude, start_time, end_time?}` · `/geo` `{latitude, longitude, date?}` · `/route` `{place, time?}` · `/compare` `{place, times[2-3]}` · `/cache/refresh`
`GET /api/orca/health|scenarios|config|trace/{id}|evidence/{id}|result/{id}|report/{id}` · `GET /` web app.
Errors are JSON `{error, status}`; bad input is 400, never a stack trace.

## 8. Blueprint additions (SEAX)
New intents: `nearest_pfz`, `closer`, `route_check`, `ocean_regions`, `scenario`, plus follow-up and "there" resolution from per-session context (`Run._update_ctx`). `decision/hazards.py` holds the buffer/intersection logic; `evidence_pack.py` builds claim records (classes official / source / computed), links answer sentences to them, and builds alerts from structured results; `eval/` measures the pipeline. See `SEAX_BLUEPRINT_COMPLIANCE.md`.

## 9. Explicit non-goals (prototype scope)
No Kubernetes/Kafka/Redis/vector DB, no ten-agent swarm, no voice, no auth, no real-time push, no learned models. Three focused agents, deterministic core.
