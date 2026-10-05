# ORCA ← SatQuery: migration audit

Purpose: separate **what actually runs** from **what is merely similar**. Every "reused" item names the file
and how it is used. Audit performed by reading the code in `SatQueryAI.zip` and importing the modules, not by
reading its README.

## A. Reused as code (imported at runtime)

| SatQuery module | What ORCA uses | Where |
|---|---|---|
| `satquery/agent/trace.py` | `Trace`, its `step()` context manager, `add()`, `of_step()`, `to_dict()`, `jsonable()` — the execution-trace format | `orca/pipeline.py`, `orca/agents/base.py`, `orca/tools/registry.py`, `orca/api/*` |
| `satquery/config.py` | `load_env_file()` (real env beats `.env`, nothing logged), `repo_root()` | `orca/config/settings.py` |
| `satquery/log.py` | `get_logger()` | pipeline, data, API |

Confirmed: these import cleanly with only the standard library + numpy (SatQuery's lazy-import rule), so ORCA inherits SatQuery's
"runs on a bare machine" property.

## B. Reused as design (re-implemented for the marine domain)

| SatQuery idea (file) | ORCA counterpart | Adaptation |
|---|---|---|
| Controller: validate → interpret → route → execute → compose (`agent/controller.py`) | `orca/pipeline.py` `Run.execute()`: input → provider → interpret → plan → capability validation → context validation → agents → decide → verify → compose | Same shape, every hop writes a trace entry, including refusals |
| `ToolSpec` + `build_menu` pruning by capability, withheld-with-reason (`agent/tools.py`) | `orca/tools/registry.py` | `requires` tokens are provider capabilities (`weather`, `ocean`, `pfz`, `advisory`, `geofence`); OFFLINE/LIVE gaps show up as *withheld tools*, distinguishable from "agent chose not to" |
| "An unbacked number never survives" composer rule (`agent/compose.py`) | `orca/explain.py` ledger + verifier + narration gate | Number-with-unit must be in the evidence ledger; `%` and assurance words always rejected; narration may not be rosier than the decision |
| Two-backend rule (deterministic always available, model optional) (`agent/backends.py`) | Template answers always; optional Mistral rephraser behind the verifier | |
| Single-file UI with CSS-variable tokens, trace tab (`api/ui.html`) | `orca/api/ui.html` | Same token scheme/tabs; map is hand-rolled SVG (no Leaflet/CDN) so it works with the network unplugged |
| Printable report (`report.py`) | `orca/report.py` | HTML only (no reportlab dependency) — print to PDF from the browser |
| `/api/...` + trace-by-id conventions (`api/app.py`) | `orca/api/server.py` | stdlib server instead of FastAPI (see README) |
| Deterministic demo data generator (`tools/make_demo_scenes.py`) | `tools/make_orca_demo.py` | Fixed, no randomness; output byte-identical |
| One-file-per-invariant tests | `tests/orca/` | unittest-style so they run without pytest |

## C. Deliberately NOT reused

| SatQuery part | Why not |
|---|---|
| VLM backends, Qwen/Colab notebooks, `models/`, `preprocess/`, `metrics/`, `eval/`, `datasets/` | satellite-imagery specific; no role in a marine evidence pipeline |
| Change detection (CVA/Otsu), fusion of bands | domain-specific to imagery |
| Raster/CRS I/O (`io/`, rasterio/pyproj) | ORCA works on point/grid observations, not rasters |
| reportlab PDF | extra dependency; HTML report suffices |
| FastAPI app | not assumable on a judging laptop |
| **The "confidence figure"** in SatQuery answers | The marine problem statement forbids invented probabilities. ORCA exposes *states + reasons + freshness*, never "87% safe". A test asserts no `probab*`/`confidence` key appears in a decision |
| `runs/sessions` data | user data; left untouched, not read |

## D. Claims in earlier planning documents that this prototype does **not** yet back

- "ORCA reuses SatQuery's router/controller/validator/tool-registry/FastAPI structure": true for controller *shape*, tool registry *pattern*, trace and config *code*; **not** FastAPI and **not** SatQuery's validator (image validation is irrelevant here).
- Leaflet/MapLibre interactive map: replaced by an offline SVG map. It supports layer toggles and hover tooltips but **no pan/zoom**.
- GeoPandas/Shapely: not used (not installable offline and not needed at this scale); geometry is in `orca/geo.py` with tests.
- Six Indian languages: two shipped (English, Tamil).
- Live INCOIS / IMD / MOSDAC integration: **not implemented** — no documented public API was used; see `DATA_SOURCES.md`.

## E. Hygiene finding

`SatQueryAI.zip` contained `.env` holding a `MISTRAL_API_KEY` value. It is git-ignored (so not in history) but it travelled inside the archive.
It was **not** copied into ORCA. Rotate it.
Also: the repo's `.gitignore` pattern `data/` would have silently excluded `orca/data/`; `.gitignore` now has explicit exceptions.
