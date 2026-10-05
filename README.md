# SEAX — Agentic Marine Intelligence for Smarter Decisions (GEOSAR)
### engine: ORCA — Marine EcOsystem Reasoning with Collaborative Agents (SIH26176)

A **working prototype** of an evidence-linked marine decision-support assistant for small-scale fishers and coastal operators,
built by extending the SatQuery AI codebase in this repository and aligned to the SEAX blueprint (**`docs/SEAX_BLUEPRINT_COMPLIANCE.md` lists exactly what matches, what deviates and what is not done**) (SatQuery is untouched and still works —
its original README is `docs/SATQUERY_README.md`).

> Ask *"Can I go fishing tomorrow at 6 AM from Nagapattinam?"* (or the same in Tamil) and get a
> state — `LOWER RISK` / `MODERATE CONDITIONS` / `HIGHER RISK` / `RESTRICTED` / `INSUFFICIENT DATA` — with an
> **evidence card**, an **interactive map**, a **WHY THIS RESULT?** trace and a **what-if comparison**.
> Every number in the answer is checked against structured evidence before you see it.

---

## 0. Read this first — what is real and what is not

**Real and tested (136 tests, all passing in the build environment):**
- The whole pipeline: *query → intent/place/time → evidence plan → capability validation → 3 specialist agents → evidence fusion → deterministic decision engine → verified explanation → card + map + trace*.
- Fact Engine ≠ Decision Engine ≠ Explanation: decisions are plain code over schema objects; nothing an LLM writes can reach them.
- Hard geofences (routes and PFZ recommendations can never enter them — enforced twice, with a 72-case property test; routes you supply can be checked via `POST /api/orca/route/check`).
- Source-conflict transparency (both values shown, selection by *configurable* priority, plus a "would the decision change?" check).
- `DEMO` / `OFFLINE` / `LIVE` modes; DEMO needs no network, key, GPU or extra package.
- Blueprint workflows: **nearest PFZ + hazards around it**, **conversation memory** ("What about tomorrow?", "Which one is closer?", "…conditions *there*"), **route restriction check**, **ocean regions with an explicit stated definition of "favourable"**, **scenario engine** ("What changes if I leave in the evening?"), an **evidence panel** (every factual sentence linked to claim records badged OFFICIAL / SOURCE DATA / SEAX-COMPUTED), and an **alert strip**.
- A **measured evaluation** (`python -m orca eval`): see §6b — including the cases it fails.
- English + Tamil answers; five further Indian scripts (hi, bn, te, kn, ml) are detected and *honestly reported as not yet supported*.

**Not real / not done — say this before a judge does:**
- **Intent understanding and planning are rule-based, not LLM-driven** (blueprint §4 expects an LLM). Unusual phrasing fails: on 7 held-out paraphrases the planner got 3 right, and one wrong reading ("Which fishing spot is closest…" → go/no-go) gives a plausible-looking wrong answer. Every answer therefore says *"Understood as: …"*. Nothing unsupported is said even then (groundedness 7/7), but the answer can be to the wrong question.
- **Not built:** PostgreSQL/PostGIS, Next.js, a pan/zoom map, duplicate-record detection, unit conversion, persistent memory, voice verified on a device (the 🎤 button uses the browser's speech API, untested here). See the compliance doc §D.
- **All DEMO data is synthetic** (Bay of Bengal–like values off Nagapattinam). Every screen says `DEMO DATA (synthetic)`.
  The geofence/coastline polygons are **illustrative shapes, not official boundaries.** Thresholds are **prototype defaults, not official criteria** — official INCOIS/IMD warnings always override them.
- **LIVE mode was not exercised against the real internet** (the build sandbox had no network). The Open-Meteo adapter is parse-tested on a fixture of the documented response shape; first real run is on you (see §7).
  LIVE deliberately provides **no PFZ / no IMD advisories / no ocean colour**: no documented public JSON API exists for them, so none is invented — those questions return `INSUFFICIENT DATA` in LIVE unless you drop official files into `orca/data/imports/`.
- The Tamil strings were written for this prototype and need review by a native speaker / fisheries-communication expert.
- The optional LLM rephraser (`ORCA_LLM=mistral`) is untested against the live API; it is behind the verifier, so failure just means the template answer is used.
- SatQuery's own pytest suite could **not** be run here (pytest not installable offline). I confirmed SatQuery still imports and `python -m satquery.cli doctor` runs; run `pytest` yourself to be sure (§6).

---

## 1. Quick start (3 commands, no installs beyond Python ≥ 3.10 + numpy)

```bash
python -m orca demo              # runs 11 scenarios, verifies every answer + evidence linkage, prints PASS/FAIL
python -m orca eval              # controlled test set (36 cases) with measured metrics
python -m orca serve             # web app at http://127.0.0.1:8100
python -m orca doctor            # what each data mode can/can't do right now
```
In the web app tap the scenario buttons: **Fishing · PFZ · Hazard · Route · WHY · WHAT-IF · தமிழ் · Avoid**.

Other commands: `python -m orca ask "…" [--card|--json]`, `python -m orca cache-refresh`, `python -m orca --mode OFFLINE ask "…"`.

*Why not FastAPI like SatQuery?* `fastapi`/`uvicorn` can't be assumed on a judging laptop; ORCA's server is stdlib-only (`http.server`) but follows SatQuery's endpoint conventions. Bound to `127.0.0.1`, unauthenticated, single-user — a demo tool, not a public service.

## 2. The 2-minute demo script

| # | Do | What it proves |
|---|---|---|
| 1 | Banner shows **DEMO DATA (synthetic)** | honesty about data |
| 2 | **Fishing** → `LOWER RISK` at 06:00: wave 0.9 m, wind 14 km/h, PFZ-A 3/3 | evidence card; every row has source/valid time/freshness |
| 3 | Ask *"…fish tomorrow at 10 AM…"* | **SOURCE DISAGREEMENT DETECTED** (INCOIS 1.41 m vs open model 2.1 m), selected by priority, difference shown, decision `MODERATE` |
| 4 | **PFZ** → PFZ-D shown **REJECTED by geofence** though the PFZ advisory lists it | hard constraint beats the advisory; red hatched area on the map |
| 5 | **Route** → routes detour around the naval area; exposure warning where a route crosses the synthetic squall | geofence-safe routing; no route is ever called "safest" |
| 6 | **WHY** → *"Why PFZ-A… ocean support 3/3: chlorophyll 0.72 mg/m³ meets…"* then **WHY THIS RESULT?** tab | explanation from the stored analysis; 8-stage chain generated from the real trace |
| 7 | **WHAT-IF** (or the Compare form) → 06:00 `LOWER` vs 10:00 `MODERATE` | real time comparison, not static text |
| 8 | **தமிழ்** | Tamil question → Tamil answer, Latin digits/units so they stay verifiable |
| 9 | Data mode → OFFLINE (no cache) → honest refusal; **Refresh offline cache**; retry | OFFLINE says *"CACHED DEMO DATA… last updated …"*; it never poses as real-time |
| 10 | Ask about **Chennai** | `INSUFFICIENT DATA`: no coverage ≠ "clear" |
| 11 | **Nearest PFZ** → "nearest valid PFZ: PFZ-A … the geographically nearest, PFZ-D, is in a hard geofence"; hazard check within 23 km; alert strip shows an OFFICIAL advisory only when it is in force | blueprint demo 1; official ≠ computed; expired/not-yet-valid warnings never applied |
| 12 | Click follow-ups **Conditions there** → **Which one is closer?** → **What about tomorrow?** | conversation memory (topic, place, "there" = the PFZ, time change) |
| 13 | **Route check** after **Route** | blueprint demo 3: intersection result per route, evidence records |
| 14 | **Leave in the evening?** | blueprint demo 4: only time changes, all data re-retrieved, decision change explained |
| 15 | **Ocean regions** | "favourable" = stated criteria (chlorophyll ≥ 0.5 mg/m³, SST 26–31 °C), shown in the answer |
| 16 | **Evidence panel** tab | each answer sentence → evidence record ids; unlinked sentences flagged red |
| 17 | Click 📍 then ask *"Where is the nearest PFZ from my location?"* | device location (far away ⇒ correctly `INSUFFICIENT DATA`: demo covers Nagapattinam only) |

## 3. Architecture (what talks to what)

```
question ──► Planner (rules; bilingual) ──► Evidence Plan ──► Capability Validator
                                                                   │ (tools the mode can't serve are withheld WITH a reason)
            ┌──────────────────────────────────────────────────────┴───────────────┐
            ▼                        ▼                                               ▼
   Weather & Hazard Agent     Ocean Agent                              Geo / PFZ / Route Agent
   wind, waves, rain,         SST, chlorophyll, SST gradient,          geofences (hard), PFZ candidates,
   lightning, advisories,     currents, tide                           ocean-support screening, routes
   cyclone status                                                                  │
            └──────────────► Observations (source, valid_time, forecast_for, issued_at, retrieved_at, freshness, quality, mode)
                                        │
                        Evidence Fusion: missing/stale reported, conflicts surfaced, priority selects
                                        ▼
                  Deterministic Decision Engine  (states + machine-readable reasons + thresholds; no probabilities)
                                        ▼
        Explanation: i18n templates ─► Verifier (unbacked number / % / assurance wording ⇒ rejected) ─► [optional LLM rephrase, gated]
                                        ▼
            Answer + Evidence Card + Map + WHY chain + Trace (+ printable report)
```
Details: `docs/ORCA_ARCHITECTURE.md`. Decision precedence: `RESTRICTED > HIGHER_RISK > INSUFFICIENT_DATA > MODERATE > LOWER`
(missing data can never *clear* a location, and can never *hide* a danger that's already proven).
`LOWER RISK` means "below ORCA's moderate thresholds", **not** "safe".

## 4. What was reused from SatQuery, and what was not

Full audit with file names and reasons: **`docs/ORCA_MIGRATION_AUDIT.md`**. Summary:

- **Imported directly (real code reuse):** `satquery.agent.trace` (Trace/step/jsonable → the execution trace), `satquery.config` (`.env` loader, repo paths), `satquery.log`.
- **Re-implemented on the same pattern (design reuse):** controller shape (validate → interpret → plan → execute → compose), `ToolSpec` + capability-pruned menu, "an unbacked number never survives" composer rule (→ `orca/explain.py`), single-file UI with the same CSS-token scheme, HTML report.
- **Deliberately not reused:** VLM/vision backends, change detection, fusion of satellite bands, raster/CRS I/O, reportlab PDF, FastAPI app. And **SatQuery's "confidence figure" is intentionally absent from ORCA** — the marine spec forbids invented probabilities.

## 5. Repository map (ORCA parts)

```
orca/
  config/settings.py      thresholds, source priority, modes (env-driven; no secrets)
  schema.py               Observation / PFZZone / Advisory / Conflict / Reason / Decision
  data/providers.py       DEMO / OFFLINE / LIVE providers, freshness + staleness, cache
  data/live.py            Open-Meteo adapter (LIVE); unavailable sources reported, never faked
  data/demo/*.json        SYNTHETIC datasets (generated by tools/make_orca_demo.py, deterministic)
  geo.py                  haversine, point-in-polygon, segment intersection, IDW (stdlib)
  decision/               geofence.py · evidence.py (fusion) · risk.py (decision engine) · route.py
  agents/                 planner.py · weather_agent.py · ocean_agent.py · geo_agent.py
  tools/registry.py       ToolSpec registry + capability-pruned menu
  explain.py              ledger + verifier + narration gate (+ optional Mistral narrator)
  pipeline.py             the controller (Orca service + Run)
  cards.py · i18n.py      evidence cards · English/Tamil catalogs
  api/server.py · ui.html stdlib server · single-file web app (SVG map, no CDN, works offline)
  evidence_pack.py        evidence records, answer→evidence linkage, alerts
  eval/                   controlled test set + metric runner (`python -m orca eval`)
  decision/hazards.py     hazards near a point, route restriction check
tests/orca/               136 unittest-style tests (pytest-compatible)
docs/                     SEAX_BLUEPRINT_COMPLIANCE · ORCA_ARCHITECTURE · ORCA_MIGRATION_AUDIT · DATA_SOURCES · SATQUERY_README
```

## 6. Tests

```bash
python -m unittest discover -s tests/orca -p "test_*.py"     # ORCA: 136 tests, ~40 s, no extra packages
python -m pytest                                              # whole repo incl. SatQuery (needs: pip install -e ".[dev,data]")
```
What the ORCA tests prove, beyond the happy path: threshold boundaries; missing/stale ⇒ `INSUFFICIENT_DATA`; proven danger beats missing data;
official ban/warning overrides thresholds; source-conflict priority is configurable; a constructed case where the decision **flips** with the source;
**no route ever enters a hard geofence** (property test) and an illegal path is caught by the post-check; a hallucinating narrator
("0.3 m, 87% safe") is rejected and the template used; an optimistic narrator cannot upgrade `INSUFFICIENT_DATA`; OFFLINE ages into `stale`;
LIVE with the network down never shows DEMO values; API returns 400 (not 500) on bad input; the UI has no CDN dependency and its map renderer is exercised under Node.

## 6b. Measured evaluation (`python -m orca eval`)

Fixed clock, synthetic data, 36 cases. **Core set (29 cases, written by the developer ⇒ a regression guard, not an accuracy estimate): 29/29.** Held-out **paraphrase set: 3/7 intents correct.** Evidence completeness 235/235 factual sentences linked; groundedness 36/36.
The evaluation found and drove the fix of real bugs while it was being built; the paraphrase failures were deliberately **not** tuned away. Full table and reading guide: `docs/SEAX_BLUEPRINT_COMPLIANCE.md` §E. Not measured: response clarity, real-world accuracy (there is no real data in DEMO).

## 7. Known limitations (honest list, in the order I'd fix them)

1. **LIVE is unverified against the real service.** Try: `python -m orca --mode LIVE ask "Can I fish tomorrow at 6 AM from Nagapattinam?"`. Expect `INSUFFICIENT DATA` (no public advisories/PFZ source) *by design*; if Open-Meteo works you'll see wave/wind from source `open_model`. If it fails, `doctor`/the trace say why.
2. Observation resolution: a question snaps to the nearest grid point (≤ 40 km, `ORCA_SNAP_KM`) and the offset is shown on every row. Ocean products are a *latest composite* (not time-resolved) and are labelled so.
3. Routes are evaluated at the **departure hour only** (not time-stepped), on a 0.02° grid; distances use a local planar approximation (<1% error). The route exposure index is ORCA-defined (`max(wave/2.5 m, wind/40 km/h)`), not a physical quantity.
4. Geofences/coastline are illustrative. Put official GIS at `orca/data/geofences_official.json` (same schema as `demo/geofences.json`) — it overrides the demo layer, including its coverage box.
5. Place matching uses a small gazetteer (8 Tamil Nadu/Puducherry ports). Only the Nagapattinam area has bundled coverage; elsewhere the answer is `INSUFFICIENT DATA`.
6. Rule-based query understanding (by design, so it works offline). Unusual phrasing ⇒ "I could not tell what you want to know", never a guess. No voice input (P2).
7. The unauthenticated local server is not hardened for public exposure.

## 8. Extending

- **New language:** add a dict to `orca/i18n.py` (missing keys fall back to English) + a keyword table in `orca/agents/planner.py`. Keep digits/units Latin so the verifier can check them.
- **New data source:** emit `Observation`s from a provider with a new `source_family`, add it to `ORCA_SOURCE_PRIORITY`. Fusion, conflicts and the decision engine need no changes.
- **Official thresholds:** edit `Thresholds` in `orca/config/settings.py` — they're copied into every decision's output.

## 9. Security note about your uploaded zip

`SatQueryAI.zip` contained a `.env` with what looks like a **real `MISTRAL_API_KEY`**. I did **not** copy it into this package. Anyone who has handled that zip has the key — **rotate it** at Mistral and keep `.env` out of archives (it is correctly git-ignored, so it never reached git).

---
*Decision support only. It does not replace official INCOIS/IMD warnings.*
