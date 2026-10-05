# Phase 2 design — the agentic prototype, as built

`PHASE1_SPEC.md` was a **forward** document: contracts to implement. This one is
**backward**. It describes the system that now exists, the decisions taken while
building it, and the places where the build departed from the plan — because the
departures are the parts a reader cannot reconstruct from the code, and the code
is where every other detail already lives.

Read `ARCHITECTURE.md` first: it states *what* the system is and the four
invariants it must hold. This document is *how* those became running code, and
what had to be decided that no plan anticipated.

- Landed on `phase2-prototype`, 26 Aug 2026, commits `9760bea`…`98021c3`
- 520 tests collected, 518 pass, 2 skipped (both download multi-GB weights)
- Runs end to end with no GPU, no model weights, no API key

---

## 1. What changed from the plan

Three departures. Each is recorded in `../CONTEXT.md` as well; the reasoning is
here.

### 1.1 The phase order inverted

The plan was: train the models (Phase 2), then build the controller (Phase 3),
then the GUI (Phase 4). What was built instead is the controller and the GUI
first, around **empty model slots**, with a deterministic implementation behind
each one.

The reason is risk, not convenience. The original order puts every demonstrable
artefact behind a five-hour Colab run that may or may not converge, on a free
tier that may or may not be available on the day. It also means the integration
work — the part with the most unknowns and the least glamour — happens last,
under time pressure, against models that are already sunk cost.

Inverting it means: the system demos today; the tool contract, the trace format
and the citation rules are settled and tested *before* a model is fitted to
them; and each trained model is a drop-in replacement for one slot rather than a
prerequisite for the first working build. The before/after delta — the
adaptation proof that gets scored — is measured by a harness that already
exists, which was Phase 1's whole justification.

The cost is real and should be stated: the demo today cannot count objects,
name a land use, or ground a free-form phrase, because those need the VLM. It
says so, every time, rather than approximating.

### 1.2 Gradio replaced by one static page

`CONTEXT.md` locked "Gradio (upgrade to React+Leaflet only if time allows)" on
speed grounds. The build produced `satquery/api/ui.html` instead: a single
self-contained page served from `/` by the same FastAPI process.

Gradio was chosen when the API was hypothetical. Once the API existed — and it
had to, because the tool contract is `/tools/v1/*` and the public surface is
specified in the approach doc §4.1 — Gradio became a *second* server, a second
dependency tree and a second place for the answer to be rendered. The page it
replaces is 500 lines of HTML/CSS/JS with no build step, no CDN and no
framework, and it renders the two things the assessment actually reads: the
withheld-tool table and the execution trace.

Speed was the criterion, and one file in one process was faster than Gradio
would have been. If the GUI needs a map, Leaflet still slots in later.

### 1.3 PDF via reportlab, not WeasyPrint

`CONTEXT.md` said "WeasyPrint/reportlab". WeasyPrint needs GTK and Pango
installed system-wide, which on Windows is not a `pip install`. The report is a
stack of headings, paragraphs, tables and images — exactly what reportlab's
Platypus flowables do without a browser engine. Where reportlab is absent the
report degrades to self-contained HTML rather than failing.

---

## 2. Shape of the system

```
POST /api/v1/query {query, image_ids | pair_id}
        │
        ▼
Controller.answer()                      satquery/agent/controller.py
  1 validate    resolve ids → reports    agent/validate.py, agent/session.py
  2 interpret   build the tool menu      agent/tools.py:build_menu
  3 route       question + menu → plan   agent/router.py
  4 execute     inject ids, run tools    agent/tools.py:run_tool
  5 compose     cite, drop, score        agent/compose.py
        │
        ▼
QueryResult ──► API response · GUI model · Markdown/HTML/PDF report
                                          satquery/report.py
```

| module | one line |
|---|---|
| `agent/session.py` | id → file workspace on disk; the indirection invariant 01 needs |
| `agent/validate.py` | per-image report, pair check, cached raster decodes |
| `agent/tools.py` | the seven tools, menu pruning, `scene_evidence()` |
| `agent/backends.py` | two implementations per served slot, and the selection policy |
| `agent/router.py` | Mistral function calling, offline rule router, plan normalisation |
| `agent/compose.py` | clause-level citation, dropping, confidence |
| `agent/controller.py` | the five steps, each writing a trace entry |
| `agent/trace.py` | the receipt: append-only, always JSON-serialisable |
| `agent/render.py` | previews and overlays, through the same transform the VLM sees |
| `api/app.py` + `api/ui.html` | HTTP surface and the whole GUI |
| `report.py` | one query → Markdown / self-contained HTML / PDF |

Supporting modules written in this phase because the loop needed them and they
did not exist: `io/raster.py`, `io/modality.py`, `preprocess/pairing.py`,
`preprocess/bands.py`, `preprocess/tiling.resample_to`, `indices.vari`,
`eval/report.py`, `cli.py`.

### One request, measured

The bi-temporal demo query on a 512×512 pair, laptop CPU, deterministic
backends:

| step | ms |
|---|---:|
| validate · resolve_inputs | 0.6 |
| interpret · build_tool_menu | 0.0 |
| route · rules | 0.0 |
| tool · indices | 336.0 |
| tool · change_map | 103.1 |
| tool · change_describe | 168.7 |
| compose · cite_and_score | 0.2 |
| **total** | **608.5** |

Nearly all of it is numpy over two 512² scenes. The orchestration itself —
validation, pruning, routing, composition — costs under a millisecond, which is
the argument for a hand-rolled loop over a framework.

---

## 3. The five steps

### Step 1 · validate

`Controller._resolve` turns ids into `ImageRecord`s and forms the pair. Two
behaviours matter:

- **Two images with no declared pair get one anyway.** The GUI should not need a
  second call before a question can be asked; the pair *check* still decides
  whether they are usable together.
- **A file that fails validation is still registered.** The user has to be able
  to see *why* their upload was refused, next to its name. `ok=False` plus the
  reader's own error, never an exception to the caller.

`validate_image` reports modality (declared beats inferred, provenance
recorded), size, bands, CRS, nodata fraction, and warnings for the cases that
have actually broken RS pipelines: constant scenes, heavy nodata, scenes too
large for a 768 px input, scenes too small to ground a box in.

### Step 2 · interpret (prune)

`build_menu(images, pair)` returns `offered` and `withheld`, where each withheld
entry carries a sentence a human can read. The requirement tokens are
deliberately coarse — `image`, `index`, `pair:bitemporal`, `pair:crossmodal` —
because the menu is a *capability* filter, not a policy engine.

This is invariant 02, and it is mechanical: the router receives JSON schemas for
the offered tools only, so a tool it was never told about cannot be called. The
A/B/C matrix in `ARCHITECTURE.md` is executed as a parametrised test
(`test_agent_tools.py::test_the_menu_matrix`) so the table and the code cannot
drift apart silently.

### Step 3 · route

Two routers behind one interface. `MistralRouter` posts to the chat-completions
endpoint with `tool_choice="any"` — mechanically forcing a plan rather than
prose — using `urllib` rather than the SDK, so the request body *is* the
auditable artefact with no client-side rewriting between it and the trace.
`RuleRouter` is keyword-based, deterministic and offline.

`build_router()` picks Mistral when `MISTRAL_API_KEY` is set and falls back to
the rule router on **any** failure — timeout, quota, malformed reply — with the
reason recorded in the plan and therefore in the trace. A demo must not die
because a free tier rate-limited it.

### Step 4 · execute

`Controller._inject` adds `image_id` / `pair_id` from the session and strips any
the router sent. Tools run in plan order through `run_tool`, which never raises:
a failure is a `ToolResult(ok=False)` with the exception text, and the plan
continues. A partial answer with an honest gap beats a 500.

### Step 5 · compose

Described in §5. It is the only component that produces the user-facing answer
object, and it is where invariant 04 is enforced rather than requested.

---

## 4. The two-backend rule

Every served slot has a model implementation and a deterministic one, and the
answer names which ran.

| slot | model | deterministic | status of the deterministic one |
|---|---|---|---|
| all five text tasks | Qwen2.5-VL-7B + QLoRA, **or** a cloud VLM | `evidence-rules` | phrases measurements; refuses counting, land-use naming, free-form grounding |
| change map | Change-Agent MCI | `cva-otsu` | CVA + robust brightness match + Otsu over a noise floor |
| optical–SAR fusion | TerraMind + head | `fusion-rules-v1` | index and backscatter rules, each stated in the output |
| routing | `mistral-small-latest` | `rules` | keyword routing |

The text slot has three implementations rather than two, because the fine-tuned
Qwen needs a GPU that a laptop does not have and the deterministic backend
cannot describe a scene:

| `SATQUERY_VLM` | backend | needs | sends imagery off the machine |
|---|---|---|---|
| `evidence` | `evidence-rules` | nothing | no |
| `mistral` | `mistral-vision:<model>` | an API key | **yes** |
| `qwen2_5_vl` | the served/tuned Qwen | a CUDA GPU, ~9 GB | no |

`mistral` reuses the *same* task prompts, the same box parser and the same
display transform as the served Qwen, so a difference in answers is a difference
in models rather than in wording. Its confidence is a fixed constant and says so
in the evidence — the API returns no token probabilities, and inventing a number
there would be a fabricated measurement in a system built to forbid exactly that.

**Selection.** `auto` promotes the model backend only when its dependencies
import *and* a GPU is visible, and never promotes the cloud backend at all. A 4-bit 7B on laptop CPU is minutes per token,
which in a live demo is indistinguishable from a hang, so the rule is explicit
rather than best-effort. Overrides: `SATQUERY_VLM`, `SATQUERY_CHANGE_BACKEND`,
`SATQUERY_FUSION_BACKEND`, `SATQUERY_ROUTER`. Every choice and its reason appear
in `satquery doctor`, in `/api/v1/health`, and in each answer's `backends` block.

**Unbuilt slots fail with an explanation.** `MciChangeMap` and `TerraMindFusion`
exist as classes whose `available()` returns `(False, why)`. Selecting one
explicitly raises with that sentence instead of an `ImportError` three frames
deep, and the slot stays visible in `doctor` rather than vanishing.

**The deterministic backends are not mocks.** They measure real quantities and
refuse what they cannot measure. Measured on synthetic scenes where the truth is
computable by hand: 7.32% reported for a 7.32% change; 0.0% for an unchanged
pair under an illumination shift; a 12.25% lake recovered as 12.25% water.

---

## 5. Decisions taken during the build

These are the ones no plan anticipated. Each is a problem found while building,
not a preference.

### 5.1 Otsu always returns a threshold

**Problem.** The first change detector reported **54% change on two identical
images** under an illumination shift. Otsu maximises between-class variance and
will happily split a unimodal histogram of pure residual noise down the middle.

**Decision.** Threshold at `max(Otsu, noise floor)`, where the floor is
`median + 3 × 1.4826 × MAD` of the change magnitude.

**Why.** The floor makes "nothing changed" an available answer. Robust
statistics specifically, because the changed pixels *are* the outliers, so a
non-robust spread estimate is inflated by the signal it is supposed to sit
under. Both thresholds are reported in the evidence, so a reader can see which
one bound.

`agent/backends.py:noise_floor` · `test_agent_backends.py::test_an_unchanged_pair_reports_no_change_despite_an_illumination_shift`

### 5.2 Brightness matching must be robust too

**Problem.** A large bright change (a new development covering a third of the
scene) dragged a mean/std brightness match, scaling the whole later image down
and smearing false change across the unchanged remainder.

**Decision.** Match on median and MAD. Gain and offset are reported, so a gain
far from 1.0 flags "these are not the same sensor".

### 5.3 Modality statistics must be per band

**Problem.** Found while preparing the demo: a **4-band optical stack was
classified as SAR**. Statistics were pooled across bands, and the visible bands
of a vegetated scene are dark while the NIR band is bright, so the pooled
histogram has the long right tail that reads as speckle.

**Consequences had it shipped.** Fusion withheld from a valid optical+SAR pair,
and the optical scene sent through a dB stretch that renders it black — both
silent, both catastrophic for the one demo that shows fusion.

**Decision.** Shape statistics (median, p99, tail ratio) are computed per band
and medianed across bands; range statistics stay global. Separately, **three or
more bands rules out SAR outright** rather than voting against it — no product
in scope ships that many co-registered power bands.

`io/modality.py:_pixel_stats` · `test_io_modality.py::test_a_multispectral_stack_is_never_read_as_sar`

### 5.4 An index whose bands are absent is withheld, never substituted

**Problem.** Most benchmark tiles and most user uploads are three-band RGB with
no NIR. NDVI is not computable. The tempting fix is to compute *something*
green-ish and call it NDVI.

**Decision.** `preprocess/bands.py` resolves band roles first; an index whose
roles are missing is absent from the tool menu with a reason. RGB-only scenes are
offered **VARI** instead — a published RGB greenness index — and every mention
of it is labelled "RGB proxy — no NIR band, so not NDVI".

**Why.** A substituted index is a number that computes, looks plausible, and is
unfalsifiable downstream. It would corrupt the citation chain at its root.

### 5.5 Evidence accumulates across a plan

**Problem.** Fusion emits class percentages and no prose. A plan of
`[fusion]` therefore ended with measurements and no answer, and the composer
correctly refused.

**Decision.** Two parts. `ToolContext.accumulated` collects each finished tool's
numeric evidence, and text tools see it on top of their own scene measurements —
so the sentence can cite what an earlier tool measured. And plan normalisation
appends a text tool after a lone `fusion` call.

**Why.** This is invariant 04 in operation: specialists emit numbers, one
component speaks. Without the accumulation the "one component" has nothing to
speak *from*.

### 5.6 The plan is normalised before it runs

**Decision.** `Plan.normalized(menu, query)` applies four repairs, each recorded
as a note in the plan and therefore in the trace:

| repair | reason |
|---|---|
| drop calls to withheld or non-existent tools | could only come from the model inventing a name |
| insert `change_map` before a lone `change_describe` | a description without the geometry behind it |
| insert `indices` before any text tool | invariant 03, applied to the plan rather than trusted to a prompt |
| append a speaking tool to a plan that only measures | invariant 04 — see below |

**Why.** It is what makes a cloud LLM safe to route with. The model may propose
anything; the pipeline's invariants are enforced on the way out, not requested in
a prompt.

**The fourth repair was found, not predicted.** On the first live query with a
real key, `mistral-small-latest` answered "what changed between these two dates"
with a plan of `[change_map]` and nothing else. `change_map` emits a mask and
geometry — by invariant 04 it may not address the user — so the query measured
the scene correctly and then answered nothing, and the composer refused. To a
user that reads as a failure when the measurement had in fact succeeded.

The fix is in two places on purpose: the system prompt now asks explicitly for a
plan ending in `vqa`, `caption`, `ground` or `change_describe`, and
`_ensure_something_speaks()` appends one when the model does not. The prompt
improves the plans; the normalisation guarantees them. Either alone is weak —
prompt compliance is probabilistic, and a guarantee with no prompt behind it
means fighting the model on every request.

`agent/router.py:_ensure_something_speaks` ·
`test_agent_router.py::test_a_plan_that_only_measures_gets_a_voice`

### 5.7 Citation is three-way, not "cite or die"

**Problem.** `ARCHITECTURE.md` says the composer "drops any clause it cannot
attribute". Applied literally that deletes every caption, since a caption is
qualitative by definition — and captioning is a mandatory task.

**Decision.** Each clause is classified:

| verdict | condition | outcome |
|---|---|---|
| **cited** | makes a numeric claim, every figure matches evidence within tolerance | kept, with the citation attached |
| **dropped** | makes a numeric claim, a figure matches nothing | **deleted** from the answer, recorded in the trace |
| **qualitative** | no numeric claim | kept, flagged, and it lowers confidence |

**Why.** The rule that matters — *a number must be backed by that number, and an
unbacked number is deleted rather than softened* — is preserved exactly. What
changes is that description is permitted and *measured*: `cited_fraction` is
reported, so a reader sees how much of an answer is measurement and how much is
prose.

### 5.8 Citation matching is unit-aware

**Problem.** "The largest region covers 3.0%" was credited to `bands = 3`. The
closest-value match won on distance.

**Decision.** A number carries the unit written next to it; `%` may only be
backed by a percentage key, `dB` only by a backscatter key. Structural facts
(`bands`, `width`, `height`) lose every tie. Small integers (≤3, no decimal
point) are not treated as claims at all — "in 2 regions" restates the input, and
demanding evidence for it would delete ordinary English.

### 5.9 Confidence is composed, never asserted

```
confidence = backend_confidence
           × (0.5 + 0.5 × cited_fraction)     how much of the answer is measured
           × (0.85 + 0.15 × modality_conf)    how sure we are of the input
           − 0.1 per dropped clause           evidence of confabulation
           − 0.02 per validation warning
clipped to [0.05, 0.95]
```

Never 0 and never 1: the system does not claim certainty, and it does not
dismiss a measurement it made. No model is asked what its confidence is.

### 5.10 Grid mode for pairs without a CRS

**Problem.** LEVIR-CD and CDVQA pairs are 8-bit PNG tiles with no
georeferencing. Requiring a CRS would refuse the entire evaluation set.

**Decision.** `check_pair` runs in **geo mode** (footprint IoU, GSD ratio,
CRS equality) when both rasters are georeferenced, and **grid mode** (identical
pixel shape) otherwise — and the verdict says which mode produced it, with grid
mode recording that co-registration is *the dataset's claim, not a measurement*.
Differing CRSs are refused rather than reprojected silently; resampling inside a
validation step is exactly the invisible transform that makes a wrong answer
untraceable.

### 5.11 A session is a folder

No database, no server-side state beyond the filesystem. `<runs_dir>/sessions/<id>/`
holds the registered rasters, their previews, per-query artefacts and one JSON
per answered query. Restarting the API loses nothing; "reset the demo" is
`rm -rf`; "ship the evidence" is a zip. The manifest is re-validated on load
rather than trusted, so a hand-edited manifest cannot inject a stale verdict.

Not thread-safe by design — one user, one GUI. Stated rather than discovered.

### 5.12 An area is not a count

**Problem.** "How many buildings are there?" was answered with "built-up cover
measures 5.0%" — true, responsive-looking, and not the question.

**Decision.** The deterministic text backend checks for count questions *before*
any concept match and refuses. Substituting an area for a count is the most
plausible-looking wrong answer this system can produce.

### 5.13 An open question must not go through a short-answer prompt

**Problem.** With a real VLM behind the text slot, "What changed between these
two dates?" came back as **"Vegetation loss"**. The tool had classified any
question as change-*VQA*, whose template instructs "the shortest possible
answer: a number, a single word, or yes/no". The answer was accurate, thin, and
cited nothing — confidence 0.35.

**Decision.** `_is_open_question()` splits on the leading word. "What",
"describe", "tell me", "explain" get the descriptive template; "is", "are", "how
many", "how much" keep the terse one.

**Result.** The same query now answers "Vegetation decreased by 5.79% in the
north-west quadrant of the scene", citing `changed_area_pct` and
`centroid_quadrant`, at confidence 0.69.

**Why it is crude on purpose.** Only a leading word decides. Erring towards
"open" costs verbosity; erring the other way loses the evidence-citing paragraph
the whole pipeline exists to produce.

### 5.14 Citation verifies the number, not the predicate

Worth stating as a known limit, because the demo shows it. In the example above
the model wrote "vegetation decreased by 5.79%" while `changed_area_pct = 5.79`
measures *total changed area*, not vegetation specifically. The composer accepts
it: the value matches an evidence key within tolerance.

So the guarantee is precisely "every number in the answer was produced by a named
tool" — not "every number is attached to the right quantity". Closing that gap
means entailment checking between a clause and an evidence key, which is a
research problem rather than a threshold. Until then the trace shows the key that
backed each claim, so a reader can see the binding and judge it.

### 5.15 The tool endpoints are public

`/tools/v1/*` is exposed, not internal. A tool contract nobody can call is a
tool contract nobody can check: an assessor can POST an image id to
`/tools/v1/indices` and get back the same number the answer cited, without going
through the loop that produced it.

### 5.16 `api/app.py` has no `from __future__ import annotations`

FastAPI resolves endpoint annotations at decoration time. The FastAPI names are
imported *inside* `create_app()` to keep the dependency optional, so as strings
those forward references cannot be resolved from module globals and every route
with an `UploadFile` or `Body` parameter fails to build. The omission is
deliberate and commented at the top of the file, because it looks like an
oversight and would be "fixed" by the next person otherwise.

---

## 6. Data contracts

Defined in code; the shapes are stable and everything downstream reads them.

| type | where | carries |
|---|---|---|
| `ToolResult` | `agent/tools.py` | `output`, `evidence`, `artifacts`, `backend`, `backend_kind`, `duration_ms`, `error` |
| `Composition` | `agent/compose.py` | `answer`, `confidence`, `clauses[]`, `citations[]`, `dropped[]`, `warnings[]` |
| `QueryResult` | `agent/controller.py` | everything above plus `menu`, `plan`, `backends`, `images`, `pair_id`, `trace` |
| `TraceEntry` | `agent/trace.py` | `step`, `name`, `status` (`ok`/`skipped`/`withheld`/`error`), `detail`, `params`, `output`, `duration_ms` |

`QueryResult.to_dict()` is simultaneously the API response, the GUI's model and
the report's source. There is no second representation of an answer anywhere in
the system, which is why a report cannot disagree with what the user saw.

`trace.jsonable()` coerces numpy scalars, arrays, Paths, dataclasses and sets on
the way in — a serialisation failure at the end of a query would lose the whole
trace, which is the one artefact the assessment reads.

---

## 7. Failure behaviour

| failure | behaviour |
|---|---|
| unreadable upload | registered with `ok=False` and the reader's error; query refuses with the reason |
| pair fails the geometric check | change and fusion tools withheld, reason in the trace and in the refusal |
| router times out / rate-limits | falls back to the offline router; `plan.fallback_reason` records it |
| router returns prose | `RouterError`; same fallback path |
| router names a tool that was withheld | dropped in normalisation, recorded as a plan note |
| one tool raises | `ToolResult(ok=False)`; plan continues; composer reports the gap |
| no text tool succeeds | composer writes the refusal itself, listing measurements and withheld tools |
| every clause quotes an unbacked number | answer becomes a refusal rather than an empty string |
| reportlab absent | report degrades to self-contained HTML with a warning |
| rasterio absent | GeoTIFF opens through PIL with a warning; geo-dependent checks skip |

---

## 8. Security notes

Scoped to what a single-user localhost demo needs, and no further.

- **Upload filenames are attacker-controlled.** `session._safe()` strips path
  separators and traversal at the single point where a name reaches the
  filesystem. Asserted in `test_api_and_report.py`.
- **The artefact route is the only URL→file mapping.** It rebuilds the path from
  the session store and checks the resolved path is inside the artefacts
  directory before serving.
- **No auth, binds to 127.0.0.1.** Building auth the challenge does not ask for
  would be scope creep; binding publicly without it would be worse. Stated in
  the module docstring so nobody deploys it by accident.
- **No imagery leaves the machine by default, and never for routing.** Invariant
  01 is the security property as well as the architectural one — relevant
  because the Mistral free tier requires a data-training opt-in. The one way
  imagery goes anywhere is `SATQUERY_VLM=mistral`, which serves the text slot
  from a cloud vision model; it is unreachable from `auto`, it announces the
  cost in the selection reason, and its `describe()` reports
  `sends_imagery: True` so anything displaying backends shows it without
  special-casing. Routing stays pixel-free under every configuration.
- **One credential, one gitignored file.** `config.load_env_file()` reads `.env`
  at import; a real environment variable always beats it. The loader returns
  which *names* it set and never their values, so it cannot print a key into a
  log or a bug report, and `satquery doctor` reports presence and file location
  only. A test asserts the key never appears in the router's request body.

---

## 9. What is deliberately not built

| gap | why | who fills it |
|---|---|---|
| nine of ten dataset adapters | each needs a real download to verify its layout; guessing produces `# VERIFY` comments and wrong numbers | Phase 4 |
| `metrics/caption.py` | BLEU/METEOR/CIDEr; `DEFAULT_METRICS` already points at it | Phase 4 |
| MCI and TerraMind vendored | research repos, not `transformers` architectures; needs a T4 attached | Phase 3 |
| tiling at inference | `tile_image`/`merge_tiles` exist and are tested; the tools resize instead. Large scenes warn | Phase 5 |
| grounding beyond land cover | needs the VLM; today only thresholded regions, and it says so | Phase 3 |
| auth, multi-user, concurrency | single-user demo tool | not planned |

---

## 10. How to check the claims in this document

| claim | command |
|---|---|
| everything passes | `python -m pytest -q` → 518 passed, 2 skipped |
| the menu matrix matches the architecture doc | `pytest tests/test_agent_tools.py::test_the_menu_matrix` |
| no imagery reaches the router | `pytest tests/test_agent_router.py -k request_body` |
| withheld tools are recorded before routing | `pytest tests/test_agent_controller.py -k withheld` |
| unbacked numbers are deleted | `pytest tests/test_agent_compose.py -k deletes_its_clause` |
| an unchanged pair reports no change | `pytest tests/test_agent_backends.py -k unchanged_pair` |
| a multispectral stack is not SAR | `pytest tests/test_io_modality.py -k multispectral` |
| which backend would serve right now | `satquery doctor` |
| the whole loop, end to end | `satquery ask "what changed?" demo/city_2019.png demo/city_2021.png --trace` |

Test distribution over the Phase 2 modules: backends 32, router 27, API and
report 27, pairing and bands 26, tools 23, compose 19, controller 18, modality
17, raster 13, CLI 10.

---

*Written 26 Aug 2026, describing `phase2-prototype` at `98021c3`.*
