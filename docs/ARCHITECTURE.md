# SatQuery AI — architecture of record

An agentic vision–language assistant for satellite imagery. Upload one or two
rasters, ask a question in plain English, get an answer that cites the numbers it
was derived from — plus an execution trace an assessor can audit line by line.

This file is what gets served, what talks to what, and the four invariants that
keep it from collapsing into a generic VLM wrapper. Rationale lives in
`../SatQuery_AI_Approach.md` §2.1, §3.1 and Part 4; project state in
`../CONTEXT.md`; and **how this became running code — with the decisions taken
during the build — is `PHASE2_DESIGN.md`**. A rendered version of this page with
proper diagrams is `architecture.html` — open it in a browser.

- ISRO / SAC challenge
- 3 served models, 1 fine-tune
- resident on one free 16 GB T4
- everything served is Apache-2.0

---

## Four rules the code enforces, not the prompt

A generic VLM wrapper is disqualified outright, so the load-bearing part of this
system is not which model answers. It is the set of properties that hold no
matter what the language model decides to do.

| # | Invariant | What it means |
|---|---|---|
| 01 | **The router never sees imagery** | Mistral classifies the task and sequences tools. The executor injects `image_id` from the session itself, so raster data is never a payload on a cloud API call. |
| 02 | **The validator prunes the tool menu** | Tools an input configuration cannot support are absent from the schema the router receives. Misrouting the obvious cases becomes impossible rather than unlikely. |
| 03 | **Deterministic tools feed the VLM** | NDVI/NDWI, SAR dB statistics, mask areas and box coordinates go *into* the prompt as structured context. This is the SAR fix: never ask a VLM to read speckle — hand it numbers and ask it to explain them. |
| 04 | **One model owns every sentence** | Specialists emit numbers only. Every user-facing clause comes from the tuned Qwen, and the composer drops any clause with no tool output behind it — so hallucination has exactly one place to be policed. |

---

## The stack

The controller is a hand-rolled plan → execute → compose loop rather than a
framework, because every hop has to be auditable. Below it, three models stay
resident at once — no request ever waits for weights to be swapped in.

```
┌─ WEB GUI (Gradio) ───────────────────────────────────────────────────────────┐
│ upload 1–2 rasters · query box · overlay viewer · confidence · trace · PDF   │
└──────────────┬────────────────────────────────────────┬──────────────────────┘
   POST /api/v1/query                    answer · confidence · evidence · trace
   {query, image_ids | pair_id}                                        ▲
               ▼                                                       │
┌─ AGENTIC CONTROLLER — hand-rolled, auditable tool loop ──────────────┴───────┐
│                                                                              │
│ ┌─1 VALIDATE───┐→┌─2 INTERPRET──┐→┌─3 ROUTE──────┐→┌─4 EXECUTE───┐→┌─5 COMPOSE┐
│ │ format·bands │ │ vqa|caption| │ │ mistral-small│ │ inject      │ │ cite     │
│ │ dtype·CRS    │ │ ground|change│ │ cloud, tools │ │ image_id,   │ │ evidence,│
│ │ modality     │ │ |fusion      │ │ NEVER pixels │ │ time & log  │ │ confidence
│ │ co-reg check │ │              │ │              │ │ every hop   │ │ emit trace
│ └──────┬───────┘ └──────────────┘ └──────▲───────┘ └──────┬──────┘ └──────────┘
│        └───── prunes the tool menu ──────┘                │                   │
│              before the router sees it                    │                   │
└───────────────────────────────────────────────────────────┼───────────────────┘
                                                            │
         POST /tools/v1/*  — strict JSON in / out, every call timed and logged
    ┌───────────────┬───────────────┬──────────────┬─────────┴─────────────────┐
    ▼               ▼               ▼              ▼
┌─DETERMINISTIC─┐ ┌─TERRAMIND-1.0─┐ ┌─CHANGE-AGENT─┐ ┌─QWEN2.5-VL-7B + RS QLoRA─┐
│ /v1/indices   │ │ /v1/fusion    │ │ MCI  ~230M   │ │ /v1/{vqa, caption,       │
│ NDVI · NDWI   │ │ + task head   │ │ /v1/change/  │ │       ground,            │
│ dB statistics │ │ S2L2A + S1GRD │ │      map     │ │       change/describe}   │
│ band compose  │ │ natively      │ │ mask ·       │ │ five mandatory tasks,    │
│ tiling·co-reg │ │ → water /     │ │ changed_area │ │ one LoRA                 │
│ numpy+rasterio│ │   built-up    │ │ _pct ·       │ │ THE ONLY VOICE THE USER  │
│ no weights, 0 │ │ Apache, 1–2GB │ │ num_regions  │ │ EVER HEARS               │
│               │ │               │ │ Apache, <4GB │ │ Apache · 4-bit · 7–9 GB  │
└───────┬───────┘ └───────┬───────┘ └──────┬───────┘ └────────────▲─────────────┘
        │                 │                │                      │
        └─────────────────┴────────────────┴──────────────────────┘
       structured evidence into the prompt — mask %, class probabilities,
       index means, dB stats.  Numbers, never prose; a specialist never
       addresses the user.
```

Evidence flows left to right; prose only ever exits the Qwen box. The three
components on the left never produce a user-facing sentence — they hand Qwen
figures it is then required to cite.

> EarthDial-4B appears in the evaluation table as the zero-shot RS baseline and
> is deliberately **not** in this serving path.

---

## One request, end to end

Two optical GeoTIFFs of the same district eighteen months apart, and the question
*“What changed between these dates, and where?”* Every step writes its own entry
into the trace.

### 1. Validator — both rasters open, and the pair is compatible

Format sniffed through rasterio, modality inferred from band count and dtype,
then the pair check: matching CRS, overlapping footprints, common ground sample
distance. A pair that fails here is refused with a reason rather than answered
badly.

```
image_a: 12 bands, uint16, EPSG:32644, 10 m GSD  → optical
image_b: 12 bands, uint16, EPSG:32644, 10 m GSD  → optical
pair:    footprint_overlap 0.98 · co_registered true · pair_type "bitemporal"
```

### 2. Registry — the tool menu is built for this configuration

Because a valid bi-temporal pair exists, the change tools are registered.
Because neither image is SAR, `fusion` is withheld — and the trace records that
it was withheld, and why.

```
offered:  vqa · caption · ground · indices · change/map · change/describe
withheld: fusion   ← reason: "no SAR image in pair"
```

### 3. Router (Mistral) — a small JSON plan, produced without ever seeing a pixel

The router receives the question, the pruned tool schema and the validation
summary, all as text. It returns a plan; it does not return an answer.

```json
{ "task":  "change_vqa",
  "tools": ["change_map", "change_describe"],
  "params": {"detail": "detailed"} }
```

### 4. Executor → MCI — the change map comes first, as evidence

The executor injects the `pair_id` itself and calls the specialist. What comes
back is geometry and arithmetic — no language at all.

```
mask_url:          /files/q123/change_mask.png
changed_area_pct:  14.2   ·  num_regions: 7
centroid_quadrant: "north-east"  ·  largest_region_pct: 6.1
```

### 5. Executor → Qwen — those numbers enter the prompt, and then the VLM speaks

The tuned Qwen receives both images *and* the MCI statistics as structured
context. This ordering is invariant 03 — the deterministic result constrains the
sentence, not the other way round.

```
POST /tools/v1/change/describe
  images:   [image_a, image_b]
  evidence: {changed_area_pct: 14.2, num_regions: 7, quadrant: "north-east"}
  question: "What changed between these dates, and where?"
```

### 6. Composer — every clause is tied back to a tool output

The composer assembles answer, confidence and evidence list, and drops any clause
it cannot attribute. Confidence comes from generation self-consistency and the
specialist's own scores — not from the model asserting a number.

```
answer: "Built-up area increased by approximately 14% between the two dates,
         concentrated in the north-east quadrant."
         ├─ "approximately 14%"   ← change_map.changed_area_pct 14.2
         └─ "north-east quadrant" ← change_map.centroid_quadrant
confidence: 0.82
```

### 7. Response — the answer, and the receipt

The GUI renders the overlay, the confidence panel and the full trace; the same
content is written to a downloadable PDF. The trace is what gets assessed, so it
is a first-class output rather than a debug log.

```
GET /api/v1/queries/q123        → answer · confidence · evidence[]
GET /api/v1/queries/q123/trace  → validation · plan · tool calls · timings
GET /api/v1/queries/q123/report → PDF
```

---

## Why misrouting the obvious cases is impossible

All seven tools exist in the codebase for every request. What differs is how many
of them are described to the router at all — and that is decided by the
validator's report, before the language model is invoked.

|  | **A** — 1 optical image | **B** — 2 optical, co-registered | **C** — optical + SAR |
|---|---|---|---|
| *validated as* | single image, no pair | `pair_type: bitemporal`, 98% overlap | `pair_type: crossmodal`, resampled |
| `vqa` | ● | ● | ● |
| `caption` | ● | ● | ● |
| `ground` | ● | ● | ● |
| `indices` | ● | ● | ● |
| `change/map` | ○ needs a pair | ● | ○ not bi-temporal |
| `change/describe` | ○ needs a pair | ● | ○ not bi-temporal |
| `fusion` | ○ needs optical + SAR | ○ needs optical + SAR | ● |

● described to the router  ·  ○ absent from the schema, reason logged to the trace

The router cannot call what it was never told about. A bi-temporal pair whose
footprints fail to overlap loses the change tools the same way, with the overlap
figure recorded in the trace — so an assessor sees a refusal with a number
attached rather than a confident wrong answer.

---

## The tool contract

Each row is one FastAPI endpoint and one entry in the function-calling schema.
The executor supplies `image_id` and `pair_id`; the router supplies everything
else.

| Endpoint | Backing model | Request → response |
|---|---|---|
| `/tools/v1/vqa` | Qwen2.5-VL + RS QLoRA | `{image_id, question}` → `{answer, confidence}` |
| `/tools/v1/caption` | same VLM | `{image_id, detail: brief\|detailed}` → `{caption, confidence}` |
| `/tools/v1/ground` | same VLM, native bbox | `{image_id, phrase}` → `{boxes: [{bbox, score}], overlay_url}` |
| `/tools/v1/change/map` | Change-Agent MCI | `{image_a, image_b}` → `{mask_url, changed_area_pct, num_regions, stats}` |
| `/tools/v1/change/describe` | same VLM, given MCI stats | `{image_a, image_b, question?}` → `{answer, confidence}` — question omitted means change description, present means change-VQA |
| `/tools/v1/fusion` | TerraMind + head + rules | `{optical_id, sar_id, targets}` → `{masks, area_pct, labels, confidence}` |
| `/tools/v1/indices` | deterministic, numpy | `{image_id, index: NDVI\|NDWI\|SAR_BACKSCATTER}` → `{map_url, mean, threshold_mask_url, area_pct}` |

Public surface (frontend ↔ backend): `POST /api/v1/images`, `GET
/api/v1/images/{id}[/preview]`, `POST /api/v1/pairs`, `POST /api/v1/query`, `GET
/api/v1/queries/{id}[/trace|/report]`, `GET /api/v1/tools`. Full descriptions in
`../SatQuery_AI_Approach.md` §4.1, and live at `/docs` once `satquery serve` is
running.

The tool endpoints are public deliberately. A tool contract nobody can call is a
tool contract nobody can check: an assessor can `POST /tools/v1/indices` with an
image id and get back the same number the answer cited, without going through
the loop that produced it.

---

## Why exactly three models are served

The free tier is a single 16 GB T4. Anything over that ceiling means loading and
unloading weights per request — a slow demo, plus a job queue to write and debug.
Collapsing six candidates to three is what buys a resident interior.

| Component | Owns | Licence | Peak VRAM |
|---|---|---|---:|
| Qwen2.5-VL-7B + QLoRA | VQA · caption · grounding · change-VQA · change-caption | Apache-2.0 | 7–9 GB |
| Change-Agent MCI | change mask + statistics, as evidence | Apache-2.0 | < 4 GB |
| TerraMind-1.0-base + head | optical–SAR fusion → water / built-up | Apache-2.0 | 1–2 GB |
| Deterministic tools | NDVI/NDWI · dB stats · tiling · co-registration | — | 0 |
| **Resident total** | typical ≈ 10 GB, worst case against a 16 GB ceiling | | **~15 GB** |
| *Rejected six-model plan* | *adding TEOChat-7B and Grounding-DINO — would not fit* | | *19–26 GB* |

The T4 is SM75, which fixes two things everywhere in the stack: fp16 only —
never bf16 — and sdpa attention, never FlashAttention-2.

---

## Where the build stands

- **Phase 1 — landed.** Loaders and evaluation harness, built deliberately
  *before* any fine-tuning, because the before/after benchmark delta is the
  adaptation proof. numpy is the only hard dependency, and every metric is
  validated against an oracle predictor so a metric bug surfaces locally rather
  than after a five-hour Colab run.
- **Phase 2 — landed.** The whole loop above, running: validator, tool menu,
  router, executor, composer, trace, the seven tool endpoints, the web app and
  the PDF report. 516 tests, all green on a laptop.

  It runs **with no GPU, no model weights and no API key**, because every served
  slot has two implementations and the answer always names the one that ran:

  | slot | model backend | deterministic backend |
  |---|---|---|
  | every sentence | Qwen2.5-VL-7B (+ RS QLoRA) | `evidence-rules` — phrases the measurements, refuses what it cannot measure |
  | change map | Change-Agent MCI | `cva-otsu` — change vector analysis, robust brightness matching, Otsu over a noise floor |
  | optical–SAR fusion | TerraMind-1.0-base + head | `fusion-rules-v1` — index and backscatter rules, each stated |
  | routing | Mistral `mistral-small-latest` | `rules` — offline keyword routing |

  The deterministic backends measure real quantities and refuse the questions
  they cannot answer, so the demo is honest rather than mocked: on a synthetic
  scene the change detector reports 7.32% for a 7.32% change and 0.0% for an
  unchanged pair under an illumination shift. What they cannot do — count
  objects, name a land use, ground a free-form phrase — they say.

  Two decisions changed during this build and are recorded in `../CONTEXT.md`:
  the GUI is a single self-contained HTML page served by the same FastAPI
  process rather than Gradio, and the phase order inverted — the app was built
  around empty slots so that each trained model is a drop-in replacement rather
  than a prerequisite.

- **Phases 3–5 — ahead.** The single QLoRA run on the mixed RS instruction set;
  specialist training (MCI on LEVIR-MCI, the TerraMind head on BigEarthNet-MM);
  the remaining dataset adapters and the captioning metrics; then hardening.
  Tuned weights get promoted per slot only where the harness says evaluation
  actually improved.

---

*Model and free-tier facts verified 26 Aug 2026. Phase 2 landed 26 Aug 2026.*
