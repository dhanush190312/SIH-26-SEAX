# SatQuery AI — Project Context

> Dense project-state file. Full reasoning, tables, and API details live in `SatQuery_AI_Approach.md` (same folder); the Phase 2 build as it was actually made, and the decisions taken inside it, are in `docs/PHASE2_DESIGN.md` — interior-model rationale in its §3.1, composition rules in §3.3. Last updated: 2026-08-26 (Phase 2 landed).

## What we're building

**SatQuery AI** — an agentic vision-language assistant for satellite imagery (ISRO/SAC challenge). User uploads 1–2 remote-sensing images (GeoTIFF/TIFF; PNG/JPEG in benchmark mode) + asks a natural-language question; an agent validates inputs, routes to specialist models, and returns an evidence-grounded answer with confidence + an auditable execution trace. Judges score the trace, the benchmark metrics, and a hidden Cartosat-2S optical + RISAT SAR eval set.

**Mandatory scope:** single-image VQA · one more single-image task (we chose captioning AND grounding) · bi-temporal change description/VQA · optical–SAR fusion · agentic orchestration · at least one component fine-tuned on open RS data (BigEarthNet prescribed) · GUI/web app with report download. A generic VLM wrapper is disqualified.

## Status

- [x] Problem decoded, approach written (`SatQuery_AI_Approach.md`)
- [x] Router model chosen & free tier verified (Mistral)
- [x] All specialist-model slots verified (weights downloadable, licenses checked, free-GPU feasible) — 2026-08-26
- [x] API spec drafted (Part 4 of approach doc)
- [x] Phase 1: data loaders + eval harness (metrics, preprocessing, run artefacts)
- [x] **Phase 2: the working prototype** — agentic loop, 7 tools, FastAPI + web app,
      PDF/Markdown reports, CLI, 516 tests. Runs end to end with no GPU, no weights
      and no API key via deterministic backends that report themselves as such.
- [ ] Phase 3: Qwen2.5-VL-7B QLoRA run; MCI on LEVIR-MCI; TerraMind head on BigEarthNet-MM
- [ ] Phase 4: dataset adapters for the remaining nine benchmarks + caption metrics,
      then the before/after table
- [ ] Phase 5: hardening + demo

**The phase order changed from the original plan** (build the models, then the app)
to: build the app around empty slots, then fill them. The prototype therefore
demos today, every trained model is a drop-in replacement for one slot, and the
before/after delta is measured by a harness that already exists.

## Decisions (locked, with rationale)

| Decision | Choice | Key facts |
|---|---|---|
| Router LLM | **Mistral `mistral-small-latest`**, free Experiment tier | Native function calling + parallel tools + JSON schema; ~1 req/sec *per model*, ~1B tokens/month; needs phone verification + data-training opt-in. `mistral-large-latest` also free for hard queries. |
| Router fallbacks | Gemini Flash-Lite (~1,000 RPD) → Groq gpt-oss-20b/120b (1,000 RPD) → **local Qwen3-4B via Ollama** (offline demo insurance) | Groq's Llama models were shut down for free tiers Jul–Aug 2026; OpenRouter free needs one-time $10 for 1,000/day; SambaNova free = 20 req/DAY. |
| Core VLM — **the only served text model** (VQA + caption + grounding + change-VQA + change-caption) | **Qwen2.5-VL-7B-Instruct + QLoRA** (Unsloth) | Apache-2.0 (⚠️ NOT the 3B — Qwen Research License, non-commercial). Beats 2023-era RS VLMs after one Colab run. One fine-tune, five mandatory tasks. |
| RS zero-shot baseline — **eval table only, NOT served** | **EarthDial-4B** | CVPR'25, multi-sensor (RGB/MS/SAR/temporal), MIT code, HF `akshaydudhane/EarthDial_4B_RGB`. Demoted 2026-08-26: untuned Qwen covers day-1 zero-shot and becomes the tuned model, so serving both doubles the prompt surface for an ensemble we cannot tune in time. |
| Change map+caption (**evidence provider, never user-facing**) | **Change-Agent MCI** (~230M) | HF `lcybuaa/Change-Agent`, Apache-2.0; only model fully re-trainable on a T4; mask + caption in one. Displaced ChangeFormer/BIT (map-only). |
| Change-VQA + change-caption | **the same tuned Qwen**, conditioned on MCI mask stats | TEOChat-7B **dropped** 2026-08-26: CDVQA + LEVIR-CC are already in the one LoRA mix, and arXiv 2604.18429 found tuned native Qwen beats structured pipelines on RS change-VQA. Dropping it also removes the stack's only Llama-2-community licence and 6–7 GB. |
| Optical–SAR fusion | **TerraMind-1.0-base** (IBM/ESA) via TerraTorch | Apache-2.0; natively ingests S2L2A+S1GRD; ships Colab notebooks + S1+S2 water-seg config. CROMA displaced: its edge was matching BigEarthNet-MM geometry, which *is* TerraMind's native input. |
| Grounding | **Qwen native bbox** (same LoRA run: DIOR-RSVG + VRSBench referring) | Grounding-DINO **dropped** 2026-08-26: a second training run + serving process + frozen BERT encoder, for a *bonus* task (captioning alone satisfies "one more single-image task"). Reopen ONLY if harness shows acc@IoU0.5 < 0.30 on VRSBench referring with time left. GeoGround rejected: NO license. |
| Fine-tune stack | **Unsloth** | Official free-T4 Colab notebook for Qwen2.5-VL-7B QLoRA; measured ~7.1 GB peak, ~49 min/3K samples/epoch. Alternative: LLaMA-Factory (multi-dataset YAML). Axolotl VLM = beta, avoid. |
| Frontend/backend | **FastAPI + one static HTML page** (was: Gradio), rasterio optional | Changed 2026-08-26 during the Phase 2 build. Gradio was chosen for speed, but the API had to exist anyway, and a single self-contained page served from `/` costs one fewer dependency, keeps everything in one process, needs no build step or CDN, and renders the trace and the withheld-tool table the way the assessment needs. Report via reportlab — WeasyPrint needs GTK/Pango system-wide on Windows. |

## Architecture (one line each)

- **Router never sees pixels** — executor injects image ids; Mistral only classifies, picks tools, composes.
- **Validator pre-filters the tool menu** by input config (fusion tool only offered when a valid optical+SAR pair is registered) → misrouting structurally impossible.
- **Tools = FastAPI endpoints, strict JSON**: `/tools/v1/{vqa, caption, ground, change/map, change/describe, fusion, indices}`. Public API: `/api/v1/{images, pairs, query, queries/{id}[/trace|/report], tools}`.
- **Every claim must cite a tool output** (mask %, box coords, class probs); composer refuses unsupported statements.
- **Deterministic tools feed the VLMs, never the reverse** — NDVI/NDWI, SAR dB stats, backscatter thresholds, mask areas, box coords go *into* the prompt as structured context. This is the SAR fix (VLMs can't read speckle) and what makes evidence-citation enforceable.
- **Three served models, one fine-tune** (locked 2026-08-26) — Qwen2.5-VL-7B+QLoRA owns every user-facing sentence (5 tasks); MCI owns change masks; TerraMind owns fusion. Whole interior ≈10 GB typical / ~15 GB worst case → resident on one 16 GB T4, no per-request model swapping (six-model plan was 19–26 GB). Everything served is Apache-2.0.
- **Execution trace** (task, validation, tools, params, timings, outputs) rendered in UI + PDF — this is what judges evaluate.

## What the prototype does without the weights

Each served slot has a deterministic stand-in, selected automatically and named
in every answer and in `satquery doctor`:

| slot | model backend | stand-in today | what the stand-in will and will not do |
|---|---|---|---|
| all five text tasks | Qwen2.5-VL-7B + QLoRA | `evidence-rules` | phrases the measurements; refuses counting, land-use naming and free-form grounding |
| change map | Change-Agent MCI | `cva-otsu` | change vector analysis with robust brightness matching, Otsu **plus a noise floor** (Otsu alone reports ~50% change on an unchanged pair — it always returns a cut); measured 7.32% on a 7.32% change, 0.0% on an illumination-shifted unchanged pair |
| optical–SAR fusion | TerraMind + head | `fusion-rules-v1` | water = low backscatter AND positive NDWI (rejects radar shadow and building shade), urban = rough SAR minus vegetation, etc. Five coarse classes, each with its rule text |
| routing | Mistral small | `rules` | keyword routing, deterministic, offline |

`auto` promotes a model backend only when its imports succeed *and* a GPU is
visible — a 4-bit 7B on laptop CPU is minutes per answer, i.e. a hung demo.

## Known gaps (as of Phase 2)

- **Nine of ten dataset adapters are unwritten** (`satquery/datasets/` has RSVQA
  only). The registry lists them; `satquery datasets` shows them as absent.
- **`metrics/caption.py` does not exist** — BLEU/METEOR/CIDEr for the captioning
  slot. `DEFAULT_METRICS` already points at it.
- **MCI and TerraMind are not vendored.** Their classes exist so the slot is
  visible and selecting one fails with an explanation rather than an ImportError.
- **No auth on the API.** It binds to localhost; this is a single-user demo tool.
- **Grounding without the VLM is land-cover only** — thresholded regions, not
  referring expressions.

## Datasets

- **Train/adapt:** BigEarthNet(-MM) (S1+S2 pairs → fusion + SAR familiarity + label-derived captions), GeoChat_Instruct (318K), VRSBench train, CDVQA train + LEVIR-CC (two-image chat format), LEVIR-MCI (MCI), DIOR-RSVG (grounding).
- **Eval (prescribed test splits):** RSVQA-LR/HR (accuracy), VRSBench (caption: BLEU/METEOR/CIDEr; grounding: acc@IoU0.5), CDVQA (accuracy). Build eval harness FIRST; before/after deltas = adaptation proof.
- ⚠️ Confirm challenge contamination rules before training on any benchmark train split.

## Hard constraints & gotchas (learned, don't relearn)

- T4 (SM75): **fp16 + sdpa only** — no bf16, no FlashAttention-2; disable flash-attn in InternVL/LLaVA codebases.
- Unsloth won't run on Kaggle **P100** (needs CUDA ≥ 7.0) — pick the 2×T4 accelerator.
- Preprocessing must be sensor-agnostic (percentile stretch, SAR linear→dB clip [-25,0]) — hidden eval is Cartosat/RISAT, not Sentinel.
- Mistral free tier: check real quotas in Admin Console → Limits after signup; don't send sensitive data (training opt-in).
- Resize RS tiles ≤768 px for VLM training; tile large scenes 512×512 with overlap at inference.
- **Never quote a strict-only before/after delta.** Exact match (the published RSVQA/VRSBench convention) scores "There are 2 buildings." against gold "2" as wrong. An un-tuned baseline answers in sentences, a tuned model answers tersely — so a strict-only comparison credits the fine-tune for learning the *output format* and reports it as learning the task. `VqaAccuracy` emits `accuracy` (headline, comparable) **and** `accuracy_lenient` (diagnostic); the report must show both.
- **Load-bearing bet:** a LoRA-tuned Qwen2.5-VL beats the 2023–24-base RS VLMs (all sit on Vicuna/LLaMA-2/Phi-3). If our own eval says otherwise, EarthDial-4B stays as ensemble partner — hence eval harness BEFORE the training run, not after.
- `D:\` root is not writable — work inside `D:\SatQueryAI\`.
- **Otsu always returns a threshold.** On a unimodal histogram (an unchanged
  image pair) it splits the noise down the middle and reports ~50% change. Any
  thresholding of a difference map needs a noise floor beside it — `max(Otsu,
  median + 3 robust sigma)` is what `cva-otsu` uses.
- **Match brightness with median/MAD, not mean/std.** A genuine change region is
  an outlier population; matching on non-robust statistics lets a new bright
  development scale the whole later image down and smear false change across the
  unchanged remainder.
- **An area is not a count.** "How many buildings" must be refused by a backend
  that can only measure built-up percentage; answering with the percentage is the
  most plausible-looking wrong answer the system can give.
- **FastAPI cannot resolve `from __future__ import annotations` in a module whose
  FastAPI names are imported lazily** — every route with an `UploadFile` or
  `Body` parameter fails to build. `satquery/api/app.py` therefore has no future
  import, deliberately.

## Working style / user prefs

- User: gowtham@openfabric.co, hackathon participant, zero budget (free tiers + Colab/Kaggle GPUs only).
- Wants: approach/planning docs in Markdown before any build; verified facts over assumptions (free tiers and model availability were web-verified 2026-08-26 — recheck if weeks pass).
