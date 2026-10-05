# SatQuery AI — Problem Explanation & Solution Approach

> An approach document only. Nothing here is implemented — this explains **what the problem statement actually is** and **how to build it**.

---

## Part 1 — What this problem actually is (plain English)

Strip away the jargon and the problem statement (an ISRO/SAC-style challenge) asks for one thing:

> **A "ChatGPT for satellite images" — a web app where a user uploads 1 or 2 satellite images, asks a question in plain English, and an AI agent picks the right specialist model(s), runs them, and answers with visual proof.**

The three things that make it hard (and why a plain GPT-4V/Gemini wrapper is explicitly disqualified):

1. **Satellite images are not normal photos.** They come as GeoTIFFs with 4–13 spectral bands, 16-bit pixel values, geographic coordinates, and sometimes they're **SAR (radar)** images — grayscale, speckled, and physically nothing like optical imagery. A generic VLM has barely seen this data. The rules therefore **mandate fine-tuning/domain adaptation** on remote-sensing data (BigEarthNet is prescribed).

2. **One image is often not enough.** The core focus is *paired* images:
   - **Bi-temporal pair** — same place, two dates → "what changed and where?"
   - **Cross-modal pair** — optical + SAR of the same place → SAR sees through clouds and captures structure; optical captures color/spectra. Together they answer things neither can alone (e.g., water under cloud cover).

3. **It must be *agentic*, not monolithic.** One model doesn't do everything. The system needs a **controller** that reads the query, inspects the inputs (how many images? optical or SAR? co-registered?), picks tools from a **registry**, runs them, fuses the outputs, and logs an **auditable execution trace** (which task, which models, which parameters). The judges evaluate that trace — not the model's internal chain-of-thought.

### The mandatory checklist (what you MUST demo to qualify)

| # | Requirement | What it means concretely |
|---|---|---|
| 1 | **RS adaptation** | At least one vision/VL component fine-tuned on BigEarthNet or other open RS data. Non-negotiable. |
| 2 | **Single-image VQA** | "How many buildings? Is there a river?" on one optical or SAR image. |
| 3 | **+1 single-image task** | Either **captioning** (describe the scene) or **grounding** (draw a box on "the water body"). Pick one. |
| 4 | **Change analysis** | Bi-temporal pair → change description or change-VQA. Change *map* is optional bonus. |
| 5 | **Optical–SAR fusion** | Co-registered pair → joint extraction (e.g., built-up + water maps using both). |
| 6 | **Agentic orchestration** | Auto task routing, input validation, tool selection, output fusion, execution summary. |
| 7 | **GUI/web app** | Upload GeoTIFF/TIFF (PNG/JPEG only for benchmark data), query box, visual results, confidence, downloadable report. |

### The prescribed datasets and what each is for

| Dataset | Role | What it contains |
|---|---|---|
| **BigEarthNet (-MM)** | **Training / adaptation** | ~590k co-registered **Sentinel-2 (optical) + Sentinel-1 (SAR)** patch pairs with multi-label land-cover tags. This is the key to both requirement #1 (adaptation) and #5 (it's literally optical–SAR paired data). |
| **VRSBench** | Eval: caption + grounding + VQA | High-res aerial images with detailed captions, referring-expression boxes, QA pairs. |
| **RSVQA** (LR/HR) | Eval: single-image VQA | Sentinel-2 / aerial images with template QA (presence, counting, comparison, area). |
| **CDVQA** | Eval: change VQA | Bi-temporal pairs with change-related QA. |
| **ISRO/SAC hidden set** | Final eval | **Cartosat-2S optical + RISAT SAR** pairs — Indian sensors, annotations withheld. Your system must generalize beyond the public benchmarks. |

### Glossary (fast)

- **SAR** — Synthetic Aperture Radar. Active microwave imaging; works at night and through clouds. Smooth water looks black, buildings look bright. Values are usually converted to decibels (dB) for display.
- **Multispectral** — image with bands beyond RGB (near-infrared, short-wave IR…). Enables indices like NDVI (vegetation) and NDWI (water).
- **Co-registered** — two images aligned pixel-for-pixel over the same ground area.
- **Grounding** — pointing to *where* in the image a phrase refers to (output = bounding box/mask).
- **Change-VQA** — answering questions about the difference between two dates ("did built-up area increase?").

---

## Part 2 — Proposed approach

### 2.1 Architecture at a glance

```
┌─────────────────────────── Web GUI ───────────────────────────┐
│  image upload (1–2 files) · query box · map/overlay viewer    │
│  confidence panel · execution-trace panel · report download   │
└──────────────────────────────┬────────────────────────────────┘
                               │ REST/WebSocket
┌──────────────────────────────▼────────────────────────────────┐
│                     AGENTIC CONTROLLER                        │
│  1. Input Validator   → format, bands, modality, CRS,         │
│                          co-registration & size checks        │
│  2. Query Interpreter → classify task (VQA / caption /        │
│                          ground / change / fusion)            │
│  3. Planner/Router    → pick tool(s) from registry,           │
│                          sequence them, set allowed params    │
│  4. Executor          → run tools, collect outputs            │
│  5. Composer          → fuse text + spatial outputs,          │
│                          confidence, evidence, trace JSON     │
└───────────┬──────────────────────┬─────────────────────┬──────┘
            ▼                      ▼                     ▼
┌────────────────────┐  ┌──────────────────┐ ┌──────────────────┐
│ Qwen2.5-VL-7B      │  │ Change-Agent MCI │ │ TerraMind-1.0    │
│ + RS QLoRA         │  │ (~230M)          │ │ -base + head     │
│                    │  │                  │ │                  │
│ VQA · caption ·    │◀─┤ change mask +    │ │ optical–SAR      │
│ grounding ·        │  │ change caption   │ │ fusion → water / │
│ change-VQA ·       │◀─┼──────────────────┼─┤ built-up maps    │
│ change-caption     │  │                  │ │                  │
└────────────────────┘  └──────────────────┘ └──────────────────┘
   the only voice the      specialists never talk to the user;
   user ever hears         they hand the VLM numeric evidence
        plus deterministic helpers: NDVI/NDWI/dB-scaling,
        band compositing, tiling, co-registration check
```

Design principle: **one adapted VLM as the only user-facing voice + two narrow specialists that exist to produce evidence + cheap deterministic tools**, glued by a router. Three served models — the entire interior stays resident on one free T4 (≈10 GB typical, ~15 GB worst case against 16 GB), so no request ever waits for a model to be swapped in.

### 2.2 The model stack (concrete picks)

> **Superseded in part by §3.2.** The table below is the pre-verification design space (Aug 2026 planning). The *committed* pick per slot — one model each, losers named and dropped — is §3.2. Where the two disagree, §3.2 wins.

| Component | Recommended base | Adaptation |
|---|---|---|
| **Core RS-VLM** (VQA + captioning) | **Qwen2.5-VL-7B** (or LLaVA-1.6) — strong, open, LoRA-friendly. Alternative: start from **GeoChat** (already RS-tuned, saves compute). | **LoRA/QLoRA fine-tune** on a mixed instruction set: RSVQA train split, VRSBench train captions/QA, + BigEarthNet-derived image–text pairs (convert multi-labels into captions/QA: "This Sentinel-2 patch contains coniferous forest and water bodies…"). This single step satisfies mandatory requirement #1. |
| **Grounding** | GeoChat's grounded output, or **Grounding-DINO fine-tuned on VRSBench/DIOR-RSVG** referring expressions. | Fine-tune on VRSBench grounding split. |
| **Change understanding** | Bi-temporal input to the same VLM (concatenate two images with a change-prompt template), fine-tuned on **LEVIR-CC** (change captions) + **CDVQA** train split. Optional mask: an off-the-shelf change-detection UNet/Siamese net (e.g., trained on LEVIR-CD) as a *tool* that outputs a change map the VLM then describes. | LoRA on change data; CD network trained/pretrained separately. |
| **Optical–SAR fusion** | **Dual-branch encoder trained on BigEarthNet-MM** (S1+S2 pairs): a compact model (e.g., two ResNet/ViT branches, late fusion) predicting multi-label land cover — from which you derive built-up and water evidence maps. Complement with deterministic rules: NDWI from optical + low-backscatter mask from SAR → fused water map; high SAR backscatter + spectral built-up indices → built-up map. | Train from scratch / fine-tune SSL4EO or BigEarthNet pretrained weights. This doubles as a second adaptation proof. |
| **SAR familiarity** | Include SAR patches (BigEarthNet S1, converted to dB, 8-bit stretched) in the VLM's fine-tuning mix so single-image VQA works on SAR inputs too. | Same LoRA run. |

**Why this stack:** it reuses one fine-tuning pipeline for three requirements (VQA, captioning, change) and uses BigEarthNet exactly where it's strongest (multisensor pairs) for the fusion requirement. Total trainable footprint stays within a single 24–48 GB GPU using QLoRA.

### 2.3 Input validation & preprocessing (underrated, heavily judged)

1. **Format sniffing** — rasterio/GDAL opens the file; reject unsupported formats with a clear message. PNG/JPEG accepted only when the user selects "benchmark mode."
2. **Modality inference** — band count + dtype + metadata heuristics (1–2 band float/complex ⇒ SAR; 3–13 bands ⇒ optical/multispectral) with user override in the UI.
3. **Normalization** — optical: per-band percentile stretch (2–98%), band selection (RGB or false-color composite); SAR: linear→dB, clip [-25, 0], speckle-aware scaling. Large scenes: tile into 512×512 with overlap, aggregate answers.
4. **Pair compatibility** — same CRS? Overlapping footprints (from geotransforms)? Same resolution (resample if needed)? Report co-registration status in the trace; refuse gracefully if footprints don't overlap.
5. Every check's result goes into the **execution trace** — this directly scores under "check the number, modality, format, metadata, and compatibility."

### 2.4 The agentic controller

> **Decision (26 Aug 2026): router LLM = Mistral La Plateforme free "Experiment" tier, model `mistral-small-latest`** (with `mistral-large-latest` also free for hard queries). Verified: free tier covers all API models with native function calling; limits are roughly 1 request/second *per model* and ~1B tokens/month (exact numbers now shown only in Admin Console → Limits). Requires phone verification and an opt-in to data training (fine for a hackathon; don't send sensitive data). Fallbacks if limits bite: Google Gemini Flash-Lite free tier (~1,000 req/day), Groq `gpt-oss-20b/120b` (1,000 req/day, 8K tokens/min), or fully-local Qwen3-4B via Ollama (no limits, ~3 GB VRAM).

- **Router:** an LLM with **function calling** over a typed tool registry (LangGraph or a plain hand-rolled tool loop — a hand-rolled loop is easier to make deterministic and auditable). Each registry entry declares: name, task type, accepted modalities/image counts, permitted parameters, output schema.
- **Hard guardrails before the LLM:** the input configuration (1 image vs 2, optical vs SAR vs mixed) *constrains* which tools are even offered. E.g., change tools are only registered when a valid bi-temporal pair is present. This makes routing errors structurally impossible for obvious cases.
- **Plan → execute → compose:** router emits a small JSON plan (`task`, `tools[]`, `params`), executor runs it, composer merges: textual answer + overlay(s) + per-tool confidence (softmax scores / self-consistency across sampled generations) + the trace.
- **Execution trace (auditable):** a JSON log — task classification, validation results, chosen tools, parameters, timings, raw tool outputs, final answer. Rendered in the UI *and* embedded in the downloadable PDF report. This is explicitly what judges evaluate.

Example flows:

| Query + inputs | Routed plan |
|---|---|
| "Describe the land cover" + 1 optical GeoTIFF | validate → preprocess → **RS-VLM caption** → compose |
| "Highlight the water body" + 1 image | validate → **grounding model** → box/mask overlay + short answer |
| "What changed between these dates and where?" + 2 images | validate pair → **CD network** (change map) → **change-VLM** describes map + images → overlay + text |
| "Use optical and SAR to identify built-up and water" + pair | validate co-registration → **fusion model** + NDWI/backscatter rules → two evidence maps + textual summary |

### 2.5 Frontend & backend

- **Backend:** Python + FastAPI. Model serving with a simple job queue (GPU-bound tools run sequentially); rasterio + numpy for geospatial I/O; results cached per upload.
- **Frontend:** for hackathon speed, **Gradio or Streamlit** gets you 90% of the required GUI (upload, chat, image overlays, downloads) in days. If polish matters, React + Leaflet/OpenLayers for a proper map viewer with georeferenced overlays. Recommendation: start Gradio, upgrade only if time allows.
- **Report:** compose answer + input thumbnails + overlays + confidence + execution trace into a PDF (reportlab/WeasyPrint).

### 2.6 Evaluation plan (mirror the judges before they do)

| Task | Benchmark (test split) | Metrics |
|---|---|---|
| VQA | RSVQA-LR/HR, VRSBench-VQA | Accuracy (per question type + overall) |
| Captioning | VRSBench captions | BLEU-4, METEOR, ROUGE-L, CIDEr |
| Grounding | VRSBench referring | Acc@IoU 0.5 / 0.7 |
| Change VQA | CDVQA | Accuracy per change-question type |
| Fusion (internal) | BigEarthNet-MM val | Multi-label mAP; water/built-up IoU vs derived masks |

Build the eval harness **early** and run it after every training iteration — the final score is normalized across these metrics, so a balanced system beats one spiky task.

### 2.7 Phased build plan

| Phase | Deliverable |
|---|---|
| **1. Foundations** | Data pipelines (BigEarthNet loaders, dB/stretch preprocessing), eval harness on all four benchmarks with the *un-tuned* base VLM → baseline numbers. |
| **2. Adaptation** | LoRA fine-tune the VLM on the mixed RS instruction set (VQA + captions + SAR patches + change data). Train the BigEarthNet-MM fusion model. Re-run eval → measure lift (this delta is your "RS adaptation" proof). |
| **3. Orchestration** | Tool registry, validator, router, trace logging. Wire all tools. |
| **4. GUI + report** | Gradio/web app, overlays, confidence display, PDF export. |
| **5. Hardening** | Test on Cartosat/RISAT-like data (LISS/EOS samples from Bhoonidhi if accessible), edge cases (huge GeoTIFFs, mismatched pairs, SAR-only VQA), demo script + recorded demo. |

### 2.8 Key risks & mitigations

- **Domain shift to Cartosat-2S/RISAT (hidden eval).** Public benchmarks are Sentinel/aerial; the hidden set is Indian sensors at different resolutions. → Make preprocessing resolution- and sensor-agnostic (percentile stretch, dB normalization), augment training with multi-resolution crops, and avoid overfitting prompts to benchmark phrasing.
- **Compute budget.** Full fine-tunes are out of reach. → QLoRA on 7B VLM; fusion model kept small (ResNet-scale); reuse RS-pretrained checkpoints (GeoChat, SSL4EO) wherever licenses allow.
- **SAR VQA quality.** VLMs are weakest here. → Lean on deterministic SAR evidence (backscatter statistics, water/built-up masks) fed *into* the VLM prompt as structured context, rather than expecting the VLM to read raw speckle.
- **Agent misrouting.** → Constrain the tool menu by validated input configuration (structural guardrail) + a small routing eval set of ~100 query/input combos with expected tool choices.
- **Hallucination penalty.** Evidence-grounded answers are the brief. → Every claim in the final answer must cite a tool output (mask area %, box coordinates, class probabilities); the composer refuses to state anything no tool produced.

### 2.9 Deliverables mapping (problem statement → this design)

- *Interactive GUI/web app* → §2.5
- *RS-adapted VL component* → LoRA-tuned VLM + BigEarthNet-MM fusion model (§2.2)
- *Specialist tools (VQA, caption/ground, change, optical–SAR)* → tool registry (§2.2, §2.4)
- *Agentic controller + auditable trace* → §2.4
- *Visual evidence, confidence, reports* → composer + PDF (§2.4, §2.5)
- *Codes, models, tests, demo* → phases 1–5 (§2.7)

---

## Part 3 — Interior models: verified picks & build approach

> Verified 26 Aug 2026 against Hugging Face / GitHub (weights actually downloadable, licenses checked, free-GPU feasibility confirmed). Router = Mistral (§2.4); these are the specialist models behind it.

### 3.1 Three served models, and what each owns

The router (Mistral, cloud) never touches pixels. Below it, **three** models are served -- down from six -- and each owns its slot outright:

| Slot | Owner | Why it owns it |
|---|---|---|
| VQA · captioning · grounding · change-VQA · change-caption | **Qwen2.5-VL-7B-Instruct + RS QLoRA** | One fine-tune covers five of the mandatory tasks. Apache-2.0. It is the only component we adapt, so it is also the mandatory "fine-tuned on open RS data" proof. |
| change mask + change caption (*as evidence*) | **Change-Agent MCI** (~230M) | Mask *and* caption from one model, and the only candidate fully re-trainable end-to-end on a free T4 → the second adaptation proof. |
| optical–SAR fusion → water / built-up maps | **TerraMind-1.0-base + task head** | Natively ingests S2L2A + S1GRD, so there is no dual-branch encoder to hand-roll; TerraTorch ships the S1+S2 config. |

...plus the deterministic tools (NDVI/NDWI, SAR dB statistics, backscatter thresholds, co-registration checks), which produce the numbers the VLM is required to cite.

Three consequences, and they are the reason the collapse is worth doing:

- **The whole interior fits on one free T4.** Qwen 4-bit 7–9 GB + MCI <4 GB (verified figure; ~0.5 GB of weights, the rest activations) + TerraMind ~1–2 GB → ≈10 GB typical, ~15 GB worst case, against 16 GB. Adding back TEOChat (6–7 GB) and Grounding-DINO (<4 GB) puts the six-model plan at 19–26 GB, which means load/unload per request — a slow demo plus a job queue we would have to write and debug.
- **Everything served is Apache-2.0.** Dropping TEOChat removes the only Llama-2-community-licensed component; there is now no licence caveat anywhere in the served stack.
- **One prompt surface, one voice.** Specialists emit numbers, never prose the user sees. Every user-facing sentence comes from one model with one prompt file — which is also the only place hallucination has to be policed.

### 3.2 Committed picks and what they displaced

One model per slot. The displaced options are recorded because a judge (or a teammate in week three) will ask why, and because two of them are worth reopening only against a stated number.

| Slot | Committed | Displaced | Why the loser lost |
|---|---|---|---|
| **Core VLM** | **Qwen2.5-VL-7B-Instruct + QLoRA** (Unsloth, ~7.1 GB peak on a free T4; Apache-2.0 — ⚠️ *not* the 3B, which is a Qwen Research licence) | EarthDial-4B (`akshaydudhane/EarthDial_4B_RGB`), GeoChat, LLaVA-1.6 | EarthDial stays in the **eval table** as the zero-shot RS baseline, but it is not served. An untuned Qwen2.5-VL-7B already covers day-1 zero-shot, and that same checkpoint then *becomes* the tuned model. Serving both buys an ensemble we have no time to tune and doubles the prompt surface. |
| **Change-VQA + change-caption** | **the same tuned Qwen**, conditioned on MCI mask statistics | TEOChat-7B (`jirvin16/TEOChat`) | Part 3's own key finding argues against it: a 2026 study (arXiv 2604.18429) found tuned native Qwen multimodal models beat structured pipelines on RS change-VQA — and CDVQA + LEVIR-CC are already in the single LoRA mix. Keeping TEOChat means a second 7B, a second licence, and 6–7 GB for a capability the fine-tune already buys. |
| **Change map + caption** | **Change-Agent MCI** (`lcybuaa/Change-Agent`, Apache-2.0) | ChangeFormer / BIT LEVIR-CD checkpoints | Map-only — they would need a separate captioner bolted on, and neither retrains as comfortably on a T4. MCI delivers mask + caption in ~230M parameters. |
| **Optical–SAR fusion** | **TerraMind-1.0-base** via TerraTorch (Apache-2.0) | CROMA (MIT), DeCUR / SSL4EO-S12 | CROMA's edge was matching BigEarthNet-MM's geometry (120×120, 12-band + 2-band) exactly — but that geometry *is* TerraMind's native S2L2A+S1GRD input, so the edge evaporates, and TerraMind ships the water-segmentation config we would otherwise write ourselves. |
| **Grounding** | **Qwen's native bbox output** (same LoRA run: DIOR-RSVG + VRSBench referring) | Grounding-DINO / MM-Grounding-DINO Swin-T; GeoGround (no licence — rejected outright) | A separate detector means a second training run, a second serving process, its own preprocessing and a frozen BERT encoder to fuss over — for a task that is a *bonus*: captioning alone already satisfies the "one more single-image task" requirement. **Reopen only if** the harness shows acc@IoU0.5 < 0.30 on VRSBench referring with build time left; that threshold lives in the eval config, not in someone's head. |

**Key finding from the research:** every dedicated RS VLM with downloadable weights sits on a 2023–24 base (Vicuna/LLaMA-2/Phi-3), so a LoRA-tuned modern Qwen2.5-VL matches or beats them after one Colab run — and a 2026 study (arXiv 2604.18429) found tuned native Qwen multimodal models beat structured pipelines on RS change-VQA. T4 gotchas: no bf16, no FlashAttention-2 — use fp16 + sdpa everywhere.

**That is the load-bearing bet of the whole interior.** If it fails on our own eval harness, EarthDial-4B stays as the ensemble partner instead of being replaced — which is precisely why the harness is built *before* the training run, not after it.

### 3.3 How the interior composes

Two properties do the real work:

- **The validator prunes the tool menu before the router sees it.** One image registered → the change and fusion tools do not exist in the schema Mistral receives. A pair with non-overlapping footprints → fusion is withheld and the trace records why. Misrouting on obvious cases becomes *structurally* impossible rather than prompt-dependent.
- **Deterministic tools feed the VLMs, not the reverse.** NDVI/NDWI, SAR dB statistics, backscatter thresholds, mask areas and box coordinates go *into* the prompt as structured context. This is the SAR fix specifically: VLMs are bad at reading speckle, so never ask them to — hand them numbers and ask them to explain. It is also what makes "every claim cites a tool output" enforceable in the composer rather than aspirational.

### 3.4 Build order (do it in this sequence)

1. **Zero-shot skeleton first** — wire the *untuned* Qwen2.5-VL-7B behind every text endpoint (VQA, caption, ground, change-VQA, change-caption), MCI behind the change map, and TerraMind behind fusion (incl. its zero-shot "Thinking-in-Modalities" LULC generation). End-to-end demo works in days, before any training — and nothing gets thrown away, because step 2 replaces the Qwen weights in place.
2. **The one big LoRA run** — QLoRA Qwen2.5-VL-7B (Unsloth, r=16–32, 1–2 epochs) on the mix: GeoChat_Instruct (318K), VRSBench train (VQA+caption+grounding), CDVQA/LEVIR-CC formatted as two-image chat, BigEarthNet-derived captions incl. SAR patches. This is the mandatory "RS adaptation" proof.
3. **Train the small specialists** — fine-tune MCI on LEVIR-MCI (change), TerraMind head on BigEarthNet-MM 19-class + Sen1Floods11-style water config (fusion).
4. **Promote tuned weights per slot only where eval improves**; keep the before/after benchmark deltas for the report. EarthDial-4B is run here as the zero-shot RS baseline row — a number in the table, not a model in the serving path.

### 3.5 Fine-tuning stack (verified 26 Aug 2026)

**Winner: Unsloth** — the only framework with an official, verified free-Colab-T4 notebook for exactly this model class (`Qwen2.5_VL_(7B)-Vision.ipynb`). Measured on a free T4: 7B QLoRA (r=8–16, batch 1 × grad-accum 4) peaks at **~7.1 GB VRAM** and runs ~3K samples/epoch in ~49 min — so ~10K RS VQA pairs ≈ 2.5–5 h/epoch, inside one Kaggle 12 h session. Recipe: `FastVisionModel.from_pretrained('unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit')`, freeze vision layers for text-answer tasks / unfreeze for grounding & change, fp16 (never bf16 on T4), lr 2e-4, adamw_8bit, 1–2 epochs, resize RS tiles to ≤768 px, export merged 4-bit or GGUF.

**Alternative: LLaMA-Factory** if you want zero-code multi-dataset mixing (register RSVQA + VRSBench + LEVIR-CC + DIOR-RSVG in `dataset_info.json`, one YAML) — but on T4 the 7B is memory-tight: set `quantization_bit: 4`, `image_max_pixels: 262144`, `upcast_layernorm: true`, fp16. ms-swift only if you later want VLM GRPO/DPO (known fp16 NaN issue on T4 — monitor loss). Axolotl's multimodal support is still beta — avoid.

**Hard constraints found:** Unsloth does NOT run on Kaggle's P100 (needs CUDA capability ≥ 7.0 — pick the 2×T4 accelerator); Qwen2.5-VL-**3B** is Qwen Research License (non-commercial + attribution) — use the Apache-2.0 **7B**; on Colab free, checkpoint LoRA adapters to Drive to survive disconnects. Also confirm the challenge's data-contamination rules before training on any benchmark's train split.

---

## Part 4 — API specification

### 4.1 Public endpoints (frontend ↔ backend)

| Method & path | Description |
|---|---|
| `POST /api/v1/images` | Upload 1 image (multipart). Runs the validator: format sniff (GeoTIFF/TIFF; PNG/JPEG only in benchmark mode), modality inference (optical/MS/SAR), band count, dtype, CRS, geotransform, size. Returns `image_id` + full **validation report**. |
| `GET /api/v1/images/{image_id}` | Metadata + validation report for an uploaded image. |
| `GET /api/v1/images/{image_id}/preview` | Display-ready PNG (optical: 2–98% stretch RGB/false-color; SAR: dB-scaled). |
| `POST /api/v1/pairs` | Register two uploaded images as a pair `{image_a, image_b, pair_type: "bitemporal"\|"crossmodal"}`. Runs co-registration/compatibility checks (CRS match, footprint overlap, resolution; resamples if needed). Returns `pair_id` + compatibility report. |
| `POST /api/v1/query` | **The agentic endpoint.** Body: `{query: str, image_ids: [..] \| pair_id}`. Mistral router classifies task → picks tools (constrained by validated input config) → executor runs them → composer returns the result (below). Async: returns `query_id` immediately. |
| `GET /api/v1/queries/{query_id}` | Status + result: `{status, answer, confidence, evidence: [{type: "overlay"\|"boxes"\|"mask"\|"stats", url/data}], trace_id}`. |
| `GET /api/v1/queries/{query_id}/trace` | **Auditable execution trace** (what judges evaluate): task classification, validation results, selected tools + parameters, per-tool raw outputs, timings, final composition. |
| `GET /api/v1/queries/{query_id}/report` | Downloadable PDF report (inputs, answer, overlays, confidence, trace). |
| `GET /api/v1/tools` | The tool registry (names, descriptions, accepted modalities/image counts, permitted parameters) — also the source of the Mistral function-calling schema. |

Example `POST /api/v1/query` result:

```json
{
  "status": "done",
  "task": "change_vqa",
  "answer": "Built-up area increased by approximately 14% between the two dates, concentrated in the north-east quadrant.",
  "confidence": 0.82,
  "evidence": [
    {"type": "mask", "name": "change_map", "url": "/files/q123/change_mask.png", "changed_area_pct": 14.2},
    {"type": "overlay", "name": "before_after", "url": "/files/q123/overlay.png"}
  ],
  "trace_id": "q123-trace"
}
```

### 4.2 Internal tool endpoints (router → specialists)

Each is one FastAPI service the Mistral router can invoke; strict JSON in/out; every call logged to the trace.

| Endpoint | Backing model | Request → Response |
|---|---|---|
| `POST /tools/v1/vqa` | LoRA Qwen2.5-VL | `{image_id, question}` → `{answer, confidence}` |
| `POST /tools/v1/caption` | same VLM | `{image_id, detail: "brief"\|"detailed"}` → `{caption, confidence}` |
| `POST /tools/v1/ground` | LoRA Qwen2.5-VL, native bbox | `{image_id, phrase}` → `{boxes: [{bbox: [x1,y1,x2,y2], score}], overlay_url}` |
| `POST /tools/v1/change/map` | Change-Agent MCI | `{image_a, image_b}` → `{mask_url, changed_area_pct, num_regions, stats}` |
| `POST /tools/v1/change/describe` | LoRA Qwen2.5-VL, given MCI mask statistics | `{image_a, image_b, question?}` → `{answer, confidence}` (question omitted ⇒ change description; present ⇒ change-VQA) |
| `POST /tools/v1/fusion` | TerraMind-1.0-base + head + deterministic rules | `{optical_id, sar_id, targets: ["built_up","water"]}` → `{masks: {built_up: url, water: url}, area_pct: {...}, labels: [...], confidence}` |
| `POST /tools/v1/indices` | deterministic (rasterio/numpy) | `{image_id, index: "NDVI"\|"NDWI"\|"SAR_BACKSCATTER"}` → `{map_url, mean, threshold_mask_url, area_pct}` |

### 4.3 Mistral tool-registry mapping

Each row of 4.2 becomes one entry in the `tools` array of the Mistral chat call, e.g.:

```json
{
  "type": "function",
  "function": {
    "name": "change_describe",
    "description": "Describe or answer a question about changes between two co-registered images of the same area taken at different times. Use ONLY when a valid bi-temporal pair is loaded.",
    "parameters": {
      "type": "object",
      "properties": {
        "question": {"type": "string", "description": "Optional change question; omit for a general change description"}
      }
    }
  }
}
```

The executor injects `image_id`/`pair_id` itself from the session (the router never handles raw pixels), and the offered tool list is **pre-filtered by the validator** — e.g. `fusion` only appears when a valid optical+SAR pair is registered.

---

**Bottom line:** this is a tool-using agent problem dressed in remote-sensing clothes. The winning shape is: **one LoRA-adapted open VLM (trained on RSVQA/VRSBench/LEVIR-CC/CDVQA + BigEarthNet-derived text) as the language brain, a small BigEarthNet-MM-trained optical–SAR fusion model + deterministic index/backscatter tools as evidence generators, and a deterministic, trace-logging router in front** — wrapped in a simple web GUI that shows its work.
