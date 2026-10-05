# SatQuery AI

Agentic vision-language assistant for satellite imagery (ISRO/SAC challenge).
Upload one or two remote-sensing images, ask a question in natural language, get
an evidence-grounded answer with a confidence figure and an auditable execution
trace.

Planning documents come first in this project and stay authoritative:

- **`SatQuery_AI_Approach.md`** — the full approach: problem decode, architecture,
  interior models (Part 3), API spec (Part 4).
- **`CONTEXT.md`** — dense project state: locked decisions, datasets, gotchas.
- **`docs/ARCHITECTURE.md`** — the architecture of record: what is served, what
  talks to what, and the four invariants the code enforces. Rendered version with
  proper diagrams: **`docs/architecture.html`** (open in a browser).
- **`docs/PHASE1_SPEC.md`** — the binding API contract for the Phase 1 build.
- **`docs/PHASE2_DESIGN.md`** — the prototype as built: the five steps, the
  two-backend rule, and the fourteen decisions taken during the build that no
  plan anticipated (why Otsu needs a noise floor under it, why citation is
  three-way, why a 4-band stack was being read as SAR).

## Try it in three commands

```bash
python -m pip install -e ".[dev,data,api]"
python tools/make_demo_scenes.py        # writes demo/ -- deterministic, ~20 MB
python -m satquery.cli serve            # http://127.0.0.1:8000
```

Upload one or two images, ask a question, read the trace, download the PDF. No
GPU, no API key and no model weights are required: the system falls back to
deterministic backends and says so on every answer (see *What actually runs*).

`demo/` holds synthetic-but-physical scenes to try it on — a land-cover map
rendered with per-class reflectance, a real NIR band, and gamma-speckled SAR:

| files | shows |
|---|---|
| `city_2019.png` + `city_2021.png` | bi-temporal change: a new development, a cleared forest patch, a widened reservoir, under a different sun angle |
| `city_2019_4band.npy` | real NDVI and NDWI (it has a NIR band), so vegetation and water are measured rather than proxied |
| `coast_optical_4band.npy` + `coast_sar_vv_vh.npy` | optical–SAR fusion, and the change tools being withheld because the pair is cross-modal |

**Optional — real LLM routing.** With no key the router is deterministic keyword
rules, which is what the tests and the offline demo use. To route with Mistral
instead (`mistral-small-latest`, native function calling, free Experiment tier):

```bash
cp .env.example .env        # PowerShell: Copy-Item .env.example .env
# put the key on the MISTRAL_API_KEY= line; .env is gitignored
satquery doctor             # confirms: router  mistral  (fallback: rules)
```

The key is read from `.env` at import by every entry point. Nothing but your
question and the validator's text summary is ever sent — no imagery, no paths
(invariant 01) — and any failure falls back to the offline router with the
reason recorded in the trace.

Or without the browser:

```bash
satquery doctor                                   # what is installed, what will serve
satquery ask "what changed?" before.tif after.tif # one question, printed with its evidence
satquery ask "how much vegetation?" scene.png --trace --report out.pdf
```

## What exists today

**Phase 2 — the end-to-end agentic prototype.** Validate → prune the tool menu →
route → execute → compose, with a trace written at every hop.

```
satquery/
  contracts.py      # Sample / Prediction / Box / RunResult + the Protocols. Frozen.
  config.py         # paths, seed, device; T4-safe dtype/attention constants
  registry.py       # the only name -> implementation mapping in the codebase
  text.py           # answer extraction + normalisation (scoring-critical)
  io/               # raster reading (rasterio optional), modality inference
  preprocess/       # stretch, SAR dB, indices, band roles, tiling, pair checks
  agent/            # the loop: session, validate, tools, backends, router,
                    #   compose, controller, trace, render
  api/              # FastAPI: /api/v1 for users, /tools/v1 for the tool contract,
                    #   and ui.html — the whole GUI, one file, no build step
  report.py         # one query -> Markdown / self-contained HTML / PDF
  datasets/         # benchmark adapters over one Sample type
  metrics/          # VQA accuracy, grounding, multi-label, change masks
  models/           # Qwen2.5-VL + EarthDial predictors, and the test doubles
  eval/             # harness, run artefacts, run reports and comparisons
  cli.py            # doctor / datasets / tools / ask / serve / session /
                    #   eval / report / compare
```

### The four invariants, enforced in code

| # | Invariant | Where it lives | How it is tested |
|---|---|---|---|
| 01 | The router never sees imagery | `agent/router.py`, `agent/controller.py` | the Mistral request body is asserted to contain no path, no id value, no base64 |
| 02 | The validator prunes the tool menu | `agent/tools.py:build_menu` | the A/B/C matrix from `docs/ARCHITECTURE.md` is a parametrised test |
| 03 | Deterministic tools feed the VLM | `agent/tools.py:scene_evidence`, `Plan.normalized` | `indices` is asserted to precede every text tool |
| 04 | One model owns every sentence | `agent/compose.py` | a specialist is asserted never to emit an answer; unbacked numbers are deleted |

### What actually runs

Each served slot has a model backend and a deterministic one, and the answer
always says which produced it:

| slot | model backend | deterministic backend (today) |
|---|---|---|
| every sentence | Qwen2.5-VL-7B (+ RS QLoRA, Phase 3) — or a cloud VLM, see below | `evidence-rules` — phrases the measurements, refuses what it cannot measure |
| change map | Change-Agent MCI | `cva-otsu` — change vector analysis, robust brightness match, Otsu cut over a noise floor |
| optical–SAR fusion | TerraMind-1.0-base + head | `fusion-rules-v1` — index and backscatter thresholds with the rule written out |
| routing | Mistral `mistral-small-latest` | `rules` — offline keyword routing |

`auto` picks a model backend only when its dependencies import *and* a GPU is
visible; otherwise the deterministic one runs and `satquery doctor` prints the
reason. Force either with `SATQUERY_VLM`, `SATQUERY_CHANGE_BACKEND`,
`SATQUERY_FUSION_BACKEND`, `SATQUERY_ROUTER`.

**A third option for the text slot, if you have no GPU.** `SATQUERY_VLM=mistral`
serves all five text tasks from a cloud vision model using the same key as the
router, which answers the questions the deterministic backend has to refuse —
counting, land use, layout, free-form grounding. It is never selected by `auto`,
because unlike every other backend **it uploads the imagery**, and that is a
decision to make deliberately rather than inherit. It is also not a substitute
for Phase 3: a general VLM is what the challenge disqualifies on its own, and
the RS adaptation that gets scored comes from the QLoRA run.

The deterministic backends are not mocks. On a synthetic scene the change
detector reports 7.32% for a 7.32% change and 0.0% for an unchanged pair under an
illumination shift; rule fusion recovers a 12.25% lake as 12.25% water. What they
will not do is count objects or name a land use — they refuse, and the refusal
lists what *was* measured.

### Design rules the code enforces rather than documents

- **numpy is the only hard dependency.** rasterio, torch, transformers, PIL,
  scipy, pandas, fastapi and reportlab are optional extras, imported inside the
  functions that need them. `satquery doctor` runs on a bare laptop.
- **Metrics are validated by an oracle, not by faith.** `OraclePredictor`
  returns ground truth and every metric must hit its ceiling on it.
- **T4 reality is a constant, not a comment.** fp16 + sdpa only, never bf16, no
  FlashAttention-2 — the free-tier GPUs are SM75.
- **Nothing hardcodes a dataset path.** `SATQUERY_DATA_<NAME>` points any
  adapter at a Kaggle input directory without moving anything else.
- **An unbacked number never survives.** The composer deletes any clause whose
  figures no tool produced, and records the deletion in the trace.

## Evaluation

```bash
python -m pytest -q                        # synthetic fixtures only, seconds
satquery datasets                          # registry + which benchmarks are local
satquery eval --dataset rsvqa_lr --predictor dummy --limit 50
satquery compare runs/baseline runs/tuned  # the before/after delta, both accuracies
```

Benchmarks are not in the repo and are not committed. Point the loaders at a
download with `SATQUERY_DATA_ROOT=/path/to/data` (or per-dataset,
`SATQUERY_DATA_RSVQA_LR=/kaggle/input/rsvqa-lr`), then run the same commands on
Colab/Kaggle with the `vlm` extra for real model numbers —
`notebooks/phase1_baseline.ipynb` does exactly that.

## Status

- **Phase 1 — landed.** Loaders, metrics and the evaluation harness.
- **Phase 2 — landed.** This prototype: the agentic loop, the seven tools, the
  API, the web app, the reports and the CLI, with deterministic backends
  standing in for the weights.
- **Phase 3–5 — next.** The QLoRA run that fills the text slot, MCI and the
  TerraMind head for the other two, then hardening. Nothing is thrown away: each
  trained model is dropped into a slot the pipeline already calls.

Remaining gaps are listed in `CONTEXT.md` under *Known gaps*.
