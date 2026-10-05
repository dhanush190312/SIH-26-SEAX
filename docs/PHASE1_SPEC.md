# Phase 1 build spec — data loaders + eval harness

This is the **binding API contract** for the Phase 1 fan-out. The spine
(`satquery/contracts.py`, `config.py`, `registry.py`, `text.py`, `log.py`,
`datasets/base.py`, `models/dummy.py`, all `__init__.py`) is already written and
**must not be edited**. Every module below implements exactly the public names
listed for it — the `__init__.py` files already import these names, so a
renamed symbol breaks the package import.

## Ground rules

1. **Own your files.** Write only the files listed for your module. Never edit a
   spine file, another module's file, or a shared doc.
2. **numpy is the only hard dependency.** `rasterio`, `pandas`, `torch`,
   `transformers`, `pycocoevalcap`, `nltk` are optional: import them *inside*
   the function that needs them, guard with `try/except ImportError`, and
   degrade with a clear message. Tests must pass with numpy + pytest only.
   Available on the dev laptop: numpy 1.26, scipy, pandas, torch 2.7 CPU,
   transformers 4.54, PIL, sklearn, matplotlib, pytest, pyyaml. **Not**
   available: rasterio, pycocoevalcap, nltk, datasets, geopandas, pyproj.
3. **Arrays are `(band, height, width)` float32**, band-first like rasterio.
   Boxes are absolute pixels `(x0, y0, x1, y1)`. See `contracts.py`.
4. **Determinism.** No `random` without a seed, no `time`-dependent output in
   anything a test asserts on. Seeds come from `config.seed` or an explicit arg.
5. **T4 reality** (relevant to model adapters): fp16 + sdpa only, never bf16,
   no FlashAttention-2. `satquery.config` exports `T4_SAFE_DTYPE` /
   `T4_SAFE_ATTENTION`.
6. **Tests.** Every module ships `tests/test_<module>.py` using the fixtures in
   `tests/conftest.py`. Prefer synthetic fixtures written into `tmp_path` over
   any real download. Dataset adapters MUST include a fixture builder that
   writes a tiny dataset in the documented on-disk layout and asserts the parser
   yields the expected samples — that test is the proof the layout is handled.
7. **Honesty about layouts.** Where an on-disk layout is recalled rather than
   verified against a real download, mark it `# VERIFY:` in code and say so in
   your `docs/datasets/<name>.md`. A wrong-but-flagged guess is fine; an
   unflagged one is not.
8. Docstrings explain *why*, in the register of the existing spine files. No
   emoji, no marketing, no "Note that". Match the surrounding code's style.

## Module contracts

### A. `satquery/io/raster.py` + `satquery/io/modality.py`

```python
HAS_RASTERIO: bool

@dataclass(frozen=True)
class RasterImage:
    array: np.ndarray            # (band, h, w) float32
    path: str
    crs: str | None              # e.g. "EPSG:4326"; None for PNG/JPEG
    transform: tuple[float, ...] | None   # affine as 6-tuple (a,b,c,d,e,f)
    nodata: float | None
    dtype: str                   # original on-disk dtype name
    tags: dict[str, str]
    @property bands/height/width -> int
    def band(self, i: int) -> np.ndarray          # 0-based
    def footprint(self) -> tuple[float,float,float,float] | None   # (minx,miny,maxx,maxy) in CRS units

def read_raster(path, *, bands: Sequence[int] | None = None, as_float: bool = True) -> RasterImage
def image_size(path) -> tuple[int, int]          # (height, width), cheap, no full decode
def write_geotiff(path, array, *, crs=None, transform=None, nodata=None) -> Path
```

`read_raster` uses rasterio when present; otherwise falls back to PIL for
PNG/JPEG (and raises `SatQueryError` with an actionable message for GeoTIFF
without rasterio — that is "benchmark mode"). `write_geotiff` without rasterio
writes a `.npy` sidecar + JSON metadata so tests can exercise the geo path on
this laptop; document that clearly.

```python
@dataclass(frozen=True)
class ModalityGuess:
    modality: Modality       # "optical" | "sar" | "unknown"
    confidence: float
    reasons: tuple[str, ...]

def infer_modality(image: RasterImage | np.ndarray, *, path: str | None = None,
                   declared: str | None = None) -> ModalityGuess
```

Heuristics (from the approach doc): 1–2 bands + float/complex dtype ⇒ SAR;
3–13 bands ⇒ optical/multispectral; filename hints (`_VV`, `_VH`, `grd`, `risat`,
`s1` vs `s2`, `cartosat`); dB-like value range (negative values, ~[-30, 5]) ⇒
SAR; `declared` always wins but is reported as such in `reasons`.

Files: `satquery/io/raster.py`, `satquery/io/modality.py`,
`tests/test_io_raster.py`, `tests/test_io_modality.py`.

### B. `satquery/preprocess/optical.py` + `indices.py`

```python
BAND_PRESETS: dict[str, tuple[int, ...]]   # e.g. {"s2_rgb": (3,2,1), "rgb": (0,1,2), "s2_false_color": (7,3,2)}

def percentile_stretch(array, *, low=2.0, high=98.0, per_band=True,
                       nodata=None, out_range=(0.0, 1.0)) -> np.ndarray
def to_uint8(array) -> np.ndarray                     # (b,h,w) float in [0,1] -> uint8
def select_composite(array, bands=None, preset=None) -> np.ndarray   # -> (3,h,w)

def ndvi(nir, red, *, eps=1e-6) -> np.ndarray
def ndwi(green, nir, *, eps=1e-6) -> np.ndarray
def ndbi(swir, nir, *, eps=1e-6) -> np.ndarray
def index_stats(index_array, *, threshold=0.0, name="index") -> dict[str, float]
```

`index_stats` returns the numeric evidence the composer must cite: mean, std,
min, max, `positive_pct` (fraction above `threshold`, ×100), and `threshold`.
Handle NaN/nodata everywhere (`np.nanpercentile` and friends); a constant band
must not produce NaN output from the stretch.

Files: `satquery/preprocess/optical.py`, `satquery/preprocess/indices.py`,
`tests/test_preprocess_optical.py`, `tests/test_preprocess_indices.py`.

### C. `satquery/preprocess/sar.py` + `tiling.py`

```python
def linear_to_db(array, *, eps=1e-8) -> np.ndarray            # 10*log10
def db_to_uint8(array, *, clip=(-25.0, 0.0)) -> np.ndarray    # clip then scale to 0..255
def lee_filter(array, *, size=5) -> np.ndarray                # speckle filter, scipy optional -> numpy fallback
def sar_water_mask(db_array, *, threshold=-18.0) -> np.ndarray # bool (h,w); low backscatter = water
def backscatter_stats(db_array, *, water_threshold=-18.0) -> dict[str, float]

@dataclass(frozen=True)
class Tile:
    array: np.ndarray; row: int; col: int; y0: int; x0: int
    @property height/width
def tile_image(array, *, size=512, overlap=64, pad=True) -> list[Tile]
def merge_tiles(tiles, *, height, width, reduce="mean") -> np.ndarray
def resize_max_side(array, *, max_side=768, order=1) -> np.ndarray
```

`resize_max_side` must work without PIL/scipy (nearest/bilinear on numpy is
fine) — it is on the path of every VLM call, and 768 px is the training cap.
`merge_tiles` must round-trip: tiling then merging an array returns it
unchanged (within float tolerance) for both divisible and non-divisible sizes.
`backscatter_stats` returns mean/std/p5/p50/p95 dB plus `water_pct`.

Files: `satquery/preprocess/sar.py`, `satquery/preprocess/tiling.py`,
`tests/test_preprocess_sar.py`, `tests/test_preprocess_tiling.py`.

### D. `satquery/preprocess/pairing.py`

```python
@dataclass(frozen=True)
class PairCheck:
    ok: bool
    same_crs: bool
    overlap_pct: float          # 0..100, of the smaller footprint
    same_shape: bool
    scale_ratio: float          # resolution ratio, 1.0 == identical
    modalities: tuple[str, str]
    reasons: tuple[str, ...]
    def to_dict(self) -> dict   # goes verbatim into the execution trace

def check_pair(a: RasterImage, b: RasterImage, *, min_overlap_pct=50.0,
               require_same_crs=True) -> PairCheck
def resample_to_match(src: RasterImage, ref: RasterImage) -> np.ndarray
```

This is the co-registration gate the validator uses to decide whether the fusion
and change tools are even offered to the router. Non-georeferenced input (PNG
benchmark mode) must degrade to a shape-only comparison with an explicit reason
string, not an exception.

Files: `satquery/preprocess/pairing.py`, `tests/test_preprocess_pairing.py`.

### E. `satquery/metrics/vqa.py`

```python
class VqaAccuracy:
    name = "vqa_accuracy"
    tasks = ("vqa", "change_vqa")
    def __init__(self, *, slice_by="question_type", numeric_tolerance=0, strict=False)
    def reset(self); def update(sample, prediction); def compute() -> dict[str, float]
```

Exact match after `satquery.text.normalize_answer`, applied to both sides;
predictions go through `extract_answer` first unless `strict=True`. Accept any
of `sample.all_answers`. Emits `accuracy`, `accuracy/<question_type>` for each
type, `n`, `n/<type>`, and `answered_pct` (non-empty predictions). RSVQA scoring
convention: also emit `average_accuracy` = unweighted mean over types, because
that is the number RSVQA papers report alongside overall accuracy.

Files: `satquery/metrics/vqa.py`, `tests/test_metrics_vqa.py`.

### F. `satquery/metrics/caption.py`

```python
class CaptionMetrics:
    name = "caption"
    tasks = ("caption", "change_caption")
    def __init__(self, *, ngram=4, use_pycocoevalcap=False, cider_sigma=6.0)
```

Pure-numpy implementations, corpus-level, matching coco-caption semantics:
`bleu_1..bleu_4` (modified n-gram precision, closest-reference length brevity
penalty, corpus-level), `rouge_l` (LCS F with β=1.2, max over references),
`cider_d` (tf-idf over corpus document frequencies, σ=6.0, length gaussian,
clipped counts, mean over n=1..4 ×10). `meteor` only when
`use_pycocoevalcap=True` and the package imports — otherwise omit the key
entirely rather than emitting a fake value. Tokenise with
`satquery.text.tokenize_caption`.

Tests must include: identical prediction/reference ⇒ BLEU-4 = 1.0 and
ROUGE-L = 1.0; a known worked example with hand-computed BLEU-1; CIDEr-D of a
single-sentence corpus against itself ⇒ 10.0; empty prediction ⇒ 0.0, no crash.

Files: `satquery/metrics/caption.py`, `tests/test_metrics_caption.py`.

### G. `satquery/metrics/grounding.py`

```python
class GroundingMetrics:
    name = "grounding"
    tasks = ("ground",)
    def __init__(self, *, thresholds=(0.5, 0.7), score_threshold=0.0, greedy_match=True)
```

Emits `acc@0.5`, `acc@0.7` (a sample counts as correct when the highest-scoring
prediction hits IoU ≥ t against *any* ground-truth box — the VRSBench referring
convention), plus `mean_iou`, `precision@t`/`recall@t` from greedy score-ordered
matching (multi-instance RS referring expressions are normal), `n`, and
`empty_pred_pct`.

Files: `satquery/metrics/grounding.py`, `tests/test_metrics_grounding.py`.

### H. `satquery/metrics/multilabel.py`

```python
class MultiLabelMetrics:
    name = "multilabel"; tasks = ("fusion_multilabel",)
    def __init__(self, *, labels=None, threshold=0.5)
    # micro/macro precision, recall, f1; mAP (average precision per label,
    # averaged) from `prediction.label_scores` when present, else binary from
    # `prediction.labels`; per-label f1 as "f1/<label>"; exact_match; n
class ChangeMaskMetrics:
    name = "change_mask"; tasks = ("change_map",)
    def __init__(self, *, positive_threshold=0.5)
    # IoU, F1, precision, recall, over-detection ratio, plus mean absolute
    # error of changed-area percentage; reads a predicted mask from
    # prediction.evidence["mask"] (np.ndarray) or ["mask_path"], and the ground
    # truth from sample.meta["mask_path"]; a sample with neither is skipped and
    # counted in "n_skipped".
```

mAP must handle the degenerate cases (label never positive, all scores equal)
without NaN. Files: `satquery/metrics/multilabel.py`,
`tests/test_metrics_multilabel.py`.

### I–M. Dataset adapters

All subclass `BaseDataset`, set `name` / `task` / `splits` / `required_paths` /
`download_hint` / `homepage`, implement `_load()`, and use `self.make_sample`
and `self.image_ref`. Registry names and class names are fixed in
`satquery/registry.py` — match them exactly. Each adapter gets
`docs/datasets/<registry_name>.md` documenting the expected layout, the source
URL, the split sizes you expect, and any `# VERIFY:` flags.

| Registry name | Class | File | Task | Notes |
|---|---|---|---|---|
| `rsvqa_lr` | `RSVQALRDataset` | `datasets/rsvqa.py` | `vqa` | Sentinel-2 LR; `question_type` ∈ {presence, count, comparison, rural_urban}; the `_questions.json` / `_answers.json` / `_images.json` triple keyed by ids, plus the `active` flag that marks unusable QA pairs (drop inactive) |
| `rsvqa_hr` | `RSVQAHRDataset` | `datasets/rsvqa.py` | `vqa` | HR aerial; types {presence, count, comparison, area}; has `test_phili` as a second test split — expose it as split `test2` |
| `vrsbench_vqa` | `VRSBenchVQADataset` | `datasets/vrsbench.py` | `vqa` | 10-type question taxonomy; keep `question_type` from the annotation |
| `vrsbench_caption` | `VRSBenchCaptionDataset` | `datasets/vrsbench.py` | `caption` | one detailed reference caption per image |
| `vrsbench_ground` | `VRSBenchGroundDataset` | `datasets/vrsbench.py` | `ground` | referring expression + box; boxes are `[x, y, w, h]` in the source — convert |
| `dior_rsvg` | `DiorRSVGDataset` | `datasets/dior_rsvg.py` | `ground` | ~38k expressions, per-image XML annotations; parse with `xml.etree` |
| `cdvqa` | `CDVQADataset` | `datasets/cdvqa.py` | `change_vqa` | image pair + question; `question_type` ∈ change/increase/decrease/largest-change style categories |
| `levir_cc` | `LevirCCDataset` | `datasets/levir.py` | `change_caption` | 5 reference change captions per pair; "no change" captions included |
| `levir_mci` | `LevirMCIDataset` | `datasets/levir.py` | `change_map` | multi-class change mask (road/building); set `meta["mask_path"]`, and `meta["classes"]` |
| `bigearthnet_mm` | `BigEarthNetMMDataset` | `datasets/bigearthnet.py` | `fusion_multilabel` | S1 (VV/VH) + S2 patch pairs, 19-class nomenclature; image[0] = optical, image[1] = SAR, both with `modality` set; also expose `to_caption(labels) -> str` for the label→caption training data, and the 19-class list as `BEN19_LABELS` |

Adapters must tolerate the common layout variants (a `.json` that is a list vs a
dict-of-lists, annotations split per-split vs one file with a `split` column)
via `first_existing` and small explicit branches — no clever auto-detection.

Files per agent: its `datasets/*.py`, its `tests/test_datasets_*.py`, its
`docs/datasets/*.md`.

### N. `satquery/eval/harness.py` + `report.py` + `satquery/cli.py`

```python
def evaluate(dataset, predictor, *, metrics=None, limit=None, run_id=None,
             config=None, progress=False, on_sample=None) -> RunResult
def save_run(result, *, config=None) -> Path        # runs/<run_id>/result.json (+ per_sample.jsonl)
def load_run(path_or_run_id, *, config=None) -> RunResult
def run_matrix(specs, *, config=None) -> list[RunResult]   # specs: list[dict]

def render_markdown(results, *, title="SatQuery eval report") -> str
def render_comparison(before, after, *, title="before / after") -> str
```

Requirements: metrics default from `registry.default_metrics_for(task)`;
predictor errors are caught per sample (counted in `n_failed`, never abort a
run); `DatasetNotAvailable` produces a skipped `RunResult` with a note rather
than an exception; `per_sample` rows carry `sample_id`, `question_type`,
prediction text/boxes/labels, per-sample correctness where the metric can give
it, and latency; `env` records python/numpy/torch versions, device, git commit
if available; wall-clock and `duration_s` filled in. `render_comparison`
computes deltas per metric key and marks regressions — this is the "before/after
adaptation proof" table for the report.

CLI (`python -m satquery.cli` and the `satquery` entry point):

```
satquery doctor                      # env + optional-dep report, exits 0
satquery datasets [--json]           # registry listing + availability + missing files
satquery eval --dataset rsvqa_lr --predictor dummy [--split test] [--limit 200]
              [--metrics vqa_accuracy] [--run-id NAME] [--out runs/] [--json]
satquery report --runs runs/a runs/b [--out report.md]
satquery compare --before runs/a --after runs/b
```

Files: `satquery/eval/harness.py`, `satquery/eval/report.py`, `satquery/cli.py`,
`tests/test_eval_harness.py`, `tests/test_cli.py`.

### O. `satquery/models/qwen_vl.py` + `earthdial.py` + `notebooks/`

```python
class QwenVLPredictor:
    name = "qwen2_5_vl"
    supported = ("vqa", "caption", "ground", "change_vqa", "change_caption")
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", *, adapter=None,
                 load_in_4bit=True, dtype=None, attn_implementation=None,
                 max_new_tokens=64, max_side=None, device=None, config=None)
    def supports(task) -> bool
    def predict(sample) -> Prediction
    def close(self)
class EarthDialPredictor:  # same shape, model_id="akshaydudhane/EarthDial_4B_RGB"
```

Both must: import torch/transformers lazily and raise a clear
`SatQueryError` when the extra is missing; default to fp16 + sdpa (never bf16 —
T4); load the model once, lazily, on first `predict`; preprocess through
`satquery.preprocess` (percentile stretch for optical, dB for SAR,
`resize_max_side`); build task-specific prompts in a module-level `PROMPTS`
dict so they are auditable and tunable in one place; run VQA output through
`satquery.text.extract_answer`; parse grounding output into `Box` via
`Box.from_normalized(..., scale=1000.0)` for Qwen's coordinate convention; set
`confidence` from mean token probability when available, else `None` (never a
made-up number); fill `evidence` with the raw decode and the prompt used;
measure `latency_ms`. Two-image tasks pass both images in one message.

Tests must run **without** downloading weights: monkeypatch the model/processor
loader and assert prompt construction, box parsing (including malformed output),
answer extraction and error handling. Mark any test that would need weights
with `@pytest.mark.skipif`.

Also write `notebooks/phase1_baseline.ipynb`: a thin Colab/Kaggle notebook that
pip-installs the extras, clones or mounts the repo, sets `SATQUERY_DATA_*`
env vars for Kaggle inputs, and calls the CLI to produce baseline numbers for
every available benchmark with `qwen2_5_vl` and `earthdial`. Keep the logic in
the package, not the notebook — cells should be a handful of lines each.

Files: `satquery/models/qwen_vl.py`, `satquery/models/earthdial.py`,
`tests/test_models_qwen.py`, `notebooks/phase1_baseline.ipynb`.
