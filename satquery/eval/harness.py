"""The eval harness: run a predictor over a benchmark, record everything.

This module exists before any training run, on purpose. The only claim this
project makes that a judge can check is the *before/after* delta on the
prescribed test splits, and a delta is only evidence if both halves were
measured by the same harness with the same normalisation and the same failure
accounting. So the harness is the artefact, and the model is the variable.

Four properties are load-bearing, and each one is a decision rather than an
implementation detail:

* **A run never dies halfway.** A predictor that raises on one sample yields a
  failed :class:`~satquery.contracts.Prediction` counted in ``n_failed``, with
  the exception's last line in that sample's row. A five-hour Colab run that
  aborts at sample 4,900 because one JPEG is truncated is a wasted GPU session.
* **A failure is wrong, not absent.** Failed predictions are still handed to the
  metrics. Dropping them would inflate every score exactly where the model is
  weakest -- the one bias a baseline harness must not have.
* **A missing dataset is not an error.** :class:`DatasetNotAvailable` produces a
  valid, skipped :class:`RunResult` with a note. The benchmarks live on
  Colab/Kaggle; ``satquery eval`` still has to behave on the dev laptop, and the
  matrix over ten datasets must not stop at the first one that is not
  downloaded.
* **Everything a number depends on is written next to it.** ``env`` carries the
  interpreter, numpy/torch versions, device, git commit and the resolved config,
  because "accuracy 0.71" without them cannot be reproduced or defended.

Determinism: nothing here seeds anything (subsampling is the dataset's job, via
``limit``), and the only wall-clock values are ``started_at``, ``duration_s`` and
measured ``latency_ms`` -- all of which tests must ignore. A predictor that
reports its own ``latency_ms`` keeps it, which is what makes a dummy run
byte-identical across invocations.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import traceback
from dataclasses import replace as _dc_replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..config import SatQueryConfig, get_config, repo_root
from ..contracts import (
    Box,
    DatasetNotAvailable,
    Metric,
    Prediction,
    Predictor,
    RunResult,
    RunSpec,
    Sample,
    SatQueryError,
    Task,
    UnsupportedTask,
    supports_task,
)
from ..log import get_logger
from ..registry import build_dataset, build_metric, build_predictor, default_metrics_for

__all__ = [
    "evaluate",
    "save_run",
    "load_run",
    "run_matrix",
    "build_metrics",
    "env_info",
    "RESULT_FILE",
    "PER_SAMPLE_FILE",
]

log = get_logger(__name__)

#: Filenames inside ``runs/<run_id>/``. Pinned here because report tooling and
#: the notebooks glob for them.
RESULT_FILE = "result.json"
PER_SAMPLE_FILE = "per_sample.jsonl"

#: How many per-sample metric failures get their own log line before the
#: harness falls back to counting them. Full detail always lands in the notes.
_MAX_LOGGED_METRIC_ERRORS = 3


# --------------------------------------------------------------------------- #
# Environment capture
# --------------------------------------------------------------------------- #


def _package_version(module_name: str) -> str | None:
    """Version of an installed distribution without importing it.

    ``importlib.metadata`` reads the installed metadata, so asking whether torch
    is present costs microseconds instead of the several seconds a real
    ``import torch`` takes -- and the harness records versions on every run.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(module_name)
    except Exception:  # PackageNotFoundError, or no metadata for a vendored copy
        return None


def _git_commit() -> str | None:
    """Current commit, or ``None`` when git is absent or this is not a checkout.

    A run record without the code version is not reproducible, but a missing
    commit must never fail a run: the Kaggle copy of this repo is frequently an
    unzipped dataset, not a clone.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def env_info(config: SatQueryConfig | None = None) -> dict[str, Any]:
    """Everything needed to reproduce (or distrust) a number in this run."""
    cfg = config or get_config()
    from .. import __version__

    return {
        "satquery": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _package_version("numpy"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "device": cfg.resolve_device(),
        "git_commit": _git_commit(),
        "config": cfg.to_dict(),
    }


# --------------------------------------------------------------------------- #
# Metric construction
# --------------------------------------------------------------------------- #


def build_metrics(
    metrics: Sequence[str | Metric] | None,
    task: Task,
    *,
    notes: list[str] | None = None,
) -> list[Metric]:
    """Resolve the metric list for a task.

    ``None`` means "whatever :func:`satquery.registry.default_metrics_for` says",
    which is how the CLI and the notebooks stay ignorant of which metric belongs
    to which benchmark. Strings are registry names; instances are passed through
    so a caller can hand in a configured metric (``VqaAccuracy(strict=True)``).

    An unknown *name* raises -- that is a typo in a command line and silence
    would produce an empty report. A name that resolves to a module which cannot
    be imported is demoted to a note, because the report is more useful with
    three metrics out of four than with an exception.
    """
    names: Sequence[str | Metric]
    if metrics is None:
        names = list(default_metrics_for(task))
        if not names:
            _note(notes, f"no default metric registered for task {task!r}")
    else:
        names = list(metrics)

    built: list[Metric] = []
    for item in names:
        if isinstance(item, str):
            try:
                metric = build_metric(item)
            except ImportError as exc:  # metric module not importable on this box
                _note(notes, f"metric {item!r} unavailable: {_short(exc)}")
                continue
        else:
            metric = item
        if not supports_task(metric, task):
            raise UnsupportedTask(
                f"metric {getattr(metric, 'name', metric)!r} does not serve task {task!r} "
                f"(serves {tuple(getattr(metric, 'tasks', ()))})"
            )
        built.append(metric)
    return built


def _note(notes: list[str] | None, message: str) -> None:
    if notes is not None and message not in notes:
        notes.append(message)


def _short(exc: BaseException) -> str:
    """Last line of the traceback: the exception type and message, nothing else.

    Rows and notes are read in a terminal and committed to the report; a full
    traceback with absolute Windows paths in every row makes both unusable.
    """
    return traceback.format_exception_only(type(exc), exc)[-1].strip()


# --------------------------------------------------------------------------- #
# Per-sample rows
# --------------------------------------------------------------------------- #


def _box_dict(box: Box) -> dict[str, Any]:
    return {
        "x0": round(float(box.x0), 3),
        "y0": round(float(box.y0), 3),
        "x1": round(float(box.x1), 3),
        "y1": round(float(box.y1), 3),
        "label": box.label,
        "score": box.score,
    }


def _gold(sample: Sample) -> Any:
    """The ground truth as one JSON-friendly value, for eyeballing a run.

    Deliberately lossy: the row is for error analysis (``grep`` the jsonl for the
    questions a model got wrong), not a second copy of the dataset.
    """
    if sample.answer:
        return sample.answer
    if sample.references:
        return list(sample.references)
    if sample.boxes:
        return [_box_dict(b) for b in sample.boxes]
    if sample.labels:
        return list(sample.labels)
    return sample.meta.get("mask_path")


def _row(
    sample: Sample,
    prediction: Prediction,
    *,
    latency_ms: float | None,
    scores: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "task": sample.task,
        "question_type": sample.question_type,
        "question": sample.question,
        "gold": _gold(sample),
        "pred_text": prediction.text,
        "pred_boxes": [_box_dict(b) for b in prediction.boxes],
        "pred_labels": list(prediction.labels),
        "confidence": prediction.confidence,
        "latency_ms": latency_ms,
        "ok": prediction.ok,
        "error": prediction.error,
        "scores": dict(scores),
    }


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def _name_of(obj: Any, fallback: str) -> str:
    name = getattr(obj, "name", None)
    return str(name) if name else fallback


def _default_run_id(dataset_name: str, split: str, predictor_name: str) -> str:
    """Deterministic run id: same run, same directory, overwritten in place.

    A timestamped default would make ``satquery eval`` non-idempotent and leave a
    dozen near-identical directories for the report to disambiguate. The
    before/after tables are keyed on *predictor*, which is already in the id.
    """
    parts = [dataset_name or "dataset", split or "split", predictor_name or "predictor"]
    return "_".join(p.replace("/", "-").replace(" ", "-") for p in parts)


def _skipped_result(
    spec: RunSpec,
    reason: str,
    *,
    config: SatQueryConfig,
    started_at: str,
    duration_s: float,
    notes: Sequence[str] = (),
) -> RunResult:
    """A run that never happened, recorded as such rather than raised.

    ``n_skipped`` is 1 -- the unit skipped is the dataset, because when the files
    are missing the sample count is by definition unknown. The note carries the
    reason verbatim (which paths were missing, and the download hint).
    """
    return RunResult(
        spec=spec,
        metrics={},
        n_samples=0,
        n_failed=0,
        n_skipped=1,
        duration_s=duration_s,
        started_at=started_at,
        per_sample=(),
        env={**env_info(config), "skipped_reason": reason},
        notes=(f"dataset unavailable, run skipped: {reason}", *notes),
    )


def evaluate(
    dataset: Any,
    predictor: Predictor,
    *,
    metrics: Sequence[str | Metric] | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    config: SatQueryConfig | None = None,
    progress: bool = False,
    on_sample: Callable[[Sample, Prediction], None] | None = None,
) -> RunResult:
    """Run ``predictor`` over ``dataset`` and return the full run record.

    ``metrics`` defaults to the task's registry default. ``limit`` caps the
    number of samples consumed *here*; prefer the dataset's own ``limit``, which
    takes a seeded random slice instead of the head of a file that is usually
    sorted by scene. ``on_sample`` is called after every prediction (progress
    bars, streaming to a UI); an exception inside it is noted, not raised, so a
    display bug cannot lose a completed run.

    Never raises for data or model problems. It *does* raise
    :class:`UnsupportedTask` for a predictor/metric that cannot serve the
    dataset's task, because that is a mistake in the invocation and running it
    would produce a table of zeros that looks like a result.
    """
    cfg = config or get_config()
    task: Task = getattr(dataset, "task", "vqa")
    dataset_name = _name_of(dataset, type(dataset).__name__)
    split = str(getattr(dataset, "split", ""))
    predictor_name = _name_of(predictor, type(predictor).__name__)
    notes: list[str] = []

    spec = RunSpec(
        run_id=run_id or _default_run_id(dataset_name, split, predictor_name),
        dataset=dataset_name,
        split=split,
        task=task,
        predictor=predictor_name,
        limit=limit if limit is not None else getattr(dataset, "limit", None),
        seed=int(getattr(dataset, "seed", cfg.seed)),
        params={"metrics": [_metric_name(m) for m in (metrics or ())]} if metrics else {},
    )

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t_start = perf_counter()

    if hasattr(predictor, "supports") and not predictor.supports(task):
        raise UnsupportedTask(
            f"predictor {predictor_name!r} does not serve task {task!r} "
            f"(dataset {dataset_name}/{split})"
        )

    metric_list = build_metrics(metrics, task, notes=notes)
    for metric in metric_list:
        metric.reset()
    spec = _dc_replace(spec, params={**dict(spec.params), "metrics": [_metric_name(m) for m in metric_list]})

    # An adapter reports missing files by raising on iteration, not on
    # construction, so availability is discovered here -- and mid-iteration too,
    # since a lazily parsing adapter can hit a missing image at sample 500.
    if hasattr(dataset, "available"):
        try:
            available = bool(dataset.available())
        except Exception as exc:  # a broken availability probe is not fatal
            _note(notes, f"availability check failed: {_short(exc)}")
            available = True
        if not available:
            reason = _unavailable_reason(dataset)
            log.warning("%s/%s unavailable, skipping run", dataset_name, split)
            return _skipped_result(
                spec, reason, config=cfg, started_at=started_at,
                duration_s=perf_counter() - t_start, notes=notes,
            )

    rows: list[dict[str, Any]] = []
    n_failed = 0
    metric_errors: dict[str, int] = {}
    metric_error_msg: dict[str, str] = {}

    try:
        for index, sample in enumerate(dataset):
            if limit is not None and index >= limit:
                break

            t0 = perf_counter()
            try:
                prediction = predictor.predict(sample)
            except Exception as exc:  # noqa: BLE001 - one bad sample must not end the run
                log.debug("predictor failed on %s", sample.sample_id, exc_info=True)
                prediction = Prediction.failed(
                    sample.sample_id, sample.task, _short(exc), predictor_name
                )
            measured_ms = (perf_counter() - t0) * 1000.0
            if not prediction.ok:
                n_failed += 1

            scores: dict[str, float] = {}
            for metric in metric_list:
                name = _metric_name(metric)
                try:
                    metric.update(sample, prediction)
                except Exception as exc:  # noqa: BLE001
                    metric_errors[name] = metric_errors.get(name, 0) + 1
                    metric_error_msg[name] = _short(exc)
                    if metric_errors[name] <= _MAX_LOGGED_METRIC_ERRORS:
                        log.warning(
                            "metric %s failed on %s: %s", name, sample.sample_id, _short(exc)
                        )
                    continue
                score = getattr(metric, "score", None)
                if callable(score):
                    try:
                        scores[name] = float(score(sample, prediction))
                    except Exception:  # noqa: BLE001 - per-sample score is a nicety
                        pass

            rows.append(
                _row(
                    sample,
                    prediction,
                    latency_ms=(
                        prediction.latency_ms
                        if prediction.latency_ms is not None
                        else round(measured_ms, 3)
                    ),
                    scores=scores,
                )
            )

            if on_sample is not None:
                try:
                    on_sample(sample, prediction)
                except Exception as exc:  # noqa: BLE001
                    _note(notes, f"on_sample callback failed: {_short(exc)}")

            if progress and (index + 1) % 50 == 0:
                log.info("%s/%s: %d samples", dataset_name, split, index + 1)

    except DatasetNotAvailable as exc:
        if not rows:
            return _skipped_result(
                spec, str(exc), config=cfg, started_at=started_at,
                duration_s=perf_counter() - t_start, notes=notes,
            )
        _note(notes, f"dataset became unavailable after {len(rows)} samples: {_short(exc)}")

    metrics_out: dict[str, float] = {}
    for metric in metric_list:
        name = _metric_name(metric)
        try:
            computed = metric.compute()
        except Exception as exc:  # noqa: BLE001
            _note(notes, f"metric {name} compute() failed: {_short(exc)}")
            continue
        for key, value in computed.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                _note(notes, f"metric {name} returned non-numeric {key!r}, dropped")
                continue
            # Two metrics in one run can both emit "n". Namespacing the loser
            # keeps both numbers instead of letting the last writer win; ":" is
            # used because "/" is reserved for the slice convention.
            if key in metrics_out and metrics_out[key] != numeric:
                metrics_out[f"{name}:{key}"] = numeric
                _note(notes, f"metric key {key!r} collided, {name}'s value kept as {name}:{key}")
            else:
                metrics_out[key] = numeric

    for name, count in sorted(metric_errors.items()):
        _note(
            notes,
            f"metric {name} failed on {count} sample(s); last error: {metric_error_msg[name]}",
        )

    duration_s = perf_counter() - t_start
    env = env_info(cfg)
    if metric_errors:
        env["metric_errors"] = dict(sorted(metric_errors.items()))
    env["metrics"] = [_describe_metric(m) for m in metric_list]
    env["predictor"] = _describe_predictor(predictor)

    log.info(
        "%s/%s %s: n=%d failed=%d in %.1fs",
        dataset_name, split, predictor_name, len(rows), n_failed, duration_s,
    )
    return RunResult(
        spec=spec,
        metrics=metrics_out,
        n_samples=len(rows),
        n_failed=n_failed,
        n_skipped=0,
        duration_s=duration_s,
        started_at=started_at,
        per_sample=tuple(rows),
        env=env,
        notes=tuple(notes),
    )


def _metric_name(metric: Any) -> str:
    if isinstance(metric, str):
        return metric
    return _name_of(metric, type(metric).__name__)


def _describe_metric(metric: Any) -> Any:
    describe = getattr(metric, "describe", None)
    if callable(describe):
        try:
            return describe()
        except Exception:  # noqa: BLE001
            pass
    return {"name": _metric_name(metric)}


def _describe_predictor(predictor: Any) -> Any:
    describe = getattr(predictor, "describe", None)
    if callable(describe):
        try:
            return describe()
        except Exception:  # noqa: BLE001
            pass
    return {"name": _name_of(predictor, type(predictor).__name__)}


def _unavailable_reason(dataset: Any) -> str:
    """Turn an adapter's availability state into one actionable sentence."""
    try:
        dataset.require_available()
    except DatasetNotAvailable as exc:
        return str(exc)
    except Exception:  # noqa: BLE001 - adapter without require_available
        pass
    missing = getattr(dataset, "missing_paths", None)
    if callable(missing):
        try:
            paths = list(missing())
        except Exception:  # noqa: BLE001
            paths = []
        if paths:
            return f"missing: {', '.join(str(p) for p in paths)}"
    root = getattr(dataset, "root", None)
    return f"files not found under {root}" if root else "available() returned False"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_run(result: RunResult, *, config: SatQueryConfig | None = None) -> Path:
    """Write ``runs/<run_id>/`` and return that directory.

    Two files, on purpose. ``result.json`` is the summary a human and the report
    renderer read, and stays small enough to paste into an issue; the per-sample
    rows go to ``per_sample.jsonl`` because a 20k-row test split is grepped and
    streamed line by line, never loaded to look at one answer.
    :func:`load_run` reassembles the two, so the pair round-trips.
    """
    cfg = config or get_config()
    run_dir = cfg.run_dir(result.spec.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = result.to_dict()
    rows = payload.pop("per_sample", [])
    payload["per_sample_file"] = PER_SAMPLE_FILE
    payload["n_per_sample"] = len(rows)
    if not payload.get("env"):
        payload["env"] = env_info(cfg)

    with open(run_dir / RESULT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, default=str)
        fh.write("\n")
    with open(run_dir / PER_SAMPLE_FILE, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")

    log.info("wrote %s", run_dir)
    return run_dir


def load_run(
    path_or_run_id: str | Path,
    *,
    config: SatQueryConfig | None = None,
) -> RunResult:
    """Read back a saved run.

    Accepts a run directory, the ``result.json`` inside it, or a bare run id
    resolved against ``config.runs_dir`` -- the three things that end up in a
    shell history or a notebook cell.
    """
    cfg = config or get_config()
    target = Path(path_or_run_id)
    if target.is_dir():
        result_path = target / RESULT_FILE
    elif target.is_file():
        result_path = target
    else:
        result_path = cfg.run_dir(str(path_or_run_id)) / RESULT_FILE

    if not result_path.is_file():
        raise SatQueryError(
            f"no run found at {path_or_run_id!r} (looked for {result_path}); "
            f"runs live under {cfg.runs_dir}"
        )

    with open(result_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows_path = result_path.parent / str(payload.get("per_sample_file") or PER_SAMPLE_FILE)
    if rows_path.is_file():
        with open(rows_path, "r", encoding="utf-8") as fh:
            payload["per_sample"] = [json.loads(line) for line in fh if line.strip()]
    return RunResult.from_dict(payload)


# --------------------------------------------------------------------------- #
# Matrix
# --------------------------------------------------------------------------- #


def run_matrix(
    specs: Iterable[Mapping[str, Any]],
    *,
    config: SatQueryConfig | None = None,
) -> list[RunResult]:
    """Run several (dataset, predictor) combinations and return every result.

    One spec dict per run::

        {"dataset": "rsvqa_lr", "split": "test", "predictor": "dummy",
         "limit": 200, "metrics": ["vqa_accuracy"], "run_id": "baseline-lr",
         "dataset_kwargs": {}, "predictor_kwargs": {"accuracy": 0.5},
         "save": True}

    ``dataset`` and ``predictor`` may also be already-built objects, which is how
    tests and notebooks reuse a loaded adapter instead of paying for a second
    parse. Nothing here raises: a spec that cannot even be constructed (unknown
    name, torch not installed, dataset not downloaded) becomes a skipped
    :class:`RunResult` carrying the reason, because the point of the matrix is a
    complete table from one Colab session.
    """
    cfg = config or get_config()
    results: list[RunResult] = []

    for raw in specs:
        spec = dict(raw)
        dataset_ref = spec.get("dataset")
        predictor_ref = spec.get("predictor")
        split = spec.get("split")
        limit = spec.get("limit")
        run_id = spec.get("run_id")
        save = bool(spec.get("save", True))

        dataset_name = dataset_ref if isinstance(dataset_ref, str) else _name_of(dataset_ref, "dataset")
        predictor_name = (
            predictor_ref if isinstance(predictor_ref, str) else _name_of(predictor_ref, "predictor")
        )
        placeholder = RunSpec(
            run_id=run_id or _default_run_id(str(dataset_name), str(split or ""), str(predictor_name)),
            dataset=str(dataset_name),
            split=str(split or ""),
            task=spec.get("task", "vqa"),
            predictor=str(predictor_name),
            limit=limit,
            seed=cfg.seed,
        )

        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            if isinstance(dataset_ref, str):
                dataset = build_dataset(
                    dataset_ref,
                    split=split,
                    limit=limit,
                    config=cfg,
                    **dict(spec.get("dataset_kwargs", {})),
                )
            else:
                dataset = dataset_ref
            if isinstance(predictor_ref, str):
                predictor = build_predictor(predictor_ref, **dict(spec.get("predictor_kwargs", {})))
            else:
                predictor = predictor_ref
            result = evaluate(
                dataset,
                predictor,
                metrics=spec.get("metrics"),
                limit=limit,
                run_id=run_id,
                config=cfg,
                progress=bool(spec.get("progress", False)),
            )
        except (SatQueryError, ImportError, KeyError, TypeError, ValueError, OSError) as exc:
            log.warning("matrix entry %s/%s skipped: %s", dataset_name, predictor_name, _short(exc))
            result = _skipped_result(
                placeholder,
                _short(exc),
                config=cfg,
                started_at=started_at,
                duration_s=0.0,
            )
            results.append(result)
            continue

        if save:
            try:
                save_run(result, config=cfg)
            except OSError as exc:  # a full disk must not lose the in-memory table
                log.warning("could not save %s: %s", result.spec.run_id, _short(exc))
        results.append(result)

    return results
