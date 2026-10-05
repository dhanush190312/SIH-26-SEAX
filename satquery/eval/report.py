"""Turning run artefacts into the table the adaptation claim rests on.

The whole reason the harness exists before the fine-tune is the before/after
delta, and a delta is only evidence if the reader can see what was compared. So
these renderers refuse to print a bare number: every table carries the sample
count, the failure count, the predictor and the environment, and
:func:`render_comparison` will not quietly compare two runs of different sizes
without saying so.

One rule is load-bearing enough to be enforced here rather than remembered.
``VqaAccuracy`` emits both ``accuracy`` (exact match, the published RSVQA
convention) and ``accuracy_lenient``; an un-tuned VLM answers "There are 2
buildings." where the gold answer is "2", so a strict-only comparison credits the
fine-tune for learning the output *format* and reports it as learning the task.
:func:`render_comparison` therefore prints both whenever both exist, and marks
the row where they disagree in direction -- that row is the one where the
headline delta is a formatting artefact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..contracts import RunResult
from ..log import get_logger

__all__ = ["render_markdown", "render_comparison", "metric_table", "load_runs"]

_LOG = get_logger(__name__)

#: Keys that are counts rather than scores; rendered as integers and never
#: shown as a delta with three decimal places.
_COUNT_KEYS = frozenset({"n", "n_labels", "n_labels_scored", "n_out_of_vocab", "n_valid"})

#: Metrics where a *lower* value is better. Everything else is assumed
#: higher-is-better, which is true of every metric this project reports today;
#: the set exists so adding a distance or an error metric later does not silently
#: print a regression as an improvement.
_LOWER_IS_BETTER: frozenset[str] = frozenset()


def render_markdown(result: RunResult | Mapping[str, Any], *, top_n: int = 5) -> str:
    """One run as a Markdown page: headline metrics, breakdowns, environment."""
    r = result.to_dict() if isinstance(result, RunResult) else dict(result)
    spec = dict(r.get("spec", {}))
    metrics = dict(r.get("metrics", {}))
    out: list[str] = []
    add = out.append

    add(f"# Run `{spec.get('run_id', '?')}`")
    add("")
    add(
        f"**{spec.get('dataset')}** / {spec.get('split')} · task `{spec.get('task')}` · "
        f"predictor `{spec.get('predictor')}`"
        + (f" · limit {spec['limit']}" if spec.get("limit") else "")
    )
    add("")
    add(
        f"{r.get('n_samples', 0)} samples · {r.get('n_failed', 0)} failed · "
        f"{r.get('n_skipped', 0)} skipped · {r.get('duration_s', 0)} s · "
        f"started {r.get('started_at', '')}"
    )
    add("")
    for note in r.get("notes", ()):
        add(f"> {note}")
    if r.get("notes"):
        add("")

    headline, breakdown = _split_metrics(metrics)
    add("## Metrics")
    add("")
    add(metric_table(headline))
    add("")
    for prefix, values in breakdown.items():
        add(f"### {prefix}")
        add("")
        add(metric_table(values))
        add("")

    env = dict(r.get("env", {}))
    if env:
        add("## Environment")
        add("")
        add("| key | value |")
        add("|---|---|")
        for key in sorted(env):
            add(f"| {key} | {_cell(env[key])} |")
        add("")

    samples = list(r.get("per_sample", ()))
    if samples and top_n:
        add(f"## First {min(top_n, len(samples))} predictions")
        add("")
        columns = _sample_columns(samples)
        add("| " + " | ".join(columns) + " |")
        add("|" + "|".join("---" for _ in columns) + "|")
        for row in samples[:top_n]:
            add("| " + " | ".join(_cell(row.get(c)) for c in columns) + " |")
        add("")

    return "\n".join(out) + "\n"


def render_comparison(
    runs: Sequence[RunResult | Mapping[str, Any]],
    *,
    baseline: int = 0,
    keys: Sequence[str] | None = None,
) -> str:
    """Several runs side by side, with the delta against ``baseline``.

    The comparison is only meaningful if the runs scored the same thing, so
    mismatched dataset, split, task or sample count produce a warning line in the
    output rather than a silently wrong table. That warning is the point: a
    before/after delta measured over different subsets is the easiest way to
    claim an improvement that is not there.
    """
    dicts = [r.to_dict() if isinstance(r, RunResult) else dict(r) for r in runs]
    if not dicts:
        return "_No runs to compare._\n"
    base = dicts[baseline]
    base_spec = dict(base.get("spec", {}))

    out: list[str] = []
    add = out.append
    add(f"# Comparison — {base_spec.get('dataset')} / {base_spec.get('split')}")
    add("")

    warnings = _comparability_warnings(dicts, baseline)
    for warning in warnings:
        add(f"> ⚠️ {warning}")
    if warnings:
        add("")

    metric_keys = list(keys) if keys else _shared_keys(dicts)
    header = ["metric"] + [
        f"{dict(d.get('spec', {})).get('predictor', '?')}"
        + (" (baseline)" if i == baseline else "")
        for i, d in enumerate(dicts)
    ]
    if len(dicts) > 1:
        header.append("delta")
    add("| " + " | ".join(header) + " |")
    add("|" + "|".join("---" for _ in header) + "|")

    for key in metric_keys:
        values = [dict(d.get("metrics", {})).get(key) for d in dicts]
        row = [f"`{key}`"] + [_number(v) for v in values]
        if len(dicts) > 1:
            row.append(_delta(values[baseline], values[-1], key))
        add("| " + " | ".join(row) + " |")
    add("")

    strict, lenient = _direction_disagreement(dicts, baseline)
    if strict is not None and lenient is not None:
        add(
            f"**Read both accuracy columns.** Strict exact match moved {strict:+.3f} and the "
            f"lenient variant moved {lenient:+.3f}."
            + (
                " They disagree in direction, which means the headline delta is mostly an "
                "answer-*format* change rather than a task improvement."
                if strict * lenient < 0
                else " They agree in direction, so the change is not purely a formatting effect."
            )
        )
        add("")

    add("| run | samples | failed | skipped | duration s |")
    add("|---|---|---|---|---|")
    for d in dicts:
        spec = dict(d.get("spec", {}))
        add(
            f"| `{spec.get('run_id')}` ({spec.get('predictor')}) | {d.get('n_samples')} "
            f"| {d.get('n_failed')} | {d.get('n_skipped')} | {d.get('duration_s')} |"
        )
    add("")
    return "\n".join(out) + "\n"


def metric_table(metrics: Mapping[str, float]) -> str:
    if not metrics:
        return "_No metrics._"
    rows = ["| metric | value |", "|---|---|"]
    for key in sorted(metrics):
        rows.append(f"| `{key}` | {_number(metrics[key])} |")
    return "\n".join(rows)


def load_runs(paths: Iterable[str | Path]) -> list[RunResult]:
    """Load ``result.json`` files (or the run directories holding them)."""
    out: list[RunResult] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            p = p / "result.json"
        out.append(RunResult.from_dict(json.loads(p.read_text(encoding="utf-8"))))
    return out


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _split_metrics(metrics: Mapping[str, float]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Headline metrics, and per-slice breakdowns grouped by their prefix.

    The ``'/'`` convention from :class:`~satquery.contracts.Metric` is what makes
    this possible: ``accuracy`` is the headline and ``accuracy/count`` belongs in
    the ``accuracy`` breakdown table.
    """
    headline: dict[str, float] = {}
    breakdown: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        if "/" in key:
            prefix, suffix = key.split("/", 1)
            breakdown.setdefault(prefix, {})[suffix] = value
        else:
            headline[key] = value
    return headline, breakdown


def _shared_keys(dicts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Metric keys worth tabulating: headline first, then slices, all sorted."""
    keys: set[str] = set()
    for d in dicts:
        keys |= set(dict(d.get("metrics", {})))
    plain = sorted(k for k in keys if "/" not in k and k not in _COUNT_KEYS)
    sliced = sorted(k for k in keys if "/" in k)
    counts = sorted(k for k in keys if k in _COUNT_KEYS)
    return plain + sliced + counts


def _comparability_warnings(dicts: Sequence[Mapping[str, Any]], baseline: int) -> list[str]:
    base = dict(dicts[baseline].get("spec", {}))
    warnings: list[str] = []
    for i, d in enumerate(dicts):
        if i == baseline:
            continue
        spec = dict(d.get("spec", {}))
        for field in ("dataset", "split", "task"):
            if spec.get(field) != base.get(field):
                warnings.append(
                    f"run `{spec.get('run_id')}` scored {field}={spec.get(field)!r} against the "
                    f"baseline's {base.get(field)!r} -- these are not comparable"
                )
        if d.get("n_samples") != dicts[baseline].get("n_samples"):
            warnings.append(
                f"run `{spec.get('run_id')}` covered {d.get('n_samples')} samples against the "
                f"baseline's {dicts[baseline].get('n_samples')}; a delta over different subsets "
                "is not a delta"
            )
    return warnings


def _direction_disagreement(
    dicts: Sequence[Mapping[str, Any]], baseline: int
) -> tuple[float | None, float | None]:
    """``(strict delta, lenient delta)`` between the baseline and the last run."""
    if len(dicts) < 2:
        return (None, None)
    first = dict(dicts[baseline].get("metrics", {}))
    last = dict(dicts[-1].get("metrics", {}))
    if "accuracy" not in first or "accuracy" not in last:
        return (None, None)
    if "accuracy_lenient" not in first or "accuracy_lenient" not in last:
        return (None, None)
    return (
        float(last["accuracy"]) - float(first["accuracy"]),
        float(last["accuracy_lenient"]) - float(first["accuracy_lenient"]),
    )


def _number(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _cell(value)
    if number.is_integer() and abs(number) >= 1:
        return f"{int(number)}"
    return f"{number:.4f}"


def _delta(before: Any, after: Any, key: str) -> str:
    if before is None or after is None:
        return "—"
    try:
        change = float(after) - float(before)
    except (TypeError, ValueError):
        return "—"
    if key in _COUNT_KEYS:
        return f"{change:+.0f}"
    arrow = "" if abs(change) < 1e-9 else ("▲" if (change > 0) != (key in _LOWER_IS_BETTER) else "▼")
    return f"{change:+.4f} {arrow}".strip()


def _sample_columns(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = ["sample_id", "question", "answer", "prediction", "correct", "error"]
    present = [c for c in preferred if any(c in row for row in samples)]
    extra = sorted({k for row in samples for k in row} - set(present))
    return present + extra[:3]


def _cell(value: Any) -> str:
    text = str("" if value is None else value)
    if len(text) > 80:
        text = text[:77] + "…"
    return text.replace("|", "\\|").replace("\n", " ")
