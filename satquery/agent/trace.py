"""The execution trace: the artefact the assessor actually reads.

Every hop the controller takes writes one :class:`TraceEntry` here -- validation,
the tool menu and what was withheld from it, the router's plan and which router
produced it, each tool call with its parameters, timing and output, and the
composer's citation decisions. The trace is a *product*, not a debug log, and
that distinction drives three rules:

* **It is written even when the query fails.** A refusal with a reason and a
  number attached is a defensible outcome; a blank screen is not.
* **It is JSON-serialisable at all times.** Numpy scalars, Paths and dataclasses
  are coerced on the way in (:func:`jsonable`), because the trace is rendered
  into the UI, the PDF and the API response, and a serialisation failure at the
  end of a query would lose the whole thing.
* **It records what did not happen too.** A tool withheld from the router is an
  entry with its reason; that is invariant 02 made auditable, and it is what
  separates "the router chose not to" from "the router could not".

Durations are measured with :func:`time.perf_counter` and stored in
milliseconds; wall-clock timestamps are ISO-8601 UTC and are the only
non-reproducible values in the structure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

__all__ = ["TraceEntry", "Trace", "jsonable", "utc_now"]


def utc_now() -> str:
    """ISO-8601 UTC, seconds resolution. One format everywhere in the trace."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jsonable(value: Any) -> Any:
    """Coerce anything the pipeline produces into JSON-safe primitives.

    Handles the four types that actually show up and would otherwise break
    ``json.dumps`` at the worst possible moment: numpy scalars and arrays (from
    the deterministic tools), :class:`~pathlib.Path` (from every artefact),
    dataclasses (from the contracts), and sets (from tool-menu bookkeeping).
    Anything else unknown degrades to ``str`` rather than raising -- losing the
    type of one field is always better than losing the trace.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return jsonable(value.to_dict())
        return jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    # numpy without importing it: scalars expose .item(), arrays .tolist()
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        return jsonable(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    return str(value)


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """One auditable step.

    ``step`` is the pipeline stage (``validate``, ``route``, ``tool``,
    ``compose``), ``name`` the specific thing that ran (``check_pair``,
    ``change_map``, ``mistral-small-latest``). ``status`` is ``ok``, ``skipped``,
    ``withheld`` or ``error`` -- the last two exist so the absence of work is
    recorded as deliberately as its presence.
    """

    step: str
    name: str
    status: str = "ok"
    detail: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "params": jsonable(self.params),
            "output": jsonable(self.output),
            "duration_ms": round(float(self.duration_ms), 2),
            "started_at": self.started_at,
        }


class Trace:
    """An append-only list of :class:`TraceEntry`, with a timing context manager.

    Usage is deliberately two-shaped: :meth:`add` for facts already known (a
    withheld tool, a validation verdict) and :meth:`step` for work whose duration
    matters::

        with trace.step("tool", "change_map", params=params) as entry:
            entry.output = run_change_map(...)

    :meth:`step` records the entry even when the body raises, marking it
    ``error`` with the exception text, and then re-raises. A tool that blows up
    must still leave a receipt.
    """

    def __init__(self, query_id: str = "", query: str = "") -> None:
        self.query_id = query_id
        self.query = query
        self.created_at = utc_now()
        self._entries: list[TraceEntry] = []

    # -- writing ------------------------------------------------------------- #

    def add(
        self,
        step: str,
        name: str,
        *,
        status: str = "ok",
        detail: str = "",
        params: Mapping[str, Any] | None = None,
        output: Mapping[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> TraceEntry:
        entry = TraceEntry(
            step=step,
            name=name,
            status=status,
            detail=detail,
            params=dict(params or {}),
            output=dict(output or {}),
            duration_ms=duration_ms,
            started_at=utc_now(),
        )
        self._entries.append(entry)
        return entry

    def step(
        self,
        step: str,
        name: str,
        *,
        detail: str = "",
        params: Mapping[str, Any] | None = None,
    ) -> "_StepContext":
        return _StepContext(self, step, name, detail, dict(params or {}))

    # -- reading ------------------------------------------------------------- #

    def __iter__(self) -> Iterator[TraceEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[TraceEntry, ...]:
        return tuple(self._entries)

    def of_step(self, step: str) -> tuple[TraceEntry, ...]:
        return tuple(e for e in self._entries if e.step == step)

    @property
    def total_ms(self) -> float:
        return round(sum(e.duration_ms for e in self._entries), 2)

    @property
    def failed(self) -> tuple[TraceEntry, ...]:
        return tuple(e for e in self._entries if e.status == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "created_at": self.created_at,
            "total_ms": self.total_ms,
            "n_steps": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trace":
        trace = cls(str(data.get("query_id", "")), str(data.get("query", "")))
        trace.created_at = str(data.get("created_at", trace.created_at))
        for raw in data.get("entries", ()):
            trace._entries.append(
                TraceEntry(
                    step=str(raw.get("step", "")),
                    name=str(raw.get("name", "")),
                    status=str(raw.get("status", "ok")),
                    detail=str(raw.get("detail", "")),
                    params=dict(raw.get("params", {})),
                    output=dict(raw.get("output", {})),
                    duration_ms=float(raw.get("duration_ms", 0.0)),
                    started_at=str(raw.get("started_at", "")),
                )
            )
        return trace


class _StepContext:
    """Mutable scratch entry for :meth:`Trace.step`; frozen into the trace on exit."""

    __slots__ = ("_trace", "_step", "_name", "detail", "params", "output", "status", "_t0")

    def __init__(
        self, trace: Trace, step: str, name: str, detail: str, params: dict[str, Any]
    ) -> None:
        self._trace = trace
        self._step = step
        self._name = name
        self.detail = detail
        self.params = params
        self.output: Mapping[str, Any] = {}
        self.status = "ok"
        self._t0 = 0.0

    def __enter__(self) -> "_StepContext":
        self._t0 = perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        duration_ms = (perf_counter() - self._t0) * 1000.0
        if exc is not None:
            self.status = "error"
            self.detail = f"{type(exc).__name__}: {exc}"
        self._trace.add(
            self._step,
            self._name,
            status=self.status,
            detail=self.detail,
            params=self.params,
            output=self.output,
            duration_ms=duration_ms,
        )
        return False  # never swallow: the caller decides what a failure means

