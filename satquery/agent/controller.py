"""The loop: validate -> interpret -> route -> execute -> compose.

Hand-rolled rather than a framework, for one reason: every hop has to be
auditable, and an agent framework's job is precisely to hide the hops. Five
steps, each writing its own trace entry, each able to fail without taking the
query with it:

1. **validate** -- resolve the ids to registered images, re-read the validation
   reports, form or fetch the pair. A query on an unusable input stops here with
   the reason.
2. **interpret / prune** -- build the tool menu from the input configuration.
   Withheld tools and their reasons are recorded before the router exists.
3. **route** -- the pruned schema plus the question go to the router. On any
   router failure the offline rule router takes over and the trace says so.
4. **execute** -- run the plan in order, injecting ``image_id`` / ``pair_id``
   from the session. One tool failing does not abort the plan; the composer sees
   the gap and says so.
5. **compose** -- assemble the answer, check every clause against the evidence,
   compute the confidence.

The result is a :class:`QueryResult`, which is simultaneously the API response,
the GUI's model and the PDF report's source. There is no second representation
of an answer anywhere in the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config import SatQueryConfig, get_config
from ..log import get_logger
from .backends import backend_status
from .compose import Composition, compose
from .router import Plan, PlannedCall, Router, RouterError, build_router
from .session import ImageRecord, PairRecord, Session
from .tools import ToolContext, ToolMenu, ToolResult, build_menu, run_tool
from .trace import Trace, jsonable

__all__ = ["Controller", "QueryResult"]

_LOG = get_logger(__name__)

#: Tools whose subject is a pair; everything else is injected with one image id.
_PAIR_TOOLS = frozenset({"change_map", "change_describe", "fusion"})

#: Tools served by the text slot, whose backend choice belongs in the trace.
_TEXT_TOOLS = frozenset({"vqa", "caption", "ground", "change_describe"})


@dataclass(frozen=True, slots=True)
class QueryResult:
    """One answered question: the answer, the receipts, and the artefacts."""

    query_id: str
    session_id: str
    query: str
    answer: str
    confidence: float
    intent: str = ""
    refused: bool = False
    citations: tuple[Mapping[str, Any], ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    tools: tuple[Mapping[str, Any], ...] = ()
    images: tuple[str, ...] = ()
    pair_id: str | None = None
    warnings: tuple[str, ...] = ()
    menu: Mapping[str, Any] = field(default_factory=dict)
    plan: Mapping[str, Any] = field(default_factory=dict)
    backends: Mapping[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    trace: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "session_id": self.session_id,
            "query": self.query,
            "answer": self.answer,
            "confidence": round(float(self.confidence), 3),
            "intent": self.intent,
            "refused": self.refused,
            "citations": [dict(c) for c in self.citations],
            "evidence": dict(self.evidence),
            "artifacts": dict(self.artifacts),
            "tools": [dict(t) for t in self.tools],
            "images": list(self.images),
            "pair_id": self.pair_id,
            "warnings": list(self.warnings),
            "menu": dict(self.menu),
            "plan": dict(self.plan),
            "backends": dict(self.backends),
            "duration_ms": round(float(self.duration_ms), 2),
            "trace": dict(self.trace),
        }


class Controller:
    """Answers one question at a time against one session."""

    def __init__(
        self,
        session: Session,
        *,
        config: SatQueryConfig | None = None,
        router: Router | None = None,
        fallback_router: Router | None = None,
    ) -> None:
        self.session = session
        self.config = config or get_config()
        if router is not None:
            self.router, self.fallback, self.router_reason = router, fallback_router, "supplied by the caller"
        else:
            self.router, self.fallback, self.router_reason = build_router()

    # -- the loop ------------------------------------------------------------ #

    def answer(
        self,
        query: str,
        *,
        image_ids: Sequence[str] | None = None,
        pair_id: str | None = None,
        query_id: str | None = None,
        save: bool = True,
    ) -> QueryResult:
        query_id = query_id or self.session.new_query_id()
        trace = Trace(query_id=query_id, query=query)

        images, pair, refusal = self._resolve(query_id, image_ids, pair_id, trace)
        if refusal is not None:
            return self._finish(query_id, query, refusal, trace, images, pair, save=save)

        menu = self._prune(images, pair, trace)
        plan = self._route(query, menu, trace)
        results = self._execute(plan, menu, images, pair, query_id, trace)
        composition = self._compose(query, results, images, menu, trace)

        result = QueryResult(
            query_id=query_id,
            session_id=self.session.session_id,
            query=query,
            answer=composition.answer,
            confidence=composition.confidence,
            intent=plan.intent,
            refused=composition.refused,
            citations=tuple(c.to_dict() for c in composition.citations),
            evidence=dict(composition.evidence),
            artifacts=_merge_artifacts(results),
            tools=tuple(r.to_dict() for r in results),
            images=tuple(r.image_id for r in images),
            pair_id=pair.pair_id if pair else None,
            warnings=composition.warnings,
            menu=menu.to_dict(),
            plan=plan.to_dict(),
            backends=self._backends(),
            duration_ms=trace.total_ms,
            trace=trace.to_dict(),
        )
        if save:
            self.session.save_query(result, trace)
        return result

    # -- steps ---------------------------------------------------------------- #

    def _resolve(
        self,
        query_id: str,
        image_ids: Sequence[str] | None,
        pair_id: str | None,
        trace: Trace,
    ) -> tuple[list[ImageRecord], PairRecord | None, Composition | None]:
        """Step 1. Turn ids into validated records, and form the pair if needed."""
        with trace.step("validate", "resolve_inputs", params={"image_ids": list(image_ids or ()), "pair_id": pair_id}) as entry:
            pair: PairRecord | None = None
            if pair_id:
                pair = self.session.pair(pair_id)
                records = [self.session.image(pair.image_a), self.session.image(pair.image_b)]
            else:
                ids = list(image_ids or [r.image_id for r in self.session.images])
                records = [self.session.image(i) for i in ids]
                if len(records) == 2:
                    # Two images and no declared pair: form one so the change and
                    # fusion tools become reachable. The check still decides
                    # whether they are usable together.
                    pair = self.session.pair_for([r.image_id for r in records]) or self.session.add_pair(
                        records[0].image_id, records[1].image_id
                    )
            entry.output = {
                "images": [r.report.to_dict() for r in records],
                "pair": pair.to_dict() if pair else None,
            }
            entry.detail = (
                f"{len(records)} image(s)"
                + (f", pair {pair.pair_id} ({pair.pair_type})" if pair else ", no pair")
            )

        usable = [r for r in records if r.ok]
        if not usable:
            problems = "; ".join(
                f"{r.image_id}: {'; '.join(r.report.errors) or 'not usable'}" for r in records
            ) or "no image was registered for this query"
            trace.add("validate", "refuse", status="error", detail=problems)
            return (
                records,
                pair,
                Composition(
                    answer=f"This query cannot be answered: {problems}.",
                    confidence=0.05,
                    refused=True,
                    warnings=(problems,),
                ),
            )
        return records, pair, None

    def _prune(
        self, images: Sequence[ImageRecord], pair: PairRecord | None, trace: Trace
    ) -> ToolMenu:
        """Step 2. Build the tool menu -- and record what was withheld, and why."""
        with trace.step("interpret", "build_tool_menu") as entry:
            menu = build_menu(images, pair)
            entry.output = menu.to_dict()
            entry.detail = (
                f"offered {len(menu.offered)}, withheld {len(menu.withheld)}"
            )
        for name, reason in menu.withheld:
            trace.add("interpret", name, status="withheld", detail=reason)
        return menu

    def _route(self, query: str, menu: ToolMenu, trace: Trace) -> Plan:
        """Step 3. Ask the router for a plan; fall back rather than fail."""
        with trace.step(
            "route",
            getattr(self.router, "name", "router"),
            detail=self.router_reason,
            params={"tools_offered": list(menu.names), "sends_imagery": False},
        ) as entry:
            try:
                plan = self.router.plan(query, menu)
            except RouterError as exc:
                if self.fallback is None:
                    entry.status = "error"
                    entry.detail = f"{exc}; no fallback router configured"
                    plan = Plan(
                        intent="unsupported",
                        calls=(),
                        router=getattr(self.router, "name", "router"),
                        fallback_reason=str(exc),
                    )
                else:
                    entry.status = "skipped"
                    entry.detail = f"{exc}; falling back to {self.fallback.name}"
                    plan = self.fallback.plan(query, menu)
                    plan = Plan(
                        intent=plan.intent,
                        calls=plan.calls,
                        router=plan.router,
                        rationale=plan.rationale,
                        fallback_reason=str(exc),
                        raw=plan.raw,
                        notes=plan.notes,
                    )
            entry.output = plan.to_dict()
        return plan

    def _execute(
        self,
        plan: Plan,
        menu: ToolMenu,
        images: Sequence[ImageRecord],
        pair: PairRecord | None,
        query_id: str,
        trace: Trace,
    ) -> list[ToolResult]:
        """Step 4. Run the plan, injecting the ids the router was never given."""
        ctx = ToolContext(
            self.session, query_id=query_id, images=list(images), pair=pair, config=self.config
        )
        results: list[ToolResult] = []
        for call in plan.calls:
            params = self._inject(call, images, pair)
            with trace.step("tool", call.tool, detail=call.reason, params=jsonable(params)) as entry:
                result = run_tool(call.tool, params, ctx)
                entry.status = "ok" if result.ok else "error"
                entry.output = dict(result.to_dict())
                if call.tool in _TEXT_TOOLS:
                    # Which text backend served this, and why it was chosen --
                    # an answer cannot be read without knowing that.
                    entry.output["backend_selection"] = ctx.vlm_reason
                if not result.ok:
                    entry.detail = result.error or "tool failed"
            results.append(result)
        if not plan.calls:
            trace.add(
                "tool",
                "none",
                status="skipped",
                detail="the plan contained no runnable tool for this input",
            )
        return results

    def _inject(
        self, call: PlannedCall, images: Sequence[ImageRecord], pair: PairRecord | None
    ) -> dict[str, Any]:
        """Invariant 01, mechanically: the ids come from the session, not the model."""
        params = {k: v for k, v in dict(call.params).items() if k not in ("image_id", "pair_id")}
        if call.tool in _PAIR_TOOLS:
            if pair is not None:
                params["pair_id"] = pair.pair_id
            return params
        usable = [r for r in images if r.ok]
        if usable:
            params["image_id"] = usable[0].image_id
        return params

    def _compose(
        self,
        query: str,
        results: Sequence[ToolResult],
        images: Sequence[ImageRecord],
        menu: ToolMenu,
        trace: Trace,
    ) -> Composition:
        """Step 5. Cite or drop every clause, then score the confidence."""
        warnings: list[str] = []
        modality_confidence = 1.0
        for record in images:
            warnings.extend(record.report.warnings)
            modality_confidence = min(modality_confidence, record.report.modality_confidence or 1.0)

        with trace.step("compose", "cite_and_score") as entry:
            composition = compose(
                query,
                results,
                validation_warnings=tuple(warnings),
                modality_confidence=modality_confidence,
                withheld=menu.withheld,
            )
            entry.output = composition.to_dict()
            entry.detail = (
                f"{len(composition.citations)} citation(s), "
                f"{len(composition.dropped)} clause(s) dropped, "
                f"confidence {composition.confidence:.2f}"
            )
        return composition

    # -- helpers -------------------------------------------------------------- #

    def _backends(self) -> dict[str, Any]:
        status = backend_status(self.config)
        status["router"] = {
            "backend": getattr(self.router, "name", "router"),
            "reason": self.router_reason,
            "fallback": getattr(self.fallback, "name", None),
        }
        return status

    def _finish(
        self,
        query_id: str,
        query: str,
        composition: Composition,
        trace: Trace,
        images: Sequence[ImageRecord],
        pair: PairRecord | None,
        *,
        save: bool,
    ) -> QueryResult:
        result = QueryResult(
            query_id=query_id,
            session_id=self.session.session_id,
            query=query,
            answer=composition.answer,
            confidence=composition.confidence,
            refused=True,
            warnings=composition.warnings,
            images=tuple(r.image_id for r in images),
            pair_id=pair.pair_id if pair else None,
            # An empty menu, not a missing one: a consumer reads the same shape
            # whether the query was answered or refused before routing.
            menu={"offered": [], "withheld": [], "context": ""},
            backends=self._backends(),
            duration_ms=trace.total_ms,
            trace=trace.to_dict(),
        )
        if save:
            self.session.save_query(result, trace)
        return result


def _merge_artifacts(results: Sequence[ToolResult]) -> dict[str, str]:
    out: dict[str, str] = {}
    for result in results:
        for key, value in dict(result.artifacts).items():
            out[f"{result.name}.{key}"] = value
    return out
