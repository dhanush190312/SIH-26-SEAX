"""Run the controlled test set and compute the blueprint section 29 metrics.

Everything is measured, nothing is asserted: the runner returns counts (n/N) per metric and per tag,
plus the list of failing cases with the first reason. The clock is fixed so runs are reproducible.
"""

from __future__ import annotations

import json
import shutil
import statistics
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import IST, OrcaConfig
from ..pipeline import Orca

TESTSET = Path(__file__).with_name("testset.json")

METRICS = ("intent", "tools", "location", "time", "result", "evidence", "grounded", "safety", "multilingual")


def load(path: Path = TESTSET) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hhmm(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(IST).strftime("%H:%M")


def _called_tools(result: dict[str, Any]) -> list[str]:
    names = []
    for e in result["trace"]["entries"]:
        if e["step"] == "agent" and "." in e["name"]:
            tool = e["name"].split(".", 1)[1]
            if tool not in ("scan_hazards", "grid"):  # internal sweeps, not registry tools
                names.append(tool)
    return sorted(set(names))


def _planned_tools(result: dict[str, Any]) -> set[str]:
    plans = [e for e in result["trace"]["entries"] if e["name"].startswith("evidence_planner")]
    out: set[str] = set()
    for e in plans:
        out |= set(e["output"]["tools"])
    return out


def evaluate_case(orca_for_mode, case: dict[str, Any], now: datetime) -> dict[str, Any]:
    exp = case["expect"]
    mode = case.get("mode")
    orca = orca_for_mode(mode)
    sid = f"eval-{case['session']}"
    for q in case.get("setup", []):
        orca.query(q, session_id=sid, mode=mode, now=now)
    t0 = time.perf_counter()
    if mode == "LIVE":
        from unittest import mock
        with mock.patch("orca.data.live._get_json", side_effect=OSError("network disabled for evaluation")):
            r = orca.query(case["query"], session_id=sid, mode=mode, now=now, location=case.get("location"))
    else:
        r = orca.query(case["query"], session_id=sid, mode=mode, now=now, location=case.get("location"))
    ms = (time.perf_counter() - t0) * 1000
    checks: dict[str, bool | None] = {m: None for m in METRICS}
    why: list[str] = []

    if "intent" in exp:
        checks["intent"] = r["intent"] == exp["intent"]
        if not checks["intent"]:
            why.append(f"intent {r['intent']!r} != {exp['intent']!r}")
    if "tools" in exp:
        called, planned = set(_called_tools(r)), _planned_tools(r)
        missing, extra = set(exp["tools"]) - called, called - planned
        checks["tools"] = not missing and not extra
        if missing:
            why.append(f"tools not called: {sorted(missing)}")
        if extra:
            why.append(f"tools called outside the plan: {sorted(extra)}")
    elif r["status"] == "ok" and r["intent"] not in ("closer", "why"):
        called, planned = set(_called_tools(r)), _planned_tools(r)
        checks["tools"] = called <= planned
        if not checks["tools"]:
            why.append(f"tools called outside the plan: {sorted(called - planned)}")
    if "place" in exp:
        got = (r.get("place") or {}).get("id") or ((r.get("place") or {}).get("name", "")).split(" ")[0].lower()
        checks["location"] = got == exp["place"]
        if not checks["location"]:
            why.append(f"place {got!r} != {exp['place']!r}")
    if "times" in exp:
        day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=exp.get("day", 1))
        got = [_hhmm(x) for x in r["times"]]
        dates_ok = all(datetime.fromisoformat(x).astimezone(IST).date() == day.date() for x in r["times"])
        checks["time"] = got == exp["times"] and dates_ok
        if not checks["time"]:
            why.append(f"times {got} (dates ok={dates_ok}) != {exp['times']}")
    res_ok = True
    if "state" in exp and r["state"] != exp["state"]:
        res_ok = False; why.append(f"state {r['state']} != {exp['state']}")
    if r["status"] != exp.get("status", "ok"):
        res_ok = False; why.append(f"status {r['status']} != {exp.get('status', 'ok')}")
    if "language" in exp and r["language"] != exp["language"]:
        res_ok = False; why.append(f"language {r['language']} != {exp['language']}")
    for s in exp.get("contains", []):
        if s not in r["answer"]:
            res_ok = False; why.append(f"answer lacks {s!r}")
    for s in exp.get("not_contains", []):
        if s in r["answer"]:
            res_ok = False; why.append(f"answer contains forbidden {s!r}")
    if "alert_class" in exp and not any(a["class"] == exp["alert_class"] for a in r["alerts"]):
        res_ok = False; why.append(f"no {exp['alert_class']} alert")
    if "no_alert_class" in exp and any(a["class"] == exp["no_alert_class"] for a in r["alerts"]):
        res_ok = False; why.append(f"unexpected {exp['no_alert_class']} alert")
    if any(k in exp for k in ("state", "status", "contains", "not_contains", "alert_class", "no_alert_class", "language")) or r["status"] != "ok":
        checks["result"] = res_ok
    link = r["evidence_linkage"]
    checks["evidence"] = not link["unlinked"]
    if link["unlinked"]:
        why.append(f"{len(link['unlinked'])} factual sentence(s) without evidence")
    checks["grounded"] = bool(r["verification"]["ok"])
    if not checks["grounded"]:
        why.append("answer failed verification")
    tags = case.get("tags", [])
    if "safety" in tags:
        checks["safety"] = bool(checks["result"])
    if "tamil" in tags:
        checks["multilingual"] = bool(checks["result"]) and (checks["intent"] is not False)
    return {"id": case["id"], "tags": tags, "query": case["query"], "checks": checks, "ok": all(v is not False for v in checks.values()),
            "why": why, "ms": round(ms, 1), "state": r["state"], "intent": r["intent"], "completeness": link["completeness"], "n_factual": link["n_factual"]}


def run(selection: str = "all", path: Path = TESTSET) -> dict[str, Any]:
    ts = load(path)
    now = datetime.fromisoformat(ts["meta"]["fixed_clock"]).astimezone(IST)
    tmp = Path(tempfile.mkdtemp(prefix="orca-eval-"))
    base = replace(OrcaConfig.from_env(mode="DEMO"), now=now, runs_dir=tmp / "runs")
    shutil.copytree(base.data_dir / "demo", tmp / "data" / "demo")
    (tmp / "data" / "cache").mkdir(parents=True)
    base = replace(base, data_dir=tmp / "data")
    orcas: dict[str | None, Orca] = {}

    def for_mode(mode: str | None) -> Orca:
        key = (mode or "DEMO").upper()
        if key not in orcas:
            orcas[key] = Orca(replace(base, mode=key))
        return orcas[key]

    cases = [c for c in ts["cases"] if selection == "all" or selection in c.get("tags", [])]
    results = []
    for c in cases:
        try:
            results.append(evaluate_case(for_mode, c, now))
        except Exception as exc:  # a crashing case is a failed case, reported
            results.append({"id": c["id"], "tags": c.get("tags", []), "query": c["query"], "checks": {m: False for m in METRICS}, "ok": False,
                            "why": [f"crashed: {type(exc).__name__}: {exc}"], "ms": 0, "state": None, "intent": None, "completeness": 0, "n_factual": 0})
    shutil.rmtree(tmp, ignore_errors=True)
    return {"selection": selection, "n_cases": len(results), "results": results, "summary": summarise(results), "meta": ts["meta"]}


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    def tally(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"cases": len(rs), "cases_fully_passed": sum(1 for r in rs if r["ok"])}
        for m in METRICS:
            vals = [r["checks"][m] for r in rs if r["checks"][m] is not None]
            out[m] = {"passed": sum(1 for v in vals if v), "of": len(vals)}
        fact = sum(r["n_factual"] for r in rs)
        linked = sum(round(r["completeness"] * r["n_factual"]) for r in rs)
        out["evidence_sentences"] = {"linked": linked, "of": fact}
        ms = [r["ms"] for r in rs if r["ms"]]
        out["latency_ms"] = {"median": round(statistics.median(ms), 1) if ms else None, "max": round(max(ms), 1) if ms else None}
        return out
    by_tag: dict[str, list] = {}
    for r in results:
        for tg in r["tags"]:
            by_tag.setdefault(tg, []).append(r)
    return {"all": tally(results), "by_tag": {k: tally(v) for k, v in sorted(by_tag.items())}}


def render(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [f"Controlled evaluation ({report['n_cases']} cases, fixed clock {report['meta']['fixed_clock']})", "",
             "NOTE: " + report["meta"]["about"], ""]
    def row(name: str, t: dict[str, Any]) -> str:
        cells = [f"{name:12s} cases {t['cases_fully_passed']}/{t['cases']}"]
        for m in METRICS:
            cells.append(f"{m} {t[m]['passed']}/{t[m]['of']}" if t[m]["of"] else f"{m} -")
        cells.append(f"evidence-sentences {t['evidence_sentences']['linked']}/{t['evidence_sentences']['of']}")
        return "  ".join(cells)
    lines.append(row("ALL", s["all"]))
    for k, v in s["by_tag"].items():
        lines.append(row(k, v))
    lines.append(f"\nlatency (ms, in-process): median {s['all']['latency_ms']['median']}, max {s['all']['latency_ms']['max']}")
    fails = [r for r in report["results"] if not r["ok"]]
    lines.append(f"\nFAILING CASES: {len(fails)}")
    for r in fails:
        lines.append(f"  - [{r['id']}] {r['query'][:70]}\n      " + "; ".join(r["why"][:3]))
    return "\n".join(lines)
