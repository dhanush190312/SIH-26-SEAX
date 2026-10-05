"""``python -m orca <command>`` -- serve | ask | demo | doctor | cache-refresh."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .api.server import SCENARIOS, refresh_cache, serve
from .config import OrcaConfig, set_config
from .pipeline import Orca


def _card_text(card: dict) -> str:
    out = [f"  RECOMMENDATION: {card['recommendation']['label']} - {card['recommendation']['headline']}"]
    for s in card["sections"]:
        out.append(f"  [{s['title']}]")
        for r in s["rows"]:
            mark = {"ok": "✓", "warn": "!", "bad": "✗", "missing": "?", "info": "·"}.get(r["status"], "·")
            out.append(f"   {mark} {r['label']}: {r['text']}")
    for c in card["conflicts"]:
        out.append(f"  ⚠ {c['headline']}: " + " | ".join(c["lines"]) + f" | {c['selected_line']} | {c['difference_line']}")
    out.append("  FRESHNESS: " + ", ".join(f"{k}: {v['text']}" for k, v in card["freshness"].items()))
    return "\n".join(out)


def _show(r: dict, *, verbose: bool) -> None:
    print(f"[{r['mode_label']}]  {r['state_label']}  (query {r['query_id']}, {len(r['trace']['entries'])} trace steps)")
    print(r["answer"])
    print(f"({r['disclaimer']})")
    if verbose and r.get("primary_card"):
        print(_card_text(r["primary_card"]))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="orca", description="ORCA - evidence-linked marine decision support (SIH26176)")
    ap.add_argument("--mode", choices=["DEMO", "LIVE", "OFFLINE"], help="data mode (default: ORCA_MODE or DEMO)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the API and web app"); s.add_argument("--host"); s.add_argument("--port", type=int)
    a = sub.add_parser("ask", help="ask one question"); a.add_argument("text"); a.add_argument("--json", action="store_true"); a.add_argument("--card", action="store_true", help="also print the evidence card")
    d = sub.add_parser("demo", help="run every demo scenario and verify each answer"); d.add_argument("--card", action="store_true")
    e = sub.add_parser("eval", help="run the controlled test set and print measured metrics")
    e.add_argument("--set", default="all", help="all | core | paraphrase | tamil | safety | ... (any tag)"); e.add_argument("--json", help="write the full report to this file")
    sub.add_parser("doctor", help="environment and data-mode check")
    sub.add_parser("cache-refresh", help="write the OFFLINE cache from DEMO (or LIVE) data")
    args = ap.parse_args(argv)

    cfg = OrcaConfig.from_env(**({"mode": args.mode} if args.mode else {}))
    set_config(cfg)
    orca = Orca(cfg)

    if args.cmd == "serve":
        serve(cfg, args.host, args.port); return 0
    if args.cmd == "ask":
        r = orca.query(args.text)
        print(json.dumps({k: v for k, v in r.items() if k != "trace"}, ensure_ascii=False, indent=2)) if args.json else _show(r, verbose=args.card)
        return 0
    if args.cmd == "eval":
        from .eval.runner import render, run
        rep = run(args.set)
        print(render(rep))
        if args.json:
            from pathlib import Path
            Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "cache-refresh":
        print(json.dumps(refresh_cache(orca), indent=2)); return 0
    if args.cmd == "doctor":
        h = orca.health()
        print(f"ORCA {h['version']}  python {sys.version.split()[0]}  default mode: {cfg.mode}  narrator: {h['narrator'] or 'none (template answers)'}")
        for m, info in h["modes"].items():
            print(f"  {m:8s} {info.get('label', info.get('note'))}")
            if "capabilities" in info:
                print("           " + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in info["capabilities"].items()))
            for e in info.get("errors", []):
                print(f"           note: {e}")
        return 0
    if args.cmd == "demo":
        bad = 0
        for sc in SCENARIOS:
            r = orca.query(sc["text"], session_id="cli-demo")
            ok = r["verification"]["ok"] and r["status"] in ("ok",) and not r["evidence_linkage"]["unlinked"]
            bad += not ok
            print(f"\n=== {sc['label']}: {sc['text']}")
            _show(r, verbose=args.card)
            print(f"--> {'PASS' if ok else 'FAIL'}: answer verified={r['verification']['ok']}, factual sentences linked to evidence={r['evidence_linkage']['n_linked']}/{r['evidence_linkage']['n_factual']}, status={r['status']}, records={len(r['evidence_records'])}")
        print(f"\n{len(SCENARIOS) - bad}/{len(SCENARIOS)} scenarios passed")
        return 1 if bad else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
