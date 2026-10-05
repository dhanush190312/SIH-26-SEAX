"""``satquery`` -- one entry point for the whole system.

Subcommands, grouped by what they are for:

*Environment*
    ``doctor``    what is installed, which backend serves each slot, and why
    ``datasets``  the registry, and which benchmarks are present locally
    ``tools``     the tool contract, and which tools a session's images support

*Answering*
    ``ask``       one question over one or two files, printed with its trace
    ``serve``     the API and the web app on localhost
    ``session``   list, inspect and clear workspaces

*Evaluation*
    ``eval``      run a predictor over a benchmark and save the run
    ``report``    render a saved run as Markdown
    ``compare``   two or more runs side by side, with the before/after delta

The import policy from the package docstring is enforced here more than
anywhere: ``doctor``, ``datasets`` and ``tools`` must run on a bare interpreter
with numpy and nothing else, because they are what a user runs *first*, on the
machine where the install went wrong. Everything heavy is imported inside the
handler that needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .config import get_config
from .contracts import SatQueryError
from .log import get_logger

__all__ = ["main", "build_parser"]

_LOG = get_logger(__name__)

_OPTIONAL = (
    ("numpy", "required, the one hard dependency"),
    ("PIL", "image IO for PNG/JPEG benchmark tiles and previews  [data]"),
    ("rasterio", "GeoTIFF + CRS; without it, PNG/JPEG only        [raster]"),
    ("pyproj", "CRS comparison in the pair check                  [raster]"),
    ("scipy", "faster connected components and speckle filter     [data]"),
    ("pandas", "tabular dataset annotations                       [data]"),
    ("torch", "VLM inference                                      [vlm]"),
    ("transformers", "VLM inference                               [vlm]"),
    ("fastapi", "the HTTP API and the web app                     [api]"),
    ("uvicorn", "serving the API                                  [api]"),
    ("reportlab", "PDF reports (HTML fallback without it)         [api]"),
)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="satquery",
        description="Agentic vision-language assistant for satellite imagery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  satquery doctor\n"
            "  satquery ask 'what changed?' before.tif after.tif\n"
            "  satquery serve --port 8000\n"
            "  satquery eval --dataset rsvqa_lr --predictor dummy --limit 50\n"
            "  satquery compare runs/base runs/tuned\n"
        ),
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output where offered")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="environment, optional dependencies and backend selection")

    p = sub.add_parser("datasets", help="registered benchmarks and whether they are present")
    p.add_argument("--check", action="store_true", help="open each adapter to test availability")

    sub.add_parser("tools", help="the tool contract, and the menu for the current session")

    p = sub.add_parser("ask", help="answer one question about one or two images")
    p.add_argument("query", help="the question, in plain English")
    p.add_argument("images", nargs="+", help="one or two raster paths")
    p.add_argument("--modality", choices=("optical", "sar"), action="append", default=None,
                   help="declare modality per image, in order (repeatable)")
    p.add_argument("--session", default=None, help="workspace to use (default: a fresh one)")
    p.add_argument("--report", default=None, help="also write a report here (.md/.html/.pdf)")
    p.add_argument("--trace", action="store_true", help="print the full execution trace")

    p = sub.add_parser("serve", help="run the API and the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="reload on source changes (development)")

    p = sub.add_parser("session", help="list, inspect or clear workspaces")
    p.add_argument("action", choices=("list", "show", "clear"), nargs="?", default="list")
    p.add_argument("name", nargs="?", default=None)

    p = sub.add_parser("eval", help="run a predictor over a benchmark")
    p.add_argument("--dataset", required=True)
    p.add_argument("--predictor", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--metrics", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-save", action="store_true")

    p = sub.add_parser("report", help="render a saved run as Markdown")
    p.add_argument("run", help="run id or path to a run directory / result.json")
    p.add_argument("--out", default=None, help="write here instead of stdout")

    p = sub.add_parser("compare", help="two or more runs side by side")
    p.add_argument("runs", nargs="+", help="run ids or paths")
    p.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = _HANDLERS[args.command]
    try:
        return handler(args)
    except SatQueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("interrupted", file=sys.stderr)
        return 130


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def _cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    cfg = get_config()
    installed = {}
    for module, why in _OPTIONAL:
        try:
            installed[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            installed[module] = False

    from .agent.backends import backend_status
    from .agent.router import build_router

    router, fallback, reason = build_router()
    backends = backend_status(cfg)
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg.to_dict(),
        "installed": installed,
        "backends": backends,
        "router": {
            "primary": getattr(router, "name", "?"),
            "fallback": getattr(fallback, "name", None),
            "reason": reason,
        },
        # Presence only. A doctor that prints a credential is a doctor nobody can
        # paste into a bug report.
        "env": {
            k: ("set" if os.environ.get(k) else "unset")
            for k in ("MISTRAL_API_KEY", "SATQUERY_VLM", "SATQUERY_ROUTER",
                      "SATQUERY_DATA_ROOT", "SATQUERY_RUNS_DIR")
        },
        "env_file": _env_file_status(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"SatQuery AI · python {payload['python']} · {payload['platform']}")
    print()
    print("dependencies")
    for module, why in _OPTIONAL:
        mark = "yes" if installed[module] else "no "
        print(f"  [{mark}] {module:<14} {why}")
    print()
    print("serving")
    for slot, info in backends.items():
        print(f"  {slot:<11} {info['backend']:<20} ({info['kind']})")
        print(f"              {info['reason']}")
    print(f"  router      {payload['router']['primary']:<20} "
          f"(fallback: {payload['router']['fallback'] or 'none'})")
    print(f"              {reason}")
    print()
    print("paths")
    for key, value in cfg.to_dict().items():
        print(f"  {key:<11} {value}")
    print()
    print("environment")
    for key, value in payload["env"].items():
        print(f"  {key:<20} {value}")
    print(f"  {'.env file':<20} {payload['env_file']}")
    if payload["env"]["MISTRAL_API_KEY"] == "unset":
        print()
        print("  no MISTRAL_API_KEY, so routing is offline (deterministic keyword rules).")
        print("  to use Mistral: copy .env.example to .env and fill in the key.")
    return 0


def _env_file_status() -> str:
    """Where a ``.env`` was found, or that none was -- never what is in it.

    The first question when a key "does not work" is whether the file is where
    the process is looking, and the answer depends on the working directory.
    """
    from .config import ENV_FILE, repo_root

    found = [str(p) for p in (Path.cwd() / ENV_FILE, repo_root() / ENV_FILE) if p.is_file()]
    return ", ".join(dict.fromkeys(found)) if found else "not present"


def _cmd_datasets(args: argparse.Namespace) -> int:
    from .registry import build_dataset, list_datasets

    cfg = get_config()
    rows = []
    for name, task, split in list_datasets():
        root = cfg.dataset_root(name)
        row = {
            "name": name,
            "task": task,
            "split": split,
            "root": str(root),
            "root_exists": root.exists(),
            "available": None,
            "note": "",
        }
        if args.check:
            try:
                adapter = build_dataset(name)
                row["available"] = bool(adapter.available())
            except Exception as exc:
                row["available"] = False
                row["note"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    width = max(len(r["name"]) for r in rows)
    print(f"{'dataset'.ljust(width)}  {'task':<18} {'split':<6} present  root")
    for r in rows:
        present = "?" if r["available"] is None else ("yes" if r["available"] else "no")
        if not args.check:
            present = "yes" if r["root_exists"] else "no"
        print(f"{r['name'].ljust(width)}  {r['task']:<18} {r['split']:<6} {present:<8} {r['root']}")
        if r["note"]:
            print(f"{' ' * width}   {r['note']}")
    print()
    print("point an adapter at a download with SATQUERY_DATA_<NAME>, or move the whole tree "
          "with SATQUERY_DATA_ROOT")
    return 0


def _cmd_tools(args: argparse.Namespace) -> int:
    from .agent.tools import TOOLS

    if args.json:
        print(json.dumps([spec.to_dict() for spec in TOOLS.values()], indent=2))
        return 0
    for spec in TOOLS.values():
        params = ", ".join(spec.parameters) or "—"
        print(f"{spec.name:<16} {spec.endpoint}")
        print(f"{'':<16} {spec.summary}")
        print(f"{'':<16} needs: {', '.join(spec.requires)} · router params: {params}")
        print()
    print("The executor injects image_id / pair_id; the router never supplies them.")
    return 0


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #


def _cmd_ask(args: argparse.Namespace) -> int:
    from .agent.controller import Controller
    from .agent.session import SessionStore

    if len(args.images) > 2:
        raise SatQueryError(f"ask takes one or two images, got {len(args.images)}")
    store = SessionStore()
    session = store.get(args.session) if args.session else store.create()

    modalities = list(args.modality or [])
    records = []
    for i, path in enumerate(args.images):
        modality = modalities[i] if i < len(modalities) else None
        record = session.add_image(path, modality=modality)
        records.append(record)
        status = "ok" if record.ok else "REJECTED"
        print(
            f"[{status}] {record.image_id}  {Path(path).name}  "
            f"{record.report.summary.get('width')}x{record.report.summary.get('height')} "
            f"{record.report.summary.get('bands')}b  {record.modality} "
            f"({record.report.modality_confidence:.2f}, {record.report.modality_source})",
            file=sys.stderr,
        )
        for message in (*record.report.errors, *record.report.warnings):
            print(f"          {message}", file=sys.stderr)

    result = Controller(session).answer(args.query, image_ids=[r.image_id for r in records])

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print()
        print(result.answer)
        print()
        print(f"confidence {result.confidence:.2f} · intent {result.intent or '—'} · "
              f"{result.duration_ms:.0f} ms · router {result.plan.get('router', '—')}")
        if result.citations:
            print()
            print("evidence")
            for c in result.citations:
                print(f"  {c['claim']:<28} <- {c['tool']}.{c['key']} = {c['value']}")
        withheld = result.menu.get("withheld") or []
        if withheld:
            print()
            print("tools withheld for this input")
            for item in withheld:
                print(f"  {item['tool']:<16} {item['reason']}")
        for warning in result.warnings:
            print(f"\n! {warning}")
        if args.trace:
            print()
            print("trace")
            for entry in result.trace.get("entries", ()):
                print(f"  {entry['step']:<10} {entry['name']:<18} {entry['status']:<9} "
                      f"{entry['duration_ms']:>8.1f} ms  {entry['detail']}")
        print()
        print(f"artefacts under {session.artifacts_dir / result.query_id}")

    if args.report:
        from .report import write_report

        out = Path(args.report)
        fmt = out.suffix.lstrip(".").lower() or "md"
        written = write_report(result.to_dict(), out, fmt=fmt, file_root=session.root)
        print(f"report: {written}", file=sys.stderr)
    return 0 if not result.refused else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    from .api.app import run

    print(f"SatQuery AI on http://{args.host}:{args.port}  (API docs at /docs)")
    run(host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_session(args: argparse.Namespace) -> int:
    from .agent.session import SessionStore

    store = SessionStore()
    if args.action == "list":
        ids = store.list_ids()
        if args.json:
            print(json.dumps({"root": str(store.root), "sessions": list(ids)}, indent=2))
        else:
            print(f"sessions under {store.root}")
            for name in ids:
                session = store.get(name)
                print(f"  {name:<20} {len(session.images)} image(s), "
                      f"{len(session.pairs)} pair(s), {len(session.query_ids)} query(ies)")
            if not ids:
                print("  (none)")
        return 0
    if not args.name:
        raise SatQueryError(f"'session {args.action}' needs a session name")
    if args.action == "clear":
        print("deleted" if store.delete(args.name) else "no such session")
        return 0
    session = store.get(args.name)
    print(json.dumps(session.to_dict(), indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval.harness import evaluate, save_run
    from .registry import build_dataset, build_predictor

    dataset = build_dataset(args.dataset, split=args.split)
    predictor = build_predictor(args.predictor)
    result = evaluate(
        dataset,
        predictor,
        metrics=args.metrics,
        limit=args.limit,
        run_id=args.run_id,
        progress=not args.json,
    )
    if not args.no_save:
        path = save_run(result)
        print(f"run saved: {path}", file=sys.stderr)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    from .eval.report import render_markdown

    print(render_markdown(result))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .eval.harness import load_run
    from .eval.report import render_markdown

    text = render_markdown(load_run(args.run))
    return _emit(text, args.out)


def _cmd_compare(args: argparse.Namespace) -> int:
    from .eval.harness import load_run
    from .eval.report import render_comparison

    text = render_comparison([load_run(r) for r in args.runs])
    return _emit(text, args.out)


def _emit(text: str, out: str | None) -> int:
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"written: {path}", file=sys.stderr)
    else:
        print(text)
    return 0


_HANDLERS = {
    "doctor": _cmd_doctor,
    "datasets": _cmd_datasets,
    "tools": _cmd_tools,
    "ask": _cmd_ask,
    "serve": _cmd_serve,
    "session": _cmd_session,
    "eval": _cmd_eval,
    "report": _cmd_report,
    "compare": _cmd_compare,
}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
