"""Stdlib HTTP server for ORCA: JSON API + the single-file web app.

Deliberately not FastAPI: FastAPI/uvicorn cannot be assumed on a judging laptop, and a
zero-dependency server means ``python -m orca serve`` works from a bare Python 3.10+.
The endpoint *conventions* are the ones SatQuery established (``/api/<name>/...`` for what a
user drives, JSON in / JSON out, a trace fetchable by query id, a single static page at ``/``).
Binds to localhost by default -- it is an unauthenticated single-user demo tool.
"""

from __future__ import annotations

import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from satquery.agent.trace import jsonable
from satquery.log import get_logger

from ..config import OrcaConfig, get_config
from ..data.providers import _demo_docs, freeze, save_cache
from ..pipeline import Orca
from ..report import render_html
from . import endpoints as ep

_LOG = get_logger("orca.api")
_UI = Path(__file__).with_name("ui.html")
_ID = re.compile(r"^o_[0-9a-f]{8}$")
MAX_BODY = 64 * 1024

SCENARIOS = [
    {"id": "fishing", "label": "Fishing", "text": "Can I fish tomorrow at 6 AM from Nagapattinam?"},
    {"id": "pfz", "label": "PFZ", "text": "Show potential fishing zones near Nagapattinam tomorrow."},
    {"id": "hazard", "label": "Hazard", "text": "Are there dangerous marine conditions tomorrow morning near Nagapattinam?"},
    {"id": "route", "label": "Route", "text": "Give me two route options avoiding restricted areas from Nagapattinam tomorrow at 10 AM."},
    {"id": "why", "label": "WHY", "text": "Why did you recommend this zone?"},
    {"id": "whatif", "label": "WHAT-IF", "text": "Compare 6 AM and 10 AM tomorrow from Nagapattinam."},
    {"id": "tamil", "label": "தமிழ்", "text": "நாளைக்கு காலை 6 மணிக்கு கடலுக்கு போகலாமா? நாகப்பட்டினம்"},
    {"id": "avoid", "label": "Avoid", "text": "Show areas I should avoid from Nagapattinam tomorrow at 10 AM."},
    {"id": "nearest", "label": "Nearest PFZ", "text": "Find the nearest PFZ and check the hazards around it tomorrow morning."},
    {"id": "routecheck", "label": "Route check", "text": "Does this route cross a restricted area?"},
    {"id": "regions", "label": "Ocean regions", "text": "Which nearby regions have high chlorophyll and favourable SST?"},
]

#: follow-ups that rely on conversation memory (blueprint section 21)
FOLLOWUPS = [
    {"label": "What about tomorrow?", "text": "What about tomorrow?"},
    {"label": "Which one is closer?", "text": "Which one is closer?"},
    {"label": "Conditions there", "text": "What are the marine conditions there tomorrow morning?"},
    {"label": "Leave in the evening?", "text": "What changes if I leave in the evening?"},
]


def refresh_cache(orca: Orca) -> dict[str, Any]:
    """Write the OFFLINE cache from DEMO data (or from LIVE data when in LIVE mode)."""
    cfg = orca.cfg
    now = cfg.clock()
    if cfg.mode == "LIVE":
        from ..data.providers import make_provider
        prov = make_provider(cfg)  # writes the cache itself when weather was fetched
        return {"origin_mode": "LIVE", "ok": prov.capabilities()["weather"], "errors": prov.errors}
    docs = freeze(_demo_docs(cfg), now)
    path = save_cache(docs, cfg.data_dir / "cache" / "bundle.json", origin_mode="DEMO", retrieved_at=now)
    return {"origin_mode": "DEMO", "ok": True, "path": str(path), "last_updated": now.isoformat(timespec="seconds")}


def make_handler(orca: Orca):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        server_version = "ORCA/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            _LOG.info("%s %s", self.address_string(), fmt % args)

        # -- plumbing -------------------------------------------------------- #
        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(jsonable(obj), ensure_ascii=False).encode("utf-8"))

        def _err(self, code: int, msg: str) -> None:
            self._json(code, {"error": msg, "status": code})

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ep.BadRequest("request body too large")
            if n == 0:
                return {}
            try:
                d = json.loads(self.rfile.read(n).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ep.BadRequest("body must be valid UTF-8 JSON") from None
            if not isinstance(d, dict):
                raise ep.BadRequest("body must be a JSON object")
            return d

        # -- verbs ----------------------------------------------------------- #
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            try:
                if path == "/":
                    self._send(200, _UI.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/orca/health":
                    with lock:
                        self._json(200, orca.health())
                elif path == "/api/orca/scenarios":
                    self._json(200, {"scenarios": SCENARIOS, "followups": FOLLOWUPS})
                elif path == "/api/orca/config":
                    self._json(200, orca.cfg.public())
                elif (m := re.match(r"^/api/orca/(trace|evidence|result|report)/([^/]+)$", path)):
                    kind, qid = m.groups()
                    if not _ID.match(qid):
                        return self._err(400, "malformed query id")
                    r = orca.get_result(qid)
                    if r is None:
                        return self._err(404, f"no such query: {qid}")
                    if kind == "trace":
                        self._json(200, r["trace"])
                    elif kind == "evidence":
                        self._json(200, {"query_id": qid, "n": len(r.get("evidence", {})), "evidence": r.get("evidence", {}),
                                         "claims": r.get("evidence_records"), "linkage": r.get("evidence_linkage"), "alerts": r.get("alerts"),
                                         "cards": r.get("cards"), "decisions": r.get("decisions"), "source_conflicts": [
                                             e["output"] for e in r["trace"]["entries"] if e["step"] == "fuse"]})
                    elif kind == "report":
                        self._send(200, render_html(r).encode("utf-8"), "text/html; charset=utf-8")
                    else:
                        self._json(200, {k: v for k, v in r.items() if k != "evidence"})
                else:
                    self._err(404, "not found")
            except Exception as exc:  # never leak a stack trace to the client
                _LOG.exception("GET %s failed", path)
                self._err(500, f"{type(exc).__name__}")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/")
            try:
                body = self._body()
                with lock:
                    if path == "/api/orca/query":
                        text = str(body.get("text", "")).strip()
                        if not text:
                            raise ep.BadRequest("'text' is required")
                        if len(text) > 500:
                            raise ep.BadRequest("question is too long (max 500 characters)")
                        lang = body.get("lang") or None
                        out = orca.query(text, session_id=str(body.get("session_id", "default"))[:64], mode=body.get("mode") or None, lang=lang,
                                         location=ep.parse_location(body.get("location")))
                    elif path == "/api/orca/ocean":
                        out = ep.ocean(orca, body)
                    elif path == "/api/orca/weather":
                        out = ep.weather(orca, body)
                    elif path == "/api/orca/geo":
                        out = ep.geo(orca, body)
                    elif path == "/api/orca/route":
                        out = ep.route(orca, body)
                    elif path == "/api/orca/route/check":
                        out = ep.route_check(orca, body)
                    elif path == "/api/orca/compare":
                        out = ep.compare(orca, body)
                    elif path == "/api/orca/cache/refresh":
                        out = refresh_cache(orca)
                    else:
                        return self._err(404, "not found")
                self._json(200, out)
            except (ep.BadRequest, ValueError) as exc:
                self._err(400, str(exc))
            except Exception as exc:
                _LOG.exception("POST %s failed", path)
                self._err(500, f"{type(exc).__name__}")

    return H


def create_server(cfg: OrcaConfig | None = None, host: str | None = None, port: int | None = None, orca: Orca | None = None) -> ThreadingHTTPServer:
    cfg = cfg or get_config()
    orca = orca or Orca(cfg)
    srv = ThreadingHTTPServer((host or cfg.host, cfg.port if port is None else port), make_handler(orca))
    srv.orca = orca  # type: ignore[attr-defined]
    return srv


def serve(cfg: OrcaConfig | None = None, host: str | None = None, port: int | None = None) -> None:
    srv = create_server(cfg, host, port)
    h, p = srv.server_address[:2]
    print(f"ORCA running at http://{h}:{p}/   (mode: {srv.orca.cfg.mode}; Ctrl+C to stop)")  # type: ignore[attr-defined]
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()
