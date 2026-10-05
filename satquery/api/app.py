"""The HTTP surface: the public API, the tool endpoints, and the web app.

Two API families, and the split is the architecture rather than a style choice:

* ``/api/v1/*`` is what a *user* drives -- register images, declare a pair, ask a
  question, fetch the trace and the report. One call does the whole loop.
* ``/tools/v1/*`` is what the *controller* drives -- one endpoint per tool,
  strict JSON in and out, no orchestration. They are exposed publicly anyway,
  because a tool contract you cannot call directly is a tool contract nobody can
  check: an assessor can hit ``/tools/v1/indices`` with an image id and see the
  same numbers the answer cited.

Everything is stateless above the session directory. There is no in-memory
request state, no login, no background worker; restart the server and the
sessions are still there, because a session is a folder. That is also the reason
the ``X-Session-Id`` header (or ``?session=``) is enough to identify a workspace
-- this is a single-user demo tool, and pretending otherwise would mean building
auth that the challenge does not ask for. It binds to localhost by default for
exactly that reason.

The GUI is served from ``/`` as one self-contained HTML file with no build step,
no CDN and no framework. Gradio was the original plan; a static page costs one
fewer dependency, keeps the whole app in a single process, and can be opened
straight from the API the judges are already looking at.
"""

# NOTE: no ``from __future__ import annotations`` here, deliberately. FastAPI
# resolves endpoint annotations at decoration time, and the FastAPI names are
# imported inside create_app() to keep the dependency optional -- as strings
# those forward references cannot be resolved from module globals, and every
# route with an UploadFile or Body parameter fails to build. Real annotation
# objects come from the enclosing scope and work.

from pathlib import Path
from typing import Any, Mapping

from ..config import get_config
from ..contracts import SatQueryError
from ..log import get_logger

__all__ = ["create_app", "run"]

_LOG = get_logger(__name__)

_UI_FILE = Path(__file__).with_name("ui.html")


def _require_fastapi() -> None:
    """Fail with an install hint rather than a bare ImportError three frames in."""
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SatQueryError(
            'the API needs FastAPI: pip install "satquery[api]" (fastapi, uvicorn, '
            "python-multipart)"
        ) from exc


def create_app(config=None, *, sessions_root=None):
    """Build the ASGI app. Imports FastAPI lazily so the CLI stays dependency-free."""
    _require_fastapi()
    from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

    from ..agent.controller import Controller
    from ..agent.session import DEFAULT_SESSION, Session, SessionError, SessionStore
    from ..agent.tools import TOOLS, ToolContext, build_menu, run_tool
    from ..report import render_html, render_markdown, write_pdf

    cfg = config or get_config()
    store = SessionStore(sessions_root, config=cfg)

    app = FastAPI(
        title="SatQuery AI",
        version="0.1.0",
        description=(
            "Agentic vision-language assistant for satellite imagery. Upload one or two "
            "rasters, ask a question, get an evidence-grounded answer with an auditable trace."
        ),
    )

    # -- plumbing ------------------------------------------------------------- #

    def session_of(request: Request, explicit: str | None = None) -> Session:
        name = explicit or request.headers.get("X-Session-Id") or request.query_params.get(
            "session"
        ) or DEFAULT_SESSION
        return store.get(name)

    @app.exception_handler(SessionError)
    async def _session_error(_request: Request, exc: SessionError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(SatQueryError)
    async def _satquery_error(_request: Request, exc: SatQueryError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # -- the web app ----------------------------------------------------------- #

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> Any:
        if not _UI_FILE.exists():  # pragma: no cover - packaging accident
            return HTMLResponse(
                "<h1>SatQuery AI</h1><p>The UI file is missing from this install; "
                "the API at <code>/docs</code> still works.</p>",
                status_code=200,
            )
        return FileResponse(_UI_FILE, media_type="text/html")

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        from ..agent.backends import backend_status
        from ..agent.router import build_router
        from ..io.raster import HAS_RASTERIO
        from ..report import HAS_REPORTLAB

        router, fallback, reason = build_router()
        return {
            "status": "ok",
            "version": "0.1.0",
            "config": cfg.to_dict(),
            "sessions_root": str(store.root),
            "backends": backend_status(cfg),
            "router": {
                "primary": getattr(router, "name", "?"),
                "fallback": getattr(fallback, "name", None),
                "reason": reason,
            },
            "optional": {"rasterio": HAS_RASTERIO, "reportlab": HAS_REPORTLAB},
        }

    # -- images ----------------------------------------------------------------- #

    @app.post("/api/v1/images")
    async def add_image(
        request: Request,
        file: UploadFile = File(..., description="GeoTIFF, PNG or JPEG"),
        modality: str | None = Form(None, description="optical | sar; omit to infer"),
        session: str | None = Form(None),
    ) -> dict[str, Any]:
        """Register one raster. A file that fails validation is still registered.

        The response carries the whole validation report -- modality with its
        confidence and reasoning, size, CRS, nodata coverage, warnings -- because
        the GUI has to be able to show the user what the system thinks it was
        given *before* they ask a question about it.
        """
        payload = await file.read()
        record = session_of(request, session).add_image(
            payload, filename=file.filename or "upload.tif", modality=modality
        )
        return record.to_dict()

    @app.get("/api/v1/images")
    async def list_images(request: Request) -> dict[str, Any]:
        s = session_of(request)
        return {"session_id": s.session_id, "images": [r.to_dict() for r in s.images]}

    @app.get("/api/v1/images/{image_id}")
    async def get_image(request: Request, image_id: str) -> dict[str, Any]:
        return session_of(request).image(image_id).to_dict()

    @app.get("/api/v1/images/{image_id}/preview")
    async def image_preview(request: Request, image_id: str) -> Any:
        """The display-transformed PNG -- exactly what the VLM is shown."""
        record = session_of(request).image(image_id)
        if not record.preview or not Path(record.preview).exists():
            raise HTTPException(status_code=404, detail="no preview was rendered for this image")
        return FileResponse(record.preview, media_type="image/png")

    # -- pairs ------------------------------------------------------------------ #

    @app.post("/api/v1/pairs")
    async def add_pair(
        request: Request,
        body: dict[str, Any] = Body(..., examples=[{"image_a": "img_1", "image_b": "img_2"}]),
    ) -> dict[str, Any]:
        """Declare two images a pair. ``image_a`` is the earlier date / the optical one."""
        s = session_of(request, body.get("session"))
        try:
            record = s.add_pair(str(body["image_a"]), str(body["image_b"]))
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"missing field {exc}") from exc
        return record.to_dict()

    @app.get("/api/v1/pairs")
    async def list_pairs(request: Request) -> dict[str, Any]:
        s = session_of(request)
        return {"session_id": s.session_id, "pairs": [p.to_dict() for p in s.pairs]}

    # -- tools listing ----------------------------------------------------------- #

    @app.get("/api/v1/tools")
    async def list_tools(request: Request, for_session: bool = Query(True)) -> dict[str, Any]:
        """Every tool, plus which are offered for what this session currently holds."""
        s = session_of(request)
        payload: dict[str, Any] = {"tools": [spec.to_dict() for spec in TOOLS.values()]}
        if for_session:
            pair = s.pairs[-1] if s.pairs else None
            payload["menu"] = build_menu(s.images, pair).to_dict()
        return payload

    # -- the query loop ------------------------------------------------------------ #

    @app.post("/api/v1/query")
    async def query(
        request: Request,
        body: dict[str, Any] = Body(
            ...,
            examples=[{"query": "What changed between these dates?", "image_ids": ["img_1", "img_2"]}],
        ),
    ) -> dict[str, Any]:
        """Validate, prune, route, execute, compose. Returns the answer and the trace."""
        text = str(body.get("query") or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="query must be a non-empty string")
        s = session_of(request, body.get("session"))
        image_ids = body.get("image_ids")
        result = Controller(s, config=cfg).answer(
            text,
            image_ids=[str(i) for i in image_ids] if image_ids else None,
            pair_id=body.get("pair_id"),
        )
        return result.to_dict()

    @app.get("/api/v1/queries")
    async def list_queries(request: Request) -> dict[str, Any]:
        s = session_of(request)
        return {"session_id": s.session_id, "queries": list(s.query_ids)}

    @app.get("/api/v1/queries/{query_id}")
    async def get_query(request: Request, query_id: str) -> dict[str, Any]:
        return session_of(request).load_query(query_id)["result"]

    @app.get("/api/v1/queries/{query_id}/trace")
    async def get_trace(request: Request, query_id: str) -> dict[str, Any]:
        stored = session_of(request).load_query(query_id)
        return stored.get("trace") or stored["result"].get("trace", {})

    @app.get("/api/v1/queries/{query_id}/report")
    async def get_report(
        request: Request,
        query_id: str,
        format: str = Query("pdf", pattern="^(pdf|md|html|json)$"),
    ) -> Any:
        """The downloadable receipt: PDF for a submission pack, md/html/json otherwise."""
        s = session_of(request)
        result = s.load_query(query_id)["result"]
        if format == "json":
            return JSONResponse(result)
        if format == "md":
            return PlainTextResponse(
                render_markdown(result),
                media_type="text/markdown",
                headers={"Content-Disposition": f'attachment; filename="{query_id}.md"'},
            )
        if format == "html":
            return HTMLResponse(render_html(result, file_root=s.root))
        written = write_pdf(result, s.root / "reports" / f"{query_id}.pdf", file_root=s.root)
        media = "application/pdf" if written.suffix == ".pdf" else "text/html"
        return FileResponse(written, media_type=media, filename=written.name)

    # -- artefacts -------------------------------------------------------------------- #

    @app.get("/files/{session_id}/{query_id}/{name}")
    async def artifact(session_id: str, query_id: str, name: str) -> Any:
        """Serve a mask, overlay or index map produced by a query.

        The path is rebuilt from the session store rather than joined onto a
        user string, and the result is checked to be inside the artefacts
        directory -- this is the one endpoint that maps a URL to a file, so it is
        the one that has to refuse ``../``.
        """
        s = store.get(session_id)
        path = (s.artifacts_dir / query_id / name).resolve()
        if not str(path).startswith(str(s.artifacts_dir.resolve())) or not path.exists():
            raise HTTPException(status_code=404, detail="no such artefact")
        return FileResponse(path)

    # -- sessions ---------------------------------------------------------------------- #

    @app.get("/api/v1/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"root": str(store.root), "sessions": list(store.list_ids())}

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        return {"session_id": session_id, "deleted": store.delete(session_id)}

    # -- the tool endpoints -------------------------------------------------------------- #

    def _tool_context(s: Session, body: Mapping[str, Any]) -> ToolContext:
        """One-shot context for a direct tool call: no plan, no accumulated evidence."""
        pair = None
        if body.get("pair_id"):
            pair = s.pair(str(body["pair_id"]))
            images = [s.image(pair.image_a), s.image(pair.image_b)]
        elif body.get("image_id"):
            images = [s.image(str(body["image_id"]))]
        else:
            images = list(s.images)
            if len(images) == 2:
                pair = s.pair_for([r.image_id for r in images])
        return ToolContext(
            s, query_id=s.new_query_id(), images=images, pair=pair, config=cfg
        )

    def _tool_route(name: str, path: str) -> None:
        spec = TOOLS[name]

        @app.post(path, name=f"tool_{name}", summary=spec.summary, description=spec.description)
        async def _endpoint(request: Request, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
            s = session_of(request, body.get("session"))
            ctx = _tool_context(s, body)
            params = {k: v for k, v in body.items() if k != "session"}
            result = run_tool(name, params, ctx)
            payload = result.to_dict()
            if not result.ok:
                return JSONResponse(payload, status_code=422)  # type: ignore[return-value]
            return payload

    for _name, _spec in TOOLS.items():
        _tool_route(_name, _spec.endpoint)

    return app


def run(host="127.0.0.1", port=8000, *, reload=False, config=None):
    """Serve the app with uvicorn. Localhost by default: there is no auth here."""
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SatQueryError('serving needs uvicorn: pip install "satquery[api]"') from exc

    if reload:  # uvicorn's reloader needs an import string, not an app object
        uvicorn.run("satquery.api.app:_reload_app", host=host, port=port, reload=True, factory=True)
        return
    uvicorn.run(create_app(config), host=host, port=port)


def _reload_app():  # pragma: no cover - only used by uvicorn --reload
    return create_app()
