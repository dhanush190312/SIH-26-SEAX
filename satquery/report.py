"""Rendering one answered query into something a human can file.

The trace is the deliverable the challenge is assessed on, so it has to leave the
process in a form that survives being emailed: Markdown for a repo or a chat, PDF
for a submission pack. Both are generated from the *same* ``QueryResult`` dict
the API returns, so a report can never disagree with the answer the user saw --
there is one representation of a query in this system and this module formats it.

What every report contains, in this order, because it is the order a sceptical
reader needs:

1. the question and the answer, with the confidence and how it was reached;
2. the citations -- each claim next to the tool and value backing it, plus
   anything the composer dropped;
3. the inputs: what was uploaded, what the validator made of it, what the pair
   check said;
4. which tools were offered, which were withheld and why -- the negative space
   matters as much as the positive;
5. the full execution trace with timings and the backend that served each slot.

PDF generation uses reportlab when it is installed and otherwise writes the HTML
next to it and says so. A missing optional dependency downgrades the format, it
never loses the report.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .log import get_logger

__all__ = [
    "render_markdown",
    "render_html",
    "write_report",
    "write_pdf",
    "HAS_REPORTLAB",
]

_LOG = get_logger(__name__)


def _has(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_REPORTLAB: bool = _has("reportlab")


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def render_markdown(result: Mapping[str, Any], *, include_trace: bool = True) -> str:
    """The whole query as one Markdown document."""
    r = dict(result)
    out: list[str] = []
    add = out.append

    add(f"# SatQuery AI — query {r.get('query_id', '?')}")
    add("")
    add(f"**Question.** {r.get('query', '')}")
    add("")
    add(f"**Answer.** {r.get('answer', '')}")
    add("")
    confidence = float(r.get("confidence", 0.0))
    add(
        f"**Confidence {confidence:.2f}** — composed from the backend's own score, the share of "
        "the answer that is measured rather than described, the input validation, and a penalty "
        "for any clause dropped as unsupported."
    )
    if r.get("refused"):
        add("")
        add("> This query was **refused or partially refused**. See the reasons below.")
    add("")

    # -- citations ---------------------------------------------------------- #
    citations = list(r.get("citations", ()))
    add("## Evidence behind the answer")
    add("")
    if citations:
        add("| claim | measurement | value | tool |")
        add("|---|---|---|---|")
        for c in citations:
            add(
                f"| {_md(c.get('claim'))} | `{_md(c.get('key'))}` | {_md(c.get('value'))} "
                f"| {_md(c.get('tool'))} |"
            )
    else:
        add("_No claim in this answer was tied to a measurement._")
    add("")

    warnings = list(r.get("warnings", ()))
    if warnings:
        add("**Caveats**")
        add("")
        for w in warnings:
            add(f"- {w}")
        add("")

    # -- inputs -------------------------------------------------------------- #
    add("## Inputs")
    add("")
    validation = _validation_entries(r)
    if validation:
        add("| image | size | bands | modality | source | notes |")
        add("|---|---|---|---|---|---|")
        for image in validation:
            summary = image.get("summary", {})
            add(
                f"| `{_md(image.get('image_id'))}` "
                f"| {summary.get('width')}×{summary.get('height')} "
                f"| {summary.get('bands')} "
                f"| {_md(image.get('modality'))} ({float(image.get('modality_confidence', 0)):.2f}) "
                f"| {_md(image.get('modality_source'))} "
                f"| {_md('; '.join(image.get('errors', []) + image.get('warnings', [])) or '—')} |"
            )
        add("")
    pair = _pair_entry(r)
    if pair:
        add(
            f"**Pair `{pair.get('pair_id')}`** — {pair.get('pair_type')}, "
            f"{'usable' if pair.get('ok') else 'not usable'}, checked in {pair.get('mode')} mode; "
            + "; ".join(pair.get("reasons", ()))
        )
        add("")

    # -- tool menu ------------------------------------------------------------ #
    menu = dict(r.get("menu", {}))
    add("## Tools offered and withheld")
    add("")
    add(f"Offered: {', '.join(f'`{t}`' for t in menu.get('offered', ())) or '_none_'}")
    add("")
    withheld = menu.get("withheld", ())
    if withheld:
        add("| withheld | reason |")
        add("|---|---|")
        for item in withheld:
            add(f"| `{_md(item.get('tool'))}` | {_md(item.get('reason'))} |")
        add("")

    # -- plan ------------------------------------------------------------------ #
    plan = dict(r.get("plan", {}))
    add("## Plan")
    add("")
    add(f"Router: `{plan.get('router', 'n/a')}` — intent `{plan.get('intent', 'n/a')}`")
    if plan.get("fallback_reason"):
        add("")
        add(f"> Fell back to the offline router: {plan['fallback_reason']}")
    add("")
    for i, call in enumerate(plan.get("calls", ()), start=1):
        params = ", ".join(f"{k}={v!r}" for k, v in dict(call.get("params", {})).items())
        add(f"{i}. `{call.get('tool')}`({params}) — {call.get('reason', '')}")
    for note in plan.get("notes", ()):
        add(f"   - note: {note}")
    add("")

    # -- tool results ----------------------------------------------------------- #
    tools = list(r.get("tools", ()))
    if tools:
        add("## Tool calls")
        add("")
        add("| tool | backend | kind | ms | result |")
        add("|---|---|---|---|---|")
        for tool in tools:
            outcome = "ok" if tool.get("ok") else f"failed: {tool.get('error')}"
            add(
                f"| `{_md(tool.get('tool'))}` | {_md(tool.get('backend'))} "
                f"| {_md(tool.get('backend_kind'))} | {tool.get('duration_ms')} | {_md(outcome)} |"
            )
        add("")
        for tool in tools:
            evidence = dict(tool.get("evidence", {}))
            if evidence:
                add(f"<details><summary>{tool.get('tool')} measurements</summary>")
                add("")
                add("```json")
                add(json.dumps(evidence, indent=2, default=str))
                add("```")
                add("")
                add("</details>")
                add("")

    artifacts = dict(r.get("artifacts", {}))
    if artifacts:
        add("## Artefacts")
        add("")
        for name, url in artifacts.items():
            add(f"- `{name}`: {url}")
        add("")

    backends = dict(r.get("backends", {}))
    if backends:
        add("## Backends that served this query")
        add("")
        add("| slot | backend | why |")
        add("|---|---|---|")
        for slot, info in backends.items():
            add(
                f"| {_md(slot)} | `{_md(info.get('backend'))}` "
                f"| {_md(info.get('reason'))} |"
            )
        add("")

    if include_trace:
        add("## Execution trace")
        add("")
        trace = dict(r.get("trace", {}))
        add(
            f"{trace.get('n_steps', 0)} steps, {trace.get('total_ms', 0)} ms total, "
            f"started {trace.get('created_at', '')}"
        )
        add("")
        add("| step | name | status | ms | detail |")
        add("|---|---|---|---|---|")
        for entry in trace.get("entries", ()):
            add(
                f"| {_md(entry.get('step'))} | {_md(entry.get('name'))} "
                f"| {_md(entry.get('status'))} | {entry.get('duration_ms')} "
                f"| {_md(entry.get('detail'))} |"
            )
        add("")

    add("---")
    add("")
    add(
        "_Generated by SatQuery AI. Every number in this report was produced by a named tool; "
        "claims the composer could not attribute were removed from the answer and are listed "
        "under the citations table._"
    )
    return "\n".join(out) + "\n"


def _md(value: Any) -> str:
    """Escape a value for a Markdown table cell (pipes break the row)."""
    return str("" if value is None else value).replace("|", "\\|").replace("\n", " ")


def _validation_entries(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per-image validation, dug out of the trace where the controller wrote it."""
    for entry in dict(result.get("trace", {})).get("entries", ()):
        if entry.get("step") == "validate" and entry.get("name") == "resolve_inputs":
            return list(entry.get("output", {}).get("images", ()))
    return []


def _pair_entry(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The pair verdict, flattened.

    The trace stores a ``PairRecord`` -- ids at the top level, the geometric
    check nested under ``check`` -- so the two halves are merged here rather than
    at four call sites, each of which would otherwise render "pair_type: None".
    """
    for entry in dict(result.get("trace", {})).get("entries", ()):
        if entry.get("step") == "validate" and entry.get("name") == "resolve_inputs":
            pair = entry.get("output", {}).get("pair")
            if not pair:
                return None
            merged = {k: v for k, v in dict(pair).items() if k != "check"}
            merged.update(dict(pair.get("check", {})))
            return merged
    return None


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


_HTML_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0 auto;
       max-width: 60rem; padding: 2rem 1.25rem 4rem; color: #16181d; background: #fff; }
h1 { font-size: 1.55rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; padding-bottom: .3rem;
     border-bottom: 1px solid #e3e6ea; }
.answer { font-size: 1.1rem; background: #f5f7fa; border-left: 4px solid #3b6ea5;
          padding: .9rem 1rem; border-radius: 0 6px 6px 0; }
.meta { color: #5b6472; font-size: .85rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .88rem; }
th, td { border: 1px solid #e3e6ea; padding: .35rem .5rem; text-align: left;
         vertical-align: top; }
th { background: #f5f7fa; font-weight: 600; }
code { background: #f0f2f5; padding: .05rem .3rem; border-radius: 3px; font-size: .85em; }
.pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .78rem;
        border: 1px solid currentColor; }
.ok { color: #1c7c4a; } .warn { color: #9a6700; } .bad { color: #b32d2e; }
.bar { height: .5rem; background: #e3e6ea; border-radius: 999px; overflow: hidden;
       max-width: 16rem; }
.bar > i { display: block; height: 100%; background: #3b6ea5; }
img { max-width: 100%; border: 1px solid #e3e6ea; border-radius: 4px; }
figure { margin: 0; } figcaption { font-size: .8rem; color: #5b6472; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; }
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e6e8ec; }
  .answer { background: #1c2027; border-left-color: #6ea8dc; }
  th { background: #1c2027; } th, td, img { border-color: #2a2f38; }
  code { background: #1c2027; } .bar { background: #2a2f38; }
  h2 { border-bottom-color: #2a2f38; }
}
"""


def render_html(result: Mapping[str, Any], *, file_root: str | Path | None = None) -> str:
    """A self-contained HTML page for one query.

    ``file_root`` points at the session directory so artefact URLs
    (``/files/<session>/<query>/<name>``) can be resolved to real files and
    inlined as data URIs -- an emailed report has to keep its overlays.
    """
    r = dict(result)
    e = html.escape
    confidence = float(r.get("confidence", 0.0))
    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>SatQuery — {e(str(r.get('query_id', '')))}</title>",
        f"<style>{_HTML_CSS}</style></head><body>",
        "<h1>SatQuery AI</h1>",
        f"<p class='meta'>Query <code>{e(str(r.get('query_id', '')))}</code> · session "
        f"<code>{e(str(r.get('session_id', '')))}</code> · "
        f"{r.get('duration_ms', 0)} ms</p>",
        f"<h2>Question</h2><p>{e(str(r.get('query', '')))}</p>",
        f"<h2>Answer</h2><p class='answer'>{e(str(r.get('answer', '')))}</p>",
        f"<p class='meta'>Confidence {confidence:.2f}</p>",
        f"<div class='bar'><i style='width:{max(2, min(100, confidence * 100)):.0f}%'></i></div>",
    ]

    citations = list(r.get("citations", ()))
    parts.append("<h2>Evidence behind the answer</h2>")
    if citations:
        parts.append("<table><tr><th>claim</th><th>measurement</th><th>value</th><th>tool</th></tr>")
        for c in citations:
            parts.append(
                f"<tr><td>{e(str(c.get('claim')))}</td><td><code>{e(str(c.get('key')))}</code></td>"
                f"<td>{e(str(c.get('value')))}</td><td>{e(str(c.get('tool')))}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p class='meta'>No claim in this answer was tied to a measurement.</p>")

    warnings = list(r.get("warnings", ()))
    if warnings:
        parts.append("<h2>Caveats</h2><ul>")
        parts.extend(f"<li class='warn'>{e(str(w))}</li>" for w in warnings)
        parts.append("</ul>")

    images = _embedded_images(r, file_root)
    if images:
        parts.append("<h2>Imagery and overlays</h2><div class='grid'>")
        for caption, src in images:
            parts.append(
                f"<figure><img src='{src}' alt='{e(caption)}'>"
                f"<figcaption>{e(caption)}</figcaption></figure>"
            )
        parts.append("</div>")

    menu = dict(r.get("menu", {}))
    parts.append("<h2>Tools offered and withheld</h2><p>")
    parts.append(
        " ".join(f"<span class='pill ok'>{e(str(t))}</span>" for t in menu.get("offered", ()))
        or "<span class='meta'>none</span>"
    )
    parts.append("</p>")
    if menu.get("withheld"):
        parts.append("<table><tr><th>withheld</th><th>reason</th></tr>")
        for item in menu["withheld"]:
            parts.append(
                f"<tr><td><code>{e(str(item.get('tool')))}</code></td>"
                f"<td>{e(str(item.get('reason')))}</td></tr>"
            )
        parts.append("</table>")

    trace = dict(r.get("trace", {}))
    parts.append("<h2>Execution trace</h2>")
    parts.append(
        f"<p class='meta'>{trace.get('n_steps', 0)} steps · {trace.get('total_ms', 0)} ms · "
        f"{e(str(trace.get('created_at', '')))}</p>"
    )
    parts.append("<table><tr><th>step</th><th>name</th><th>status</th><th>ms</th><th>detail</th></tr>")
    for entry in trace.get("entries", ()):
        css = {"ok": "ok", "error": "bad", "withheld": "warn", "skipped": "warn"}.get(
            str(entry.get("status")), ""
        )
        parts.append(
            f"<tr><td>{e(str(entry.get('step')))}</td><td>{e(str(entry.get('name')))}</td>"
            f"<td class='{css}'>{e(str(entry.get('status')))}</td>"
            f"<td>{entry.get('duration_ms')}</td><td>{e(str(entry.get('detail')))}</td></tr>"
        )
    parts.append("</table>")

    backends = dict(r.get("backends", {}))
    if backends:
        parts.append("<h2>Backends</h2><table><tr><th>slot</th><th>backend</th><th>why</th></tr>")
        for slot, info in backends.items():
            parts.append(
                f"<tr><td>{e(str(slot))}</td><td><code>{e(str(info.get('backend')))}</code></td>"
                f"<td>{e(str(info.get('reason')))}</td></tr>"
            )
        parts.append("</table>")

    parts.append(
        "<p class='meta'>Generated by SatQuery AI. Every number here was produced by a named "
        "tool; unattributable claims were removed from the answer before it was shown.</p>"
    )
    parts.append("</body></html>")
    return "".join(parts)


def _embedded_images(
    result: Mapping[str, Any], file_root: str | Path | None
) -> list[tuple[str, str]]:
    """``(caption, data-uri)`` for every artefact that resolves to a real file."""
    if file_root is None:
        return []
    import base64
    import mimetypes

    root = Path(file_root)
    out: list[tuple[str, str]] = []
    for name, url in dict(result.get("artifacts", {})).items():
        path = _resolve_artifact(root, str(url))
        if path is None or not path.exists():
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        out.append((name, f"data:{mime};base64,{data}"))
    return out


def _resolve_artifact(session_root: Path, url: str) -> Path | None:
    """``/files/<session>/<query>/<file>`` -> a path inside the session directory."""
    parts = [p for p in url.split("/") if p]
    if len(parts) < 4 or parts[0] != "files":
        return None
    return session_root / "artifacts" / parts[2] / parts[3]


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #


def write_report(
    result: Mapping[str, Any],
    path: str | Path,
    *,
    fmt: str = "md",
    file_root: str | Path | None = None,
) -> Path:
    """Write ``md``, ``html`` or ``pdf``. Returns the path actually written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "md":
        p.write_text(render_markdown(result), encoding="utf-8")
        return p
    if fmt == "html":
        p.write_text(render_html(result, file_root=file_root), encoding="utf-8")
        return p
    if fmt == "pdf":
        return write_pdf(result, p, file_root=file_root)
    raise ValueError(f"unknown report format {fmt!r}; use md, html or pdf")


def write_pdf(
    result: Mapping[str, Any], path: str | Path, *, file_root: str | Path | None = None
) -> Path:
    """Render the query as a PDF, or as HTML with a warning if reportlab is absent.

    reportlab rather than WeasyPrint: WeasyPrint needs GTK/Pango on Windows,
    which is a system install rather than a pip install, and the report's layout
    is a stack of headings, paragraphs, tables and images -- exactly what
    reportlab's Platypus flowables do without a browser engine.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not HAS_REPORTLAB:
        fallback = p.with_suffix(".html")
        fallback.write_text(render_html(result, file_root=file_root), encoding="utf-8")
        _LOG.warning(
            "reportlab is not installed, so the report was written as HTML: %s "
            "(pip install reportlab)",
            fallback,
        )
        return fallback

    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image as PdfImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    r = dict(result)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, textColor=colors.HexColor("#5b6472"))
    answer = ParagraphStyle(
        "answer", parent=body, fontSize=11.5, leading=16, leftIndent=6, spaceBefore=4,
        backColor=colors.HexColor("#f2f5f9"), borderPadding=6,
    )

    def cell(text: Any, style: ParagraphStyle = small) -> Paragraph:
        return Paragraph(html.escape(str("" if text is None else text)), style)

    def table(rows: Sequence[Sequence[Any]], widths: Sequence[float]) -> Table:
        t = Table([[cell(c) for c in row] for row in rows], colWidths=list(widths), repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dbe0")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return t

    width = A4[0] - 36 * mm
    flow: list[Any] = [
        Paragraph("SatQuery AI — query report", styles["Title"]),
        Paragraph(
            f"Query <b>{html.escape(str(r.get('query_id', '')))}</b> · session "
            f"{html.escape(str(r.get('session_id', '')))} · {r.get('duration_ms', 0)} ms",
            small,
        ),
        Spacer(1, 8),
        Paragraph("<b>Question</b>", body),
        Paragraph(html.escape(str(r.get("query", ""))), body),
        Spacer(1, 6),
        Paragraph("<b>Answer</b>", body),
        Paragraph(html.escape(str(r.get("answer", ""))), answer),
        Spacer(1, 6),
        Paragraph(
            f"<b>Confidence {float(r.get('confidence', 0.0)):.2f}</b> — composed from the "
            "backend score, the measured share of the answer, input validation and a penalty "
            "for dropped clauses.",
            small,
        ),
        Spacer(1, 10),
    ]

    citations = list(r.get("citations", ()))
    flow.append(Paragraph("<b>Evidence behind the answer</b>", body))
    if citations:
        flow.append(
            table(
                [["claim", "measurement", "value", "tool"]]
                + [[c.get("claim"), c.get("key"), c.get("value"), c.get("tool")] for c in citations],
                [width * 0.34, width * 0.28, width * 0.18, width * 0.20],
            )
        )
    else:
        flow.append(Paragraph("No claim in this answer was tied to a measurement.", small))
    flow.append(Spacer(1, 10))

    menu = dict(r.get("menu", {}))
    flow.append(Paragraph("<b>Tools offered and withheld</b>", body))
    flow.append(Paragraph("Offered: " + (", ".join(menu.get("offered", ())) or "none"), small))
    if menu.get("withheld"):
        flow.append(Spacer(1, 4))
        flow.append(
            table(
                [["withheld", "reason"]]
                + [[i.get("tool"), i.get("reason")] for i in menu["withheld"]],
                [width * 0.22, width * 0.78],
            )
        )
    flow.append(Spacer(1, 10))

    images = _artifact_paths(r, file_root)
    if images:
        flow.append(Paragraph("<b>Imagery and overlays</b>", body))
        for caption, image_path in images[:6]:
            try:
                flow.append(PdfImage(str(image_path), width=width * 0.48, height=width * 0.48))
                flow.append(Paragraph(html.escape(caption), small))
                flow.append(Spacer(1, 6))
            except Exception as exc:  # a broken PNG must not lose the report
                _LOG.warning("could not embed %s: %s", image_path, exc)

    flow.append(PageBreak())
    trace = dict(r.get("trace", {}))
    flow.append(Paragraph("<b>Execution trace</b>", body))
    flow.append(
        Paragraph(
            f"{trace.get('n_steps', 0)} steps · {trace.get('total_ms', 0)} ms · "
            f"{trace.get('created_at', '')}",
            small,
        )
    )
    flow.append(Spacer(1, 4))
    flow.append(
        table(
            [["step", "name", "status", "ms", "detail"]]
            + [
                [
                    entry.get("step"),
                    entry.get("name"),
                    entry.get("status"),
                    entry.get("duration_ms"),
                    entry.get("detail"),
                ]
                for entry in trace.get("entries", ())
            ],
            [width * 0.12, width * 0.18, width * 0.10, width * 0.08, width * 0.52],
        )
    )

    backends = dict(r.get("backends", {}))
    if backends:
        flow.append(Spacer(1, 10))
        flow.append(Paragraph("<b>Backends that served this query</b>", body))
        flow.append(
            table(
                [["slot", "backend", "why"]]
                + [[k, v.get("backend"), v.get("reason")] for k, v in backends.items()],
                [width * 0.16, width * 0.24, width * 0.60],
            )
        )

    SimpleDocTemplate(
        str(p),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"SatQuery AI — {r.get('query_id', '')}",
        author="SatQuery AI",
    ).build(flow)
    return p


def _artifact_paths(
    result: Mapping[str, Any], file_root: str | Path | None
) -> list[tuple[str, Path]]:
    if file_root is None:
        return []
    root = Path(file_root)
    out: list[tuple[str, Path]] = []
    for name, url in dict(result.get("artifacts", {})).items():
        path = _resolve_artifact(root, str(url))
        if path is not None and path.exists():
            out.append((name, path))
    return out
