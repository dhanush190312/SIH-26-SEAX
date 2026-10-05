"""Printable HTML report for one ORCA result (reuses the idea of SatQuery's report module;
no reportlab dependency -- open in a browser and print to PDF)."""

from __future__ import annotations

import html
import json
from typing import Any, Mapping


def _e(x: Any) -> str:
    return html.escape(str(x))


def render_html(r: Mapping[str, Any]) -> str:
    card = r.get("primary_card")
    parts = [f"<h1>ORCA report {_e(r['query_id'])}</h1>",
             f"<p><b>Data mode:</b> {_e(r['mode_label'])}</p>",
             f"<p><b>Question:</b> {_e(r['query'])}</p>",
             f"<h2>{_e(r['state_label'])}</h2><p>{_e(r['answer'])}</p><p><i>{_e(r['disclaimer'])}</i></p>"]
    for c in r.get("cards", [])[:6]:
        parts.append(f"<h3>{_e(c['title'])} - {_e(c['recommendation']['label'])}</h3><p>{_e(c['recommendation']['headline'])}</p>")
        for s in c["sections"]:
            parts.append(f"<h4>{_e(s['title'])}</h4><table>")
            for row in s["rows"]:
                parts.append(f"<tr><td>{_e(row['label'])}</td><td>{_e(row['text'])}</td><td>{_e(row.get('source') or '')}</td><td>{_e(row.get('valid_time') or '')}</td></tr>")
            parts.append("</table>")
        for cf in c["conflicts"]:
            parts.append(f"<p><b>{_e(cf['headline'])}</b> " + "; ".join(_e(x) for x in cf["lines"] + [cf["selected_line"], cf["difference_line"]]) + "</p>")
    parts.append("<h2>Why this result</h2><ol>")
    for st in r.get("why", []):
        parts.append(f"<li><b>{_e(st['stage'])}</b><pre>{_e(json.dumps(st['content'], ensure_ascii=False, indent=1)[:2500])}</pre></li>")
    parts.append("</ol>")
    css = "body{font:14px system-ui;max-width:52rem;margin:2rem auto}table{border-collapse:collapse}td{border:1px solid #ccc;padding:.2rem .5rem}pre{white-space:pre-wrap;background:#f6f6f6;padding:.5rem}"
    return f"<!doctype html><meta charset=utf-8><title>ORCA {_e(r['query_id'])}</title><style>{css}</style>" + "".join(parts)
