"""Tests for the HTTP surface and the report renderers.

The API is the contract the GUI and any assessor both use, so the assertions are
about shape and status codes as much as content: a tool that cannot run must
answer 422 with a readable reason rather than 500 with a stack trace, and the
artefact endpoint must refuse a path that climbs out of the session.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from satquery.report import HAS_REPORTLAB, render_html, render_markdown, write_report

fastapi = pytest.importorskip("fastapi", reason="the API extra is not installed")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("SATQUERY_ROUTER", "rules")
    monkeypatch.setenv("SATQUERY_VLM", "evidence")


@pytest.fixture
def client(tmp_config):
    from satquery.api.app import create_app

    return TestClient(create_app(tmp_config))


@pytest.fixture
def png_bytes(rgb_scene):
    from PIL import Image

    def _make(**kwargs) -> bytes:
        array = np.transpose(rgb_scene(**kwargs), (1, 2, 0))
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue()

    return _make


def register(client, png_bytes, name="scene.png", **kwargs) -> str:
    response = client.post("/api/v1/images", files={"file": (name, png_bytes(**kwargs), "image/png")})
    assert response.status_code == 200, response.text
    return response.json()["image_id"]


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


def test_health_reports_which_backend_serves_each_slot(client):
    body = client.get("/api/v1/health").json()

    assert body["status"] == "ok"
    assert set(body["backends"]) >= {"text", "change_map", "fusion"}
    assert body["router"]["primary"] == "rules"
    assert "rasterio" in body["optional"]


def test_the_web_app_is_served_from_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "SatQuery AI" in response.text
    # No external fetches: the demo has to work on a laptop with no network, so
    # nothing may be loaded over http from another host.
    assert 'src="http' not in response.text and "src='http" not in response.text
    assert 'href="http' not in response.text and "href='http" not in response.text


def test_registering_an_image_returns_its_whole_validation_report(client, png_bytes):
    body = client.post(
        "/api/v1/images", files={"file": ("scene.png", png_bytes(), "image/png")}
    ).json()

    validation = body["validation"]
    assert validation["ok"] and validation["modality"] == "optical"
    assert validation["modality_reasons"]
    assert validation["summary"]["bands"] == 3


def test_an_unreadable_upload_is_registered_with_its_error_not_rejected(client):
    """The GUI has to be able to show the user why their file failed."""
    body = client.post(
        "/api/v1/images", files={"file": ("broken.png", b"not an image", "image/png")}
    ).json()

    assert body["validation"]["ok"] is False
    assert body["validation"]["errors"]


def test_the_preview_is_a_png(client, png_bytes):
    image_id = register(client, png_bytes)
    response = client.get(f"/api/v1/images/{image_id}/preview")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_the_tool_menu_reflects_what_the_session_holds(client, png_bytes):
    register(client, png_bytes)

    menu = client.get("/api/v1/tools").json()["menu"]

    assert set(menu["offered"]) == {"vqa", "caption", "ground", "indices"}
    assert {w["tool"] for w in menu["withheld"]} == {"change_map", "change_describe", "fusion"}
    assert all(w["reason"] for w in menu["withheld"])


def test_a_full_query_returns_answer_trace_and_artifacts(client, png_bytes):
    a = register(client, png_bytes, name="before.png")
    b = register(client, png_bytes, name="after.png", shift=16)
    pair = client.post("/api/v1/pairs", json={"image_a": a, "image_b": b}).json()
    assert pair["check"]["pair_type"] == "bitemporal"

    result = client.post(
        "/api/v1/query", json={"query": "What changed between these dates?", "pair_id": pair["pair_id"]}
    ).json()

    assert result["answer"] and result["confidence"] > 0
    assert result["trace"]["n_steps"] >= 5
    assert result["artifacts"]
    for url in result["artifacts"].values():
        served = client.get(url)
        assert served.status_code == 200
        assert served.headers["content-type"] == "image/png"


def test_an_empty_query_is_a_422(client, png_bytes):
    register(client, png_bytes)
    assert client.post("/api/v1/query", json={"query": "   "}).status_code == 422


def test_an_unknown_image_is_a_404(client):
    assert client.get("/api/v1/images/img_nope").status_code == 404


def test_the_stored_query_can_be_fetched_again(client, png_bytes):
    register(client, png_bytes)
    query_id = client.post("/api/v1/query", json={"query": "Describe this image."}).json()["query_id"]

    assert client.get(f"/api/v1/queries/{query_id}").json()["query_id"] == query_id
    assert client.get(f"/api/v1/queries/{query_id}/trace").json()["n_steps"] > 0
    assert query_id in client.get("/api/v1/queries").json()["queries"]


@pytest.mark.parametrize("fmt,mime", [("json", "application/json"), ("md", "text/markdown"),
                                      ("html", "text/html")])
def test_reports_render_in_every_text_format(client, png_bytes, fmt, mime):
    register(client, png_bytes)
    query_id = client.post("/api/v1/query", json={"query": "Describe this image."}).json()["query_id"]

    response = client.get(f"/api/v1/queries/{query_id}/report", params={"format": fmt})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(mime)
    assert len(response.content) > 500


@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab is not installed")
def test_the_pdf_report_is_a_pdf(client, png_bytes):
    register(client, png_bytes)
    query_id = client.post("/api/v1/query", json={"query": "Describe this image."}).json()["query_id"]

    response = client.get(f"/api/v1/queries/{query_id}/report", params={"format": "pdf"})

    assert response.status_code == 200
    assert response.content[:4] == b"%PDF"


# --------------------------------------------------------------------------- #
# The tool endpoints
# --------------------------------------------------------------------------- #


def test_a_tool_endpoint_can_be_called_directly(client, png_bytes):
    """An assessor must be able to reproduce a cited number without the loop."""
    image_id = register(client, png_bytes)

    body = client.post("/tools/v1/indices", json={"image_id": image_id, "index": "VARI"}).json()

    assert body["ok"] and body["output"]["index"] == "VARI"
    assert 0 <= body["output"]["area_pct"] <= 100


def test_a_tool_that_cannot_run_answers_422_with_a_reason(client, png_bytes):
    image_id = register(client, png_bytes)

    response = client.post("/tools/v1/indices", json={"image_id": image_id, "index": "NDVI"})

    assert response.status_code == 422
    assert "NDVI needs bands" in response.json()["error"]


def test_every_registered_tool_has_a_route(client):
    from satquery.agent.tools import TOOLS

    paths = {route.path for route in client.app.routes}
    assert {spec.endpoint for spec in TOOLS.values()} <= paths


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def test_the_artifact_route_refuses_to_climb_out_of_the_session(client):
    for attempt in ("/files/default/../../secrets/x.png", "/files/default/q1/..%2F..%2Fx.png"):
        assert client.get(attempt).status_code in (404, 400)


def test_an_upload_filename_cannot_escape_the_images_directory(client, png_bytes, tmp_config):
    body = client.post(
        "/api/v1/images",
        files={"file": ("../../evil.png", png_bytes(), "image/png")},
    ).json()

    from pathlib import Path

    stored = Path(body["path"]).resolve()
    assert str(stored).startswith(str((tmp_config.runs_dir / "sessions").resolve()))
    assert ".." not in stored.name


def test_sessions_are_isolated_from_each_other(client, png_bytes):
    client.post(
        "/api/v1/images",
        files={"file": ("a.png", png_bytes(), "image/png")},
        headers={"X-Session-Id": "alpha"},
    )

    alpha = client.get("/api/v1/images", headers={"X-Session-Id": "alpha"}).json()
    beta = client.get("/api/v1/images", headers={"X-Session-Id": "beta"}).json()

    assert len(alpha["images"]) == 1 and beta["images"] == []


def test_a_session_can_be_deleted(client, png_bytes):
    register(client, png_bytes)
    assert client.delete("/api/v1/sessions/default").json()["deleted"] is True
    assert client.get("/api/v1/images").json()["images"] == []


# --------------------------------------------------------------------------- #
# Report rendering, without the API
# --------------------------------------------------------------------------- #


@pytest.fixture
def answered(session, scene_file):
    from satquery.agent.controller import Controller

    before = session.add_image(scene_file("before.png", size=96))
    after = session.add_image(scene_file("after.png", size=96, shift=12))
    result = Controller(session).answer(
        "What changed between these dates?", image_ids=[before.image_id, after.image_id]
    )
    return session, result


def test_markdown_carries_answer_evidence_menu_and_trace(answered):
    _session, result = answered

    text = render_markdown(result.to_dict())

    assert result.answer[:40] in text
    assert "## Evidence behind the answer" in text
    assert "## Tools offered and withheld" in text
    assert "## Execution trace" in text
    assert "fusion" in text  # the withheld tool is named
    assert "| claim |" in text


def test_markdown_reports_the_pair_verdict_not_a_none(answered):
    _session, result = answered
    text = render_markdown(result.to_dict())

    assert "bitemporal" in text
    assert "None, not usable" not in text


def test_html_embeds_the_artefacts_so_the_report_travels(answered):
    session, result = answered

    html = render_html(result.to_dict(), file_root=session.root)

    assert html.count("data:image") == len(result.artifacts)
    assert "<title>" in html and "prefers-color-scheme" in html


def test_write_report_picks_the_format_from_the_extension(answered, tmp_path):
    session, result = answered

    md = write_report(result.to_dict(), tmp_path / "r.md", fmt="md")
    html = write_report(result.to_dict(), tmp_path / "r.html", fmt="html", file_root=session.root)

    assert md.read_text(encoding="utf-8").startswith("# SatQuery AI")
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")


@pytest.mark.skipif(not HAS_REPORTLAB, reason="reportlab is not installed")
def test_the_pdf_embeds_the_overlays(answered, tmp_path):
    session, result = answered

    pdf = write_report(result.to_dict(), tmp_path / "r.pdf", fmt="pdf", file_root=session.root)

    assert pdf.suffix == ".pdf"
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert pdf.stat().st_size > 5_000


def test_an_unknown_format_is_rejected(answered, tmp_path):
    _session, result = answered
    with pytest.raises(ValueError, match="unknown report format"):
        write_report(result.to_dict(), tmp_path / "r.rtf", fmt="rtf")
