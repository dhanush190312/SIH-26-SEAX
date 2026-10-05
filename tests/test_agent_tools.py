"""Tests for the tool menu and the tool implementations.

The centre of this file is :func:`test_the_menu_matrix`, which is the table from
``docs/ARCHITECTURE.md`` executed rather than asserted in prose. If that table
and this test ever disagree, one of them is a lie, and a reader has no way to
tell which -- so the doc's three input configurations are enumerated here with
their expected offered/withheld sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.agent.tools import (
    THRESHOLDS,
    TOOLS,
    ToolContext,
    build_menu,
    run_tool,
    scene_evidence,
)
from satquery.contracts import SatQueryError


@pytest.fixture
def optical_png(session, scene_file):
    return session.add_image(scene_file("optical.png", size=128))


@pytest.fixture
def later_png(session, scene_file):
    return session.add_image(scene_file("later.png", size=128, shift=16))


@pytest.fixture
def sar_tif(session, tmp_path, write_raster):
    """A two-band linear-power scene with dark water and a bright block."""
    g = np.random.default_rng(5)
    array = g.gamma(4.0, 0.03, size=(2, 128, 128)).astype(np.float32)
    array[:, :32, :32] *= 0.02  # water
    array[:, 80:110, 10:40] *= 15.0  # buildings
    path = write_raster(tmp_path / "s1_vv.tif", array)
    return session.add_image(path, modality="sar")


def context(session, images, pair=None, config=None) -> ToolContext:
    return ToolContext(
        session, query_id="q_test", images=list(images), pair=pair, config=config or session.config
    )


# --------------------------------------------------------------------------- #
# The menu: invariant 02
# --------------------------------------------------------------------------- #


def test_the_menu_matrix(session, optical_png, later_png, sar_tif):
    """docs/ARCHITECTURE.md's table, executed.

    A: one optical image · B: two optical images · C: optical + SAR.
    """
    single = build_menu([optical_png], None)
    bitemporal_pair = session.add_pair(optical_png.image_id, later_png.image_id)
    bitemporal = build_menu([optical_png, later_png], bitemporal_pair)
    crossmodal_pair = session.add_pair(optical_png.image_id, sar_tif.image_id)
    crossmodal = build_menu([optical_png, sar_tif], crossmodal_pair)

    expected = {
        #            A single   B bitemporal  C crossmodal
        "vqa":             (True, True, True),
        "caption":         (True, True, True),
        "ground":          (True, True, True),
        "indices":         (True, True, True),
        "change_map":      (False, True, False),
        "change_describe": (False, True, False),
        "fusion":          (False, False, True),
    }
    assert set(expected) == set(TOOLS), "the matrix must cover every registered tool"
    for tool, (a, b, c) in expected.items():
        assert single.has(tool) is a, f"{tool} in configuration A"
        assert bitemporal.has(tool) is b, f"{tool} in configuration B"
        assert crossmodal.has(tool) is c, f"{tool} in configuration C"


def test_every_withheld_tool_carries_a_reason_a_human_can_read(session, optical_png):
    menu = build_menu([optical_png], None)

    withheld = dict(menu.withheld)
    assert set(withheld) == {"change_map", "change_describe", "fusion"}
    for tool, reason in withheld.items():
        assert len(reason) > 20 and reason == reason.lower() or reason[0].islower(), reason
    assert "no image pair" in withheld["change_map"]


def test_the_reason_names_the_actual_mismatch(session, optical_png, later_png, sar_tif):
    bitemporal = build_menu(
        [optical_png, later_png], session.add_pair(optical_png.image_id, later_png.image_id)
    )
    crossmodal = build_menu(
        [optical_png, sar_tif], session.add_pair(optical_png.image_id, sar_tif.image_id)
    )

    assert "optical + optical" in dict(bitemporal.withheld)["fusion"]
    assert "crossmodal" in dict(crossmodal.withheld)["change_map"]


def test_an_unusable_pair_withholds_both_families(session, optical_png, scene_file):
    """A pair that fails the geometric check is not a pair."""
    other = session.add_image(scene_file("small.png", size=64))
    pair = session.add_pair(optical_png.image_id, other.image_id)
    assert not pair.ok

    menu = build_menu([optical_png, other], pair)

    assert not menu.has("change_map") and not menu.has("fusion")
    assert "not usable" in dict(menu.withheld)["change_map"]


def test_the_router_only_ever_sees_a_text_summary(session, optical_png, sar_tif):
    """Invariant 01 at the menu layer: no paths, no arrays, in what is sent."""
    menu = build_menu([optical_png, sar_tif], None)

    assert optical_png.image_id in menu.context_summary
    assert optical_png.path not in menu.context_summary
    assert "optical" in menu.context_summary and "sar" in menu.context_summary


def test_tool_schemas_never_expose_image_id(session, optical_png):
    """The executor injects ids; a schema that asks for one invites a guess."""
    for schema in build_menu([optical_png], None).schemas():
        properties = schema["function"]["parameters"]["properties"]
        assert "image_id" not in properties and "pair_id" not in properties


def test_asking_the_menu_for_a_withheld_tool_explains_itself(session, optical_png):
    menu = build_menu([optical_png], None)
    with pytest.raises(SatQueryError, match="withheld"):
        menu.spec("fusion")
    with pytest.raises(SatQueryError, match="unknown tool"):
        menu.spec("teleport")


# --------------------------------------------------------------------------- #
# Evidence: invariant 03
# --------------------------------------------------------------------------- #


def test_scene_evidence_reports_what_the_bands_allow(optical_png):
    evidence = scene_evidence(optical_png, max_side=256)

    assert evidence["index_source"] == "rgb"
    assert evidence["indices_available"] == ["VARI"]
    assert "vari_mean" in evidence and "ndvi_mean" not in evidence
    assert "RGB proxy" in evidence["vegetation_index"]
    assert evidence["vegetation_threshold"] == THRESHOLDS["vari_vegetation"]


def test_scene_evidence_on_sar_is_backscatter_not_indices(sar_tif):
    evidence = scene_evidence(sar_tif, max_side=256)

    assert evidence["modality"] == "sar"
    assert evidence["backscatter_p5"] < evidence["backscatter_p50"] < evidence["backscatter_p95"]
    assert evidence["water_threshold_db"] == THRESHOLDS["sar_water_db"]
    # The 32x32 water block of a 128x128 scene is 6.25% of it.
    assert evidence["water_pct"] == pytest.approx(6.25, abs=1.0)


def test_working_data_is_kept_out_of_the_reported_evidence(session, optical_png):
    result = run_tool("indices", {"image_id": optical_png.image_id}, context(session, [optical_png]))

    assert result.ok
    assert not any(key.startswith("_") for key in result.evidence)
    assert "concept_boxes" not in result.evidence


# --------------------------------------------------------------------------- #
# Running tools
# --------------------------------------------------------------------------- #


def test_indices_writes_a_map_and_reports_its_threshold(session, optical_png):
    ctx = context(session, [optical_png])

    result = run_tool("indices", {"image_id": optical_png.image_id, "index": "VARI"}, ctx)

    assert result.ok and result.backend_kind == "deterministic"
    assert result.output["index"] == "VARI"
    assert 0.0 <= result.output["area_pct"] <= 100.0
    assert result.artifacts["map_url"].startswith("/files/")
    written = session.artifacts_dir / "q_test"
    assert any(written.iterdir())


def test_an_index_whose_bands_are_missing_is_refused_with_the_alternative(session, optical_png):
    result = run_tool(
        "indices", {"image_id": optical_png.image_id, "index": "NDVI"}, context(session, [optical_png])
    )

    assert not result.ok
    assert "NDVI needs bands nir, red" in result.error
    assert "available: VARI" in result.error


def test_an_unknown_index_lists_the_known_ones(session, optical_png):
    result = run_tool(
        "indices", {"image_id": optical_png.image_id, "index": "NDXX"}, context(session, [optical_png])
    )
    assert not result.ok and "known:" in result.error


def test_change_map_measures_the_shift(session, optical_png, later_png):
    pair = session.add_pair(optical_png.image_id, later_png.image_id)
    ctx = context(session, [optical_png, later_png], pair)

    result = run_tool("change_map", {"pair_id": pair.pair_id}, ctx)

    assert result.ok
    assert result.output["changed_area_pct"] > 1.0
    assert set(result.artifacts) == {"mask_url", "overlay_url"}
    assert result.backend == "cva-otsu"


def test_change_describe_runs_the_map_itself_if_the_plan_forgot(session, optical_png, later_png):
    """Invariant 03 is not optional because a router produced a sloppy plan."""
    pair = session.add_pair(optical_png.image_id, later_png.image_id)
    ctx = context(session, [optical_png, later_png], pair)

    result = run_tool("change_describe", {}, ctx)

    assert result.ok
    assert "changed_area_pct" in result.evidence
    assert ctx._change is not None


@pytest.mark.parametrize("question,expected", [
    ("What changed between these dates?", "change_caption"),
    ("Describe the differences.", "change_caption"),
    ("Tell me what is different", "change_caption"),
    ("Is there any change?", "change_vqa"),
    ("How much of the scene changed?", "change_vqa"),
    ("How many new buildings appeared?", "change_vqa"),
    ("", "change_caption"),
])
def test_an_open_change_question_gets_the_descriptive_prompt(
    session, optical_png, later_png, question, expected
):
    """A paragraph question must not go through the short-answer template.

    Observed with a real VLM behind the slot: "what changed between these two
    dates" sent through the change-VQA template -- which demands "the shortest
    possible answer: a number, a single word, or yes/no" -- came back as
    "Vegetation loss". Accurate, thin, and citing nothing, where the descriptive
    prompt produced a sentence that cited two measurements.
    """
    pair = session.add_pair(optical_png.image_id, later_png.image_id)
    ctx = context(session, [optical_png, later_png], pair)

    result = run_tool("change_describe", {"question": question} if question else {}, ctx)

    assert result.ok
    assert result.output["task"] == expected


def test_a_pair_tool_without_a_pair_fails_cleanly(session, optical_png):
    result = run_tool("change_map", {}, context(session, [optical_png]))

    assert not result.ok
    assert "needs an image pair" in result.error
    assert result.duration_ms >= 0.0


def test_an_unknown_tool_is_a_result_not_an_exception(session, optical_png):
    result = run_tool("teleport", {}, context(session, [optical_png]))
    assert not result.ok and "unknown tool" in result.error


def test_evidence_accumulates_across_a_plan(session, optical_png, later_png):
    """What lets a text tool cite the change statistics a previous tool measured."""
    pair = session.add_pair(optical_png.image_id, later_png.image_id)
    ctx = context(session, [optical_png, later_png], pair)

    run_tool("change_map", {"pair_id": pair.pair_id}, ctx)

    assert "changed_area_pct" in ctx.accumulated
    assert "changed_area_pct" in ctx.evidence_for_text(optical_png)


def test_vqa_and_caption_route_through_the_text_backend(session, optical_png):
    ctx = context(session, [optical_png])

    caption = run_tool("caption", {"image_id": optical_png.image_id}, ctx)
    vqa = run_tool(
        "vqa", {"image_id": optical_png.image_id, "question": "How much vegetation is there?"}, ctx
    )

    assert caption.ok and vqa.ok
    assert caption.backend_kind in ("deterministic", "model")
    assert "vegetation" in vqa.output["answer"].lower()


def test_vqa_without_a_question_is_refused(session, optical_png):
    result = run_tool("vqa", {"image_id": optical_png.image_id}, context(session, [optical_png]))
    assert not result.ok and "needs a question" in result.error


def test_fusion_needs_one_of_each_modality(session, optical_png, later_png):
    pair = session.add_pair(optical_png.image_id, later_png.image_id)
    result = run_tool("fusion", {"pair_id": pair.pair_id}, context(session, [optical_png, later_png], pair))

    assert not result.ok
    assert "one optical and one SAR" in result.error


def test_fusion_labels_a_crossmodal_pair(session, optical_png, sar_tif):
    pair = session.add_pair(optical_png.image_id, sar_tif.image_id)
    ctx = context(session, [optical_png, sar_tif], pair)

    result = run_tool("fusion", {"pair_id": pair.pair_id}, ctx)

    assert result.ok
    assert result.output["labels"], "a scene with water, vegetation and buildings must label"
    assert all(f"{label}_pct" in result.evidence for label in result.output["labels"])
    assert all(label in result.output["rationale"] for label in result.output["labels"])


def test_every_tool_result_serialises(session, optical_png):
    import json

    result = run_tool("caption", {"image_id": optical_png.image_id}, context(session, [optical_png]))
    json.dumps(result.to_dict())
