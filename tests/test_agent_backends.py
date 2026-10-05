"""Tests for the deterministic backends -- the ones answering today.

Every expected number here is computable by hand from the synthetic scene, which
is the only way to tell "the change detector works" apart from "the change
detector returned a plausible-looking float". The unchanged-pair case is the most
important test in the file: Otsu always returns a threshold, so a detector
without a noise floor reports ~50% change on two identical images and nobody
notices until a judge asks.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.agent.backends import (
    FUSION_LABELS,
    CvaChangeMap,
    EvidenceVLM,
    MciChangeMap,
    RuleFusion,
    TerraMindFusion,
    format_evidence_block,
    label_components,
    noise_floor,
    otsu_threshold,
    select_change_backend,
    select_fusion_backend,
    select_vlm,
)
from satquery.contracts import ImageRef, SatQueryError


@pytest.fixture
def gen() -> np.random.Generator:
    return np.random.default_rng(11)


def base_scene(gen, size=128, level=120.0, spread=12.0) -> np.ndarray:
    return np.clip(gen.normal(level, spread, size=(3, size, size)), 0, 255).astype(np.uint8)


def illuminate(image: np.ndarray, gain: float = 1.08, offset: float = 4.0) -> np.ndarray:
    """A sun-angle difference: the global shift that fools a naive difference."""
    return np.clip(image.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Change vector analysis
# --------------------------------------------------------------------------- #


def test_a_known_change_is_measured_to_within_a_tenth_of_a_percent(gen):
    before = base_scene(gen)
    after = illuminate(before).astype(np.float32)
    after[:, 20:50, 70:110] = 235.0  # 30 x 40 px of a 128 x 128 scene = 7.32%

    result = CvaChangeMap().detect(before, after.astype(np.uint8))

    assert result.stats["changed_area_pct"] == pytest.approx(7.32, abs=0.15)
    assert result.stats["num_regions"] == 1
    assert result.stats["centroid_quadrant"] == "north-east"
    assert result.backend == "cva-otsu" and result.kind == "deterministic"


def test_an_unchanged_pair_reports_no_change_despite_an_illumination_shift(gen):
    """Otsu alone splits a unimodal noise histogram and reports ~50% change."""
    before = base_scene(gen)

    result = CvaChangeMap().detect(before, illuminate(before))

    assert result.stats["changed_area_pct"] == 0.0
    assert result.stats["num_regions"] == 0
    assert result.stats["threshold_noise_floor"] > 0.0


def test_the_noise_floor_is_what_rescues_the_unchanged_case(gen):
    """Pin the mechanism, not just the outcome, so a refactor cannot lose it."""
    before = base_scene(gen)
    after = illuminate(before)
    magnitude = np.abs(after.astype(np.float32) - before.astype(np.float32)).mean(axis=0)

    assert noise_floor(magnitude) > otsu_threshold(magnitude)


def test_a_small_change_is_still_found(gen):
    before = base_scene(gen)
    after = illuminate(before).astype(np.float32)
    after[:, 60:70, 60:70] += 40.0  # 100 px = 0.61%

    result = CvaChangeMap().detect(before, np.clip(after, 0, 255).astype(np.uint8))

    assert result.stats["changed_area_pct"] == pytest.approx(0.61, abs=0.2)
    assert result.stats["centroid_quadrant"] == "central"


def test_speckle_below_the_minimum_region_is_dropped(gen):
    before = base_scene(gen)
    after = illuminate(before).astype(np.float32)
    after[:, 5, 5] = 255.0  # a single hot pixel

    result = CvaChangeMap(min_region_px=16).detect(before, np.clip(after, 0, 255).astype(np.uint8))

    assert result.stats["num_regions"] == 0


def test_brightness_matching_is_robust_to_the_change_itself(gen):
    """A large bright change must not drag the correction and smear false change."""
    before = base_scene(gen)
    after = before.astype(np.float32)
    after[:, :, :40] = 250.0  # a third of the scene turns bright

    result = CvaChangeMap().detect(before, after.astype(np.uint8))

    # The changed third is found; the untouched two-thirds are not.
    assert 28.0 <= result.stats["changed_area_pct"] <= 34.0


def test_mismatched_grids_are_refused_rather_than_broadcast(gen):
    with pytest.raises(SatQueryError, match="one grid"):
        CvaChangeMap().detect(base_scene(gen, 64), base_scene(gen, 96))


def test_two_regions_are_counted_separately_and_ranked(gen):
    before = base_scene(gen)
    after = illuminate(before).astype(np.float32)
    after[:, 10:30, 10:40] = 240.0  # 600 px
    after[:, 90:100, 90:100] = 240.0  # 100 px

    result = CvaChangeMap().detect(before, np.clip(after, 0, 255).astype(np.uint8))

    assert result.stats["num_regions"] == 2
    assert result.regions[0]["area_px"] > result.regions[1]["area_px"]


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #


def test_otsu_finds_the_valley_of_a_bimodal_histogram():
    values = np.concatenate([np.full(500, 2.0), np.full(500, 20.0)])
    assert 2.0 < otsu_threshold(values) < 20.0


def test_otsu_on_a_constant_array_says_nothing_is_above_it():
    assert otsu_threshold(np.full(100, 5.0)) > 5.0


def test_connected_components_finds_disjoint_blobs():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:14, 10:16] = True

    labels, count = label_components(mask)

    assert count == 2
    assert set(np.unique(labels)) == {0, 1, 2}
    assert sorted(np.bincount(labels.reshape(-1))[1:]) == [9, 24]


def test_connected_components_joins_a_diagonal_snake_four_connected():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1, 1:5] = True
    mask[1:5, 4] = True

    _labels, count = label_components(mask)

    assert count == 1


def test_the_numpy_fallback_agrees_with_scipy(monkeypatch):
    """Both paths must partition identically; only label numbering may differ."""
    pytest.importorskip("scipy")
    mask = np.zeros((30, 30), dtype=bool)
    mask[2:8, 2:8] = True
    mask[20:25, 20:28] = True
    mask[15, 15] = True

    with_scipy, n_scipy = label_components(mask)

    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("scipy disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    without_scipy, n_numpy = label_components(mask)

    assert n_scipy == n_numpy == 3
    assert sorted(np.bincount(with_scipy.reshape(-1))[1:]) == sorted(
        np.bincount(without_scipy.reshape(-1))[1:]
    )


# --------------------------------------------------------------------------- #
# Rule-based fusion
# --------------------------------------------------------------------------- #


def masks_for(size=100):
    """Quadrant masks standing in for the index/backscatter thresholds."""
    water_optical = np.zeros((size, size), bool)
    water_sar = np.zeros((size, size), bool)
    vegetation = np.zeros((size, size), bool)
    rough = np.zeros((size, size), bool)
    water_optical[:30, :30] = True   # 9% of the scene
    water_sar[:30, :30] = True
    water_sar[80:, 80:] = True       # radar shadow: dark but not wet
    vegetation[:, 60:] = True        # 40%
    rough[60:80, :20] = True         # 4% built-up
    return {
        "water_optical": water_optical,
        "water_sar": water_sar,
        "vegetation": vegetation,
        "rough": rough,
    }


def test_water_needs_both_sensors_to_agree():
    result = RuleFusion(min_area_pct=1.0).classify(
        optical_evidence={}, sar_evidence={}, masks=masks_for()
    )

    # 9% where both agree; the 400 px of radar shadow is rejected.
    assert result.area_pct["water_bodies"] == pytest.approx(9.0, abs=0.1)
    assert "both sensors" in result.rationale["water_bodies"]


def test_urban_fabric_comes_from_roughness_minus_vegetation():
    result = RuleFusion(min_area_pct=1.0).classify(
        optical_evidence={}, sar_evidence={}, masks=masks_for()
    )
    assert result.area_pct["urban_fabric"] == pytest.approx(4.0, abs=0.1)
    assert "double-bounce" in result.rationale["urban_fabric"]


def test_classes_below_the_area_floor_are_scored_but_not_asserted():
    tiny = masks_for()
    tiny["rough"] = np.zeros((100, 100), bool)
    tiny["rough"][40:45, 40:45] = True  # 25 px = 0.25%, clear of water and vegetation

    result = RuleFusion(min_area_pct=2.0).classify(
        optical_evidence={}, sar_evidence={}, masks=tiny
    )

    assert "urban_fabric" not in result.labels
    assert result.area_pct["urban_fabric"] == pytest.approx(0.25, abs=0.01)


def test_one_sensor_only_still_produces_a_labelled_result_with_a_caveat():
    only_optical = {"water_optical": masks_for()["water_optical"], "vegetation": masks_for()["vegetation"]}

    result = RuleFusion(min_area_pct=1.0).classify(
        optical_evidence={}, sar_evidence={}, masks=only_optical
    )

    assert "water_bodies" in result.labels
    assert "one sensor only" in result.rationale["water_bodies"]


def test_every_label_is_reported_even_at_zero():
    result = RuleFusion().classify(optical_evidence={}, sar_evidence={}, masks={})
    assert set(result.label_scores) == set(FUSION_LABELS)
    assert set(result.area_pct) == set(FUSION_LABELS)


# --------------------------------------------------------------------------- #
# The evidence-only text backend
# --------------------------------------------------------------------------- #


def image_ref(modality="optical") -> ImageRef:
    return ImageRef(image_id="i1", path="i1.png", modality=modality)


def test_it_answers_a_measurable_question_from_the_evidence():
    answer = EvidenceVLM().answer(
        "vqa", [image_ref()], question="How much of the scene is vegetation?",
        evidence={"vegetation_pct": 41.2},
    )
    assert "41.2" in answer.text and not answer.refused


def test_punctuation_does_not_hide_the_concept():
    """"is this vegetation?" has to match the same way "vegetation" does."""
    answer = EvidenceVLM().answer(
        "vqa", [image_ref()], question="Is there water?", evidence={"water_pct": 12.0}
    )
    assert answer.text.startswith("yes") and "12.0" in answer.text


def test_a_substring_is_not_a_concept_match():
    answer = EvidenceVLM().answer(
        "vqa", [image_ref()], question="What is the watershed boundary?", evidence={"water_pct": 12.0}
    )
    assert answer.refused


def test_counting_is_refused_with_the_reason():
    answer = EvidenceVLM().answer(
        "vqa", [image_ref()], question="How many buildings are there?", evidence={"built_up_pct": 5.0}
    )
    assert answer.refused
    assert "counting objects" in answer.meta["reason"]


def test_an_open_change_question_gets_the_description_not_one_statistic():
    answer = EvidenceVLM().answer(
        "change_vqa",
        [image_ref(), image_ref()],
        question="What changed between these dates?",
        evidence={"changed_area_pct": 14.2, "num_regions": 7, "centroid_quadrant": "north-east"},
    )
    assert "14.2" in answer.text and "north-east" in answer.text and "7" in answer.text


def test_a_yes_no_change_question_gets_yes_or_no():
    answer = EvidenceVLM().answer(
        "change_vqa", [image_ref(), image_ref()], question="Is there any change?",
        evidence={"changed_area_pct": 0.2},
    )
    assert answer.text.startswith("no")


def test_grounding_refuses_but_names_what_it_could_locate():
    answer = EvidenceVLM().answer(
        "ground", [image_ref()], question="the largest warehouse",
        evidence={"concept_boxes": {"water": [object()], "vegetation": []}},
    )
    assert answer.refused
    assert "water" in answer.meta["reason"]


def test_a_caption_states_only_what_was_measured():
    answer = EvidenceVLM().answer(
        "caption", [image_ref()],
        evidence={"width": 512, "height": 512, "vegetation_pct": 62.0, "water_pct": 3.0},
    )
    assert "62.0" in answer.text and "512x512" in answer.text
    assert "no vision-language model" in answer.text


def test_fusion_labels_are_narrated_largest_first():
    answer = EvidenceVLM().answer(
        "vqa", [image_ref()], question="What land cover is present?",
        evidence={"water_bodies_pct": 9.0, "urban_fabric_pct": 21.0, "backscatter_mean": -11.2},
    )
    assert answer.text.index("urban fabric") < answer.text.index("water bodies")
    assert "-11.2 dB" in answer.text


# --------------------------------------------------------------------------- #
# Selection and prompting
# --------------------------------------------------------------------------- #


def test_the_evidence_block_is_flat_labelled_lines():
    block = format_evidence_block({"water_pct": 12.25, "_masks": {}, "note": "x", "n": 3})

    assert "- water_pct: 12.2" in block
    assert "_masks" not in block  # working data never reaches a prompt
    assert block.startswith("Measured evidence")
    assert "do not make it" in block  # the instruction that makes citation possible


def test_unavailable_model_slots_explain_themselves_rather_than_importerror():
    ok, why = MciChangeMap.available()
    assert not ok and "Phase 3" in why
    ok, why = TerraMindFusion.available()
    assert not ok and "head" in why


def test_selection_falls_back_and_records_the_reason(monkeypatch):
    monkeypatch.delenv("SATQUERY_CHANGE_BACKEND", raising=False)
    monkeypatch.delenv("SATQUERY_FUSION_BACKEND", raising=False)
    monkeypatch.setenv("SATQUERY_VLM", "evidence")

    vlm, vlm_why = select_vlm()
    change, change_why = select_change_backend()
    fusion, fusion_why = select_fusion_backend()

    assert vlm.name == "evidence-rules" and "requested" in vlm_why
    assert change.name == "cva-otsu" and change_why.startswith("auto:")
    assert fusion.name == "fusion-rules-v1" and fusion_why.startswith("auto:")


def test_asking_for_an_unbuilt_backend_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("SATQUERY_CHANGE_BACKEND", "mci")
    with pytest.raises(SatQueryError, match="Phase 3"):
        select_change_backend()


def test_an_unbuildable_vlm_downgrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("SATQUERY_VLM", "no_such_predictor")

    backend, why = select_vlm()

    assert backend.name == "evidence-rules"
    assert "could not be built" in why
