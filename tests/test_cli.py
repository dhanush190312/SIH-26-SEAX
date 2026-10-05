"""Smoke tests for the command line.

``doctor``, ``datasets`` and ``tools`` are what a user runs first, on the machine
where the install went wrong, so the property under test is that they run at all
-- with a bare environment, no session, no benchmark downloaded and nothing
served. ``ask`` is checked end to end because the CLI is the offline demo path.
"""

from __future__ import annotations

import json

import pytest

from satquery.cli import main


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_config):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setenv("SATQUERY_ROUTER", "rules")
    monkeypatch.setenv("SATQUERY_VLM", "evidence")
    monkeypatch.setenv("SATQUERY_RUNS_DIR", str(tmp_config.runs_dir))
    monkeypatch.setenv("SATQUERY_DATA_ROOT", str(tmp_config.data_root))


def test_doctor_runs_and_names_every_slot(capsys):
    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "dependencies" in out and "serving" in out
    for slot in ("text", "change_map", "fusion", "router"):
        assert slot in out


def test_doctor_json_is_machine_readable(capsys):
    assert main(["--json", "doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["backends"]["text"]["backend"]
    assert "MISTRAL_API_KEY" in payload["env"]


def test_datasets_lists_the_registry_without_importing_an_adapter(capsys):
    assert main(["datasets"]) == 0

    out = capsys.readouterr().out
    assert "rsvqa_lr" in out and "bigearthnet_mm" in out
    assert "SATQUERY_DATA_" in out  # the escape hatch is advertised


def test_tools_prints_the_contract(capsys):
    assert main(["tools"]) == 0

    out = capsys.readouterr().out
    for name in ("vqa", "caption", "ground", "indices", "change_map", "change_describe", "fusion"):
        assert name in out
    assert "injects image_id" in out


def test_ask_answers_over_two_files_and_writes_a_report(capsys, tmp_path, scene_file):
    before = scene_file("before.png", size=96)
    after = scene_file("after.png", size=96, shift=12)
    report = tmp_path / "report.md"

    code = main(
        ["ask", "What changed between these dates?", str(before), str(after),
         "--session", "cli-test", "--report", str(report)]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "%" in captured.out and "confidence" in captured.out
    assert "evidence" in captured.out
    assert report.exists() and report.read_text(encoding="utf-8").startswith("# SatQuery AI")
    # The per-image validation goes to stderr so stdout stays pipeable.
    assert "optical" in captured.err


def test_ask_json_emits_the_whole_result(capsys, scene_file):
    code = main(["--json", "ask", "Describe this image.", str(scene_file(size=96)),
                 "--session", "cli-json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["answer"] and payload["trace"]["entries"]


def test_ask_returns_a_nonzero_code_when_it_refuses(capsys, scene_file):
    code = main(["ask", "How many boats are there?", str(scene_file(size=96)),
                 "--session", "cli-refuse"])

    assert code == 1
    assert "counting objects" in capsys.readouterr().out


def test_ask_rejects_more_than_two_images(capsys, scene_file):
    paths = [str(scene_file(f"s{i}.png", size=64)) for i in range(3)]
    assert main(["ask", "q", *paths]) == 2
    assert "one or two images" in capsys.readouterr().err


def test_session_list_and_clear(capsys, scene_file):
    main(["ask", "Describe this image.", str(scene_file(size=96)), "--session", "cli-session"])
    capsys.readouterr()

    assert main(["session", "list"]) == 0
    assert "cli-session" in capsys.readouterr().out

    assert main(["session", "clear", "cli-session"]) == 0
    assert "deleted" in capsys.readouterr().out


def test_eval_on_a_missing_dataset_is_a_skip_not_a_crash(capsys):
    """Benchmarks live on Colab; the laptop still has to behave."""
    assert main(["eval", "--dataset", "rsvqa_lr", "--predictor", "dummy", "--limit", "3"]) == 0

    out = capsys.readouterr().out
    assert "rsvqa_lr" in out
