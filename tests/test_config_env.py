"""Tests for ``.env`` loading.

This is the only file in the project that moves a credential, so the properties
worth pinning are the safety ones: a real environment variable wins, a value is
never returned or logged, and nothing here can make an import fail.
"""

from __future__ import annotations

import os

import pytest

from satquery.config import ENV_FILE, load_env_file


@pytest.fixture
def env_file(tmp_path):
    def _write(text: str):
        path = tmp_path / ENV_FILE
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("MISTRAL_API_KEY", "SATQUERY_ROUTER", "SATQUERY_VLM", "SQ_TEST_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_a_plain_assignment_is_loaded(env_file):
    load_env_file(env_file("SQ_TEST_KEY=hello"))
    assert os.environ["SQ_TEST_KEY"] == "hello"


def test_comments_blanks_and_export_prefixes(env_file):
    load_env_file(
        env_file("# a comment\n\n   \nexport SQ_TEST_KEY=value\n# SATQUERY_VLM=ignored\n")
    )
    assert os.environ["SQ_TEST_KEY"] == "value"
    assert "SATQUERY_VLM" not in os.environ


@pytest.mark.parametrize("raw,expected", [
    ('SQ_TEST_KEY="quoted"', "quoted"),
    ("SQ_TEST_KEY='single'", "single"),
    ("SQ_TEST_KEY=  spaced  ", "spaced"),
    ("SQ_TEST_KEY=", ""),
])
def test_value_forms(env_file, raw, expected):
    load_env_file(env_file(raw))
    assert os.environ["SQ_TEST_KEY"] == expected


def test_a_dollar_sign_survives_verbatim(env_file):
    """No interpolation: an API key is an opaque string, not a shell expression."""
    load_env_file(env_file("SQ_TEST_KEY=ab$HOME/c$1"))
    assert os.environ["SQ_TEST_KEY"] == "ab$HOME/c$1"


def test_a_real_environment_variable_wins(env_file, monkeypatch):
    """``MISTRAL_API_KEY=... satquery serve`` must beat a stale value in the file."""
    monkeypatch.setenv("SQ_TEST_KEY", "from-the-shell")

    loaded = load_env_file(env_file("SQ_TEST_KEY=from-the-file"))

    assert os.environ["SQ_TEST_KEY"] == "from-the-shell"
    assert "SQ_TEST_KEY" not in loaded


def test_override_is_available_but_off_by_default(env_file):
    os.environ["SQ_TEST_KEY"] = "first"
    load_env_file(env_file("SQ_TEST_KEY=second"), override=True)
    assert os.environ["SQ_TEST_KEY"] == "second"


def test_no_value_is_ever_returned(env_file):
    """The return says *which* names were set, never what they were set to."""
    loaded = load_env_file(env_file("SQ_TEST_KEY=super-secret-value"))

    assert list(loaded) == ["SQ_TEST_KEY"]
    assert "super-secret-value" not in repr(loaded)


def test_a_missing_file_is_a_no_op(tmp_path):
    assert load_env_file(tmp_path / "nothing-here") == {}


def test_a_malformed_line_is_skipped_not_raised(env_file):
    load_env_file(env_file("this line has no equals sign\nSQ_TEST_KEY=fine\n=novalue\n"))
    assert os.environ["SQ_TEST_KEY"] == "fine"


def test_an_unreadable_file_does_not_break_the_import(tmp_path, monkeypatch):
    """An import-time loader that can raise is an import-time crash."""
    path = tmp_path / ENV_FILE
    path.write_text("SQ_TEST_KEY=x", encoding="utf-8")

    def explode(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", explode)
    assert load_env_file(path) == {}


def test_the_key_never_reaches_the_router_trace(env_file, monkeypatch):
    """A credential in the environment must not travel in a plan or a trace."""
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-should-never-appear")
    from satquery.agent.router import MistralRouter
    from satquery.agent.tools import TOOLS, ToolMenu

    menu = ToolMenu(offered=(TOOLS["caption"],), withheld=(), context_summary="img_1: optical")
    router = MistralRouter()
    body = router.request_body("describe this", menu)

    import json

    assert "sk-should-never-appear" not in json.dumps(body)
