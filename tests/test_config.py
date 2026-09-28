import os
import subprocess

import pytest

from src import config


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in ("TABPFN_TOKEN", "TABPFN_LIVE"):
        # setenv before delenv so teardown restores the original state even when
        # a test sets the variable through load_dotenv rather than monkeypatch.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    # Keep a developer's real .env out of these tests.
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")


def test_env_file_is_loaded(tmp_path, clean_env):
    env_file = tmp_path / ".env"
    env_file.write_text("TABPFN_TOKEN=abc123\n")
    config.load_env(env_file)
    assert os.environ["TABPFN_TOKEN"] == "abc123"


def test_shell_variable_wins_over_env_file(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("TABPFN_TOKEN=from-file\n")
    config.load_env(env_file)
    assert os.environ["TABPFN_TOKEN"] == "from-shell"


def test_missing_token_raises_with_instructions(clean_env):
    with pytest.raises(RuntimeError, match="TABPFN_TOKEN"):
        config.require_tabpfn_token()


def test_present_token_passes(clean_env, monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "abc123")
    config.require_tabpfn_token()


def test_live_is_opt_in(clean_env, monkeypatch):
    assert not config.live_requested()
    assert config.live_requested(cli_flag=True)
    monkeypatch.setenv("TABPFN_LIVE", "1")
    assert config.live_requested()


def test_dotenv_is_git_ignored():
    result = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=config.ROOT, check=False)
    assert result.returncode == 0, ".env must be listed in .gitignore"


def test_env_example_has_no_secret():
    text = (config.ROOT / ".env.example").read_text()
    token_lines = [ln for ln in text.splitlines() if ln.startswith("TABPFN_TOKEN=")]
    assert token_lines == ["TABPFN_TOKEN="]
