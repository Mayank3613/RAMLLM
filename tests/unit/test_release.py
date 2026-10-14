"""Unit tests for Milestone 8 release, update, clean, completion, and version tools."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ramllm.cli.app import app
from ramllm.services.upgrade_service import (
    check_for_updates,
    detect_install_method,
    get_upgrade_command,
)

runner = CliRunner()


def test_detect_install_method_brew(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.executable", "/opt/homebrew/Cellar/ramllm/0.1.0/bin/python")
    assert detect_install_method() == "brew"


def test_detect_install_method_pipx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.executable", "/home/user/.local/pipx/venvs/ramllm/bin/python")
    assert detect_install_method() == "pipx"


def test_detect_install_method_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.executable", "/home/user/.local/share/uv/tools/ramllm/bin/python")
    assert detect_install_method() == "uv"


def test_get_upgrade_command() -> None:
    assert get_upgrade_command("brew") == "brew upgrade ramllm"
    assert get_upgrade_command("pipx") == "pipx upgrade ramllm"
    assert get_upgrade_command("uv") == "uv tool upgrade ramllm"
    assert "pip install --upgrade" in get_upgrade_command("pip")


@patch("httpx.get")
def test_check_for_updates_available(mock_get: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"info": {"version": "99.0.0"}}
    mock_get.return_value = mock_resp

    curr, latest, has_update = check_for_updates()
    assert latest == "99.0.0"
    assert has_update is True


@patch("httpx.get")
def test_check_for_updates_network_error(mock_get: MagicMock) -> None:
    mock_get.side_effect = Exception("Connection refused")
    curr, latest, has_update = check_for_updates()
    assert latest is None
    assert has_update is False


def test_ramllm_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "RAMLLM v" in result.stdout


def test_ramllm_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "RAMLLM v" in result.stdout
    assert "Backend:" in result.stdout


def test_ramllm_tour_command() -> None:
    result = runner.invoke(app, ["tour"])
    assert result.exit_code == 0
    assert "Welcome to RAMLLM" in result.stdout
    assert "VRAM" in result.stdout
    assert "COMFORTABLE" in result.stdout


@patch("ramllm.cli.commands.update_cmd.check_for_updates")
def test_ramllm_update_check_only(mock_check: MagicMock) -> None:
    mock_check.return_value = ("0.1.0", "0.2.0", True)
    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 0
    assert "A new version of RAMLLM is available" in result.stdout
    assert "0.2.0" in result.stdout


@patch("ramllm.cli.commands.update_cmd.check_for_updates")
def test_ramllm_update_already_latest(mock_check: MagicMock) -> None:
    mock_check.return_value = ("0.1.0", "0.1.0", False)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "RAMLLM is already up to date" in result.stdout


def test_ramllm_clean_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_models = tmp_path / "models"
    test_models.mkdir()
    fake_part = test_models / "model.gguf.part"
    fake_part.write_bytes(b"0" * 1024)

    monkeypatch.setattr("ramllm.cli.commands.clean_cmd.get_models_dir", lambda: test_models)

    result = runner.invoke(app, ["clean", "--dry-run"])
    assert result.exit_code == 0
    assert "[DRY-RUN]" in result.stdout
    assert fake_part.exists()


def test_ramllm_clean_partials_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_models = tmp_path / "models"
    test_models.mkdir()
    fake_part = test_models / "model.gguf.part"
    fake_part.write_bytes(b"0" * 2048)

    monkeypatch.setattr("ramllm.cli.commands.clean_cmd.get_models_dir", lambda: test_models)

    result = runner.invoke(app, ["clean", "--partials", "--yes"])
    assert result.exit_code == 0
    assert "Successfully cleaned" in result.stdout
    assert not fake_part.exists()


def test_ramllm_completion_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    # Test zsh completion installation
    result = runner.invoke(app, ["completion", "install", "zsh"])
    assert result.exit_code == 0
    zfunc_file = tmp_path / ".zfunc" / "_ramllm"
    assert zfunc_file.exists()
    assert "_ramllm" in zfunc_file.read_text()

    # Test bash completion installation
    result = runner.invoke(app, ["completion", "install", "bash"])
    assert result.exit_code == 0
    bash_file = tmp_path / ".local" / "share" / "bash-completion" / "completions" / "ramllm"
    assert bash_file.exists()
    assert "_RAMLLM_COMPLETE=bash_complete" in bash_file.read_text()
