"""Unit tests for RAMLLM doctor diagnostics and remediation."""

from pathlib import Path

from typer.testing import CliRunner

from ramllm.cli.main import app
from ramllm.services.doctor_service import execute_fix, run_diagnostics

runner = CliRunner()


def test_run_diagnostics() -> None:
    items = run_diagnostics()
    assert len(items) >= 3
    categories = [i.category for i in items]
    assert "Storage" in categories
    assert "Backend" in categories
    assert "OS & Enforcement" in categories


def test_execute_fix_directories(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path / "ramllm_home"))
    ok, msg = execute_fix("ensure_directories")
    assert ok is True
    assert (tmp_path / "ramllm_home").exists()


def test_execute_fix_unknown() -> None:
    ok, msg = execute_fix("invalid_action_xyz")
    assert ok is False
    assert "Unknown fix action" in msg


def test_cli_doctor_command() -> None:
    res = runner.invoke(app, ["doctor", "--yes"])
    assert res.exit_code == 0
    assert "RAMLLM Diagnostic Doctor" in res.stdout


def test_cli_doctor_plain() -> None:
    res = runner.invoke(app, ["--plain", "doctor", "--yes"])
    assert res.exit_code == 0
    assert "Storage: Directories" in res.stdout
