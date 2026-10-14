"""Unit tests for the RAMLLM CLI commands using CliRunner."""

from pathlib import Path

from tests.fixtures.make_gguf import create_tiny_dense_gguf
from typer.testing import CliRunner

from ramllm.cli.main import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Run bigger models on smaller machines" in result.stdout


def test_cli_profile_json() -> None:
    result = runner.invoke(app, ["profile", "--json"])
    assert result.exit_code == 0
    assert '"cpu_name"' in result.stdout
    assert '"ram_total_bytes"' in result.stdout


def test_cli_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Backend" in result.stdout


def test_cli_plan_with_fixtures(tmp_path: Path) -> None:
    model_path = tmp_path / "test_dense.gguf"
    create_tiny_dense_gguf(model_path)
    hw_path = "tests/fixtures/hardware/laptop_8gb_no_gpu.json"

    result = runner.invoke(
        app,
        [
            "plan",
            str(model_path),
            "--hw-json",
            hw_path,
            "--explain",
        ],
    )
    assert result.exit_code == 0
    assert "Execution Plans for test_dense.gguf" in result.stdout
    assert "Plan #1 Reasoning & Breakdown" in result.stdout


def test_cli_models_list() -> None:
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
