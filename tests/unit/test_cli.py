"""Unit tests for the RAMLLM CLI commands using CliRunner."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
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


def test_cli_bench_help() -> None:
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "Run empirical benchmark" in result.stdout


def test_cli_serve_help() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Start OpenAI-compatible API server" in result.stdout


def test_cli_plan_no_budgets_3_profiles(tmp_path: Path) -> None:
    model_path = tmp_path / "test_dense.gguf"
    create_tiny_dense_gguf(model_path)

    profiles = [
        "tests/fixtures/hardware/laptop_8gb_no_gpu.json",
        "tests/fixtures/hardware/laptop_16gb_gpu_6gb.json",
        "tests/fixtures/hardware/apple_m_16gb_unified.json",
    ]

    for p in profiles:
        result = runner.invoke(
            app,
            [
                "plan",
                str(model_path),
                "--hw-json",
                p,
            ],
        )
        assert result.exit_code == 0
        assert "Plan Summary" in result.stdout
        assert "Speed:" in result.stdout
        assert "Memory:" in result.stdout
        assert "Where it runs:" in result.stdout
        assert "Using up to" in result.stdout


def test_cli_run_dry_run_no_budgets(tmp_path: Path) -> None:
    model_path = tmp_path / "test_dense.gguf"
    create_tiny_dense_gguf(model_path)
    hw_path = "tests/fixtures/hardware/laptop_8gb_no_gpu.json"

    result = runner.invoke(
        app,
        [
            "run",
            str(model_path),
            "--hw-json",
            hw_path,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "[DRY-RUN]" in result.stdout
    assert "Plan Summary" in result.stdout


def test_cli_config_commands() -> None:
    # config list
    res_list = runner.invoke(app, ["config", "list"])
    assert res_list.exit_code == 0
    assert "Configuration" in res_list.stdout

    # config path
    res_path = runner.invoke(app, ["config", "path"])
    assert res_path.exit_code == 0
    assert "config.toml" in res_path.stdout

    # config set and get
    res_set = runner.invoke(app, ["config", "set", "mode", "fast"])
    assert res_set.exit_code == 0

    res_get = runner.invoke(app, ["config", "get", "mode"])
    assert res_get.exit_code == 0
    assert "fast" in res_get.stdout

    # reset
    res_reset = runner.invoke(app, ["config", "reset", "mode"])
    assert res_reset.exit_code == 0


def test_cli_models_search_and_info() -> None:
    # models search
    res_search = runner.invoke(app, ["models", "search", "qwen"])
    assert res_search.exit_code == 0
    assert "Curated Model Catalog" in res_search.stdout
    assert "qwen2.5-0.5b" in res_search.stdout

    # models search with --fits
    res_fits = runner.invoke(app, ["models", "search", "--fits"])
    assert res_fits.exit_code == 0
    assert "Curated Model Catalog" in res_fits.stdout

    # models info
    res_info = runner.invoke(app, ["models", "info", "qwen2.5-0.5b"])
    assert res_info.exit_code == 0
    assert "Qwen 2.5 0.5B Instruct" in res_info.stdout
    assert "Available Quantizations" in res_info.stdout
    assert "Q4_K_M" in res_info.stdout


def test_cli_models_du_and_clean() -> None:
    res_du = runner.invoke(app, ["models", "du"])
    assert res_du.exit_code == 0
    assert "Model Disk Usage" in res_du.stdout

    res_clean = runner.invoke(app, ["models", "clean", "--yes"])
    assert res_clean.exit_code == 0


def test_cli_models_list_json() -> None:
    res = runner.invoke(app, ["models", "list", "--json"])
    assert res.exit_code == 0
    assert res.stdout.strip().startswith("[")


def test_cli_bench_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ramllm.telemetry.calibration import BenchmarkRecord

    model_path = tmp_path / "test_model.gguf"
    create_tiny_dense_gguf(model_path)
    export_path = tmp_path / "report.json"

    mock_rec = BenchmarkRecord(
        schema_version=2,
        timestamp="2026-10-17T12:00:00",
        model_path=str(model_path),
        plan_id="full_gpu_f16",
        arch="llama",
        quant="f16",
        backend_version="0.6.0",
        strategy="full_gpu",
        predicted_decode_low=10.0,
        predicted_decode_high=15.0,
        measured_decode_tok_s=12.5,
        predicted_prefill_low=50.0,
        predicted_prefill_high=70.0,
        measured_prefill_tok_s=60.0,
        peak_ram_rss_bytes=1024**3,
        ttft_s=0.1,
        decode_within_predicted=True,
        decode_error_pct=0.0,
        prefill_within_predicted=True,
        prefill_error_pct=0.0,
    )

    monkeypatch.setattr("ramllm.cli.app.run_benchmark", lambda **kwargs: mock_rec)

    res = runner.invoke(app, ["bench", str(model_path), "--export", str(export_path)])
    assert res.exit_code == 0
    assert "In Range?" in res.stdout
    assert "Error %" in res.stdout
    assert "✓ Yes" in res.stdout
    assert export_path.exists()

    with open(export_path) as f:
        data = json.load(f)
    assert data["model"]["name"] == "test_model.gguf"
    assert data["strategy"] == "full_gpu"
    assert data["benchmark"]["measured_decode_tok_s"] == 12.5


def test_cli_plan_cross_check_fit(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny_dense_fit.gguf"
    create_tiny_dense_gguf(model_path)
    hw_path = "tests/fixtures/hardware/laptop_16gb_gpu_6gb.json"

    res = runner.invoke(
        app,
        [
            "plan",
            str(model_path),
            "--hw-json",
            hw_path,
            "--cross-check-fit",
        ],
    )
    assert res.exit_code == 0
    assert "llama.cpp -fit Cross-Check Comparison" in res.stdout
    assert "GPU Offload" in res.stdout
    assert "Speed Forecast" in res.stdout


def test_cli_serve_loopback_protection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI serve refuses binding to 0.0.0.0 without API key or --insecure flag."""
    model_path = tmp_path / "tiny_serve_sec.gguf"
    create_tiny_dense_gguf(model_path)

    # 1. 0.0.0.0 without key and without --insecure must fail upfront
    res_refuse = runner.invoke(app, ["serve", str(model_path), "--host", "0.0.0.0"])
    assert res_refuse.exit_code == 1
    assert "Refusing to bind to non-loopback" in res_refuse.output

    # 2. Mock uvicorn.run and create_app so foreground server does not block event loop
    mock_run = MagicMock()
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", mock_run)
    monkeypatch.setattr("ramllm.api.server.create_app", lambda *args, **kwargs: MagicMock())

    res_insecure = runner.invoke(
        app, ["serve", str(model_path), "--host", "0.0.0.0", "--insecure", "--port", "9999"]
    )
    assert "Refusing to bind to non-loopback" not in res_insecure.output
    assert mock_run.called
