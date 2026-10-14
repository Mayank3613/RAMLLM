"""Unit tests for RAMLLM setup wizard and home menu."""

from pathlib import Path

from typer.testing import CliRunner

from ramllm.cli.app import app
from ramllm.cli.setup.state import (
    SetupState,
    clear_setup_state,
    is_setup_completed,
    load_setup_state,
    save_setup_state,
)

runner = CliRunner()


def test_setup_state_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    state = SetupState(
        engine_installed=True,
        mode="quality",
        memory_preset="max",
        selected_model_id="qwen2.5-0.5b",
        finished=True,
    )
    save_setup_state(state)
    assert is_setup_completed()

    loaded = load_setup_state()
    assert loaded.engine_installed is True
    assert loaded.mode == "quality"
    assert loaded.memory_preset == "max"
    assert loaded.selected_model_id == "qwen2.5-0.5b"
    assert loaded.finished is True

    clear_setup_state()
    assert not is_setup_completed()


def test_cli_setup_help() -> None:
    res = runner.invoke(app, ["setup", "--help"])
    assert res.exit_code == 0
    assert "Run the interactive onboarding wizard" in res.stdout


def test_cli_setup_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))
    monkeypatch.setenv("RAMLLM_CONFIG", str(tmp_path / "config.toml"))

    res = runner.invoke(app, ["setup", "--skip"])
    assert res.exit_code == 0
    assert "Setup skipped" in res.stdout
    assert is_setup_completed()


def test_cli_setup_non_interactive_yes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))
    monkeypatch.setenv("RAMLLM_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("RAMLLM_MODELS_DIR", str(tmp_path / "models"))

    # Seed mock backend detection
    fake_backend = tmp_path / "llama-server"
    fake_backend.write_bytes(b"mock")
    fake_backend.chmod(0o755)
    monkeypatch.setenv("RAMLLM_BACKEND_PATH", str(fake_backend))

    # Seed local cached model so it doesn't attempt real network download
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    fake_model = models_dir / "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    from tests.fixtures.make_gguf import create_tiny_dense_gguf

    create_tiny_dense_gguf(fake_model)
    monkeypatch.setattr(
        "ramllm.cli.setup.wizard.resolve_model_reference", lambda *a, **kw: fake_model
    )

    res = runner.invoke(app, ["setup", "--yes", "--model", "qwen2.5-0.5b", "--mode", "fast"])
    assert res.exit_code == 0
    assert "RAMLLM Setup Complete" in res.stdout
    assert is_setup_completed()
