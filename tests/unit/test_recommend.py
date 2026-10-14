"""Unit tests for model recommendation engine and CLI commands (Milestone 4)."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from tests.fixtures.make_gguf import create_tiny_dense_gguf, create_tiny_moe_gguf
from typer.testing import CliRunner

from ramllm.cli.main import app
from ramllm.core.planner.recommend import recommend_models
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import UserRequest
from ramllm.registry.catalog import ModelCatalogEntry


def test_recommend_returns_only_feasible_models(tmp_path: Path) -> None:
    """Test that recommend filters out INFEASIBLE models for the requested hardware profile."""
    # 1. Create a tiny feasible model
    tiny_path = tmp_path / "tiny_model.gguf"
    create_tiny_dense_gguf(tiny_path)

    tiny_entry = ModelCatalogEntry(
        path=str(tiny_path),
        name=tiny_path.name,
        arch="llama",
        n_params=500_000_000,
        n_layers=4,
        quant="Q4_K_M",
        total_weight_bytes=500 * 1024 * 1024,
        is_moe=False,
        added_at=datetime.now(),
    )

    # 2. Create an infeasible entry (simulate huge 70B model with allow_streaming=False)
    huge_path = tmp_path / "huge_model.gguf"
    create_tiny_dense_gguf(huge_path)  # File exists, but we mock analyze_gguf to return 70B

    huge_entry = ModelCatalogEntry(
        path=str(huge_path),
        name=huge_path.name,
        arch="llama",
        n_params=70_000_000_000,
        n_layers=80,
        quant="Q4_K_M",
        total_weight_bytes=40 * 1024**3,
        is_moe=False,
        added_at=datetime.now(),
    )

    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    req = UserRequest(model="", allow_streaming=False)

    # Patch analyze_gguf for huge model
    from ramllm.core.schemas import ModelInfo

    orig_analyze = __import__("ramllm.core.analyzer.gguf", fromlist=["analyze_gguf"]).analyze_gguf

    def mock_analyze(p: Path) -> ModelInfo:
        if "huge_model" in str(p):
            total_b = 40 * 1024**3
            return ModelInfo(
                path=str(p),
                arch="llama",
                n_params=70_000_000_000,
                n_layers=80,
                hidden_size=8192,
                n_heads=64,
                n_kv_heads=8,
                head_dim=128,
                context_length_max=4096,
                quant="Q4_K_M",
                total_weight_bytes=total_b,
                per_layer_bytes=[total_b // 80] * 80,
                non_layer_bytes=500 * 1024**2,
                is_moe=False,
            )
        return orig_analyze(p)

    with patch("ramllm.core.planner.recommend.analyze_gguf", side_effect=mock_analyze):
        recs = recommend_models(hw, req, catalog=[tiny_entry, huge_entry])

    # Acceptance requirement: recommend returns ONLY non-INFEASIBLE models
    assert len(recs) == 1
    assert recs[0].name == "tiny_model.gguf"
    assert recs[0].best_plan.verdict != "INFEASIBLE"
    assert "huge_model.gguf" not in [r.name for r in recs]


def test_recommend_moe_and_dense_ranking(tmp_path: Path) -> None:
    """Test ranking of diverse feasible models (dense vs MoE)."""
    dense_path = tmp_path / "model_dense.gguf"
    create_tiny_dense_gguf(dense_path)
    moe_path = tmp_path / "model_moe.gguf"
    create_tiny_moe_gguf(moe_path)

    dense_entry = ModelCatalogEntry(
        path=str(dense_path),
        name=dense_path.name,
        arch="llama",
        n_params=1_000_000_000,
        n_layers=4,
        quant="Q4_K_M",
        total_weight_bytes=500 * 1024 * 1024,
        is_moe=False,
        added_at=datetime.now(),
    )
    moe_entry = ModelCatalogEntry(
        path=str(moe_path),
        name=moe_path.name,
        arch="qwen3moe",
        n_params=2_000_000_000,
        n_layers=2,
        quant="Q8_0",
        total_weight_bytes=800 * 1024 * 1024,
        is_moe=True,
        added_at=datetime.now(),
    )

    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    req = UserRequest(model="")

    recs = recommend_models(hw, req, catalog=[dense_entry, moe_entry])
    assert len(recs) == 2
    # Verify all recommendations have positive scores and valid reasons
    for r in recs:
        assert r.score > 0
        assert len(r.reasons) > 0


def test_cli_recommend_command(tmp_path: Path) -> None:
    """Test `ramllm recommend` CLI invocation with --json and --hw-json."""
    runner = CliRunner()
    model_path = tmp_path / "cli_tiny.gguf"
    create_tiny_dense_gguf(model_path)

    # Register model in catalog
    from ramllm.registry.catalog import add_model

    add_model(model_path)

    res = runner.invoke(
        app,
        [
            "recommend",
            "--hw-json",
            "tests/fixtures/hardware/laptop_16gb_gpu_6gb.json",
            "--json",
        ],
    )
    assert res.exit_code == 0
    assert "cli_tiny.gguf" in res.output or "[]" in res.output


def test_cli_models_verify_and_remove(tmp_path: Path) -> None:
    """Test `ramllm models verify` and `ramllm models remove` CLI commands."""
    runner = CliRunner()
    model_path = tmp_path / "verify_tiny.gguf"
    create_tiny_dense_gguf(model_path)

    from ramllm.registry.catalog import add_model

    add_model(model_path)

    # Verify command
    v_res = runner.invoke(app, ["models", "verify", "verify_tiny.gguf"])
    assert v_res.exit_code == 0
    assert "✔ verify_tiny.gguf" in v_res.output

    # Remove command
    r_res = runner.invoke(app, ["models", "remove", "verify_tiny.gguf"])
    assert r_res.exit_code == 0
    assert "✔ Removed" in r_res.output
