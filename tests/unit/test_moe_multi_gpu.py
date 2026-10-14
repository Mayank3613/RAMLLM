"""Unit tests for MoE-aware placement and Multi-GPU pipeline offloading (Milestone 3)."""

from typer.testing import CliRunner

from ramllm.backends import LlamaCppAdapter
from ramllm.cli.main import _parse_gpu_args, app
from ramllm.core.planner import create_execution_plans
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import ModelInfo, UserRequest


def _make_dummy_moe_model(
    size_gb: float = 14.0,
    n_layers: int = 28,
    n_experts: int = 8,
    n_experts_used: int = 2,
    shared_ratio: float = 0.10,
) -> ModelInfo:
    """Create a synthetic MoE model descriptor for testing."""
    total_bytes = int(size_gb * 1024**3)
    layer_bytes = total_bytes // n_layers
    expert_bytes = int(layer_bytes * 0.70)
    shared_bytes = int(layer_bytes * shared_ratio)

    return ModelInfo(
        path="/models/qwen-moe-test.gguf",
        arch="qwen3moe",
        n_params=30_000_000_000,
        n_layers=n_layers,
        hidden_size=2048,
        n_heads=16,
        n_kv_heads=4,
        head_dim=128,
        context_length_max=8192,
        quant="Q4_K_M",
        total_weight_bytes=total_bytes,
        per_layer_bytes=[layer_bytes] * n_layers,
        non_layer_bytes=int(300 * 1024**2),
        is_moe=True,
        n_experts=n_experts,
        n_experts_used=n_experts_used,
        expert_bytes_per_layer=[expert_bytes] * n_layers,
        shared_expert_bytes_per_layer=[shared_bytes] * n_layers,
    )


def _make_dummy_dense_model(size_gb: float = 32.0, n_layers: int = 64) -> ModelInfo:
    """Create a synthetic dense model descriptor for testing."""
    total_bytes = int(size_gb * 1024**3)
    layer_bytes = total_bytes // n_layers
    return ModelInfo(
        path="/models/dense-test.gguf",
        arch="llama",
        n_params=32_000_000_000,
        n_layers=n_layers,
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        context_length_max=8192,
        quant="Q4_K_M",
        total_weight_bytes=total_bytes,
        per_layer_bytes=[layer_bytes] * n_layers,
        non_layer_bytes=int(400 * 1024**2),
        is_moe=False,
    )


def test_dual_gpu_pipeline_split_proportional() -> None:
    """Test that two identical GPUs proportionally split layers with -sm layer and -ts flags."""
    hw = load_synthetic_profile("tests/fixtures/hardware/dual_gpu_workstation.json")
    model = _make_dummy_dense_model(size_gb=32.0, n_layers=64)
    req = UserRequest(model=model.path, min_context=2048, target_context=2048)

    plans = create_execution_plans(model, hw, req)
    assert len(plans) > 0

    # Locate multi-GPU split plan
    multi_gpu_plans = [p for p in plans if "multi_gpu_split" in p.id]
    assert len(multi_gpu_plans) > 0
    plan = multi_gpu_plans[0]

    # Verify both GPUs received layers
    assert len(plan.placement.gpu_layers) == 2
    assert 0 in plan.placement.gpu_layers
    assert 1 in plan.placement.gpu_layers
    layers_gpu0 = len(plan.placement.gpu_layers[0])
    layers_gpu1 = len(plan.placement.gpu_layers[1])
    assert layers_gpu0 > 0
    assert layers_gpu1 > 0
    assert layers_gpu0 + layers_gpu1 == 64

    # Verify backend arguments
    assert "-sm" in plan.backend_args
    assert "layer" in plan.backend_args
    assert "-ts" in plan.backend_args
    ts_idx = plan.backend_args.index("-ts")
    ts_val = plan.backend_args[ts_idx + 1]
    assert len(ts_val.split(",")) == 2
    fractions = [float(f) for f in ts_val.split(",")]
    assert 0.95 <= sum(fractions) <= 1.05
    assert "-mg" in plan.backend_args

    # Verify adapter flag generation
    adapter = LlamaCppAdapter()
    built_args = adapter.build_args(plan)
    assert "-sm" in built_args
    assert "layer" in built_args
    assert "-ts" in built_args


def test_heterogeneous_gpu_bandwidth_mismatch_warning() -> None:
    """Test that mismatched GPU bandwidth (RTX 4090 + RTX 3060) triggers mismatch warning."""
    hw = load_synthetic_profile("tests/fixtures/hardware/mismatched_dual_gpu.json")
    model = _make_dummy_dense_model(size_gb=24.0, n_layers=48)
    req = UserRequest(model=model.path, min_context=2048, target_context=2048)

    plans = create_execution_plans(model, hw, req)
    multi_gpu_plans = [p for p in plans if "multi_gpu_split" in p.id]
    assert len(multi_gpu_plans) > 0
    plan = multi_gpu_plans[0]

    # Check for mismatch warning in plan
    assert len(plan.warnings) > 0
    assert any("Heterogeneous GPU mismatch" in w and "GPU 1" in w for w in plan.warnings)


def test_gpu_indices_filtering() -> None:
    """Test that specifying gpu_indices restricts placement to only chosen GPUs."""
    hw = load_synthetic_profile("tests/fixtures/hardware/dual_gpu_workstation.json")
    model = _make_dummy_dense_model(size_gb=16.0, n_layers=32)

    # Request only GPU 1
    req = UserRequest(
        model=model.path,
        gpu_indices=[1],
        min_context=2048,
        target_context=2048,
    )
    plans = create_execution_plans(model, hw, req)
    assert len(plans) > 0
    top_plan = plans[0]

    # GPU 0 must NOT have any layers assigned
    assert 0 not in top_plan.placement.gpu_layers
    if top_plan.placement.gpu_layers:
        assert 1 in top_plan.placement.gpu_layers


def test_moe_cpu_routed_experts_plan() -> None:
    """Test that MoE model generates attention/shared on GPU and routed experts on CPU."""
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_moe_model(size_gb=12.0, n_layers=28)
    req = UserRequest(model=model.path, min_context=2048, target_context=2048)

    plans = create_execution_plans(model, hw, req)
    moe_plans = [p for p in plans if p.id == "moe_cpu_routed_experts"]
    assert len(moe_plans) == 1
    moe_plan = moe_plans[0]

    assert moe_plan.placement.expert_placement == "cpu"
    assert "-cmoe" in moe_plan.backend_args
    assert any("MoE Optimized Placement" in exp for exp in moe_plan.explanation)

    # Decode speed must be positive and realistic
    assert moe_plan.speed.decode_tok_s_low > 2.0


def test_moe_partial_offload_candidates() -> None:
    """Test that partial MoE offload (-ncmoe) plans are generated."""
    hw = load_synthetic_profile("tests/fixtures/hardware/desktop_32gb_gpu_24gb.json")
    model = _make_dummy_moe_model(size_gb=18.0, n_layers=28)
    req = UserRequest(model=model.path, min_context=2048, target_context=2048)

    plans = create_execution_plans(model, hw, req)
    partial_moe = [p for p in plans if "moe_partial_cpu" in p.id]
    assert len(partial_moe) > 0

    plan = partial_moe[0]
    assert "-ncmoe" in plan.backend_args
    adapter = LlamaCppAdapter()
    built_args = adapter.build_args(plan)
    assert "-ncmoe" in built_args


def test_cli_gpu_args_parsing() -> None:
    """Test helper parsing for --gpus and --vram CLI arguments."""
    # 1. Single GPU
    idxs, vram = _parse_gpu_args("1", "8GB")
    assert idxs == [1]
    assert vram == {1: 8 * 1024**3}

    # 2. Dual GPU with uniform VRAM
    idxs, vram = _parse_gpu_args("0,1", "8GB")
    assert idxs == [0, 1]
    assert vram == {0: 8 * 1024**3, 1: 8 * 1024**3}

    # 3. Dual GPU with per-GPU VRAM
    idxs, vram = _parse_gpu_args("0,1", "16GB,8GB")
    assert idxs == [0, 1]
    assert vram == {0: 16 * 1024**3, 1: 8 * 1024**3}

    # 4. None cases
    idxs, vram = _parse_gpu_args(None, None)
    assert idxs is None
    assert vram is None


def test_cli_plan_with_gpus_option(tmp_path: object) -> None:
    """Test running `ramllm plan` with --gpus and --hw-json."""
    from tests.fixtures.make_gguf import create_tiny_dense_gguf

    runner = CliRunner()
    model_path = tmp_path / "tiny_dense.gguf"  # type: ignore[operator]
    create_tiny_dense_gguf(model_path)

    res = runner.invoke(
        app,
        [
            "plan",
            str(model_path),
            "--hw-json",
            "tests/fixtures/hardware/dual_gpu_workstation.json",
            "--gpus",
            "0,1",
            "--json",
        ],
    )
    assert res.exit_code == 0
    assert "full_gpu_f16" in res.output or "Strategy" in res.output or "id" in res.output
