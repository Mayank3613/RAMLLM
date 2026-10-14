"""Unit and property tests for Planner across synthetic profiles."""

from hypothesis import given
from hypothesis import strategies as st

from ramllm.core.planner import calculate_safety_margins, create_execution_plans
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import ModelInfo, UserRequest


def _make_dummy_model(size_gb: float = 4.0, n_layers: int = 32, is_moe: bool = False) -> ModelInfo:
    total_bytes = int(size_gb * 1024**3)
    layer_bytes = total_bytes // n_layers
    return ModelInfo(
        path="/models/test.gguf",
        arch="llama",
        n_params=7_000_000_000,
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
        is_moe=is_moe,
        n_experts=8 if is_moe else None,
        n_experts_used=2 if is_moe else None,
        expert_bytes_per_layer=[int(layer_bytes * 0.75)] * n_layers if is_moe else None,
    )


def test_safety_margins_reservation() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    req = UserRequest(model="test.gguf")

    usable_ram, usable_vram = calculate_safety_margins(hw, req)
    # Available is ~6.4 GB, reserve is max(1.5 GB, 10% of 8 GB = 0.85 GB) -> reserve = 1.5 GB
    assert usable_ram < hw.ram_available_bytes
    assert usable_ram <= int(hw.ram_available_bytes - 1.5 * 1024**3)
    assert not usable_vram


def test_golden_profiles_planning() -> None:
    fixtures = [
        ("tests/fixtures/hardware/laptop_8gb_no_gpu.json", 3.5, "CONSTRAINED"),
        ("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json", 4.0, "COMFORTABLE"),
        ("tests/fixtures/hardware/desktop_32gb_gpu_24gb.json", 16.0, "COMFORTABLE"),
        ("tests/fixtures/hardware/apple_m_16gb_unified.json", 6.0, "COMFORTABLE"),
        ("tests/fixtures/hardware/apple_m_32gb_unified.json", 14.0, "COMFORTABLE"),
    ]

    for hw_path, model_gb, expected_min_verdict in fixtures:
        hw = load_synthetic_profile(hw_path)
        model = _make_dummy_model(size_gb=model_gb)
        req = UserRequest(model=model.path, min_context=2048, target_context=2048)

        plans = create_execution_plans(model, hw, req)
        assert len(plans) > 0
        best_plan = plans[0]
        assert best_plan.verdict == expected_min_verdict
        assert best_plan.speed.decode_tok_s_low > 0.0
        assert len(best_plan.explanation) > 0
        assert len(best_plan.backend_args) > 0


def test_infeasible_model_remedies() -> None:
    # 70B model (40 GB) on an 8 GB laptop with streaming disabled
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    model = _make_dummy_model(size_gb=40.0, n_layers=80)
    req = UserRequest(model=model.path, allow_streaming=False)

    plans = create_execution_plans(model, hw, req)
    assert len(plans) == 1
    assert plans[0].verdict == "INFEASIBLE"
    assert len(plans[0].warnings) > 0
    # Must propose increasing RAM budget or lowering context
    assert any("Increase RAM" in w or "budget" in w for w in plans[0].warnings)


def test_moe_expert_cpu_candidate_selected() -> None:
    # 16 GB laptop with 6 GB VRAM: 14 GB MoE model
    # Full GPU doesn't fit (14 GB > 6 GB VRAM), but Attention + Shared on GPU fits!
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(size_gb=12.0, n_layers=28, is_moe=True)
    req = UserRequest(model=model.path, min_context=2048)

    plans = create_execution_plans(model, hw, req)
    assert len(plans) > 0
    # Check if MoE CPU offload is among the candidates
    moe_plans = [p for p in plans if p.placement.expert_placement == "cpu"]
    assert len(moe_plans) > 0
    assert "-cmoe" in moe_plans[0].backend_args


@given(
    vram_extra_gb=st.integers(min_value=0, max_value=16),
)
def test_monotonicity_more_vram_never_slower(vram_extra_gb: int) -> None:
    # Invariant: Adding more VRAM should never result in a slower top plan
    hw_base = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(size_gb=8.0)
    req = UserRequest(model=model.path)

    plans_base = create_execution_plans(model, hw_base, req)

    # Now add VRAM
    gpu_boosted = hw_base.gpus[0].model_copy(
        update={
            "vram_total_bytes": hw_base.gpus[0].vram_total_bytes + vram_extra_gb * 1024**3,
            "vram_free_bytes": hw_base.gpus[0].vram_free_bytes + vram_extra_gb * 1024**3,
        }
    )
    hw_boosted = hw_base.model_copy(update={"gpus": [gpu_boosted]})
    plans_boosted = create_execution_plans(model, hw_boosted, req)

    assert plans_boosted[0].speed.decode_tok_s_low >= plans_base[0].speed.decode_tok_s_low - 0.1
