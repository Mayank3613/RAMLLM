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


def test_boundary_overhead_slows_fragmented_placement() -> None:
    """Verify that placement with more CPU<->GPU boundaries is predicted slower."""
    from ramllm.core.planner.cost_model import estimate_speed

    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")

    # Contiguous split: 1 boundary
    speed_contiguous = estimate_speed(
        hw=hw,
        bytes_gpu=4 * 1024**3,
        bytes_cpu=2 * 1024**3,
        bytes_disk=0,
        bytes_xfer_h2d=0,
        kv_read_bytes_gpu=500 * 1024**2,
        kv_read_bytes_cpu=200 * 1024**2,
        active_params=7_000_000_000,
        num_boundaries=1,
    )

    # Fragmented split: 10 boundaries (alternating layers)
    speed_fragmented = estimate_speed(
        hw=hw,
        bytes_gpu=4 * 1024**3,
        bytes_cpu=2 * 1024**3,
        bytes_disk=0,
        bytes_xfer_h2d=0,
        kv_read_bytes_gpu=500 * 1024**2,
        kv_read_bytes_cpu=200 * 1024**2,
        active_params=7_000_000_000,
        num_boundaries=10,
    )

    assert speed_fragmented.decode_tok_s_low < speed_contiguous.decode_tok_s_low
    assert speed_fragmented.decode_tok_s_high < speed_contiguous.decode_tok_s_high


def test_slowness_tolerance_thresholds() -> None:
    from ramllm.core.planner.feasibility import SlownessTolerance, classify_verdict

    # Default tolerance: 8.0 tok/s comfortable, 1.0 tok/s constrained
    # A plan with 6 tok/s is CONSTRAINED under default tolerance
    v_default = classify_verdict(
        decode_tok_s_low=5.0, decode_tok_s_high=6.0, has_disk_reads=False, fits_budget=True
    )
    assert v_default == "CONSTRAINED"

    # With higher min_tps = 10.0, 6 tok/s falls below tolerance -> BATCH_ONLY
    strict_tol = SlownessTolerance(
        min_tok_s=10.0, comfortable_tok_s=25.0, comfortable_low_tok_s=15.0
    )
    v_strict = classify_verdict(
        decode_tok_s_low=5.0,
        decode_tok_s_high=6.0,
        has_disk_reads=False,
        fits_budget=True,
        tolerance=strict_tol,
    )
    assert v_strict == "BATCH_ONLY"

    # If fits_budget is False, always INFEASIBLE
    assert classify_verdict(5.0, 6.0, False, False, tolerance=strict_tol) == "INFEASIBLE"


def test_cpu_compute_bound_bottleneck() -> None:
    from ramllm.core.planner.cost_model import estimate_speed

    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    # Force a very slow / constrained CPU GFLOPs
    hw_slow_cpu = hw.model_copy(update={"cpu_effective_gflops": 5.0})

    speed = estimate_speed(
        hw=hw_slow_cpu,
        bytes_gpu=0,
        bytes_cpu=4 * 1024**3,
        bytes_disk=0,
        bytes_xfer_h2d=0,
        kv_read_bytes_gpu=0,
        kv_read_bytes_cpu=500 * 1024**2,
        active_params=7_000_000_000,
    )

    # 2 * 7B FLOPs / 5 GFLOPs = 14 / 5 = 2.8s per token -> compute bound (~0.3 tok/s)
    assert speed.bottleneck == "compute"


def test_flash_attention_v_quant_downgrade() -> None:
    # On pure CPU fallback, V-quantization should be downgraded to f16
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    model = _make_dummy_model(size_gb=3.0, n_layers=16)
    req = UserRequest(model=model.path, use_gpu=False, allow_kv_quant=True)

    plans = create_execution_plans(model, hw, req)
    assert len(plans) > 0
    # Every CPU plan that used quantized KV should have type_v="f16" and type_k quantized
    for p in plans:
        if p.kv.location == "cpu" and p.kv.type_k in ("q8_0", "q4_0"):
            assert p.kv.type_v == "f16"
            assert any("Flash Attention" in w for w in p.warnings)


@given(
    vram_extra_gb=st.integers(min_value=1, max_value=12),
)
def test_property_layer_offload_monotonicity_with_vram(vram_extra_gb: int) -> None:
    """Property: Increasing available VRAM never decreases offloaded GPU layers in the optimal plan."""
    hw_base = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(size_gb=8.0, n_layers=32)
    req = UserRequest(model=model.path)

    plans_base = create_execution_plans(model, hw_base, req)
    ngl_base = sum(len(layers) for layers in plans_base[0].placement.gpu_layers.values())

    gpu_boosted = hw_base.gpus[0].model_copy(
        update={
            "vram_total_bytes": hw_base.gpus[0].vram_total_bytes + vram_extra_gb * 1024**3,
            "vram_free_bytes": hw_base.gpus[0].vram_free_bytes + vram_extra_gb * 1024**3,
        }
    )
    hw_boosted = hw_base.model_copy(update={"gpus": [gpu_boosted]})
    plans_boosted = create_execution_plans(model, hw_boosted, req)
    ngl_boosted = sum(len(layers) for layers in plans_boosted[0].placement.gpu_layers.values())

    assert ngl_boosted >= ngl_base


@given(
    ram_gb=st.integers(min_value=8, max_value=64),
    vram_gb=st.integers(min_value=4, max_value=24),
)
def test_property_strict_memory_bounds(ram_gb: int, vram_gb: int) -> None:
    """Property: Feasible plans must never exceed hardware RAM or per-GPU VRAM limits."""
    hw_base = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    gpu = hw_base.gpus[0].model_copy(
        update={
            "vram_total_bytes": vram_gb * 1024**3,
            "vram_free_bytes": vram_gb * 1024**3,
        }
    )
    hw = hw_base.model_copy(
        update={
            "ram_total_bytes": ram_gb * 1024**3,
            "ram_available_bytes": int(ram_gb * 0.9 * 1024**3),
            "gpus": [gpu],
        }
    )
    model = _make_dummy_model(size_gb=6.0, n_layers=24)
    req = UserRequest(model=model.path, min_context=2048, target_context=2048)

    plans = create_execution_plans(model, hw, req)
    for p in plans:
        if p.verdict != "INFEASIBLE":
            assert p.memory.ram_peak_bytes <= hw.ram_total_bytes
            for g_idx, v_peak in p.memory.vram_peak_bytes.items():
                g_total = next(g.vram_total_bytes for g in hw.gpus if g.index == g_idx)
                assert v_peak <= g_total


@given(
    n_layers=st.integers(min_value=8, max_value=48),
)
def test_property_placement_completeness_and_exclusivity(n_layers: int) -> None:
    """Property: Every model layer is assigned to exactly one tier (GPU, CPU, or Disk)."""
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(size_gb=4.0, n_layers=n_layers)
    req = UserRequest(model=model.path)

    plans = create_execution_plans(model, hw, req)
    assert len(plans) > 0
    for p in plans:
        if p.verdict != "INFEASIBLE":
            assigned_layers: list[int] = []
            for gpu_layer_list in p.placement.gpu_layers.values():
                assigned_layers.extend(gpu_layer_list)
            assigned_layers.extend(p.placement.cpu_layers)
            assigned_layers.extend(p.placement.disk_streamed_layers)

            # Check exclusivity: no duplicates
            assert len(assigned_layers) == len(set(assigned_layers))
            # Check completeness: all layers 0..n_layers-1 accounted for
            assert set(assigned_layers) == set(range(n_layers))


@given(
    ctx=st.sampled_from([1024, 2048, 4096]),
)
def test_property_plan_determinism_and_invariance(ctx: int) -> None:
    """Property: Plan generation is deterministic given identical hardware profile and request."""
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(size_gb=4.0, n_layers=32)
    req = UserRequest(model=model.path, min_context=ctx, target_context=ctx)

    plans_a = create_execution_plans(model, hw, req)
    plans_b = create_execution_plans(model, hw, req)

    assert len(plans_a) == len(plans_b)
    for p_a, p_b in zip(plans_a, plans_b, strict=True):
        assert p_a.id == p_b.id
        assert p_a.verdict == p_b.verdict
        assert p_a.placement == p_b.placement
        assert p_a.kv == p_b.kv
        assert p_a.backend_args == p_b.backend_args
