"""Milestone 5 Chaos & Robustness Suite: OOM recovery, driver loss fallback, memory pressure, and fault injection."""

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from ramllm.api.server import api_app, state
from ramllm.backends.mock import MockBackendAdapter
from ramllm.cli.main import app as cli_app
from ramllm.core.governor.monitor import MemoryGovernor, TierMetrics
from ramllm.core.governor.recovery import (
    FailureType,
    RecoveryExhaustedError,
    classify_failure,
    format_failure_cause,
    launch_with_recovery,
)
from ramllm.core.planner.reduction import get_reduced_plan
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import (
    ExecutionPlan,
    KvConfig,
    MemoryEstimate,
    ModelInfo,
    Placement,
    SpeedEstimate,
    UserRequest,
)


def _make_dummy_model(is_moe: bool = False) -> ModelInfo:
    """Create a synthetic ModelInfo for testing."""
    n_layers = 32
    per_layer = [200 * 1024 * 1024] * n_layers  # 200 MB per layer
    if is_moe:
        expert_per_layer = [150 * 1024 * 1024] * n_layers
        return ModelInfo(
            path="tests/fixtures/models/synthetic_moe.gguf",
            arch="qwen2moe",
            quant="Q4_K_M",
            n_layers=n_layers,
            hidden_size=4096,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            context_length_max=32768,
            n_params=14_000_000_000,
            is_moe=True,
            n_experts=8,
            n_experts_used=2,
            per_layer_bytes=per_layer,
            expert_bytes_per_layer=expert_per_layer,
            shared_expert_bytes_per_layer=[0] * n_layers,
            non_layer_bytes=500 * 1024 * 1024,
            total_weight_bytes=sum(per_layer) + 500 * 1024 * 1024,
        )
    return ModelInfo(
        path="tests/fixtures/models/synthetic_dense.gguf",
        arch="llama",
        quant="Q4_K_M",
        n_layers=n_layers,
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        context_length_max=32768,
        n_params=7_000_000_000,
        is_moe=False,
        per_layer_bytes=per_layer,
        non_layer_bytes=500 * 1024 * 1024,
        total_weight_bytes=sum(per_layer) + 500 * 1024 * 1024,
    )


def _make_sample_plan(
    ngl: int = 24,
    cpu_moe: bool = False,
    type_k: str = "f16",
    context: int = 4096,
) -> ExecutionPlan:
    gpu_layers = {0: list(range(ngl))} if ngl > 0 else {}
    cpu_layers = list(range(ngl, 32))
    return ExecutionPlan(
        id=f"plan_gpu_{ngl}L_{type_k}",
        model_variant="Q4_K_M",
        backend="llamacpp",
        placement=Placement(
            gpu_layers=gpu_layers,
            cpu_layers=cpu_layers,
            expert_placement="cpu" if cpu_moe else None,
        ),
        kv=KvConfig(
            location="gpu" if ngl > 0 else "cpu",
            type_k=type_k,  # type: ignore[arg-type]
            type_v=type_k,  # type: ignore[arg-type]
            context=context,
        ),
        memory=MemoryEstimate(
            vram_peak_bytes={0: 6 * 1024**3} if ngl > 0 else {},
            ram_peak_bytes=4 * 1024**3,
        ),
        speed=SpeedEstimate(
            decode_tok_s_low=12.0,
            decode_tok_s_high=18.0,
            prefill_tok_s_low=60.0,
            prefill_tok_s_high=90.0,
            bottleneck="vram_bw",
        ),
        verdict="COMFORTABLE",
        backend_args=["-m", "/models/test.gguf", "-ngl", str(ngl), "-c", str(context)],
        explanation=["Sample plan for chaos tests."],
    )


def test_failure_classification() -> None:
    plan_with_gpu = _make_sample_plan(ngl=16)
    plan_pure_cpu = _make_sample_plan(ngl=0)

    # 1. Driver loss / accelerator unavailable
    assert (
        classify_failure(
            "CUDA error: all CUDA-capable devices are busy or unavailable", plan_with_gpu
        )
        == FailureType.DRIVER_LOSS
    )
    assert (
        classify_failure("cudaErrorNoDevice: device reset detected", plan_with_gpu)
        == FailureType.DRIVER_LOSS
    )
    assert (
        classify_failure("Metal device failure during command buffer dispatch", plan_with_gpu)
        == FailureType.DRIVER_LOSS
    )

    # 2. GPU OOM
    assert (
        classify_failure("CUDA out of memory: tried to allocate 4.2 GiB", plan_with_gpu)
        == FailureType.GPU_OOM
    )
    assert (
        classify_failure("ggml_cuda_init: failed to allocate buffer size 2048 MB", plan_with_gpu)
        == FailureType.GPU_OOM
    )
    assert (
        classify_failure("Process exited with code 137 (SIGKILL)", plan_with_gpu)
        == FailureType.GPU_OOM
    )
    assert (
        classify_failure("out of memory while allocating layers", plan_with_gpu)
        == FailureType.GPU_OOM
    )

    # 3. Host OOM
    assert (
        classify_failure("std::bad_alloc: cannot allocate memory", plan_pure_cpu)
        == FailureType.HOST_OOM
    )
    assert classify_failure("cgroup memory limit exceeded", plan_pure_cpu) == FailureType.HOST_OOM
    assert (
        classify_failure("Process exited with code 137 (SIGKILL)", plan_pure_cpu)
        == FailureType.HOST_OOM
    )

    # 4. Timeout
    assert (
        classify_failure(TimeoutError("llama-server timed out"), plan_with_gpu)
        == FailureType.TIMEOUT
    )

    # 5. Cause formatting
    cause = format_failure_cause("CUDA out of memory", FailureType.GPU_OOM)
    assert "GPU memory allocation failed" in cause


def test_plan_reduction_steps() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)

    # Initial plan: 32 GPU layers, f16 KV, 4096 context
    plan0 = _make_sample_plan(ngl=32, type_k="f16", context=4096)

    # Attempt 1: Reduce GPU layers by ~25% (32 -> 24)
    plan1 = get_reduced_plan(plan0, model, hw, req, failure_type="gpu_oom", attempt=1)
    assert plan1 is not None
    ngl1 = sum(len(layer_list) for layer_list in plan1.placement.gpu_layers.values())
    assert ngl1 < 32
    assert plan1.kv.type_k == "f16"
    assert "Reduced GPU layers" in plan1.warnings[0]

    # Attempt 2: Further reduce GPU layers and quantize KV cache to q8_0
    plan2 = get_reduced_plan(plan1, model, hw, req, failure_type="gpu_oom", attempt=2)
    assert plan2 is not None
    ngl2 = sum(len(layer_list) for layer_list in plan2.placement.gpu_layers.values())
    assert ngl2 < ngl1
    assert plan2.kv.type_k == "q8_0"

    # Attempt 3: Pure CPU fallback and q4_0 KV
    plan3 = get_reduced_plan(plan2, model, hw, req, failure_type="gpu_oom", attempt=3)
    assert plan3 is not None
    ngl3 = sum(len(layer_list) for layer_list in plan3.placement.gpu_layers.values())
    assert ngl3 == 0
    assert plan3.kv.type_k == "q4_0"
    assert len(plan3.placement.cpu_layers) == model.n_layers

    # Attempt 4: Halve context length down to min_context
    plan4 = get_reduced_plan(plan3, model, hw, req, failure_type="host_oom", attempt=4)
    assert plan4 is not None
    assert plan4.kv.context <= 2048


def test_driver_loss_immediate_cpu_fallback() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)

    plan_gpu = _make_sample_plan(ngl=24, type_k="f16", context=4096)
    reduced = get_reduced_plan(plan_gpu, model, hw, req, failure_type="driver_loss", attempt=1)
    assert reduced is not None
    assert sum(len(layer_list) for layer_list in reduced.placement.gpu_layers.values()) == 0
    assert len(reduced.placement.cpu_layers) == model.n_layers
    assert any("GPU acceleration became unavailable" in w for w in reduced.warnings)


def test_moe_routed_expert_cpu_offload_on_gpu_oom() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=True)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)

    # Initial plan with MoE experts on GPU
    plan_moe_gpu = _make_sample_plan(ngl=32, cpu_moe=False, type_k="f16", context=4096)
    reduced = get_reduced_plan(plan_moe_gpu, model, hw, req, failure_type="gpu_oom", attempt=1)
    assert reduced is not None
    assert reduced.placement.expert_placement == "cpu"
    assert "-cmoe" in reduced.backend_args


def test_launch_with_recovery_succeeds_after_oom() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)
    initial_plan = _make_sample_plan(ngl=32, type_k="f16", context=4096)

    # Adapter fails first 2 launches with CUDA OOM, then succeeds
    adapter = MockBackendAdapter(
        fail_launches=2,
        fail_error=RuntimeError("CUDA out of memory: failed to allocate 8.0 GiB buffer"),
    )

    retry_events: list[tuple[int, str]] = []

    def on_retry(att: int, p: ExecutionPlan, cause: str) -> None:
        retry_events.append((att, cause))

    proc, final_plan, logs = launch_with_recovery(
        adapter=adapter,
        plan=initial_plan,
        model=model,
        hw=hw,
        req=req,
        max_retries=3,
        auto_retry=True,
        on_retry=on_retry,
    )

    assert adapter.launch_count == 3
    assert len(retry_events) == 2
    assert len(logs) == 2
    # Final plan should have fewer GPU layers
    assert sum(len(layer_list) for layer_list in final_plan.placement.gpu_layers.values()) < 32
    assert proc.pid == 99999


def test_launch_with_recovery_driver_loss_fallback() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)
    initial_plan = _make_sample_plan(ngl=24, type_k="f16", context=4096)

    # Adapter fails any GPU offload with driver reset
    adapter = MockBackendAdapter(fail_if_gpu=True)

    proc, final_plan, logs = launch_with_recovery(
        adapter=adapter,
        plan=initial_plan,
        model=model,
        hw=hw,
        req=req,
        max_retries=3,
        auto_retry=True,
    )

    # 1st attempt with GPU failed, 2nd attempt with CPU succeeded
    assert adapter.launch_count == 2
    assert sum(len(layer_list) for layer_list in final_plan.placement.gpu_layers.values()) == 0
    assert any("GPU acceleration became unavailable" in w for w in final_plan.warnings)


def test_no_auto_retry_flag_enforcement() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)
    initial_plan = _make_sample_plan(ngl=32, type_k="f16", context=4096)

    adapter = MockBackendAdapter(
        fail_launches=1,
        fail_error=RuntimeError("CUDA out of memory: failed to allocate 8.0 GiB buffer"),
    )

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        launch_with_recovery(
            adapter=adapter,
            plan=initial_plan,
            model=model,
            hw=hw,
            req=req,
            auto_retry=False,
        )

    assert "Auto-retry is disabled (--no-auto-retry)" in str(exc_info.value)
    assert adapter.launch_count == 1
    assert len(exc_info.value.remedies) > 0


def test_recovery_exhausted_after_max_retries() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    req = UserRequest(model=model.path, min_context=2048, target_context=4096)
    initial_plan = _make_sample_plan(ngl=32, type_k="f16", context=4096)

    # Adapter that always fails
    adapter = MockBackendAdapter(
        fail_launches=10,
        fail_error=RuntimeError("CUDA out of memory: buffer allocation error"),
    )

    with pytest.raises(RecoveryExhaustedError) as exc_info:
        launch_with_recovery(
            adapter=adapter,
            plan=initial_plan,
            model=model,
            hw=hw,
            req=req,
            max_retries=2,
            auto_retry=True,
        )

    assert "Failed to launch after 3 recovery attempts" in str(exc_info.value)
    assert adapter.launch_count == 3
    assert exc_info.value.failure_type == FailureType.GPU_OOM


def test_memory_governor_pressure_response() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    plan = _make_sample_plan(ngl=16)
    adapter = MockBackendAdapter()
    proc = adapter.launch(plan)

    pressure_received = []

    def on_pressure_cb(metrics: TierMetrics) -> None:
        pressure_received.append(metrics)

    # Set pressure threshold absurdly high (500 GB) so current RAM available is below threshold
    governor = MemoryGovernor(
        process_handle=proc,
        hw=hw,
        sample_interval_s=0.05,
        pressure_threshold_ram_bytes=500 * 1024**3,
        on_pressure=on_pressure_cb,
    )
    governor.start()

    try:
        import time

        time.sleep(0.2)
        metrics = governor.get_latest_metrics()
        assert metrics.pressure_detected
        assert metrics.process_alive
        assert any("Critical system memory pressure" in w for w in metrics.warnings)
        assert len(pressure_received) > 0
    finally:
        governor.stop()


def test_api_replan_with_recovery_and_impossible_budget() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    plan = _make_sample_plan(ngl=16)
    model = _make_dummy_model(is_moe=False)

    adapter = MockBackendAdapter()
    proc = adapter.launch(plan, port=9099)
    governor = MemoryGovernor(proc, hw, sample_interval_s=0.1)
    governor.start()

    state.model_path = model.path
    state.model_info = model
    state.hw_profile = hw
    state.user_request = UserRequest(model=model.path, min_context=2048, target_context=4096)
    state.active_plan = plan
    state.backend_process = proc
    state.governor = governor
    state.adapter = adapter

    client = TestClient(api_app)

    try:
        # Replan with valid budget
        resp = client.post(
            "/ramllm/replan",
            json={"context": 2048, "ram_budget_bytes": 10 * 1024**3},
        )
        assert resp.status_code == 200
        replan_data = resp.json()
        assert replan_data["kv"]["context"] == 2048

        # Replan with impossible budget (100 MB RAM, 0 VRAM, no streaming for 7B model) -> returns 422, NOT 500
        bad_resp = client.post(
            "/ramllm/replan",
            json={
                "ram_budget_bytes": 100 * 1024 * 1024,
                "vram_budget_bytes": {0: 0},
                "allow_streaming": False,
            },
        )
        assert bad_resp.status_code == 422
        assert "Cannot replan" in bad_resp.json()["detail"]
    finally:
        if state.governor:
            state.governor.stop()
        state.governor = None
        state.backend_process = None
        state.active_plan = None


def test_cli_run_fails_soft_without_raw_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    # Model file doesn't exist
    res = runner.invoke(cli_app, ["run", "/nonexistent/model.gguf"])
    assert res.exit_code == 1
    assert "Model file not found" in res.output or "not found" in res.output
    # Must never dump python traceback
    assert "Traceback (most recent call last)" not in res.output


def test_real_world_memory_pressure_drill() -> None:
    """Simulate sudden memory pressure during inference triggering governor detection and downgrade."""
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = _make_dummy_model(is_moe=False)
    initial_plan = _make_sample_plan(ngl=24, context=4096)

    # 1. Simulate VRAM pressure spike -> reduced plan offloads fewer layers to GPU
    vram_pressure_plan = get_reduced_plan(
        initial_plan,
        model,
        hw,
        UserRequest(model=model.path, min_context=2048, target_context=4096),
        failure_type="gpu_oom",
        attempt=1,
    )
    assert vram_pressure_plan is not None
    vram_layers = sum(len(layers) for layers in vram_pressure_plan.placement.gpu_layers.values())
    assert vram_layers < 24

    # 2. Simulate RAM pressure spike -> reduced plan decreases context length or quantizes KV
    ram_pressure_plan = get_reduced_plan(
        initial_plan,
        model,
        hw,
        UserRequest(model=model.path, min_context=2048, target_context=4096),
        failure_type="ram_oom",
        attempt=1,
    )
    assert ram_pressure_plan is not None
    assert (
        ram_pressure_plan.kv.context < initial_plan.kv.context
        or ram_pressure_plan.kv.type_k != "f16"
    )

    # 3. Simulate sudden driver loss -> falls back to pure CPU execution
    driver_loss_plan = get_reduced_plan(
        initial_plan,
        model,
        hw,
        UserRequest(model=model.path, min_context=2048, target_context=4096),
        failure_type="driver_loss",
        attempt=1,
    )
    assert driver_loss_plan is not None
    assert sum(len(layers) for layers in driver_loss_plan.placement.gpu_layers.values()) == 0
    assert len(driver_loss_plan.placement.cpu_layers) == 32
