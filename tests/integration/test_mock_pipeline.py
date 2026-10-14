"""Integration tests for RAMLLM execution pipeline using Mock backend."""

from pathlib import Path

from tests.fixtures.make_gguf import create_tiny_dense_gguf

from ramllm.backends.mock import MockBackendAdapter
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.planner import create_execution_plans
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import UserRequest


def test_end_to_end_mock_pipeline(tmp_path: Path) -> None:
    # 1. Setup synthetic model & hardware
    model_path = tmp_path / "pipeline_model.gguf"
    create_tiny_dense_gguf(model_path)
    model_info = analyze_gguf(model_path)

    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")

    # 2. User request
    req = UserRequest(
        model=str(model_path),
        vram_budget_bytes={0: 4 * 1024**3},
        ram_budget_bytes=8 * 1024**3,
        min_context=2048,
        target_context=2048,
        preference="balanced",
    )

    # 3. Plan
    plans = create_execution_plans(model_info, hw, req)
    assert len(plans) > 0
    chosen_plan = plans[0]
    assert chosen_plan.verdict in ("COMFORTABLE", "CONSTRAINED")

    # 4. Launch with Mock backend
    backend = MockBackendAdapter()
    install = backend.detect()
    assert install is not None

    args = backend.build_args(chosen_plan, port=8888)
    assert "--port" in args
    assert "8888" in args

    proc = backend.launch(chosen_plan, port=8888)
    assert proc.pid == 99999
    assert proc.port == 8888

    # 5. Verify Memory Report
    report = backend.memory_report(proc)
    assert report.ram_rss_bytes == chosen_plan.memory.ram_peak_bytes
    assert report.vram_used_bytes == chosen_plan.memory.vram_peak_bytes

    # 6. Stop
    backend.stop(proc)
