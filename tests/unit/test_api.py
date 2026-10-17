"""Unit tests for FastAPI endpoints and RAMLLM extensions."""

from pathlib import Path

from starlette.testclient import TestClient
from tests.unit.test_backends import _make_sample_plan

from ramllm.api.server import api_app, state
from ramllm.backends.mock import MockBackendAdapter
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.profiler import load_synthetic_profile

client = TestClient(api_app)


def test_healthz_endpoint() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_models_endpoint() -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) > 0


def test_ramllm_plan_and_metrics_endpoints() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    plan = _make_sample_plan()
    adapter = MockBackendAdapter()
    proc = adapter.launch(plan, port=9099)
    governor = MemoryGovernor(proc, hw, sample_interval_s=0.1)
    governor.start()

    state.active_plan = plan
    state.governor = governor

    try:
        # Test /ramllm/plan
        plan_resp = client.get("/ramllm/plan")
        assert plan_resp.status_code == 200
        plan_data = plan_resp.json()
        assert plan_data["id"] == plan.id
        assert plan_data["verdict"] == plan.verdict

        # Test /ramllm/metrics
        metrics_resp = client.get("/ramllm/metrics")
        assert metrics_resp.status_code == 200
        metrics_data = metrics_resp.json()
        assert "ram_rss_bytes" in metrics_data
        assert "predicted_ram_bytes" in metrics_data
        assert "budget_exceeded" in metrics_data
        assert not metrics_data["budget_exceeded"]
    finally:
        governor.stop()
        state.governor = None
        state.active_plan = None


def test_ramllm_replan_endpoint(tmp_path: Path) -> None:
    from tests.fixtures.make_gguf import create_tiny_dense_gguf

    from ramllm.core.analyzer.gguf import analyze_gguf
    from ramllm.core.schemas import UserRequest

    model_path = tmp_path / "replan_api_test.gguf"
    create_tiny_dense_gguf(model_path)
    model_info = analyze_gguf(model_path)
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")

    state.model_path = str(model_path)
    state.model_info = model_info
    state.hw_profile = hw
    state.user_request = UserRequest(model=str(model_path), min_context=2048, target_context=2048)
    state.adapter = MockBackendAdapter()
    state.internal_port = 8092

    try:
        # Replan with new context length and RAM budget
        resp = client.post(
            "/ramllm/replan",
            json={"context": 1024, "ram_budget_bytes": 6 * 1024**3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["kv"]["context"] == 1024
        assert state.active_plan is not None
        assert state.active_plan.kv.context == 1024
    finally:
        if state.governor:
            state.governor.stop()
        if state.backend_process:
            state.adapter.stop(state.backend_process)
        state.governor = None
        state.backend_process = None
        state.active_plan = None
        state.model_path = ""
