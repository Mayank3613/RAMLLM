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


def test_api_key_authentication() -> None:
    """Verify that API key is enforced on protected endpoints and healthz remains open."""
    state.api_key = "secret-ramllm-test-key"
    try:
        # /healthz is exempt
        resp_health = client.get("/healthz")
        assert resp_health.status_code == 200

        # Protected endpoint without key -> 401
        resp_unauth = client.get("/v1/models")
        assert resp_unauth.status_code == 401
        assert "Invalid or missing API key" in resp_unauth.json()["error"]["message"]

        # Protected endpoint with wrong key -> 401
        resp_wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert resp_wrong.status_code == 401

        # Protected endpoint with valid Bearer token -> 200
        resp_bearer = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer secret-ramllm-test-key"},
        )
        assert resp_bearer.status_code == 200

        # Protected endpoint with valid X-API-Key header -> 200
        resp_xkey = client.get(
            "/v1/models",
            headers={"X-API-Key": "secret-ramllm-test-key"},
        )
        assert resp_xkey.status_code == 200
    finally:
        state.api_key = None


def test_cors_middleware() -> None:
    """Verify configurable CORS preflight and response headers."""
    state.cors_origins = ["http://localhost:3000", "https://myclient.local"]
    try:
        # Preflight OPTIONS request
        resp_opt = client.options(
            "/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp_opt.status_code == 204
        assert resp_opt.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

        # Normal GET with Origin
        resp_get = client.get(
            "/healthz",
            headers={"Origin": "https://myclient.local"},
        )
        assert resp_get.status_code == 200
        assert resp_get.headers.get("Access-Control-Allow-Origin") == "https://myclient.local"

        # Disallowed origin
        resp_disallowed = client.get(
            "/healthz",
            headers={"Origin": "https://malicious-site.com"},
        )
        assert resp_disallowed.status_code == 200
        assert "Access-Control-Allow-Origin" not in resp_disallowed.headers
    finally:
        state.cors_origins = []
