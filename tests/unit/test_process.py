"""Unit and CLI integration tests for process management (Milestone 6)."""

import http.server
import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

import pytest
from tests.ux.test_harness import assert_clean_process_state
from typer.testing import CliRunner

from ramllm.cli.app import app
from ramllm.services.paths import get_logs_dir, get_run_dir
from ramllm.services.process_registry import (
    RunningEngine,
    generate_engine_id,
    get_running_engine,
    list_running_engines,
    register_engine,
    unregister_engine,
)

runner = CliRunner()


class MockReplanServer(http.server.BaseHTTPRequestHandler):
    """Mock server responding to /ramllm/replan."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        if self.path.endswith("/ramllm/replan"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = (
                json.loads(self.rfile.read(content_length).decode("utf-8"))
                if content_length > 0
                else {}
            )
            ctx = body.get("context", 4096)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "id": f"plan-ctx-{ctx}",
                "verdict": "COMFORTABLE",
                "kv": {"context": ctx},
            }
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def mock_replan_port():
    """Start mock replan server on a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), MockReplanServer)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


# =====================================================================
# 1. PROCESS REGISTRY TESTS
# =====================================================================


def test_generate_engine_id() -> None:
    id1 = generate_engine_id("Qwen2.5-7B-Instruct")
    assert id1 == "qwen2-5-7b-instruct-1"


def test_register_and_get_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    log_file = tmp_path / "logs" / "test-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Engine boot logs\n", encoding="utf-8")

    eng = RunningEngine(
        id="test-model-1",
        pid=os.getpid(),  # Current process is definitely alive
        port=8080,
        model_name="test-model",
        model_path="/tmp/test.gguf",
        plan_id="plan-test",
        log_file=str(log_file),
    )
    register_engine(eng)

    loaded = get_running_engine("test-model-1")
    assert loaded is not None
    assert loaded.id == "test-model-1"
    assert loaded.pid == os.getpid()
    assert loaded.base_url == "http://127.0.0.1:8080"
    assert loaded.api_url == "http://127.0.0.1:8080/v1"
    assert loaded.is_alive() is True

    # List engines
    all_engines = list_running_engines()
    assert len(all_engines) == 1
    assert all_engines[0].id == "test-model-1"

    # Unregister
    unregister_engine("test-model-1")
    assert get_running_engine("test-model-1") is None


def test_stale_engine_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    # Register an engine with an impossible PID
    dead_eng = RunningEngine(
        id="dead-engine-1",
        pid=99999999,
        port=8080,
        model_name="dead-model",
        model_path="/tmp/dead.gguf",
        plan_id="plan-dead",
        log_file="/tmp/dead.log",
    )
    register_engine(dead_eng)

    # get_running_engine should detect it's dead, clean up, and return None
    assert get_running_engine("dead-engine-1") is None

    # list_running_engines with clean_stale=True should remove dead files
    run_file = get_run_dir() / "dead-engine-1.json"
    assert not run_file.exists()


# =====================================================================
# 2. CLI COMMAND TESTS: ps, status, stop, logs, replan
# =====================================================================


def test_cli_ps_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))
    res = runner.invoke(app, ["ps"])
    assert res.exit_code == 0
    assert "No RAMLLM engines are currently running" in res.stdout


def test_cli_ps_with_running_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    eng = RunningEngine(
        id="qwen-1",
        pid=os.getpid(),
        port=8080,
        model_name="Qwen2.5-7B",
        model_path="/tmp/qwen.gguf",
        plan_id="plan-1",
        log_file="/tmp/qwen.log",
    )
    register_engine(eng)

    res = runner.invoke(app, ["ps"])
    assert res.exit_code == 0
    assert "qwen-1" in res.stdout
    assert "8080" in res.stdout

    # Test json output
    res_json = runner.invoke(app, ["ps", "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.stdout)
    assert len(data) == 1
    assert data[0]["id"] == "qwen-1"

    unregister_engine("qwen-1")


def test_cli_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))
    res = runner.invoke(app, ["status"])
    assert res.exit_code == 0
    assert "RAMLLM System Status" in res.stdout
    assert "Backend Engine" in res.stdout
    assert "RAM:" in res.stdout


def test_cli_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "my-engine-1.log"
    log_file.write_text("line 1\nline 2\nline 3\nAll systems operational\n", encoding="utf-8")

    eng = RunningEngine(
        id="my-engine-1",
        pid=os.getpid(),
        port=8080,
        model_name="my-model",
        model_path="/tmp/model.gguf",
        plan_id="plan-1",
        log_file=str(log_file),
    )
    register_engine(eng)

    res = runner.invoke(app, ["logs", "my-engine-1", "--lines", "2"])
    assert res.exit_code == 0
    assert "All systems operational" in res.stdout
    assert "line 3" in res.stdout

    unregister_engine("my-engine-1")


def test_cli_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    eng = RunningEngine(
        id="to-stop-1",
        pid=os.getpid(),
        port=8080,
        model_name="to-stop",
        model_path="/tmp/stop.gguf",
        plan_id="plan-1",
        log_file="/tmp/stop.log",
    )
    register_engine(eng)

    # Stop specific engine
    # Note: stop_engine on current process PID would kill the test runner if allowed,
    # so we test unregistration logic and stop nonexistent / mock
    unregister_engine("to-stop-1")

    res_missing = runner.invoke(app, ["stop", "nonexistent-id"])
    assert res_missing.exit_code == 1
    assert "No running engine found" in res_missing.stdout


def test_cli_replan(mock_replan_port: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_HOME", str(tmp_path))

    eng = RunningEngine(
        id="replan-eng-1",
        pid=os.getpid(),
        port=mock_replan_port,
        model_name="replan-model",
        model_path="/tmp/replan.gguf",
        plan_id="plan-1",
        log_file="/tmp/replan.log",
    )
    register_engine(eng)

    res = runner.invoke(app, ["replan", "replan-eng-1", "--ctx", "8192"])
    assert res.exit_code == 0
    assert "Replan applied successfully" in res.stdout
    assert "plan-ctx-8192" in res.stdout

    unregister_engine("replan-eng-1")
    assert_clean_process_state()
