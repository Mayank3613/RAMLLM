"""Unit and integration tests for RAMLLM live TUI dashboard (Milestone 7)."""

import http.server
import json
import socket
import threading
from typing import Any

import pytest
from typer.testing import CliRunner

from ramllm.cli.app import app
from ramllm.cli.dashboard.app import DashboardApp
from ramllm.cli.dashboard.widgets import EventsPanel, MemoryPanel, PlanPanel, SpeedPanel
from ramllm.core.schemas import (
    ExecutionPlan,
    KvConfig,
    MemoryEstimate,
    Placement,
    SpeedEstimate,
)

runner = CliRunner()


class MockDashboardServer(http.server.BaseHTTPRequestHandler):
    """Mock server responding to /ramllm/metrics, /ramllm/plan, /ramllm/replan."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.endswith("/ramllm/plan"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            plan = ExecutionPlan(
                id="test-plan-dashboard",
                model_variant="qwen-7b",
                backend="mock",
                placement=Placement(gpu_layers={0: list(range(32))}, cpu_layers=[]),
                kv=KvConfig(location="gpu", type_k="q8_0", type_v="q8_0", context=4096),
                memory=MemoryEstimate(vram_peak_bytes={0: 5 * 1024**3}, ram_peak_bytes=2 * 1024**3),
                speed=SpeedEstimate(
                    decode_tok_s_low=28.0,
                    decode_tok_s_high=34.0,
                    prefill_tok_s_low=120.0,
                    prefill_tok_s_high=150.0,
                    bottleneck="vram_bw",
                ),
                verdict="COMFORTABLE",
                warnings=[],
            )
            self.wfile.write(plan.model_dump_json().encode())
        elif self.path.endswith("/ramllm/metrics"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            metrics = {
                "vram_used_bytes": {0: 4 * 1024**3},
                "ram_rss_bytes": int(1.5 * 1024**3),
                "ram_available_bytes": 10 * 1024**3,
                "disk_read_bytes_s": 0,
                "warnings": ["External memory pressure detected"],
            }
            self.wfile.write(json.dumps(metrics).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path.endswith("/ramllm/replan"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "context": 8192}')
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def mock_dashboard_url():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), MockDashboardServer)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


# =====================================================================
# 1. WIDGET UNIT TESTS
# =====================================================================


def test_memory_panel_updates() -> None:
    panel = MemoryPanel()
    panel.update_metrics(
        ram_rss_bytes=4 * 1024**3,
        ram_avail_bytes=12 * 1024**3,
        ram_total_bytes=16 * 1024**3,
        vram_used_bytes=3 * 1024**3,
        vram_total_bytes=6 * 1024**3,
        warnings=None,
    )
    assert panel.ram_bar.progress == 25.0
    assert panel.vram_bar.progress == 50.0

    # With warning
    panel.update_metrics(
        ram_rss_bytes=14 * 1024**3,
        ram_avail_bytes=2 * 1024**3,
        ram_total_bytes=16 * 1024**3,
        warnings=["Low memory"],
    )
    assert "Low memory" in str(panel.status_label.render())


def test_speed_and_plan_panels() -> None:
    speed = SpeedPanel()
    speed.update_speed(current_tok_s=31.5, avg_tok_s=29.0, ttft_s=0.45)
    assert "31.5" in str(speed.speed_digits.render())

    plan_panel = PlanPanel()
    dummy_plan = ExecutionPlan(
        id="plan-ui-test",
        model_variant="variant-1",
        backend="mock",
        placement=Placement(gpu_layers={0: [1, 2, 3]}, cpu_layers=[4, 5]),
        kv=KvConfig(location="gpu", type_k="q4_0", type_v="q4_0", context=2048),
        memory=MemoryEstimate(ram_peak_bytes=1024**3),
        speed=SpeedEstimate(
            decode_tok_s_low=20.0,
            decode_tok_s_high=25.0,
            prefill_tok_s_low=80.0,
            prefill_tok_s_high=100.0,
            bottleneck="ram_bw",
        ),
        verdict="CONSTRAINED",
    )
    plan_panel.update_plan(dummy_plan)
    rendered_text = str(plan_panel.content_label.render())
    assert "plan-ui-test" in rendered_text
    assert "CONSTRAINED" in rendered_text


def test_events_panel() -> None:
    events = EventsPanel()
    for i in range(12):
        events.add_event(f"Event {i}")
    # History is capped at 8
    assert len(events.events) == 8
    assert "Event 11" in events.events[-1]


# =====================================================================
# 2. TEXTUAL APP ASYNC TESTS (pilot testing)
# =====================================================================


@pytest.mark.asyncio
async def test_dashboard_textual_app(mock_dashboard_url: str) -> None:
    app_instance = DashboardApp(target_url=mock_dashboard_url, engine_id="test-engine")

    async with app_instance.run_test() as pilot:
        # Check initial composition
        assert pilot.app.query_one("#mem-panel") is not None
        assert pilot.app.query_one("#speed-panel") is not None
        assert pilot.app.query_one("#plan-panel") is not None
        assert pilot.app.query_one("#events-panel") is not None

        # Fetch plan & metrics roundtrip
        app_instance.fetch_plan()
        app_instance.fetch_metrics()
        await pilot.pause(0.1)

        # Trigger replan action via keybinding
        await pilot.press("r")
        await pilot.pause(0.1)

        # Trigger help
        await pilot.press("?")
        await pilot.pause(0.1)


# =====================================================================
# 3. CLI COMMAND TESTS
# =====================================================================


def test_cli_dashboard_help() -> None:
    res = runner.invoke(app, ["dashboard", "--help"])
    assert res.exit_code == 0
    assert "live visual TUI dashboard" in res.stdout


def test_cli_dashboard_no_engines_error() -> None:
    res = runner.invoke(app, ["dashboard"])
    assert res.exit_code == 1
    assert "No running RAMLLM engine found" in res.stdout
