from typing import Any

import httpx
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from ramllm.cli.dashboard.widgets import EventsPanel, MemoryPanel, PlanPanel, SpeedPanel
from ramllm.core.profiler import probe_hardware
from ramllm.core.schemas import ExecutionPlan

DASHBOARD_CSS = """
Screen {
    background: #111217;
    color: #e2e8f0;
}

Header {
    background: #1a1c23;
    color: #38bdf8;
    dock: top;
}

Footer {
    background: #1a1c23;
    color: #94a3b8;
    dock: bottom;
}

.panel-box {
    border: solid #2e3344;
    background: #161821;
    padding: 1 2;
    margin: 1;
    height: auto;
}

#header-info {
    padding: 0 1;
    color: #94a3b8;
    background: #1e222d;
    height: 1;
}

.bold-highlight {
    text-style: bold;
    color: #4ade80;
}
"""


class DashboardApp(App[None]):
    """Calm, live-updating Textual TUI dashboard for monitoring memory governor and speed."""

    CSS = DASHBOARD_CSS
    TITLE = "RAMLLM Live Dashboard"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "replan_prompt", "Replan Context"),
        ("s", "stop_server", "Stop Engine"),
        ("?", "show_help", "Help"),
    ]

    def __init__(
        self,
        target_url: str = "http://127.0.0.1:8080",
        engine_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_url = target_url.rstrip("/")
        self.engine_id = engine_id or "local-engine"
        self.mem_panel = MemoryPanel(id="mem-panel")
        self.speed_panel = SpeedPanel(id="speed-panel")
        self.plan_panel = PlanPanel(id="plan-panel")
        self.events_panel = EventsPanel(id="events-panel")
        self.header_info = Static("Connecting to engine...", id="header-info")
        self._last_plan_id: str | None = None

        # Probe local hardware once for baseline maximum capacities
        self.hw = probe_hardware(quick=True)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield self.header_info
        with Vertical():
            yield self.mem_panel
            with Horizontal():
                yield self.speed_panel
                yield self.plan_panel
            yield self.events_panel
        yield Footer()

    def on_mount(self) -> None:
        """Schedule periodic 1-second metric polling."""
        self.set_interval(1.0, self.fetch_metrics)
        self.fetch_plan()
        self.fetch_metrics()

    def fetch_plan(self) -> None:
        """Fetch active execution plan from /ramllm/plan."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.target_url}/ramllm/plan")
                if res.status_code == 200:
                    plan = ExecutionPlan.model_validate(res.json())
                    self.plan_panel.update_plan(plan)
                    if plan.id != self._last_plan_id:
                        if self._last_plan_id is not None:
                            self.events_panel.add_event(f"Replan active: '{plan.id}'")
                        self._last_plan_id = plan.id
        except Exception:
            pass

    def fetch_metrics(self) -> None:
        """Fetch latest governor metrics from /ramllm/metrics."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self.target_url}/ramllm/metrics")
                if res.status_code == 200:
                    data = res.json()
                    rss = data.get("ram_rss_bytes", 0)
                    avail = data.get("ram_available_bytes", self.hw.ram_available_bytes)
                    tot_ram = self.hw.ram_total_bytes
                    warnings = data.get("warnings", [])

                    vram_dict = data.get("vram_used_bytes", {})
                    vram_used = sum(vram_dict.values())
                    vram_tot = sum(g.vram_total_bytes for g in self.hw.gpus) if self.hw.gpus else 0

                    self.mem_panel.update_metrics(
                        ram_rss_bytes=rss,
                        ram_avail_bytes=avail,
                        ram_total_bytes=tot_ram,
                        vram_used_bytes=vram_used,
                        vram_total_bytes=vram_tot,
                        warnings=warnings,
                    )

                    if warnings:
                        for w in warnings:
                            self.events_panel.add_event(w)

                    self.header_info.update(
                        f" Engine ID: {self.engine_id}  ·  Endpoint: {self.target_url}  ·  Status: Active"
                    )
                else:
                    self.header_info.update(
                        f" Engine ID: {self.engine_id}  ·  Status: Responding ({res.status_code})"
                    )
        except Exception:
            self.header_info.update(
                f" Engine ID: {self.engine_id}  ·  Endpoint: {self.target_url}  ·  Status: Waiting for engine..."
            )

    def action_replan_prompt(self) -> None:
        """Trigger rapid replan to 8192 context length."""
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.post(f"{self.target_url}/ramllm/replan", json={"context": 8192})
                if res.status_code == 200:
                    self.events_panel.add_event("Requested replan: context set to 8192")
                    self.fetch_plan()
                else:
                    self.events_panel.add_event(f"Replan failed: {res.text}")
        except Exception as e:
            self.events_panel.add_event(f"Replan network error: {e}")

    def action_stop_server(self) -> None:
        """Signal engine stop and quit dashboard."""
        from ramllm.services.process_registry import get_running_engine, stop_engine

        eng = get_running_engine(self.engine_id)
        if eng:
            stop_engine(eng)
            self.events_panel.add_event("Stopped engine process.")
        self.exit()

    def action_show_help(self) -> None:
        self.events_panel.add_event("Keybindings: [q] Quit · [r] Replan 8k · [s] Stop · [?] Help")
