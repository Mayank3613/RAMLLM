"""Textual widgets for RAMLLM Live Dashboard."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar, Static

from ramllm.core.schemas import ExecutionPlan


class MemoryPanel(Static):
    """Memory usage panel displaying per-tier consumption, bars, and headroom."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ram_bar = ProgressBar(total=100, show_eta=False, id="ram-progress")
        self.vram_bar = ProgressBar(total=100, show_eta=False, id="vram-progress")
        self.ram_label = Label("RAM: Measuring...", id="ram-label")
        self.vram_label = Label("GPU VRAM: Measuring...", id="vram-label")
        self.status_label = Label("Status: Nominal", id="mem-status-label")

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-box"):
            yield Label("[b cyan]Memory Headroom & Consumption[/b cyan]")
            yield self.ram_label
            yield self.ram_bar
            yield self.vram_label
            yield self.vram_bar
            yield self.status_label

    def update_metrics(
        self,
        ram_rss_bytes: int,
        ram_avail_bytes: int,
        ram_total_bytes: int,
        vram_used_bytes: int = 0,
        vram_total_bytes: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        """Update progress bars and text labels from fresh metric sample."""
        # RAM calculations
        ram_pct = (
            min(100.0, max(0.0, (ram_rss_bytes / ram_total_bytes) * 100))
            if ram_total_bytes > 0
            else 0.0
        )
        self.ram_bar.progress = ram_pct
        rss_gb = ram_rss_bytes / (1024**3)
        avail_gb = ram_avail_bytes / (1024**3)
        total_gb = ram_total_bytes / (1024**3)
        self.ram_label.update(
            f"RAM: {rss_gb:.1f} GB RSS / {total_gb:.1f} GB Total ({avail_gb:.1f} GB Headroom)"
        )

        # GPU VRAM calculations
        if vram_total_bytes > 0:
            vram_pct = min(100.0, max(0.0, (vram_used_bytes / vram_total_bytes) * 100))
            self.vram_bar.progress = vram_pct
            v_used_gb = vram_used_bytes / (1024**3)
            v_tot_gb = vram_total_bytes / (1024**3)
            self.vram_label.update(f"GPU VRAM: {v_used_gb:.1f} GB Used / {v_tot_gb:.1f} GB Total")
            self.vram_bar.display = True
            self.vram_label.display = True
        else:
            self.vram_label.update("GPU VRAM: CPU-only mode (No discrete GPU)")
            self.vram_bar.display = False

        if warnings:
            self.status_label.update(f"[yellow b]⚠ Warning: {warnings[0]}[/yellow b]")
        else:
            self.status_label.update("[green]✔ Status: Headroom safe[/green]")


class SpeedPanel(Static):
    """Inference throughput, tokens per second, and latency panel."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.speed_digits = Label("0.0 tok/s", id="speed-digits", classes="bold-highlight")
        self.details_label = Label("TTFT: -  ·  Average: -", id="speed-details")

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-box"):
            yield Label("[b cyan]Generation Throughput & Latency[/b cyan]")
            yield self.speed_digits
            yield self.details_label

    def update_speed(
        self, current_tok_s: float, avg_tok_s: float = 0.0, ttft_s: float = 0.0
    ) -> None:
        self.speed_digits.update(f"[b green]{current_tok_s:.1f} tok/s[/b green]")
        ttft_str = f"{ttft_s:.2f}s" if ttft_s > 0 else "-"
        avg_str = f"{avg_tok_s:.1f} tok/s" if avg_tok_s > 0 else "-"
        self.details_label.update(f"TTFT: {ttft_str}  ·  Session Average: {avg_str}")


class PlanPanel(Static):
    """Execution plan specifications, layer placement, and bottleneck panel."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.content_label = Label("Loading execution plan...", id="plan-content")

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-box"):
            yield Label("[b cyan]Execution Plan & Hardware Assignment[/b cyan]")
            yield self.content_label

    def update_plan(self, plan: ExecutionPlan | None) -> None:
        if not plan:
            self.content_label.update("[dim]No active plan loaded.[/dim]")
            return

        gpu_l = sum(len(layers) for layers in plan.placement.gpu_layers.values())
        cpu_l = len(plan.placement.cpu_layers)
        verdict_color = "green" if plan.verdict == "COMFORTABLE" else "yellow"

        lines = [
            f"Plan ID: [b]{plan.id}[/b]  ·  Verdict: [{verdict_color} b]{plan.verdict}[/{verdict_color} b]",
            f"Layers: {gpu_l} on GPU, {cpu_l} on CPU  ·  KV Cache: {plan.kv.context} ({plan.kv.type_k})",
            f"Bottleneck: [b]{plan.speed.bottleneck.upper()}[/b]  ·  Expected Speed: {plan.speed.decode_tok_s_low}–{plan.speed.decode_tok_s_high} tok/s",
        ]
        self.content_label.update("\n".join(lines))


class EventsPanel(Static):
    """Timeline of governor events, auto-adaptations, and memory pressure warnings."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.events: list[str] = []
        self.events_label = Label("No governor events recorded.", id="events-label")

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-box"):
            yield Label("[b cyan]Governor Events & Adaptations[/b cyan]")
            yield self.events_label

    def add_event(self, text: str) -> None:
        import time

        timestamp = time.strftime("%H:%M:%S")
        self.events.append(f"[{timestamp}] {text}")
        if len(self.events) > 8:
            self.events.pop(0)
        self.events_label.update("\n".join(reversed(self.events)))
