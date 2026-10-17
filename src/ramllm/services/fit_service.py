"""Cross-check service for comparing RAMLLM plans against upstream llama.cpp -fit advisory."""

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from ramllm.backends.base import BackendInstall
from ramllm.core.schemas import ExecutionPlan


@dataclass
class FitComparisonResult:
    """Structured comparison between RAMLLM execution plan and upstream llama.cpp -fit output."""

    available: bool
    binary_path: str | None = None
    target_model: str = ""
    target_context: int = 0
    fit_gpu_layers: int | None = None
    fit_context: int | None = None
    ramllm_gpu_layers: int = 0
    ramllm_context: int = 0
    ramllm_kv_type: str = "f16"
    ramllm_speed_forecast: str = ""
    ramllm_verdict: str = ""
    differences: list[str] = field(default_factory=list)
    advisory_notes: list[str] = field(default_factory=list)
    raw_output: str = ""
    error_message: str | None = None


def parse_fit_params_output(text: str) -> tuple[int | None, int | None]:
    """Extract suggested GPU layers and context length from llama-fit-params output."""
    gpu_layers: int | None = None
    context: int | None = None

    ngl_match = re.search(
        r"(?:-ngl|--gpu-layers|n_gpu_layers\s*=|fit_params:.*-ngl)\s*[:=]?\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if ngl_match:
        gpu_layers = int(ngl_match.group(1))

    ctx_match = re.search(
        r"(?:-c|--ctx-size|ctx_size\s*=|fit_params:.*-c)\s*[:=]?\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if ctx_match:
        context = int(ctx_match.group(1))

    return gpu_layers, context


def run_fit_cross_check(
    model_path: Path | str,
    target_ctx: int,
    plan: ExecutionPlan,
    install: BackendInstall | None = None,
) -> FitComparisonResult:
    """Execute llama-fit-params non-interactively and compare suggested placement with RAMLLM plan."""
    p = Path(model_path).resolve()
    target_model_name = p.name

    ramllm_ngl = sum(len(layers) for layers in plan.placement.gpu_layers.values())
    ramllm_ctx = plan.kv.context
    ramllm_kv = f"{plan.kv.type_k}/{plan.kv.type_v}"
    speed_forecast = f"~{plan.speed.decode_tok_s_low:.1f}–{plan.speed.decode_tok_s_high:.1f} tok/s"

    # Locate binary
    bin_path: str | None = None
    if install and install.capabilities.fit_params_binary_path:
        bin_path = install.capabilities.fit_params_binary_path
    elif install and install.capabilities.has_fit_params:
        bin_path = shutil.which("llama-fit-params")

    if not bin_path:
        bin_path = shutil.which("llama-fit-params")

    if not bin_path:
        return FitComparisonResult(
            available=False,
            target_model=target_model_name,
            target_context=target_ctx,
            ramllm_gpu_layers=ramllm_ngl,
            ramllm_context=ramllm_ctx,
            ramllm_kv_type=ramllm_kv,
            ramllm_speed_forecast=speed_forecast,
            ramllm_verdict=plan.verdict,
            error_message="llama-fit-params executable not found. Install llama.cpp upstream utilities (e.g. brew install llama.cpp) to run live cross-checks.",
            advisory_notes=[
                "RAMLLM plans forward with analytical memory & speed modeling, whereas upstream -fit statically downscales parameters at launch.",
                "RAMLLM includes dynamic runtime memory governance, preventing out-of-memory crashes if system load shifts.",
            ],
        )

    # Run tool non-interactively
    cmd = [bin_path, "-m", str(p), "-c", str(target_ctx), "-fitp", "on"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        raw = f"{res.stdout}\n{res.stderr}"
    except Exception as e:
        return FitComparisonResult(
            available=False,
            binary_path=bin_path,
            target_model=target_model_name,
            target_context=target_ctx,
            ramllm_gpu_layers=ramllm_ngl,
            ramllm_context=ramllm_ctx,
            ramllm_kv_type=ramllm_kv,
            ramllm_speed_forecast=speed_forecast,
            ramllm_verdict=plan.verdict,
            error_message=f"Execution of {bin_path} failed: {e}",
        )

    fit_ngl, fit_c = parse_fit_params_output(raw)

    differences: list[str] = []
    if fit_ngl is not None:
        if ramllm_ngl > fit_ngl:
            differences.append(
                f"RAMLLM places more layers on GPU ({ramllm_ngl} vs {fit_ngl}) due to intelligent buffer sizing and KV quantization."
            )
        elif ramllm_ngl < fit_ngl:
            differences.append(
                f"RAMLLM reserves safer OS margins ({ramllm_ngl} vs {fit_ngl} layers) to protect against system memory paging freezes."
            )
        else:
            differences.append(f"Both tools recommend offloading {ramllm_ngl} layers to GPU.")
    else:
        differences.append("Upstream -fit did not suggest a specific layer split for this model.")

    if plan.kv.type_k != "f16":
        differences.append(
            f"RAMLLM enables KV-cache quantization ({ramllm_kv}), saving VRAM/RAM compared to upstream -fit's unquantized FP16 default."
        )

    differences.append(
        f"RAMLLM projects expected inference speed ({speed_forecast} [{plan.verdict}]), whereas -fit operates blind to performance consequences."
    )

    advisory_notes = [
        "Advisory: Upstream llama.cpp -fit only scales parameters at engine boot; RAMLLM monitors and adapts memory live during inference.",
    ]

    return FitComparisonResult(
        available=True,
        binary_path=bin_path,
        target_model=target_model_name,
        target_context=target_ctx,
        fit_gpu_layers=fit_ngl,
        fit_context=fit_c,
        ramllm_gpu_layers=ramllm_ngl,
        ramllm_context=ramllm_ctx,
        ramllm_kv_type=ramllm_kv,
        ramllm_speed_forecast=speed_forecast,
        ramllm_verdict=plan.verdict,
        differences=differences,
        advisory_notes=advisory_notes,
        raw_output=raw,
    )


def render_fit_comparison_table(result: FitComparisonResult, console: Console) -> None:
    """Print a formatted Rich comparison table between RAMLLM and upstream -fit."""
    table = Table(title="llama.cpp -fit Cross-Check Comparison (Advisory)", box=box.ROUNDED)
    table.add_column("Dimension", style="cyan")
    table.add_column("Upstream llama.cpp (-fit)", style="yellow")
    table.add_column("RAMLLM Planned Plan", style="green")
    table.add_column("Key Difference & Advantage", style="white")

    if not result.available:
        fit_gpu_desc = "Tool unavailable"
        fit_kv_desc = "f16 (default)"
        fit_perf_desc = "No forecast"
        fit_gov_desc = "Static downscale only"
    else:
        fit_gpu_desc = (
            f"{result.fit_gpu_layers} layers"
            if result.fit_gpu_layers is not None
            else "Dynamic auto-fit"
        )
        fit_kv_desc = "f16 (unquantized)"
        fit_perf_desc = "None (speed blind)"
        fit_gov_desc = "Static downscale at boot"

    table.add_row(
        "GPU Offload (-ngl)",
        fit_gpu_desc,
        f"{result.ramllm_gpu_layers} layers",
        "RAMLLM profiles memory topology and boundaries before loading",
    )
    table.add_row(
        "KV-Cache Format",
        fit_kv_desc,
        result.ramllm_kv_type,
        "RAMLLM auto-quantizes KV when beneficial to unlock higher contexts",
    )
    table.add_row(
        "Speed Forecast",
        fit_perf_desc,
        f"{result.ramllm_speed_forecast} [{result.ramllm_verdict}]",
        "RAMLLM predicts tokens/sec and classifies slowness thresholds",
    )
    table.add_row(
        "OOM Protection",
        fit_gov_desc,
        "Live Governor (1 Hz sampling + recovery)",
        "RAMLLM actively detects memory spikes and automatically recovers",
    )

    console.print(table)

    if result.error_message:
        console.print(f"[yellow]Note: {result.error_message}[/yellow]")

    if result.differences:
        console.print("[bold]Key Comparison Highlights:[/bold]")
        for diff in result.differences:
            console.print(f"  • {diff}")
