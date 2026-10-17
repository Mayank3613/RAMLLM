"""Central Typer application for RAMLLM CLI with grouped help and typo matching."""

import json
import platform
from pathlib import Path
from typing import Annotated, Literal

import httpx
import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from ramllm.backends import LlamaCppAdapter
from ramllm.cli.commands.backend_cmd import backend_app
from ramllm.cli.commands.chat_cmd import chat_command
from ramllm.cli.commands.clean_cmd import clean_command
from ramllm.cli.commands.completion_cmd import completion_app
from ramllm.cli.commands.config_cmd import config_app
from ramllm.cli.commands.dashboard_cmd import dashboard_command
from ramllm.cli.commands.doctor_cmd import doctor_command
from ramllm.cli.commands.explain_cmd import explain_command
from ramllm.cli.commands.models_cmd import models_app
from ramllm.cli.commands.process_cmd import (
    logs_command,
    ps_command,
    replan_command,
    status_command,
    stop_command,
)
from ramllm.cli.commands.serve_cmd import serve_command
from ramllm.cli.commands.setup_cmd import setup_command
from ramllm.cli.commands.tour_cmd import tour_command
from ramllm.cli.commands.update_cmd import update_command
from ramllm.cli.commands.version_cmd import print_version_info, version_command
from ramllm.cli.typo import SuggestGroup
from ramllm.cli.ui.console import get_console, print_error, print_success
from ramllm.cli.ui.theme import set_plain_mode
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.governor.recovery import RecoveryExhaustedError, launch_with_recovery
from ramllm.core.planner import create_execution_plans, recommend_models
from ramllm.core.profiler import load_synthetic_profile, probe_hardware
from ramllm.core.schemas import ExecutionPlan, UserRequest
from ramllm.services.model_resolver import resolve_model_reference
from ramllm.services.plan_service import format_plain_summary, resolve_budgets, translate_mode
from ramllm.telemetry.bench import run_benchmark
from ramllm.telemetry.calibration import get_calibration_factor
from ramllm.util.parsing import parse_bytes, parse_context

app = typer.Typer(
    cls=SuggestGroup,
    name="ramllm",
    help="Run bigger models on smaller machines — RAM-adaptive, GPU-aware LLM inference manager.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-command groups
app.add_typer(models_app, name="models")
app.add_typer(backend_app, name="backend")
app.add_typer(config_app, name="config")
app.add_typer(completion_app, name="completion")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", "-v", help="Show RAMLLM and backend version")
    ] = False,
    plain: Annotated[
        bool, typer.Option("--plain", help="Disable rich formatting, emojis, and colors")
    ] = False,
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable verbose debug logging and stack traces")
    ] = False,
) -> None:
    """RAMLLM global configuration callback."""
    if version:
        print_version_info()
        raise typer.Exit()

    if plain:
        set_plain_mode(True)

    if ctx.invoked_subcommand is None:
        import sys

        if sys.stdin.isatty():
            from ramllm.cli.home.menu import run_home_menu
            from ramllm.cli.setup.state import is_setup_completed
            from ramllm.cli.setup.wizard import run_wizard
            from ramllm.registry import list_models

            if not is_setup_completed() and not list_models():
                run_wizard()
            else:
                run_home_menu()
        else:
            typer.echo(ctx.get_help())
            raise typer.Exit()


def _parse_gpu_args(
    gpus_str: str | None, vram_str: str | None
) -> tuple[list[int] | None, dict[int, int] | None]:
    """Parse comma-separated GPU indices and VRAM budgets."""
    gpu_indices: list[int] | None = None
    if gpus_str:
        gpu_indices = [int(x.strip()) for x in gpus_str.split(",") if x.strip()]

    vram_dict: dict[int, int] | None = None
    if vram_str:
        if "," in vram_str:
            parts = [parse_bytes(p.strip()) for p in vram_str.split(",")]
            indices = gpu_indices if gpu_indices else list(range(len(parts)))
            vram_dict = {
                idx: val
                for i, idx in enumerate(indices)
                if i < len(parts) and (val := parts[i]) is not None
            }
        else:
            single_val = parse_bytes(vram_str)
            if single_val is not None:
                if gpu_indices and len(gpu_indices) > 1:
                    vram_dict = dict.fromkeys(gpu_indices, single_val)
                else:
                    target_idx = gpu_indices[0] if (gpu_indices and len(gpu_indices) == 1) else 0
                    vram_dict = {target_idx: single_val}
    return gpu_indices, vram_dict


# GET STARTED
app.command("setup", rich_help_panel="GET STARTED")(setup_command)
app.command("doctor", rich_help_panel="GET STARTED")(doctor_command)
app.command("status", rich_help_panel="GET STARTED")(status_command)
app.command("tour", rich_help_panel="GET STARTED")(tour_command)


# SYSTEM
app.command("explain", rich_help_panel="SYSTEM")(explain_command)
app.command("ps", rich_help_panel="SYSTEM")(ps_command)
app.command("stop", rich_help_panel="SYSTEM")(stop_command)
app.command("logs", rich_help_panel="SYSTEM")(logs_command)
app.command("replan", rich_help_panel="SYSTEM")(replan_command)
app.command("dashboard", rich_help_panel="SYSTEM")(dashboard_command)
app.command("clean", rich_help_panel="SYSTEM")(clean_command)
app.command("update", rich_help_panel="SYSTEM")(update_command)
app.command("version", rich_help_panel="SYSTEM")(version_command)


@app.command(rich_help_panel="SYSTEM")
def profile(
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-run bandwidth benchmarks")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output raw JSON")] = False,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Load synthetic hardware profile")
    ] = None,
) -> None:
    """Profile hardware topology, memory headroom, and compute/memory bandwidth."""
    console = get_console()
    if hw_json:
        hw = load_synthetic_profile(hw_json)
    else:
        with console.status("[bold blue]Profiling hardware topology and memory bandwidth..."):
            hw = probe_hardware(refresh=refresh)

    if as_json:
        typer.echo(hw.model_dump_json(indent=2))
        return

    table = Table(title="Hardware Profile", box=box.ROUNDED)
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Specification", style="white")

    table.add_row("Operating System", f"{hw.os.upper()} ({platform.machine()})")
    table.add_row(
        "CPU",
        f"{hw.cpu_name} ({hw.cpu_cores_physical} physical, {hw.cpu_cores_logical} logical cores)",
    )
    table.add_row("CPU Features", ", ".join(hw.cpu_features) if hw.cpu_features else "standard")
    table.add_row(
        "System RAM",
        f"{hw.ram_total_bytes / (1024**3):.1f} GB total ({hw.ram_available_bytes / (1024**3):.1f} GB available)",
    )
    table.add_row("RAM Bandwidth", f"{hw.ram_bandwidth_bytes_s / (1024**3):.1f} GB/s")
    table.add_row(
        "Storage Drive",
        f"{hw.disk_kind.upper()} ({hw.disk_read_bandwidth_bytes_s / (1024**3):.1f} GB/s read)",
    )

    if hw.gpus:
        for g in hw.gpus:
            unified_str = " (Unified Memory)" if g.unified_memory else ""
            table.add_row(
                f"GPU {g.index}",
                f"{g.name} [{', '.join(g.api_support)}]{unified_str}\n"
                f"VRAM: {g.vram_total_bytes / (1024**3):.1f} GB total ({g.vram_free_bytes / (1024**3):.1f} GB free)\n"
                f"Bandwidth: {(g.vram_bandwidth_bytes_s or 0) / (1024**3):.1f} GB/s",
            )
    else:
        table.add_row("GPU", "[yellow]No discrete GPU detected (CPU mode)[/yellow]")

    console.print(table)


# MODELS
@app.command(rich_help_panel="MODELS")
def recommend(
    vram: Annotated[
        str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB or 8GB,6GB)")
    ] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    memory: Annotated[
        str | None,
        typer.Option("--memory", help="Total memory budget across RAM and VRAM (e.g. 12GB)"),
    ] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[
        str, typer.Option("--context", help="Target context window (e.g. 4096, 8k)")
    ] = "4096",
    mode: Annotated[
        Literal["fast", "balanced", "quality", "max"] | None,
        typer.Option("--mode", help="Performance and quality tradeoff mode"),
    ] = None,
    prefer: Annotated[
        Literal["fastest", "balanced", "fit_at_any_cost"] | None,
        typer.Option("--prefer", hidden=True),
    ] = None,
    cpu_only: Annotated[bool, typer.Option("--cpu-only", help="Disable GPU offloading")] = False,
    no_kv_quant: Annotated[
        bool, typer.Option("--no-kv-quant", help="Disallow quantized KV cache")
    ] = False,
    min_tps: Annotated[
        float | None, typer.Option("--min-tps", help="Minimum tolerable tokens/second threshold")
    ] = None,
    ubatch: Annotated[
        int, typer.Option("--ubatch", help="Micro-batch size for activation buffers")
    ] = 512,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Synthetic hardware profile")
    ] = None,
    cross_check_fit: Annotated[
        bool,
        typer.Option(
            "--cross-check-fit",
            help="Cross-check execution plan against upstream llama.cpp -fit advisory",
        ),
    ] = False,
) -> None:
    """Recommend the best-quality models that fit within hardware budgets (offline catalog)."""
    console = get_console()

    if hw_json:
        hw = load_synthetic_profile(hw_json)
    else:
        hw = probe_hardware(quick=True)

    budget = resolve_budgets(
        hw, ram_str=ram, vram_str=vram, memory_total_str=memory, preset="balanced", gpus_str=gpus
    )
    if mode:
        pref, allow_kv, ctx_len = translate_mode(mode, context, no_kv_quant)
    else:
        pref = prefer or "balanced"
        allow_kv = not no_kv_quant
        ctx_len = parse_context(context)

    gpu_indices = [g.index for g in hw.gpus] if hw.gpus else None
    if gpus:
        gpu_indices = [int(x.strip()) for x in gpus.split(",") if x.strip()]

    req = UserRequest(
        model="",
        vram_budget_bytes=budget.vram_bytes_per_gpu,
        ram_budget_bytes=budget.ram_bytes,
        min_context=ctx_len,
        target_context=ctx_len,
        preference=pref,
        use_gpu=not cpu_only,
        gpu_indices=gpu_indices,
        allow_kv_quant=allow_kv,
        min_tps=min_tps,
        ubatch_size=ubatch,
    )

    recs = recommend_models(hw, req)

    if as_json:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in recs], indent=2))
        return

    console.print(f"[dim]{budget.explanation}[/dim]\n")

    if not recs:
        console.print(
            "[yellow]No feasible models found in catalog for this hardware configuration.[/yellow]\n"
            "Use `ramllm models add <path>` or `ramllm models scan <dir>` to register models."
        )
        return

    table = Table(title="Model Recommendations", box=box.ROUNDED)
    table.add_column("Rank", style="bold cyan", width=6)
    table.add_column("Model", style="white")
    table.add_column("Params", style="magenta")
    table.add_column("Quant", style="green")
    table.add_column("Strategy", style="yellow")
    table.add_column("Speed", style="cyan")
    table.add_column("Verdict", style="bold green")
    table.add_column("Key Advantage", style="dim")

    for i, r in enumerate(recs, 1):
        bp = r.best_plan
        speed_str = f"~{bp.speed.decode_tok_s_low}–{bp.speed.decode_tok_s_high} tok/s"
        advantage = r.reasons[0] if r.reasons else ""
        table.add_row(
            f"#{i}",
            r.name,
            f"{r.n_params / 1e9:.1f}B",
            r.quant,
            bp.id,
            speed_str,
            bp.verdict,
            advantage,
        )

    console.print(table)


@app.command(rich_help_panel="MODELS")
def plan(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    vram: Annotated[
        str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB or 8GB,6GB)")
    ] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    memory: Annotated[
        str | None,
        typer.Option("--memory", help="Total memory budget across RAM and VRAM (e.g. 12GB)"),
    ] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[
        str, typer.Option("--context", help="Target context window (e.g. 4096, 8k)")
    ] = "4096",
    mode: Annotated[
        Literal["fast", "balanced", "quality", "max"] | None,
        typer.Option("--mode", help="Performance and quality tradeoff mode"),
    ] = None,
    prefer: Annotated[
        Literal["fastest", "balanced", "fit_at_any_cost"] | None,
        typer.Option("--prefer", hidden=True),
    ] = None,
    cpu_only: Annotated[bool, typer.Option("--cpu-only", help="Disable GPU offloading")] = False,
    no_kv_quant: Annotated[
        bool, typer.Option("--no-kv-quant", help="Disallow quantized KV cache")
    ] = False,
    min_tps: Annotated[
        float | None, typer.Option("--min-tps", help="Minimum tolerable tokens/second threshold")
    ] = None,
    ubatch: Annotated[
        int, typer.Option("--ubatch", help="Micro-batch size for activation buffers")
    ] = 512,
    details: Annotated[
        bool, typer.Option("--details", help="Display full candidate comparison table")
    ] = False,
    explain: Annotated[bool, typer.Option("--explain", help="Print detailed reasoning")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Synthetic hardware profile")
    ] = None,
    cross_check_fit: Annotated[
        bool,
        typer.Option(
            "--cross-check-fit",
            help="Cross-check execution plan against upstream llama.cpp -fit advisory",
        ),
    ] = False,
) -> None:
    """Plan model execution strategy given hardware constraints and memory budgets."""
    console = get_console()

    if hw_json:
        hw = load_synthetic_profile(hw_json)
    else:
        hw = probe_hardware(quick=True)

    try:
        resolved_model = resolve_model_reference(model, hw=hw)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e

    model_info = analyze_gguf(resolved_model)

    budget = resolve_budgets(
        hw, ram_str=ram, vram_str=vram, memory_total_str=memory, preset="balanced", gpus_str=gpus
    )

    if mode:
        pref, allow_kv, ctx_len = translate_mode(mode, context, no_kv_quant)
    else:
        pref = prefer or "balanced"
        allow_kv = not no_kv_quant
        ctx_len = parse_context(context)

    gpu_indices = [g.index for g in hw.gpus] if hw.gpus else None
    if gpus:
        gpu_indices = [int(x.strip()) for x in gpus.split(",") if x.strip()]

    req = UserRequest(
        model=str(resolved_model.resolve()),
        vram_budget_bytes=budget.vram_bytes_per_gpu,
        ram_budget_bytes=budget.ram_bytes,
        min_context=ctx_len,
        target_context=ctx_len,
        preference=pref,
        use_gpu=not cpu_only,
        gpu_indices=gpu_indices,
        allow_kv_quant=allow_kv,
        min_tps=min_tps,
        ubatch_size=ubatch,
    )

    plans = create_execution_plans(model_info, hw, req)

    if as_json:
        data = [p.model_dump(mode="json") for p in plans]
        if cross_check_fit:
            from ramllm.services.fit_service import run_fit_cross_check

            fit_res = run_fit_cross_check(resolved_model, ctx_len, plans[0])
            for d in data:
                d["fit_cross_check"] = fit_res.__dict__
        typer.echo(json.dumps(data, indent=2))
        return

    # Plain-English summary first
    summary = format_plain_summary(plans[0], resolved_model.name, hw)
    console.print(Panel(summary, title="Plan Summary", box=box.ROUNDED))
    console.print(f"[dim]{budget.explanation}[/dim]\n")

    if details or explain or plans[0].verdict == "INFEASIBLE":
        weight_desc = (
            f"Total: {model_info.total_weight_bytes / (1024**3):.1f} GB resident, Active: {model_info.active_weight_bytes / (1024**3):.1f} GB/tok"
            if model_info.is_moe
            else f"{model_info.total_weight_bytes / (1024**3):.1f} GB"
        )
        table = Table(
            title=f"Execution Plans for {resolved_model.name} ({weight_desc})",
            box=box.ROUNDED,
        )
        table.add_column("Rank", style="bold cyan", width=6)
        table.add_column("Strategy", style="white")
        table.add_column("VRAM Peak", style="magenta")
        table.add_column("RAM Peak", style="blue")
        table.add_column("Decode Speed", style="green")
        table.add_column("Bottleneck", style="yellow")
        table.add_column("Verdict", style="bold")

        for i, p in enumerate(plans):
            vram_gb = sum(p.memory.vram_peak_bytes.values()) / (1024**3)
            ram_gb = p.memory.ram_peak_bytes / (1024**3)
            vram_str = f"{vram_gb:.1f} GB" if vram_gb > 0 else "0.0 GB"
            ram_str = f"{ram_gb:.1f} GB"
            speed_str = f"~{p.speed.decode_tok_s_low:.1f}–{p.speed.decode_tok_s_high:.1f} tok/s"

            verdict_color = {
                "COMFORTABLE": "[green]COMFORTABLE[/green]",
                "CONSTRAINED": "[yellow]CONSTRAINED[/yellow]",
                "BATCH_ONLY": "[magenta]BATCH_ONLY[/magenta]",
                "INFEASIBLE": "[red]INFEASIBLE[/red]",
            }.get(p.verdict, p.verdict)

            table.add_row(
                f"#{i + 1}",
                p.id,
                vram_str,
                ram_str,
                speed_str,
                p.speed.bottleneck,
                verdict_color,
            )

        console.print(table)

        if explain or plans[0].verdict == "INFEASIBLE":
            console.print("\n[bold]Plan #1 Reasoning & Breakdown:[/bold]")
            for line in plans[0].explanation:
                console.print(f"  • {line}")
            if plans[0].warnings:
                console.print("\n[bold yellow]Remedies & Recommendations:[/bold yellow]")
                for w in plans[0].warnings:
                    console.print(f"  [yellow]• {w}[/yellow]")
    else:
        console.print(f"[dim]More detail: ramllm plan {model} --details[/dim]")

    if cross_check_fit:
        from ramllm.services.fit_service import render_fit_comparison_table, run_fit_cross_check

        fit_res = run_fit_cross_check(resolved_model, ctx_len, plans[0])
        render_fit_comparison_table(fit_res, console)


@app.command(rich_help_panel="MODELS")
def bench(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    plan_id: Annotated[
        str | None, typer.Option("--plan-id", help="Benchmark specific plan")
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Port for llama-server")] = 8085,
    export: Annotated[
        Path | None, typer.Option("--export", help="Export benchmark report to JSON")
    ] = None,
) -> None:
    """Run empirical benchmark on model and update calibration factor."""
    console = get_console()
    if not model.exists():
        print_error(f"Model file not found: {model}")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Benchmarking {model.name}...[/bold blue]")
    with console.status("[bold blue]Running standard prompt suite and sampling metrics..."):
        try:
            record = run_benchmark(model_path=model, plan_id=plan_id, port=port)
        except Exception as e:
            print_error(f"Benchmark failed: {e}")
            raise typer.Exit(code=1) from e

    table = Table(title="Benchmark Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Predicted Range", style="yellow")
    table.add_column("Measured", style="bold")
    table.add_column("Error %", justify="right")
    table.add_column("In Range?", justify="center")

    def fmt_err(err: float | None) -> str:
        if err is None:
            return "-"
        sign = "+" if err > 0 else ""
        return f"{sign}{err:.1f}%"

    def fmt_within(within: bool | None) -> str:
        if within is None:
            return "-"
        return "[green]✓ Yes[/green]" if within else "[red]✗ No[/red]"

    table.add_row(
        "Decode Speed",
        f"~{record.predicted_decode_low:.1f}–{record.predicted_decode_high:.1f} tok/s",
        f"{record.measured_decode_tok_s:.1f} tok/s",
        fmt_err(record.decode_error_pct),
        fmt_within(record.decode_within_predicted),
    )
    table.add_row(
        "Prefill Speed",
        f"~{record.predicted_prefill_low:.1f}–{record.predicted_prefill_high:.1f} tok/s",
        f"{record.measured_prefill_tok_s:.1f} tok/s",
        fmt_err(record.prefill_error_pct),
        fmt_within(record.prefill_within_predicted),
    )
    table.add_row("Time to First Token (TTFT)", "-", f"{record.ttft_s:.3f} s", "-", "-")
    table.add_row(
        "Peak RAM RSS",
        "-",
        f"{record.peak_ram_rss_bytes / (1024**2):.1f} MB",
        "-",
        "-",
    )

    console.print(table)
    new_factor = get_calibration_factor(
        model_path=str(model.resolve()),
        arch=record.arch,
        quant=record.quant,
        strategy=record.strategy,
    )
    console.print(
        f"[dim]Saved to bench.jsonl. Active machine calibration factor: {new_factor:.2f}x[/dim]"
    )

    if export:
        report_data = {
            "timestamp": record.timestamp,
            "model": {
                "name": model.name,
                "path": str(model.resolve()),
                "arch": record.arch,
                "quant": record.quant,
            },
            "strategy": record.strategy,
            "backend_version": record.backend_version,
            "benchmark": record.model_dump(),
            "active_calibration_factor": new_factor,
        }
        export.parent.mkdir(parents=True, exist_ok=True)
        with open(export, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        console.print(f"[green]Report exported to {export}[/green]")


# CHAT & RUN
app.command("chat", rich_help_panel="CHAT & RUN")(chat_command)


@app.command(rich_help_panel="CHAT & RUN")
def run(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    vram: Annotated[str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB)")] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    memory: Annotated[
        str | None,
        typer.Option("--memory", help="Total memory budget across RAM and VRAM (e.g. 12GB)"),
    ] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[str, typer.Option("--context", help="Target context window")] = "4096",
    mode: Annotated[
        Literal["fast", "balanced", "quality", "max"] | None,
        typer.Option("--mode", help="Performance and quality tradeoff mode"),
    ] = None,
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Single prompt generation")
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Port for llama-server")] = 8080,
    min_tps: Annotated[
        float | None, typer.Option("--min-tps", help="Minimum tolerable tokens/second threshold")
    ] = None,
    ubatch: Annotated[
        int, typer.Option("--ubatch", help="Micro-batch size for activation buffers")
    ] = 512,
    live: Annotated[bool, typer.Option("--live", help="Monitor memory live with governor")] = False,
    auto_retry: Annotated[
        bool,
        typer.Option(
            "--auto-retry/--no-auto-retry",
            help="Automatically adapt to smaller plans on failure or OOM",
        ),
    ] = True,
    max_retries: Annotated[
        int, typer.Option("--max-retries", help="Maximum auto-retry recovery attempts")
    ] = 3,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Path to synthetic hardware JSON profile")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Verify execution plan and parameters without starting engine"
        ),
    ] = False,
) -> None:
    """Run model with budget enforcement, buffer validation, and execution."""
    console = get_console()
    if hw_json:
        hw = load_synthetic_profile(hw_json)
    else:
        hw = probe_hardware(quick=True)

    try:
        resolved_model = resolve_model_reference(model, hw=hw)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e

    model_info = analyze_gguf(resolved_model)

    budget = resolve_budgets(
        hw, ram_str=ram, vram_str=vram, memory_total_str=memory, preset="balanced", gpus_str=gpus
    )

    if mode:
        pref, _, ctx_len = translate_mode(mode, context)
    else:
        pref = "balanced"
        ctx_len = parse_context(context)

    gpu_indices = [g.index for g in hw.gpus] if hw.gpus else None
    if gpus:
        gpu_indices = [int(x.strip()) for x in gpus.split(",") if x.strip()]

    req = UserRequest(
        model=str(resolved_model.resolve()),
        vram_budget_bytes=budget.vram_bytes_per_gpu,
        ram_budget_bytes=budget.ram_bytes,
        min_context=ctx_len,
        target_context=ctx_len,
        preference=pref,
        gpu_indices=gpu_indices,
        min_tps=min_tps,
        ubatch_size=ubatch,
    )

    plans = create_execution_plans(model_info, hw, req)
    chosen_plan = plans[0]

    # Plain-English summary first
    summary = format_plain_summary(chosen_plan, resolved_model.name, hw)
    console.print(Panel(summary, title="Plan Summary", box=box.ROUNDED))
    console.print(f"[dim]{budget.explanation}[/dim]\n")

    if dry_run:
        console.print("[yellow][DRY-RUN] Execution plan verified. Skipping launch.[/yellow]")
        return

    if chosen_plan.verdict == "INFEASIBLE":
        console.print(
            "[red bold]Cannot launch: Model is INFEASIBLE under requested budgets.[/red bold]"
        )
        for w in chosen_plan.warnings:
            console.print(f"  • {w}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Starting execution plan '{chosen_plan.id}'...[/bold green]")
    console.print(
        f"Predicted Speed: {chosen_plan.speed.decode_tok_s_low}–{chosen_plan.speed.decode_tok_s_high} tok/s | Bottleneck: {chosen_plan.speed.bottleneck}"
    )

    adapter = LlamaCppAdapter()
    install = adapter.detect()
    if not install:
        print_error(
            "llama-server executable not found. Run `ramllm backend install` or `ramllm doctor --fix`."
        )
        raise typer.Exit(code=1)

    def _on_retry(attempt: int, next_plan: ExecutionPlan, cause: str) -> None:
        console.print(f"\n[bold yellow][!] Launch failed: {cause}[/bold yellow]")
        console.print(
            f"[cyan]↻ Automatically adapting to smaller plan (attempt {attempt}/{max_retries}): '{next_plan.id}'...[/cyan]"
        )
        for w in next_plan.warnings:
            console.print(f"  [dim]• {w}[/dim]")

    try:
        with console.status("[bold blue]Launching backend and verifying memory buffers..."):
            proc, active_plan, recovery_logs = launch_with_recovery(
                adapter=adapter,
                plan=chosen_plan,
                model=model_info,
                hw=hw,
                req=req,
                port=port,
                timeout_s=40.0,
                max_retries=max_retries,
                auto_retry=auto_retry,
                on_retry=_on_retry,
            )
    except RecoveryExhaustedError as err:
        console.print(f"\n[bold red]Launch failed and recovery exhausted:[/bold red] {err.message}")
        if err.remedies:
            console.print("\n[bold yellow]Recommended remedies:[/bold yellow]")
            for r in err.remedies:
                console.print(f"  [yellow]• {r}[/yellow]")
        raise typer.Exit(code=1) from err
    except Exception as e:
        print_error(f"Launch failed: {e}")
        raise typer.Exit(code=1) from e

    governor: MemoryGovernor | None = None
    if live:
        governor = MemoryGovernor(proc, hw)
        governor.start()

    try:
        if recovery_logs:
            print_success(f"Session recovered successfully with reduced plan '{active_plan.id}'")
        mem_report = proc.initial_memory_report
        pred_vram = sum(active_plan.memory.vram_peak_bytes.values()) / (1024**3)
        actual_ram_mb = mem_report.ram_rss_bytes / (1024**2)

        console.print("\n[bold]Memory Verification (Predicted vs Actual):[/bold]")
        console.print(f"  • Predicted VRAM: {pred_vram:.1f} GB")
        console.print(f"  • Actual Process RSS: {actual_ram_mb:.1f} MB")
        if mem_report.model_buffer_bytes:
            console.print(
                f"  • Logged Model Buffer: {mem_report.model_buffer_bytes / (1024**2):.1f} MB"
            )
        if mem_report.kv_buffer_bytes:
            console.print(
                f"  • Logged KV Cache Buffer: {mem_report.kv_buffer_bytes / (1024**2):.1f} MB"
            )

        if governor:
            metrics = governor.get_latest_metrics()
            console.print(
                f"  • Active Governor Sampling: Available RAM = {metrics.ram_available_bytes / (1024**2):.1f} MB"
            )

        test_prompt = prompt or "Hello! Introduce yourself briefly."
        console.print(f"\n[cyan]> {test_prompt}[/cyan]")

        endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": test_prompt}],
            "max_tokens": 128,
            "stream": False,
        }
        res = httpx.post(endpoint, json=payload, timeout=60.0)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"]
            console.print(f"[white]{content}[/white]\n")
        else:
            print_error(f"Error from backend API: {res.text}")

    finally:
        if governor:
            governor.stop()
        console.print("[dim]Stopping backend engine...[/dim]")
        adapter.stop(proc)


app.command("serve", rich_help_panel="CHAT & RUN")(serve_command)
