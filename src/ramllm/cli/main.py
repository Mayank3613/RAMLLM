"""RAMLLM Command Line Interface."""

import json
import platform
from pathlib import Path
from typing import Annotated, Literal

import httpx
import typer
import uvicorn
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ramllm.api.server import create_app
from ramllm.backends import LlamaCppAdapter
from ramllm.backends.llamacpp.install import detect_llamacpp
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.governor.enforcement import get_current_enforcement_status
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.planner import create_execution_plans, recommend_models
from ramllm.core.profiler import load_synthetic_profile, probe_hardware
from ramllm.core.schemas import UserRequest
from ramllm.registry import (
    add_model,
    list_models,
    pull_model,
    remove_model,
    scan_directory,
    verify_model,
)
from ramllm.telemetry.bench import run_benchmark
from ramllm.telemetry.calibration import get_calibration_factor
from ramllm.util.parsing import parse_bytes, parse_context

app = typer.Typer(
    name="ramllm",
    help="Run bigger models on smaller machines — RAM-adaptive, GPU-aware LLM inference manager.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Manage local model catalog.", no_args_is_help=True)
app.add_typer(models_app, name="models")

console = Console()


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
                idx: parts[i]
                for i, idx in enumerate(indices)
                if i < len(parts) and parts[i] is not None
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


@app.command()
def profile(
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-run bandwidth benchmarks")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output raw JSON")] = False,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Load synthetic hardware profile")
    ] = None,
) -> None:
    """Profile hardware topology, memory headroom, and compute/memory bandwidth."""
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


@app.command()
def doctor() -> None:
    """Diagnose backend engines, accelerators, and OS memory enforcement levels."""
    console.print(Panel("[bold]RAMLLM Diagnostic Doctor[/bold]", box=box.ROUNDED))

    # 1. Backend check
    install = detect_llamacpp()
    if install:
        console.print(
            f"[green]✔[/green] Backend: [bold]llama.cpp[/bold] found at `{install.binary_path}`"
        )
        if install.version:
            console.print(f"  Version: {install.version}")
        caps = install.capabilities
        cap_list = []
        if caps.partial_gpu_offload:
            cap_list.append("GPU Offload")
        if caps.kv_quantization:
            cap_list.append("KV Quantization")
        if caps.per_tensor_placement:
            cap_list.append("MoE CPU Offload")
        if caps.flash_attention:
            cap_list.append("Flash Attention")
        console.print(f"  Capabilities: {', '.join(cap_list)}")
    else:
        console.print("[red]✘[/red] Backend: No llama.cpp found on PATH.")

    # 2. OS Memory Enforcement check
    status = get_current_enforcement_status()
    if status.can_enforce:
        console.print(
            f"[green]✔[/green] OS Enforcement: {status.mechanism} active ({status.detail})"
        )
    elif status.supported:
        console.print(f"[yellow]![/yellow] OS Enforcement: {status.mechanism} ({status.detail})")
    else:
        console.print(f"[yellow]![/yellow] OS Enforcement: {status.detail}")


@models_app.command("add")
def models_add(
    model_path: Annotated[Path, typer.Argument(help="Path to .gguf model file")],
) -> None:
    """Register a local GGUF model into the catalog."""
    with console.status(f"Analyzing {model_path}..."):
        entry = add_model(model_path)
    console.print(
        f"[green]✔ Added {entry.name} ({entry.quant}, {entry.total_weight_bytes / (1024**3):.1f} GB) to catalog.[/green]"
    )


@models_app.command("list")
def models_list() -> None:
    """List all registered models in the local catalog."""
    entries = list_models()
    if not entries:
        console.print("No models in catalog. Use `ramllm models add <path>` to add one.")
        return

    table = Table(title="Model Catalog", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Arch", style="white")
    table.add_column("Quant", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("MoE", style="yellow")
    table.add_column("Path", style="dim")

    for e in entries:
        table.add_row(
            e.name,
            e.arch,
            e.quant,
            f"{e.total_weight_bytes / (1024**3):.1f} GB",
            "Yes" if e.is_moe else "No",
            e.path,
        )
    console.print(table)


@models_app.command("scan")
def models_scan(
    directory: Annotated[Path, typer.Argument(help="Directory containing models")],
) -> None:
    """Scan directory recursively for GGUF models."""
    with console.status(f"Scanning {directory}..."):
        added = scan_directory(directory)
    console.print(f"[green]✔ Found and registered {len(added)} models.[/green]")


@models_app.command("remove")
def models_remove(
    name: Annotated[str, typer.Argument(help="Model name, path, or stem to remove")],
) -> None:
    """Remove a model entry from the local catalog."""
    removed = remove_model(name)
    if removed:
        console.print(f"[green]✔ Removed '{name}' from catalog.[/green]")
    else:
        console.print(f"[yellow]Model '{name}' not found in catalog.[/yellow]")


@models_app.command("verify")
def models_verify(
    model: Annotated[
        str | None,
        typer.Argument(help="Model name or file path to verify (or all if omitted)"),
    ] = None,
) -> None:
    """Compute and verify SHA-256 integrity hash of model file(s)."""
    targets = [model] if model else [e.name for e in list_models()]
    if not targets:
        console.print("No models in catalog to verify.")
        return

    for t in targets:
        with console.status(f"Verifying SHA-256 for {t}..."):
            matched, h, msg = verify_model(t)
        if matched:
            console.print(f"[green]✔ {t}[/green]: {msg} (SHA256: {h[:16]}...)")
        else:
            console.print(f"[red]✘ {t}[/red]: {msg}")


@models_app.command("pull")
def models_pull(
    target: Annotated[
        str,
        typer.Argument(help="Hugging Face model or direct URL (e.g. 'org/repo:file.gguf')"),
    ],
    dest: Annotated[Path | None, typer.Option("--dest", help="Destination directory")] = None,
    sha256: Annotated[str | None, typer.Option("--sha256", help="Expected SHA256 checksum")] = None,
) -> None:
    """Download a model with resumable streaming and checksum verification (offline-first)."""
    console.print(f"[bold blue]Downloading model from {target}...[/bold blue]")
    with console.status("[bold blue]Downloading model chunks..."):
        try:
            saved_path = pull_model(target, dest_dir=dest, expected_sha256=sha256)
            console.print(f"[green]✔ Downloaded and registered: {saved_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error downloading model: {e}[/red]")
            raise typer.Exit(code=1) from e


@app.command()
def recommend(
    vram: Annotated[
        str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB or 8GB,6GB)")
    ] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[
        str, typer.Option("--context", help="Target context window (e.g. 4096, 8k)")
    ] = "4096",
    prefer: Annotated[
        Literal["fastest", "balanced", "fit_at_any_cost"], typer.Option("--prefer")
    ] = "balanced",
    cpu_only: Annotated[bool, typer.Option("--cpu-only", help="Disable GPU offloading")] = False,
    no_kv_quant: Annotated[
        bool, typer.Option("--no-kv-quant", help="Disallow quantized KV cache")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Synthetic hardware profile")
    ] = None,
) -> None:
    """Recommend the best-quality models that fit within hardware budgets (offline catalog)."""
    gpu_indices, vram_dict = _parse_gpu_args(gpus, vram)
    ram_bytes = parse_bytes(ram)
    ctx_len = parse_context(context)

    if hw_json:
        hw = load_synthetic_profile(hw_json)
    else:
        hw = probe_hardware(quick=True)

    req = UserRequest(
        model="",
        vram_budget_bytes=vram_dict,
        ram_budget_bytes=ram_bytes,
        min_context=ctx_len,
        target_context=ctx_len,
        preference=prefer,
        use_gpu=not cpu_only,
        gpu_indices=gpu_indices,
        allow_kv_quant=not no_kv_quant,
    )

    recs = recommend_models(hw, req)

    if as_json:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in recs], indent=2))
        return

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


@app.command()
def plan(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    vram: Annotated[
        str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB or 8GB,6GB)")
    ] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[
        str, typer.Option("--context", help="Target context window (e.g. 4096, 8k)")
    ] = "4096",
    prefer: Annotated[
        Literal["fastest", "balanced", "fit_at_any_cost"], typer.Option("--prefer")
    ] = "balanced",
    cpu_only: Annotated[bool, typer.Option("--cpu-only", help="Disable GPU offloading")] = False,
    no_kv_quant: Annotated[
        bool, typer.Option("--no-kv-quant", help="Disallow quantized KV cache")
    ] = False,
    explain: Annotated[bool, typer.Option("--explain", help="Print detailed reasoning")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Output JSON")] = False,
    hw_json: Annotated[
        Path | None, typer.Option("--hw-json", help="Synthetic hardware profile")
    ] = None,
) -> None:
    """Plan model execution strategy given hardware constraints and memory budgets."""
    gpu_indices, vram_dict = _parse_gpu_args(gpus, vram)
    ram_bytes = parse_bytes(ram)
    ctx_len = parse_context(context)

    # 1. Profile hardware
    if hw_json:
        hw = load_synthetic_profile(hw_json)
    else:
        hw = probe_hardware(quick=True)

    # 2. Analyze model
    if not model.exists():
        console.print(f"[red]Error: Model file '{model}' not found.[/red]")
        raise typer.Exit(code=1)

    model_info = analyze_gguf(model)

    # 3. Formulate UserRequest
    req = UserRequest(
        model=str(model.resolve()),
        vram_budget_bytes=vram_dict,
        ram_budget_bytes=ram_bytes,
        min_context=ctx_len,
        target_context=ctx_len,
        preference=prefer,
        use_gpu=not cpu_only,
        gpu_indices=gpu_indices,
        allow_kv_quant=not no_kv_quant,
    )

    # 4. Generate and rank plans
    plans = create_execution_plans(model_info, hw, req)

    if as_json:
        data = [p.model_dump(mode="json") for p in plans]
        typer.echo(json.dumps(data, indent=2))
        return

    # Render plans
    table = Table(
        title=f"Execution Plans for {model.name} ({model_info.total_weight_bytes / (1024**3):.1f} GB)",
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


@app.command()
def run(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    vram: Annotated[str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB)")] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[str, typer.Option("--context", help="Target context window")] = "4096",
    prompt: Annotated[
        str | None, typer.Option("--prompt", "-p", help="Single prompt generation")
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Port for llama-server")] = 8080,
    live: Annotated[bool, typer.Option("--live", help="Monitor memory live with governor")] = False,
) -> None:
    """Run model with budget enforcement, buffer validation, and interactive chat."""
    if not model.exists():
        console.print(f"[red]Model file not found: {model}[/red]")
        raise typer.Exit(code=1)

    gpu_indices, vram_dict = _parse_gpu_args(gpus, vram)
    ram_bytes = parse_bytes(ram)
    ctx_len = parse_context(context)

    hw = probe_hardware(quick=True)
    model_info = analyze_gguf(model)

    req = UserRequest(
        model=str(model.resolve()),
        vram_budget_bytes=vram_dict,
        ram_budget_bytes=ram_bytes,
        min_context=ctx_len,
        target_context=ctx_len,
        gpu_indices=gpu_indices,
    )

    plans = create_execution_plans(model_info, hw, req)
    chosen_plan = plans[0]

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
        console.print(
            "[red]Error: llama-server executable not found on PATH. Run `ramllm doctor`.[/red]"
        )
        raise typer.Exit(code=1)

    with console.status("[bold blue]Launching llama-server and loading model buffers..."):
        try:
            proc = adapter.launch(chosen_plan, port=port, timeout_s=40.0)
        except Exception as e:
            console.print(f"[red]Launch failed: {e}[/red]")
            raise typer.Exit(code=1) from e

    governor: MemoryGovernor | None = None
    if live:
        governor = MemoryGovernor(proc, hw)
        governor.start()

    try:
        # Buffer sizes comparison table
        mem_report = proc.initial_memory_report
        pred_vram = sum(chosen_plan.memory.vram_peak_bytes.values()) / (1024**3)
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

        # Execute prompt if provided
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
            console.print(f"[red]Error from backend API: {res.text}[/red]")

    finally:
        if governor:
            governor.stop()
        console.print("[dim]Stopping backend engine...[/dim]")
        adapter.stop(proc)


@app.command()
def bench(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    plan_id: Annotated[
        str | None, typer.Option("--plan-id", help="Benchmark specific plan")
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Port for llama-server")] = 8085,
) -> None:
    """Run empirical benchmark on model and update calibration factor."""
    if not model.exists():
        console.print(f"[red]Model file not found: {model}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Benchmarking {model.name}...[/bold blue]")
    with console.status("[bold blue]Running standard prompt suite and sampling metrics..."):
        try:
            record = run_benchmark(model_path=model, plan_id=plan_id, port=port)
        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            raise typer.Exit(code=1) from e

    table = Table(title="Benchmark Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Predicted", style="yellow")
    table.add_column("Measured", style="green")

    table.add_row(
        "Decode Speed",
        f"~{record.predicted_decode_low:.1f}–{record.predicted_decode_high:.1f} tok/s",
        f"{record.measured_decode_tok_s:.1f} tok/s",
    )
    table.add_row(
        "Prefill Speed",
        f"~{record.predicted_prefill_low:.1f}–{record.predicted_prefill_high:.1f} tok/s",
        f"{record.measured_prefill_tok_s:.1f} tok/s",
    )
    table.add_row("Time to First Token (TTFT)", "-", f"{record.ttft_s:.3f} s")
    table.add_row("Peak RAM RSS", "-", f"{record.peak_ram_rss_bytes / (1024**2):.1f} MB")

    console.print(table)
    new_factor = get_calibration_factor(str(model.resolve()))
    console.print(
        f"[dim]Saved to bench.jsonl. Active machine calibration factor: {new_factor:.2f}x[/dim]"
    )


@app.command()
def serve(
    model: Annotated[Path, typer.Argument(help="Path to GGUF model")],
    vram: Annotated[str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB)")] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[str, typer.Option("--context", help="Target context window")] = "4096",
    port: Annotated[int, typer.Option("--port", help="Port to listen")] = 8080,
    host: Annotated[str, typer.Option("--host", help="Host interface to bind")] = "127.0.0.1",
) -> None:
    """Start OpenAI-compatible API server with active memory governor."""
    if not model.exists():
        console.print(f"[red]Model file not found: {model}[/red]")
        raise typer.Exit(code=1)

    gpu_indices, vram_dict = _parse_gpu_args(gpus, vram)
    ram_bytes = parse_bytes(ram)
    ctx_len = parse_context(context)

    app_instance = create_app(
        model_path=str(model.resolve()),
        vram_budget_bytes=vram_dict,
        ram_budget_bytes=ram_bytes,
        context=ctx_len,
        gpu_indices=gpu_indices,
    )

    console.print(f"[bold green]Starting RAMLLM API server on http://{host}:{port}...[/bold green]")
    console.print(f"OpenAI endpoint: http://{host}:{port}/v1/chat/completions")
    console.print(f"Metrics endpoint: http://{host}:{port}/ramllm/metrics")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
