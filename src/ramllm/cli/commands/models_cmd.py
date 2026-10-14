"""CLI commands for managing, searching, inspecting, and pulling models."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.table import Table

from ramllm.cli.ui.console import get_console, print_error, print_success
from ramllm.cli.ui.errors import (
    ChecksumMismatchError,
    DiskSpaceExhaustedError,
    DownloadFailedError,
    GatedModelAuthError,
    render_error,
)
from ramllm.core.profiler import probe_hardware
from ramllm.registry import (
    add_model,
    list_models,
    remove_model,
    scan_directory,
    verify_model,
)
from ramllm.services.model_resolver import _download_with_progress
from ramllm.services.model_service import (
    clean_models_dir,
    estimate_model_fit,
    get_curated_model,
    get_models_disk_usage,
    pick_best_quant,
    search_curated_models,
)
from ramllm.services.paths import get_models_dir

models_app = typer.Typer(
    help="Search, download, inspect, and manage local models.",
    no_args_is_help=True,
    rich_help_panel="MODELS",
)


@models_app.command("search")
def models_search(
    query: Annotated[
        str | None, typer.Argument(help="Search query for name, family, or description")
    ] = None,
    tag: Annotated[
        str | None, typer.Option("--tag", "-t", help="Filter by tag (e.g. chat, coding, small)")
    ] = None,
    fits: Annotated[
        bool, typer.Option("--fits", help="Only show models that fit your computer's memory")
    ] = False,
) -> None:
    """Search the curated model catalog with hardware fit predictions."""
    console = get_console()
    hw = probe_hardware(quick=True)

    matches = search_curated_models(query=query, tag=tag)
    if not matches:
        console.print("[yellow]No models matched your search query.[/yellow]")
        return

    table = Table(title="Curated Model Catalog", box=box.ROUNDED)
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Model Name", style="white")
    table.add_column("Params", style="magenta")
    table.add_column("Fits?", style="bold")
    table.add_column("Tags", style="dim")
    table.add_column("Notes", style="white")

    shown = 0
    for m in matches:
        verdict, plain_fit, color = estimate_model_fit(m, hw)
        if fits and verdict == "INFEASIBLE":
            continue

        symbol = "✔" if verdict == "COMFORTABLE" else ("~" if verdict == "CONSTRAINED" else "✘")
        fit_str = f"[{color}]{symbol} {plain_fit}[/{color}]"
        params_str = f"{m.params_b:.1f}B" if m.params_b >= 1.0 else f"{int(m.params_b * 1000)}M"
        tags_str = ", ".join(m.tags[:3])

        table.add_row(m.id, m.name, params_str, fit_str, tags_str, m.notes)
        shown += 1

    if shown == 0 and fits:
        console.print(
            "[yellow]No models matching the query fit in your current memory budget.[/yellow]"
        )
        return

    console.print(table)
    console.print(
        f"[dim]Showing {shown} models. Inspect details with `ramllm models info <ID>`.[/dim]"
    )


@models_app.command("info")
def models_info(
    model: Annotated[str, typer.Argument(help="Model friendly ID, catalog name, or path")],
) -> None:
    """Inspect model details, available quantizations, and compatibility."""
    console = get_console()
    hw = probe_hardware(quick=True)

    curated = get_curated_model(model)
    if not curated:
        # Check local catalog
        for entry in list_models():
            if (
                entry.name.lower() == model.lower()
                or Path(entry.path).name.lower() == model.lower()
            ):
                console.print(f"[bold cyan]{entry.name}[/bold cyan] (Local Catalog Model)")
                console.print(f"Path: {entry.path}")
                console.print(f"Architecture: {entry.arch}")
                console.print(f"Quantization: {entry.quant}")
                console.print(f"Size: {entry.total_weight_bytes / (1024**3):.2f} GB")
                console.print(f"SHA-256: {entry.sha256 or '(not computed)'}")
                return
        print_error(f"Model '{model}' not found in curated catalog or local registry.")
        raise typer.Exit(code=1)

    console.print(f"[bold cyan]{curated.name}[/bold cyan] ({curated.id})\n")
    console.print(f"[white]{curated.notes}[/white]\n")
    console.print(f"  • Family: {curated.family}")
    params_str = (
        f"{curated.params_b:.1f}B"
        if curated.params_b >= 1.0
        else f"{int(curated.params_b * 1000)}M"
    )
    console.print(f"  • Parameters: {params_str} ({curated.architecture.upper()})")
    if curated.active_params_b:
        console.print(f"  • Active Parameters: {curated.active_params_b:.1f}B")
    console.print(f"  • License: {curated.license}")
    console.print(f"  • Recommended Min RAM: {curated.recommended_min_memory_gb} GB")
    console.print(f"  • Tags: {', '.join(curated.tags)}")

    table = Table(title=f"Available Quantizations for {curated.name}", box=box.ROUNDED)
    table.add_column("Quant", style="cyan", width=10)
    table.add_column("Size", style="green", width=10)
    table.add_column("Fits on Your Machine?", style="bold")
    table.add_column("Hugging Face File", style="dim")

    for f in curated.files:
        verdict, plain_fit, color = estimate_model_fit(curated, hw, quant=f.quant)
        symbol = "✔" if verdict == "COMFORTABLE" else ("~" if verdict == "CONSTRAINED" else "✘")
        fit_str = f"[{color}]{symbol} {plain_fit}[/{color}]"
        size_str = f"{f.size_bytes / (1024**3):.1f} GB" if f.size_bytes > 0 else "N/A"
        table.add_row(f.quant, size_str, fit_str, f.hf_file)

    console.print()
    console.print(table)
    console.print(f"\n[dim]To pull this model: `ramllm models pull {curated.id}`[/dim]")


@models_app.command("pull")
def models_pull(
    target: Annotated[
        str,
        typer.Argument(help="Model friendly ID (e.g. qwen2.5-7b) or Hugging Face repo:file.gguf"),
    ],
    quant: Annotated[
        str | None,
        typer.Option("--quant", "-q", help="Specific quantization (e.g. Q4_K_M, Q5_K_M)"),
    ] = None,
    dest: Annotated[Path | None, typer.Option("--dest", help="Destination directory")] = None,
    sha256: Annotated[str | None, typer.Option("--sha256", help="Expected SHA256 checksum")] = None,
) -> None:
    """Download a model with automatic quant selection, resume support, and checksum verification."""
    console = get_console()
    models_dir = dest or get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if target is a curated model ID
    model_id = target
    req_quant = quant
    if "@" in target:
        model_id, req_quant = target.split("@", 1)

    curated = get_curated_model(model_id)
    if curated:
        hw = probe_hardware(quick=True)
        if req_quant:
            matched = next((f for f in curated.files if f.quant.upper() == req_quant.upper()), None)
            if not matched:
                available = ", ".join(f.quant for f in curated.files)
                print_error(
                    f"Quantization '{req_quant}' not available for '{curated.id}'. Available: {available}"
                )
                raise typer.Exit(code=2)
            target_file = matched
        else:
            target_file = pick_best_quant(curated, hw)
            console.print(
                f"[dim]Auto-selected optimal quantization '{target_file.quant}' for your system.[/dim]"
            )

        hf_target = f"{target_file.hf_repo}:{target_file.hf_file}"
        expected_sha = target_file.sha256 or sha256
        expected_size = target_file.size_bytes
    else:
        hf_target = target
        expected_sha = sha256
        expected_size = 0

    try:
        saved_path = _download_with_progress(
            hf_target=hf_target,
            dest_dir=models_dir,
            expected_sha256=expected_sha,
            expected_size=expected_size,
            console=console,
        )
        print_success(f"Downloaded and registered: {saved_path}")
    except (
        DiskSpaceExhaustedError,
        GatedModelAuthError,
        ChecksumMismatchError,
        DownloadFailedError,
    ) as e:
        render_error(e, console=console)
        raise typer.Exit(code=e.exit_code) from e
    except Exception as e:
        print_error(f"Error downloading model: {e}")
        raise typer.Exit(code=1) from e


@models_app.command("list")
def models_list(
    as_json: Annotated[bool, typer.Option("--json", help="Output models as JSON")] = False,
) -> None:
    """List all registered models in the local catalog."""
    console = get_console()
    entries = list_models()
    if as_json:
        typer.echo(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return

    if not entries:
        console.print("No models in catalog. Use `ramllm models add <path>` to add one.")
        return

    hw = probe_hardware(quick=True)
    table = Table(title="Model Catalog", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Arch", style="white")
    table.add_column("Quant", style="magenta")
    table.add_column("Size", style="green")
    table.add_column("Fits?", style="bold")
    table.add_column("MoE", style="yellow")
    table.add_column("Path", style="dim")

    for e in entries:
        # Simple fit estimation for registered model
        total_ram = hw.ram_available_bytes
        total_vram = sum(g.vram_total_bytes for g in hw.gpus) if hw.gpus else 0
        fits_str = (
            "[green]✔ runs[/green]"
            if e.total_weight_bytes <= (total_vram + total_ram) * 0.85
            else "[red]✘ heavy[/red]"
        )

        table.add_row(
            e.name,
            e.arch,
            e.quant,
            f"{e.total_weight_bytes / (1024**3):.1f} GB",
            fits_str,
            "Yes" if e.is_moe else "No",
            e.path,
        )
    console.print(table)


@models_app.command("add")
def models_add(
    model_path: Annotated[Path, typer.Argument(help="Path to .gguf model file")],
) -> None:
    """Register a local GGUF model into the catalog."""
    console = get_console()
    with console.status(f"Analyzing {model_path}..."):
        entry = add_model(model_path)
    console.print(
        f"[green]✔ Added {entry.name} ({entry.quant}, {entry.total_weight_bytes / (1024**3):.1f} GB) to catalog.[/green]"
    )


@models_app.command("scan")
def models_scan(
    directory: Annotated[Path, typer.Argument(help="Directory containing models")],
) -> None:
    """Scan directory recursively for GGUF models."""
    console = get_console()
    with console.status(f"Scanning {directory}..."):
        added = scan_directory(directory)
    console.print(f"[green]✔ Found and registered {len(added)} models.[/green]")


@models_app.command("remove")
def models_remove(
    name: Annotated[str, typer.Argument(help="Model name, path, or stem to remove")],
) -> None:
    """Remove a model entry from the local catalog."""
    console = get_console()
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
    console = get_console()
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


@models_app.command("du")
def models_du() -> None:
    """Show disk space usage of stored models and available drive capacity."""
    console = get_console()
    usage = get_models_disk_usage()

    table = Table(title="Model Disk Usage", box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Kind", style="white")
    table.add_column("Size", style="bold green")

    for m in usage["model_files"]:
        table.add_row(m["name"], "Model (GGUF)", f"{m['size_bytes'] / (1024**3):.2f} GB")

    for p in usage["partial_files"]:
        table.add_row(
            p["name"], "[yellow]Partial download[/yellow]", f"{p['size_bytes'] / (1024**3):.2f} GB"
        )

    console.print(table)
    console.print(
        f"\n[bold]Total models directory size:[/bold] {usage['total_used_bytes'] / (1024**3):.2f} GB"
    )
    console.print(
        f"[bold]Free disk space on volume:[/bold] {usage['free_disk_bytes'] / (1024**3):.1f} GB"
    )
    console.print(f"[dim]Directory: {usage['models_dir']}[/dim]")


@models_app.command("clean")
def models_clean(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
) -> None:
    """Clean up incomplete downloads and temporary files."""
    console = get_console()
    usage = get_models_disk_usage()
    partials = usage["partial_files"]

    if not partials:
        console.print(
            "[green]✔ No incomplete or temporary files found in models directory.[/green]"
        )
        return

    total_mb = sum(p["size_bytes"] for p in partials) / (1024**2)
    console.print(f"Found {len(partials)} incomplete download files ({total_mb:.1f} MB):")
    for p in partials:
        console.print(f"  • {p['name']} ({p['size_bytes'] / (1024**2):.1f} MB)")

    if not yes:
        confirmed = typer.confirm("Delete these files to reclaim disk space?", default=True)
        if not confirmed:
            console.print("[dim]Aborted clean.[/dim]")
            return

    count, reclaimed = clean_models_dir(remove_partials=True)
    console.print(
        f"[green]✔ Removed {count} incomplete files, reclaimed {reclaimed / (1024**2):.1f} MB.[/green]"
    )
