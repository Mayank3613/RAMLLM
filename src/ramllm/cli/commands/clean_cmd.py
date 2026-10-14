"""Clean command for reclaiming disk space (ramllm clean)."""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ramllm.cli.ui.console import get_console, print_success
from ramllm.services.paths import (
    get_backends_dir,
    get_chats_dir,
    get_logs_dir,
    get_models_dir,
)


def _calc_dir_size(directory: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for a directory."""
    if not directory.exists():
        return 0, 0
    count = 0
    total = 0
    for p in directory.glob("**/*"):
        if p.is_file():
            count += 1
            total += p.stat().st_size
    return count, total


def clean_command(
    all_data: Annotated[
        bool,
        typer.Option(
            "--all", "-a", help="Clean all cached models, managed engines, chats, and logs"
        ),
    ] = False,
    models: Annotated[
        bool,
        typer.Option("--models", help="Delete downloaded model files"),
    ] = False,
    backends: Annotated[
        bool,
        typer.Option("--backends", help="Delete managed engine binaries"),
    ] = False,
    chats: Annotated[
        bool,
        typer.Option("--chats", help="Delete conversation transcript files"),
    ] = False,
    logs: Annotated[
        bool,
        typer.Option("--logs", help="Delete log files"),
    ] = False,
    partials: Annotated[
        bool,
        typer.Option("--partials", help="Delete only incomplete/partial downloads (.part, .tmp)"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show files and disk space to reclaim without deleting"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("-y", "--yes", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Free disk space by removing cached models, managed engines, or logs."""
    console = get_console()

    # If no specific flags selected, default to partials and logs
    clean_all = all_data
    clean_models = clean_all or models
    clean_backends = clean_all or backends
    clean_chats = clean_all or chats
    clean_logs = clean_all or logs
    clean_partials = clean_all or partials or (not (models or backends or chats or logs))

    targets: list[tuple[str, Path, int, int]] = []

    if clean_models:
        c, b = _calc_dir_size(get_models_dir())
        targets.append(("Downloaded Models", get_models_dir(), c, b))
    elif clean_partials:
        # Only partial downloads
        m_dir = get_models_dir()
        p_count = 0
        p_bytes = 0
        if m_dir.exists():
            for f in m_dir.glob("*"):
                if f.is_file() and (f.suffix in (".part", ".tmp")):
                    p_count += 1
                    p_bytes += f.stat().st_size
        targets.append(("Partial Downloads (.part, .tmp)", m_dir, p_count, p_bytes))

    if clean_backends:
        c, b = _calc_dir_size(get_backends_dir())
        targets.append(("Engine Binaries", get_backends_dir(), c, b))

    if clean_chats:
        c, b = _calc_dir_size(get_chats_dir())
        targets.append(("Chat Transcripts", get_chats_dir(), c, b))

    if clean_logs:
        c, b = _calc_dir_size(get_logs_dir())
        targets.append(("Log Files", get_logs_dir(), c, b))

    total_bytes = sum(t[3] for t in targets)
    total_files = sum(t[2] for t in targets)

    if total_bytes == 0 and total_files == 0:
        console.print("[dim]Everything is already clean. No files to remove.[/dim]")
        return

    table = Table(title="Disk Space Reclamation Plan", border_style="cyan")
    table.add_column("Category", style="bold cyan")
    table.add_column("Files", justify="right")
    table.add_column("Reclaimable Space", justify="right")

    for cat, _, count, b in targets:
        mb = b / (1024**2)
        size_str = f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"
        table.add_row(cat, str(count), size_str)

    tot_mb = total_bytes / (1024**2)
    tot_str = f"{tot_mb / 1024:.2f} GB" if tot_mb >= 1024 else f"{tot_mb:.1f} MB"
    table.add_section()
    table.add_row("Total", str(total_files), tot_str, style="bold green")

    console.print(table)

    if dry_run:
        console.print("[yellow][DRY-RUN] No files were deleted.[/yellow]")
        return

    confirmed = yes or typer.confirm(
        f"Proceed with deleting these {total_files} file(s)?", default=True
    )
    if not confirmed:
        console.print("[dim]Aborted by user.[/dim]")
        return

    # Delete files
    deleted_count = 0
    reclaimed_bytes = 0

    for cat, dir_path, _, _ in targets:
        if cat.startswith("Partial"):
            if dir_path.exists():
                for f in dir_path.glob("*"):
                    if f.is_file() and (f.suffix in (".part", ".tmp")):
                        sz = f.stat().st_size
                        f.unlink(missing_ok=True)
                        deleted_count += 1
                        reclaimed_bytes += sz
        elif dir_path.exists():
            for f in dir_path.glob("*"):
                if f.is_file():
                    sz = f.stat().st_size
                    f.unlink(missing_ok=True)
                    deleted_count += 1
                    reclaimed_bytes += sz
                elif f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)

    print_success(
        f"Successfully cleaned {deleted_count} file(s) and reclaimed {tot_str} of disk space."
    )
