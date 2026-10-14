"""30-second quick tour of RAMLLM concepts (ramllm tour)."""

from rich import box
from rich.panel import Panel
from rich.table import Table

from ramllm.cli.ui.console import get_console


def tour_command() -> None:
    """Take a 30-second tour of memory tiers, verdicts, and key RAMLLM commands."""
    console = get_console()

    console.print(
        Panel(
            "[bold cyan]Welcome to RAMLLM[/bold cyan] — Run bigger models on smaller machines\n"
            "[dim]A 30-second tour of how RAMLLM manages your memory and performance.[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )

    # Section 1: Memory Tiers
    table_tiers = Table(title="1. Memory Tiers", box=box.SIMPLE)
    table_tiers.add_column("Tier", style="bold magenta", width=12)
    table_tiers.add_column("Speed", style="green", width=14)
    table_tiers.add_column("How RAMLLM Uses It", style="white")

    table_tiers.add_row(
        "VRAM",
        "Lightning fast",
        "Top priority. Holds model weights, attention compute, and active KV cache.",
    )
    table_tiers.add_row(
        "System RAM",
        "Fast & spacious",
        "Large memory pool. Safely hosts offloaded layers and fallback context.",
    )
    table_tiers.add_row(
        "Disk / Swap",
        "Slow safety net",
        "Emergency storage. RAMLLM actively monitors and avoids swapping to prevent freezes.",
    )
    console.print(table_tiers)
    console.print()

    # Section 2: Plan Verdicts
    table_verdicts = Table(title="2. Plan Verdicts", box=box.SIMPLE)
    table_verdicts.add_column("Verdict", width=15)
    table_verdicts.add_column("User Experience", style="white")

    table_verdicts.add_row(
        "[bold green]COMFORTABLE[/bold green]",
        "Runs smoothly at top speed with plenty of safety margin.",
    )
    table_verdicts.add_row(
        "[bold yellow]CONSTRAINED[/bold yellow]",
        "Works reliably, but operates near memory limits or with split layers.",
    )
    table_verdicts.add_row(
        "[bold magenta]BATCH_ONLY[/bold magenta]",
        "Runs primarily on CPU/RAM. Great for overnight or automated jobs, slow for interactive chat.",
    )
    table_verdicts.add_row(
        "[bold red]INFEASIBLE[/bold red]",
        "Exceeds safe system limits. Blocked to protect your operating system from crashing.",
    )
    console.print(table_verdicts)
    console.print()

    # Section 3: Essential Commands
    console.print("[bold]3. Key Commands to Try Next:[/bold]")
    console.print(
        "  • [cyan]ramllm setup[/cyan]       — Guided hardware detection & first model download"
    )
    console.print(
        "  • [cyan]ramllm recommend[/cyan]   — See exactly which models fit your computer best"
    )
    console.print(
        "  • [cyan]ramllm chat[/cyan]        — Start an interactive, memory-aware conversation"
    )
    console.print(
        "  • [cyan]ramllm dashboard[/cyan]   — Monitor live GPU/RAM usage, tokens/sec, and governor"
    )
    console.print(
        "  • [cyan]ramllm explain <t>[/cyan] — Plain-English explanation of terms (e.g. `explain vram`)\n"
    )
