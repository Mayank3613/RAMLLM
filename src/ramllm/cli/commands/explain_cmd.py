"""CLI command to explain technical terms and error codes in plain language."""

from typing import Annotated

import typer
from rich import box
from rich.panel import Panel

from ramllm.cli.ui.console import get_console, print_warning
from ramllm.cli.ui.glossary import ERROR_EXPLANATIONS, GLOSSARY_TERMS, get_glossary_entry


def explain_command(
    term: Annotated[
        str | None,
        typer.Argument(help="Term or error code to explain (e.g. vram, quantization, RAM-E012)"),
    ] = None,
) -> None:
    """Explain technical inference terms or error codes in plain English."""
    console = get_console()

    if not term:
        console.print("[bold cyan]RAMLLM Plain-Language Glossary[/bold cyan]\n")
        console.print("Available terms:")
        for k in sorted(GLOSSARY_TERMS.keys()):
            console.print(f"  • [bold]{k}[/bold]: {GLOSSARY_TERMS[k]['title']}")
        console.print("\nAvailable error codes:")
        for ec in sorted(ERROR_EXPLANATIONS.keys()):
            console.print(f"  • [bold]{ec.upper()}[/bold]")
        console.print("\nUsage: `ramllm explain <term>`")
        return

    entry = get_glossary_entry(term)
    if not entry:
        print_warning(f"Term '{term}' not found in glossary.")
        console.print("Run `ramllm explain` without arguments to view all available terms.")
        return

    body = f"[bold]{entry['title']}[/bold]\n\n{entry['explanation']}"
    if entry.get("tip"):
        body += f"\n\n[dim]Tip: {entry['tip']}[/dim]"

    console.print(Panel(body, box=box.ROUNDED))
