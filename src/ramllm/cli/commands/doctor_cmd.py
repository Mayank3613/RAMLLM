"""CLI command for diagnosing system, engine, and enforcement levels, with optional auto-fix."""

from typing import Annotated

import typer
from rich import box
from rich.panel import Panel

from ramllm.cli.ui.console import get_console, print_error, print_success, print_warning
from ramllm.cli.ui.theme import icons
from ramllm.services.doctor_service import execute_fix, run_diagnostics


def doctor_command(
    fix: Annotated[
        bool, typer.Option("--fix", help="Automatically attempt safe repairs for identified issues")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Assume yes to confirmation prompts")
    ] = False,
) -> None:
    """Diagnose backend engines, hardware topology, catalogs, and OS memory enforcement levels."""
    console = get_console()
    console.print(Panel("[bold]RAMLLM Diagnostic Doctor[/bold]", box=box.ROUNDED))

    items = run_diagnostics()

    for item in items:
        if item.severity == "ok":
            console.print(f"[green]{icons.ok}[/green] [bold]{item.category}: {item.title}[/bold]")
            console.print(f"  {item.detail}")
        elif item.severity == "notice":
            console.print(
                f"[yellow]{icons.warn}[/yellow] [bold]{item.category}: {item.title}[/bold]"
            )
            console.print(f"  {item.detail}")
            if item.fix_suggestion:
                console.print(f"  [dim]Suggestion: {item.fix_suggestion}[/dim]")
        else:  # problem
            console.print(f"[red]{icons.fail}[/red] [bold]{item.category}: {item.title}[/bold]")
            console.print(f"  [red]{item.detail}[/red]")
            if item.fix_suggestion:
                console.print(f"  [yellow]Try: {item.fix_suggestion}[/yellow]")

    import sys

    problems = [it for it in items if it.severity == "problem"]
    fixable = [it for it in items if it.fix_action]

    should_fix = fix or (problems and yes)
    if not should_fix and problems and sys.stdin.isatty() and not yes:
        should_fix = typer.confirm("\nAttempt automatic fixes for detected issues?", default=True)

    if fixable and should_fix:
        console.print("\n[bold blue]Applying automatic fixes...[/bold blue]")
        for f in fixable:
            if f.fix_action:
                ok, msg = execute_fix(f.fix_action)
                if ok:
                    print_success(msg)
                else:
                    print_error(msg)
        # Re-run diagnostics
        items = run_diagnostics()
        problems = [it for it in items if it.severity == "problem"]

    console.print("")
    if not problems:
        print_success("Everything looks good. Try `ramllm chat`.")
    else:
        print_warning(
            f"Found {len(problems)} problem(s). Run `ramllm doctor --fix` or review suggestions above."
        )
