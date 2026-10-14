"""Live TUI dashboard command (ramllm dashboard)."""

import sys
import time
from typing import Annotated

import typer

from ramllm.cli.ui.console import get_console, print_error
from ramllm.services.process_registry import get_running_engine, list_running_engines


def dashboard_command(
    target: Annotated[
        str | None,
        typer.Argument(
            help="ID, port, or base URL of running engine to monitor (auto-detected if omitted)"
        ),
    ] = None,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch", "-w", help="Run in linear terminal text watch mode without full TUI"
        ),
    ] = False,
) -> None:
    """Open a live visual TUI dashboard monitoring memory headroom, governor, and speed."""
    console = get_console()

    target_url = "http://127.0.0.1:8080"
    target_id = "local-engine"

    if target:
        if target.startswith("http://") or target.startswith("https://"):
            target_url = target
            target_id = "custom-url"
        elif target.isdigit():
            target_url = f"http://127.0.0.1:{target}"
            target_id = f"port-{target}"
        else:
            eng = get_running_engine(target)
            if eng:
                target_url = eng.base_url
                target_id = eng.id
            else:
                print_error(
                    f"No running engine found with ID '{target}'. Run `ramllm ps` to list active engines."
                )
                raise typer.Exit(code=1)
    else:
        # Auto-attach to the only / last running engine
        running = list_running_engines(clean_stale=True)
        if running:
            chosen = running[-1]
            target_url = chosen.base_url
            target_id = chosen.id
        else:
            console.print("[yellow]No running RAMLLM engine found to monitor.[/yellow]")
            console.print(
                "[dim]Start an engine first with:[/dim] [cyan]ramllm serve <model> --detach[/cyan]"
            )
            console.print(
                "[dim]Or attach to a specific URL:[/dim] [cyan]ramllm dashboard http://127.0.0.1:8080[/cyan]\n"
            )
            raise typer.Exit(code=1)

    # Fallback to linear text watch mode if terminal is non-interactive or explicitly requested
    if watch or not sys.stdin.isatty():
        from ramllm.cli.commands.process_cmd import status_command

        console.print(f"[dim]Monitoring {target_id} at {target_url} (Ctrl-C to stop):[/dim]\n")
        try:
            while True:
                status_command()
                time.sleep(2.0)
        except KeyboardInterrupt:
            return

    # Run Textual TUI Application
    from ramllm.cli.dashboard.app import DashboardApp

    app = DashboardApp(target_url=target_url, engine_id=target_id)
    app.run()
