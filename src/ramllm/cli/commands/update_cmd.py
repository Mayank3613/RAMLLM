"""Self-update command (ramllm update)."""

import shlex
import subprocess
from typing import Annotated

import typer

from ramllm.cli.ui.console import get_console, print_error, print_success
from ramllm.cli.ui.theme import icons
from ramllm.services.upgrade_service import (
    check_for_updates,
    detect_install_method,
    get_upgrade_command,
)


def update_command(
    check_only: Annotated[
        bool,
        typer.Option("--check", help="Check for available updates without applying them"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("-y", "--yes", help="Automatically proceed with upgrade if available"),
    ] = False,
) -> None:
    """Check for newer RAMLLM releases and update using your installation method."""
    console = get_console()

    with console.status("[dim]Checking for RAMLLM updates on PyPI...[/dim]"):
        current_ver, latest_ver, has_update = check_for_updates()

    method = detect_install_method()

    if not latest_ver:
        console.print(
            f"[yellow]Could not reach PyPI to verify the latest release (Current: v{current_ver}).[/yellow]"
        )
        console.print("[dim]Check your network connection and try again.[/dim]")
        return

    if not has_update:
        console.print(
            f"[green]{icons.ok} RAMLLM is already up to date[/green] (v{current_ver}, installed via {method})."
        )
        return

    upgrade_cmd = get_upgrade_command(method)

    console.print("\n[bold cyan]🚀 A new version of RAMLLM is available![/bold cyan]")
    console.print(f"  • Current version:  [dim]v{current_ver}[/dim]")
    console.print(f"  • Latest version:   [bold green]v{latest_ver}[/bold green]")
    console.print(f"  • Install method:   [cyan]{method}[/cyan]\n")

    console.print(f"[bold]Recommended upgrade command:[/bold]\n  [cyan]{upgrade_cmd}[/cyan]\n")

    if check_only:
        return

    do_upgrade = yes
    if not do_upgrade:
        do_upgrade = typer.confirm("Would you like to run the upgrade command now?", default=True)

    if do_upgrade:
        console.print(f"[dim]Running: {upgrade_cmd}[/dim]\n")
        try:
            ret = subprocess.run(shlex.split(upgrade_cmd), check=False)
            if ret.returncode == 0:
                print_success("RAMLLM upgraded successfully!")
            else:
                print_error(f"Upgrade command exited with status {ret.returncode}.")
        except Exception as e:
            print_error(f"Failed to execute upgrade command: {e}")
