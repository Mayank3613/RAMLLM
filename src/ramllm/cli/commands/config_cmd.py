"""CLI commands for managing RAMLLM configuration."""

import os
import subprocess
from typing import Annotated

import typer
from rich import box
from rich.table import Table

from ramllm.cli.ui.console import get_console, print_error, print_success
from ramllm.services.config_service import (
    RamllmConfig,
    get_config_value,
    load_config,
    reset_config,
    set_config_value,
)
from ramllm.services.paths import get_config_file

config_app = typer.Typer(
    help="Manage persistent RAMLLM configuration (settings).",
    no_args_is_help=True,
    rich_help_panel="SYSTEM",
)


@config_app.command("list")
def config_list() -> None:
    """Display all current configuration settings."""
    console = get_console()
    cfg = load_config()

    table = Table(title="RAMLLM Configuration Settings", box=box.ROUNDED)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="bold green")
    table.add_column("Description", style="white")

    for k, field in RamllmConfig.model_fields.items():
        val = getattr(cfg, k)
        desc = field.description or ""
        table.add_row(k, str(val) if val is not None else "(none)", desc)

    console.print(table)
    console.print(f"[dim]Config file: {get_config_file()}[/dim]")


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Configuration setting key to inspect")],
) -> None:
    """Print the value of a configuration setting."""
    try:
        val = get_config_value(key)
        print(val if val is not None else "")
    except KeyError as e:
        print_error(str(e))
        raise typer.Exit(code=2) from e


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Configuration setting key")],
    value: Annotated[str, typer.Argument(help="New value for setting")],
) -> None:
    """Update a configuration setting value."""
    try:
        new_val = set_config_value(key, value)
        print_success(f"Set '{key}' = {new_val}")
    except (KeyError, ValueError) as e:
        print_error(str(e))
        raise typer.Exit(code=2) from e


@config_app.command("path")
def config_path() -> None:
    """Print the absolute filesystem path of config.toml."""
    print(get_config_file())


@config_app.command("reset")
def config_reset(
    key: Annotated[
        str | None, typer.Argument(help="Optional specific key to reset to default")
    ] = None,
) -> None:
    """Reset all settings or a specific setting to factory default."""
    try:
        reset_config(key)
        if key:
            print_success(f"Reset '{key}' to default value.")
        else:
            print_success("All configuration settings reset to defaults.")
    except KeyError as e:
        print_error(str(e))
        raise typer.Exit(code=2) from e


@config_app.command("edit")
def config_edit() -> None:
    """Open config.toml in system text editor."""
    cfg_file = get_config_file()
    if not cfg_file.exists():
        reset_config()

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        editor = "notepad" if os.name == "nt" else "nano"

    try:
        subprocess.run([editor, str(cfg_file)], check=False)
    except Exception as e:
        print_error(f"Could not open editor '{editor}': {e}")
        get_console().print(f"Edit manually at: {cfg_file}")
