"""CLI command wrapper for ramllm setup wizard."""

from typing import Annotated

import typer

from ramllm.cli.setup.wizard import run_wizard


def setup_command(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Run non-interactively with defaults")
    ] = False,
    skip: Annotated[
        bool, typer.Option("--skip", help="Mark setup as completed without changes")
    ] = False,
    model: Annotated[
        str | None, typer.Option("--model", help="Pre-select starting model ID")
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Pre-select inference mode (fast|balanced|quality|max)"),
    ] = None,
) -> None:
    """Run the interactive onboarding wizard to configure RAMLLM and set up models."""
    run_wizard(yes=yes, skip=skip, model_id=model, mode_pref=mode)
