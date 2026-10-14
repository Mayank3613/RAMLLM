"""CLI commands for managing llama.cpp engine installations."""

from typing import Annotated

import typer
from rich import box
from rich.table import Table

from ramllm.cli.ui.console import get_console, print_error, print_success, print_warning
from ramllm.services.backend_manager import (
    discover_backend,
    install_backend,
    list_managed_backends,
    remove_managed_backend,
    use_managed_backend,
)

backend_app = typer.Typer(
    help="Manage, install, update, and switch llama.cpp engine builds.",
    no_args_is_help=True,
    rich_help_panel="SYSTEM",
)


@backend_app.command("install")
def backend_install(
    tag: Annotated[
        str | None, typer.Option("--tag", help="Specific release tag to install")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Automatically accept download prompt")
    ] = False,
) -> None:
    """Download and set up prebuilt llama.cpp engine optimized for your hardware."""
    console = get_console()

    def consent_cb(asset_name: str, size_bytes: int) -> bool:
        if yes:
            return True
        mb = size_bytes / (1024**2)
        console.print(f"About to download engine build [bold]{asset_name}[/bold] (~{mb:.1f} MB).")
        return typer.confirm("Do you want to proceed with the download?", default=True)

    try:
        with console.status("[bold blue]Querying releases and downloading engine..."):
            install = install_backend(tag=tag, consent_callback=consent_cb)

        apis_str = ", ".join(install.capabilities.apis) if install.capabilities.apis else "CPU"
        print_success(
            f"Engine ready: llama.cpp {install.version or 'latest'} (Accelerators: {apis_str})"
        )
        console.print(f"  Binary installed at: `{install.binary_path}`")
    except Exception as e:
        print_error(f"Backend installation failed: {e}")
        raise typer.Exit(code=1) from e


@backend_app.command("list")
def backend_list() -> None:
    """List all managed and active llama.cpp engine builds."""
    console = get_console()
    active_disc = discover_backend()

    installs = list_managed_backends()
    if not installs and not active_disc:
        console.print("No llama.cpp installations found. Run `ramllm backend install` to get one.")
        return

    table = Table(title="llama.cpp Engine Installations", box=box.ROUNDED)
    table.add_column("Tag / Source", style="cyan")
    table.add_column("Binary Path", style="white")
    table.add_column("Version", style="magenta")
    table.add_column("Status", style="green")

    if active_disc:
        table.add_row(
            f"Active ({active_disc.source})",
            active_disc.install.binary_path,
            active_disc.install.version or "unknown",
            "[bold green]ACTIVE[/bold green]",
        )

    for m in installs:
        is_active = m.get("is_active", False)
        status_str = "[green]ACTIVE[/green]" if is_active else "Installed"
        table.add_row(
            m["tag"],
            m.get("binary_path") or "-",
            m.get("version") or "-",
            status_str,
        )

    console.print(table)


@backend_app.command("use")
def backend_use(
    target: Annotated[str, typer.Argument(help="Tag name or absolute path to llama-server binary")],
) -> None:
    """Switch active engine build to a managed tag or custom file path."""
    success = use_managed_backend(target)
    if success:
        print_success(f"Active backend switched to: {target}")
    else:
        print_error(
            f"Could not use '{target}'. Verify that the tag exists or the file path is executable."
        )
        raise typer.Exit(code=1)


@backend_app.command("remove")
def backend_remove(
    tag: Annotated[str, typer.Argument(help="Tag of managed installation to remove")],
) -> None:
    """Remove a managed engine installation from storage."""
    removed = remove_managed_backend(tag)
    if removed:
        print_success(f"Removed managed backend tag: {tag}")
    else:
        print_warning(f"No managed backend found with tag '{tag}'.")


@backend_app.command("update")
def backend_update(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Assume yes to download prompts")
    ] = False,
) -> None:
    """Fetch and install the newest available llama.cpp release."""
    console = get_console()
    console.print("[bold blue]Checking for newer llama.cpp releases...[/bold blue]")
    backend_install(tag=None, yes=yes)
