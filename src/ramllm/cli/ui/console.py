"""Shared Rich Console and printing utilities."""

from rich.console import Console

from ramllm.cli.ui.theme import is_plain_mode

_console: Console | None = None


def get_console() -> Console:
    """Get the active Rich console instance."""
    global _console
    if _console is None:
        if is_plain_mode():
            _console = Console(no_color=True, highlight=False)
        else:
            _console = Console()
    return _console


def print_success(message: str) -> None:
    from ramllm.cli.ui.theme import icons

    c = get_console()
    if is_plain_mode():
        c.print(f"{icons.ok} {message}")
    else:
        c.print(f"[bold green]{icons.ok}[/bold green] {message}")


def print_error(message: str) -> None:
    from ramllm.cli.ui.theme import icons

    c = get_console()
    if is_plain_mode():
        c.print(f"{icons.fail} {message}")
    else:
        c.print(f"[bold red]{icons.fail}[/bold red] {message}")


def print_warning(message: str) -> None:
    from ramllm.cli.ui.theme import icons

    c = get_console()
    if is_plain_mode():
        c.print(f"{icons.warn} {message}")
    else:
        c.print(f"[bold yellow]{icons.warn}[/bold yellow] {message}")


def print_info(message: str) -> None:
    c = get_console()
    c.print(message)
