"""Version command (ramllm version / ramllm --version)."""

from ramllm import __version__
from ramllm.backends import LlamaCppAdapter
from ramllm.cli.ui.console import get_console


def print_version_info() -> None:
    """Print RAMLLM and backend version information."""
    console = get_console()
    adapter = LlamaCppAdapter()
    install = adapter.detect()
    if install:
        version_str = f" v{install.version}" if install.version else ""
        backend_str = f"{install.name}{version_str} at {install.binary_path}"
    else:
        backend_str = "llama.cpp (not installed)"

    console.print(f"RAMLLM v{__version__}")
    console.print(f"Backend: {backend_str}")


def version_command() -> None:
    """Show RAMLLM and backend version details."""
    print_version_info()
