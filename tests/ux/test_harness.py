"""UX Test Harness for RAMLLM CLI testing.

Provides utilities for:
- Testing Typer commands with CliRunner
- Capturing Rich console output with record=True
- Testing --plain and --json outputs
- Process hygiene checks to detect orphaned processes
"""

import os
from typing import Any

import psutil
from rich.console import Console
from typer.testing import CliRunner

runner = CliRunner()


def run_cli_command(app: Any, args: list[str], env: dict[str, str] | None = None) -> Any:
    """Run CLI command using CliRunner with optional custom environment variables."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return runner.invoke(app, args, env=merged_env)


def render_to_plain_text(renderable: Any) -> str:
    """Render a Rich renderable object into plain text without ANSI escape sequences."""
    console = Console(record=True, color_system=None, force_terminal=False, width=100)
    console.print(renderable)
    return console.export_text()


def find_orphaned_processes(cmdline_substring: str = "llama-server") -> list[int]:
    """Find running processes containing the specified substring in their command line."""
    orphaned_pids: list[int] = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            cmdline = " ".join(proc.info["cmdline"] or [])
            if cmdline_substring in cmdline or cmdline_substring in (proc.info["name"] or ""):
                orphaned_pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return orphaned_pids


def assert_clean_process_state(cmdline_substring: str = "llama-server") -> None:
    """Assert that no orphaned backend or mock processes are currently running."""
    orphans = find_orphaned_processes(cmdline_substring)
    assert not orphans, f"Found orphaned processes matching '{cmdline_substring}': PIDs {orphans}"
