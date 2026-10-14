"""Process management commands: ps, stop, logs, status, and replan."""

import time
from pathlib import Path
from typing import Annotated, Any

import httpx
import psutil
import typer
from rich.table import Table

from ramllm.cli.ui.console import get_console, print_error, print_success
from ramllm.cli.ui.theme import icons
from ramllm.core.profiler import probe_hardware
from ramllm.services.backend_manager import discover_backend
from ramllm.services.model_service import get_models_disk_usage
from ramllm.services.process_registry import (
    RunningEngine,
    get_running_engine,
    list_running_engines,
    stop_all_engines,
    stop_engine,
)


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-friendly format (e.g. 5m 12s, 2h 15m)."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m = s // 60
    if m < 60:
        return f"{m}m {s % 60}s"
    h = m // 60
    return f"{h}h {m % 60}m"


def ps_command(
    as_json: Annotated[bool, typer.Option("--json", help="Output raw JSON array")] = False,
) -> None:
    """List all running RAMLLM engine instances and resource usage."""
    console = get_console()
    engines = list_running_engines(clean_stale=True)

    if as_json:
        import json

        data = [e.model_dump() for e in engines]
        console.print(json.dumps(data, indent=2))
        return

    if not engines:
        console.print("[dim]No RAMLLM engines are currently running.[/dim]")
        console.print(
            "[dim]Start one with `ramllm chat <model>` or `ramllm serve <model> --detach`[/dim]"
        )
        return

    table = Table(title="Running Engines", border_style="cyan")
    table.add_column("ID", style="bold cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("PID", style="dim", justify="right")
    table.add_column("Port", justify="right")
    table.add_column("Uptime", justify="right")
    table.add_column("RAM RSS", justify="right")
    table.add_column("Mode", style="magenta")

    for eng in engines:
        rss_str = "-"
        try:
            if psutil.pid_exists(eng.pid):
                p = psutil.Process(eng.pid)
                rss_bytes = p.memory_info().rss
                for ch in p.children(recursive=True):
                    rss_bytes += ch.memory_info().rss
                rss_str = f"{rss_bytes / (1024**2):.1f} MB"
        except Exception:
            pass

        uptime_str = _format_uptime(eng.uptime_seconds)
        table.add_row(
            eng.id,
            eng.model_name,
            str(eng.pid),
            str(eng.port),
            uptime_str,
            rss_str,
            eng.mode,
        )

    console.print(table)


def stop_command(
    target: Annotated[
        str | None,
        typer.Argument(help="ID or PID of the running engine to stop"),
    ] = None,
    all_engines: Annotated[
        bool,
        typer.Option("--all", "-a", help="Stop all currently running engines"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force immediate kill without graceful wait"),
    ] = False,
) -> None:
    """Stop running RAMLLM engine instances cleanly."""
    console = get_console()

    if all_engines:
        stopped = stop_all_engines()
        if stopped:
            print_success(f"Stopped {len(stopped)} engine(s): {', '.join(stopped)}")
        else:
            console.print("[dim]No running engines found.[/dim]")
        return

    if not target:
        # If only one engine running, default to stopping it
        running = list_running_engines(clean_stale=True)
        if len(running) == 1:
            eng = running[0]
            stop_engine(eng, timeout_s=0.5 if force else 5.0)
            print_success(f"Stopped engine '{eng.id}' (PID {eng.pid})")
            return
        elif len(running) > 1:
            console.print(
                "[yellow]Multiple engines running. Specify which ID to stop, or use --all:[/yellow]"
            )
            for r in running:
                console.print(f"  • {r.id} (PID {r.pid}, port {r.port})")
            raise typer.Exit(code=1)
        else:
            console.print("[dim]No running engines found to stop.[/dim]")
            return

    # Look up by ID or PID
    engine: RunningEngine | None = None
    if target.isdigit():
        target_pid = int(target)
        for r in list_running_engines(clean_stale=True):
            if r.pid == target_pid:
                engine = r
                break
    else:
        engine = get_running_engine(target)

    if not engine:
        print_error(
            f"No running engine found with ID or PID '{target}'. Use `ramllm ps` to view active engines."
        )
        raise typer.Exit(code=1)

    stop_engine(engine, timeout_s=0.5 if force else 5.0)
    print_success(f"Stopped engine '{engine.id}' (PID {engine.pid})")


def logs_command(
    target: Annotated[
        str | None,
        typer.Argument(help="ID of the running engine to view logs for"),
    ] = None,
    lines: Annotated[
        int,
        typer.Option("-n", "--lines", help="Number of recent log lines to display"),
    ] = 50,
    follow: Annotated[
        bool,
        typer.Option("-f", "--follow", help="Stream log updates continuously"),
    ] = False,
) -> None:
    """View or stream stdout/stderr log output of a running or recent engine."""
    console = get_console()
    engines = list_running_engines(clean_stale=False)

    chosen_log_path: Path | None = None
    chosen_id: str = ""

    if target:
        eng = get_running_engine(target)
        if eng:
            chosen_log_path = Path(eng.log_file)
            chosen_id = eng.id
        else:
            # Check if log file exists directly in logs dir
            from ramllm.services.paths import get_logs_dir

            candidate = get_logs_dir() / f"{target}.log"
            if candidate.exists():
                chosen_log_path = candidate
                chosen_id = target
    elif engines:
        # Default to the most recently started running engine
        chosen_log_path = Path(engines[-1].log_file)
        chosen_id = engines[-1].id

    if not chosen_log_path or not chosen_log_path.exists():
        print_error(
            f"No log file found for '{target or 'engine'}'. View running engines with `ramllm ps`."
        )
        raise typer.Exit(code=1)

    console.print(f"[dim]Showing logs for {chosen_id} ({chosen_log_path}):[/dim]\n")

    # Read last N lines
    try:
        content = chosen_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = content[-lines:] if lines > 0 else content
        for line in tail:
            console.print(line, highlight=False)

        if follow:
            with open(chosen_log_path, encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)  # Seek to end
                while True:
                    line = f.readline()
                    if line:
                        console.print(line.rstrip("\n"), highlight=False)
                    else:
                        time.sleep(0.2)
    except KeyboardInterrupt:
        pass


def status_command() -> None:
    """Print a comprehensive one-screen status dashboard for RAMLLM on this system."""
    console = get_console()

    console.print("[bold cyan]RAMLLM System Status[/bold cyan]\n")

    # 1. Backend Engine
    discovery = discover_backend()
    if discovery:
        inst = discovery.install
        ver_tag = f" v{inst.version}" if inst.version else ""
        console.print(
            f"[green]{icons.ok} Backend Engine:[/green] llama-server{ver_tag} ({inst.binary_path})"
        )
    else:
        console.print(
            f"[yellow]{icons.fail} Backend Engine:[/yellow] Not installed. Run `ramllm backend install`."
        )

    # 2. Hardware Capabilities
    hw = probe_hardware(quick=True)
    ram_gb = hw.ram_total_bytes / (1024**3)
    free_ram_gb = hw.ram_available_bytes / (1024**3)
    console.print(f"  • RAM: {ram_gb:.1f} GB total ({free_ram_gb:.1f} GB free)")
    if hw.gpus:
        for g in hw.gpus:
            vram_gb = g.vram_total_bytes / (1024**3)
            free_vram_gb = g.vram_free_bytes / (1024**3)
            console.print(
                f"  • GPU #{g.index}: {g.name} ({vram_gb:.1f} GB total, {free_vram_gb:.1f} GB free)"
            )
    else:
        console.print("  • GPU: None detected (CPU execution mode)")

    # 3. Models Library
    models_info = get_models_disk_usage()
    model_count = len(models_info["model_files"])
    total_gb = models_info["total_model_bytes"] / (1024**3)
    console.print(
        f"\n[bold]Models Library:[/bold] {model_count} model(s) stored ({total_gb:.1f} GB)"
    )

    # 4. Running Engines
    engines = list_running_engines(clean_stale=True)
    console.print(f"\n[bold]Active Engines:[/bold] {len(engines)} running")
    for e in engines:
        uptime = _format_uptime(e.uptime_seconds)
        console.print(
            f"  • [cyan]{e.id}[/cyan] ({e.model_name}) at {e.base_url} [dim]({uptime})[/dim]"
        )

    console.print()


def replan_command(
    target: Annotated[str, typer.Argument(help="ID or port of running engine to replan")],
    ctx: Annotated[int | None, typer.Option("--ctx", help="New context window length")] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="New RAM budget (e.g. 8GB)")] = None,
    vram: Annotated[str | None, typer.Option("--vram", help="New VRAM budget (e.g. 6GB)")] = None,
) -> None:
    """Dynamically replan an active background server without losing state."""
    console = get_console()
    engine = get_running_engine(target)

    target_url: str
    if engine:
        target_url = f"{engine.base_url}/ramllm/replan"
    elif target.isdigit():
        target_url = f"http://127.0.0.1:{target}/ramllm/replan"
    else:
        print_error(f"Running engine '{target}' not found. Check `ramllm ps`.")
        raise typer.Exit(code=1)

    from ramllm.util.parsing import parse_bytes

    body: dict[str, Any] = {}
    if ctx:
        body["context"] = ctx
    if ram:
        body["ram_budget_bytes"] = parse_bytes(ram)
    if vram:
        body["vram_budget_bytes"] = {0: parse_bytes(vram)}

    console.print(f"[dim]Sending replan request to {target_url}...[/dim]")
    try:
        resp = httpx.post(target_url, json=body, timeout=40.0)
        if resp.status_code == 200:
            data = resp.json()
            plan_id = data.get("id", "updated")
            verdict = data.get("verdict", "")
            print_success(f"Replan applied successfully: '{plan_id}' ({verdict})")
        else:
            print_error(f"Replan failed ({resp.status_code}): {resp.text}")
            raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Failed to connect to engine: {e}")
        raise typer.Exit(code=1) from e
