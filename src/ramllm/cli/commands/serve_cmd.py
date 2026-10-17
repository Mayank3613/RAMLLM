"""Serve command for running RAMLLM API servers in foreground or detached background."""

import os
import socket
import subprocess
import sys
import time
from typing import Annotated

import httpx
import typer
from rich.panel import Panel

from ramllm.cli.ui.console import get_console, print_error
from ramllm.cli.ui.theme import icons
from ramllm.core.profiler import probe_hardware
from ramllm.services.config_service import load_config
from ramllm.services.model_resolver import resolve_model_reference
from ramllm.services.model_service import search_curated_models
from ramllm.services.process_registry import (
    RunningEngine,
    generate_engine_id,
    get_default_engine_log_file,
    list_running_engines,
    register_engine,
    unregister_engine,
)
from ramllm.util.parsing import parse_bytes, parse_context


def _find_available_port(start_port: int, host: str = "127.0.0.1") -> int:
    """Find the next open port starting from start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    return start_port


def _parse_gpu_args(
    gpus_str: str | None, vram_str: str | None
) -> tuple[list[int] | None, dict[int, int] | None]:
    """Parse comma-separated GPUs and VRAM limit flags."""
    gpu_indices: list[int] | None = None
    if gpus_str:
        gpu_indices = [int(x.strip()) for x in gpus_str.split(",") if x.strip()]

    vram_dict: dict[int, int] | None = None
    if vram_str:
        if "," in vram_str:
            parts = [parse_bytes(p.strip()) for p in vram_str.split(",")]
            indices = gpu_indices if gpu_indices else list(range(len(parts)))
            vram_dict = {
                idx: val
                for i, idx in enumerate(indices)
                if i < len(parts) and (val := parts[i]) is not None
            }
        else:
            single_val = parse_bytes(vram_str)
            if single_val is not None:
                if gpu_indices and len(gpu_indices) > 1:
                    vram_dict = dict.fromkeys(gpu_indices, single_val)
                else:
                    target_idx = gpu_indices[0] if (gpu_indices and len(gpu_indices) == 1) else 0
                    vram_dict = {target_idx: single_val}
    return gpu_indices, vram_dict


def serve_command(
    model: Annotated[
        str | None,
        typer.Argument(help="Path or name of GGUF model (uses config default if omitted)"),
    ] = None,
    vram: Annotated[str | None, typer.Option("--vram", help="VRAM budget (e.g. 6GB)")] = None,
    ram: Annotated[str | None, typer.Option("--ram", help="RAM budget (e.g. 12GB)")] = None,
    gpus: Annotated[
        str | None, typer.Option("--gpus", help="Comma-separated GPU indices to use (e.g. 0,1)")
    ] = None,
    context: Annotated[
        str, typer.Option("--context", help="Target context window length")
    ] = "4096",
    port: Annotated[
        int, typer.Option("--port", help="Port to listen (auto-selected if taken)")
    ] = 8080,
    host: Annotated[str, typer.Option("--host", help="Host interface to bind")] = "127.0.0.1",
    detach: Annotated[
        bool,
        typer.Option("-d", "--detach", help="Run server in the background as a detached service"),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="API key required for completions (recommended on 0.0.0.0)"),
    ] = None,
    min_tps: Annotated[
        float | None, typer.Option("--min-tps", help="Minimum tolerable tokens/second threshold")
    ] = None,
    ubatch: Annotated[
        int, typer.Option("--ubatch", help="Micro-batch size for activation buffers")
    ] = 512,
    auto_retry: Annotated[
        bool,
        typer.Option(
            "--auto-retry/--no-auto-retry",
            help="Automatically adapt to smaller plans on failure or OOM",
        ),
    ] = True,
    max_retries: Annotated[
        int, typer.Option("--max-retries", help="Maximum auto-retry recovery attempts")
    ] = 3,
    engine_id: Annotated[
        str | None,
        typer.Option("--engine-id", hidden=True, help="Internal explicit engine ID for daemon"),
    ] = None,
    insecure: Annotated[
        bool,
        typer.Option(
            "--insecure",
            help="Allow binding to non-loopback interface (0.0.0.0) without an API key",
        ),
    ] = False,
    cors_origins: Annotated[
        str | None,
        typer.Option(
            "--cors-origins",
            help="Comma-separated allowed CORS origins (e.g. 'http://localhost:3000')",
        ),
    ] = None,
) -> None:
    """Start OpenAI-compatible API server with active memory governor."""
    console = get_console()
    cfg = load_config()

    # 1. Resolve Target Model
    target_model_str = model or cfg.default_model
    if not target_model_str:
        curated = search_curated_models()
        if curated:
            target_model_str = curated[0].id
            console.print(
                f"[dim]No model specified. Using recommended model '{target_model_str}'[/dim]"
            )
        else:
            print_error("No model specified and no default model found in config.")
            raise typer.Exit(code=1)

    try:
        resolved_model = resolve_model_reference(target_model_str, auto_download=True)
    except Exception as e:
        print_error(f"Failed to resolve model '{target_model_str}': {e}")
        raise typer.Exit(code=1) from e

    # 2. Port and Host Resolution
    actual_port = _find_available_port(port, host=host)
    if actual_port != port:
        console.print(
            f"[yellow]Notice: Port {port} is occupied. Using next free port: {actual_port}[/yellow]"
        )

    # Check network exposure and enforce loopback policy
    active_api_key = api_key or os.environ.get("RAMLLM_API_KEY")
    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    if not is_loopback:
        if not active_api_key and not insecure:
            print_error(
                f"Security Error: Refusing to bind to non-loopback interface '{host}' without an API key.\n"
                "Pass an API key with '--api-key <KEY>', set RAMLLM_API_KEY, or pass '--insecure' to proceed."
            )
            raise typer.Exit(code=1)
        elif not active_api_key and insecure:
            console.print(
                f"[bold red][!] SECURITY WARNING: Binding to {host} without an API key (--insecure enabled). "
                "Anyone on your network can execute inference requests.[/bold red]"
            )
        elif active_api_key:
            console.print(
                f"[bold yellow][!] Notice: Binding to {host} with API key authentication enabled.[/bold yellow]"
            )

    cors_list = [o.strip() for o in cors_origins.split(",") if o.strip()] if cors_origins else []

    # 3. Memory Concurrency Warning
    running_instances = list_running_engines(clean_stale=True)
    if running_instances:
        hw = probe_hardware(quick=True)
        free_gb = hw.ram_available_bytes / (1024**3)
        console.print(
            f"[yellow]Notice: {len(running_instances)} other engine(s) running ({free_gb:.1f} GB RAM available).[/yellow]"
        )

    # 4. DETACHED BACKGROUND SERVICE
    if detach:
        assigned_id = engine_id or generate_engine_id(resolved_model.stem)
        log_path = get_default_engine_log_file(assigned_id)

        child_cmd = [
            sys.executable,
            "-m",
            "ramllm.cli.main",
            "serve",
            str(resolved_model),
            "--port",
            str(actual_port),
            "--host",
            host,
            "--context",
            str(context),
            "--engine-id",
            assigned_id,
        ]
        if vram:
            child_cmd.extend(["--vram", vram])
        if ram:
            child_cmd.extend(["--ram", ram])
        if gpus:
            child_cmd.extend(["--gpus", gpus])
        if active_api_key:
            child_cmd.extend(["--api-key", active_api_key])
        if insecure:
            child_cmd.append("--insecure")
        if cors_origins:
            child_cmd.extend(["--cors-origins", cors_origins])
        if min_tps is not None:
            child_cmd.extend(["--min-tps", str(min_tps)])
        if ubatch != 512:
            child_cmd.extend(["--ubatch", str(ubatch)])

        # Launch detached subprocess
        with open(log_path, "w", encoding="utf-8") as log_file:
            if sys.platform == "win32":
                creationflags = (
                    0x00000008 | 0x00000200
                )  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                proc = subprocess.Popen(
                    child_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            else:
                proc = subprocess.Popen(
                    child_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

        # Wait for service health check
        health_url = f"http://{host}:{actual_port}/healthz"
        t0 = time.time()
        healthy = False

        with console.status(f"[bold cyan]Starting server in background (id: {assigned_id})..."):
            while time.time() - t0 < 35.0:
                ret = proc.poll()
                if ret is not None:
                    # Process died early
                    break
                try:
                    res = httpx.get(health_url, timeout=0.8)
                    if res.status_code == 200:
                        healthy = True
                        break
                except Exception:
                    pass
                time.sleep(0.4)

        if not healthy:
            last_logs = ""
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                last_logs = "\n".join(lines[-20:])
            print_error(
                f"Server failed to start within timeout.\nLog file: {log_path}\nRecent logs:\n{last_logs}"
            )
            raise typer.Exit(code=1)

        # Register running engine
        engine_record = RunningEngine(
            id=assigned_id,
            pid=proc.pid,
            port=actual_port,
            host=host,
            model_name=resolved_model.stem,
            model_path=str(resolved_model),
            plan_id="active",
            log_file=str(log_path),
            mode="serve",
            api_key=active_api_key,
        )
        register_engine(engine_record)

        # Print beautiful connection help panel
        key_hint = f'"{active_api_key}"' if active_api_key else '"not-needed"'
        panel_content = (
            f"[bold green]{icons.ok} Server running (id: {assigned_id}) at http://{host}:{actual_port}/v1[/bold green]\n\n"
            f'  [dim]Use with OpenAI SDK:[/dim]  base_url="http://{host}:{actual_port}/v1", api_key={key_hint}\n'
            f"  [dim]Open WebUI / LangChain:[/dim] use the same base URL.\n"
            f"  [dim]Manage:[/dim]  ramllm ps  ·  ramllm logs {assigned_id} -f  ·  ramllm stop {assigned_id}"
        )
        console.print(Panel(panel_content, border_style="green", expand=False))
        return

    # 5. FOREGROUND SERVER EXECUTION
    gpu_indices, vram_dict = _parse_gpu_args(gpus, vram)
    ram_bytes = parse_bytes(ram)
    ctx_len = parse_context(context)

    assigned_id = engine_id or generate_engine_id(resolved_model.stem)
    log_file_path = get_default_engine_log_file(assigned_id)

    from ramllm.api.server import create_app

    app_instance = create_app(
        model_path=str(resolved_model.resolve()),
        vram_budget_bytes=vram_dict,
        ram_budget_bytes=ram_bytes,
        context=ctx_len,
        gpu_indices=gpu_indices,
        auto_retry=auto_retry,
        max_retries=max_retries,
        min_tps=min_tps,
        ubatch=ubatch,
        api_key=active_api_key,
        cors_origins=cors_list,
    )

    # Register in process table
    record = RunningEngine(
        id=assigned_id,
        pid=os.getpid(),
        port=actual_port,
        host=host,
        model_name=resolved_model.stem,
        model_path=str(resolved_model),
        plan_id="active",
        log_file=str(log_file_path),
        mode="serve",
        api_key=active_api_key,
    )
    register_engine(record)

    console.print(
        f"[bold green]Starting RAMLLM API server on http://{host}:{actual_port}...[/bold green]"
    )
    console.print(f"OpenAI endpoint: http://{host}:{actual_port}/v1/chat/completions")
    console.print(f"Metrics endpoint: http://{host}:{actual_port}/ramllm/metrics")

    try:
        import uvicorn

        uvicorn.run(app_instance, host=host, port=actual_port, log_level="info")
    finally:
        unregister_engine(assigned_id)
