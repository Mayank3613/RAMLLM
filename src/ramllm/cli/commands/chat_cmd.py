"""Interactive chat CLI command (ramllm chat)."""

import socket
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from ramllm.backends.llamacpp.adapter import LlamaCppAdapter
from ramllm.cli.ui.console import get_console, print_error
from ramllm.cli.ui.errors import InsufficientMemoryError, render_error
from ramllm.cli.ui.theme import icons
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.governor.recovery import launch_with_recovery
from ramllm.core.planner import create_execution_plans
from ramllm.core.profiler import probe_hardware
from ramllm.core.schemas import UserRequest
from ramllm.services.backend_manager import install_backend
from ramllm.services.config_service import load_config
from ramllm.services.model_resolver import resolve_model_reference
from ramllm.services.model_service import estimate_model_fit, search_curated_models
from ramllm.services.plan_service import resolve_budgets, translate_mode


def _find_free_port(start_port: int = 8085) -> int:
    """Find the next available localhost TCP port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def chat_command(
    model: Annotated[
        str | None,
        typer.Argument(
            help="Model name, local file path, or Hugging Face ID (e.g. 'qwen-2.5-7b', 'llama-3.2-3b')",
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option("-p", "--prompt", help="Single prompt for non-interactive execution"),
    ] = None,
    system: Annotated[
        str | None,
        typer.Option("--system", help="Custom system instruction prompt"),
    ] = None,
    ctx: Annotated[
        str | None,
        typer.Option("--ctx", help="Context window limit (e.g. 4096, 8192)"),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Execution mode ('fast', 'balanced', 'quality', 'max')"),
    ] = None,
    memory: Annotated[
        str | None,
        typer.Option(
            "--memory",
            help="Memory budget preset ('conservative', 'balanced', 'max') or bytes ('8GB')",
        ),
    ] = None,
    temp: Annotated[
        float,
        typer.Option("--temp", help="Sampling temperature"),
    ] = 0.7,
    top_p: Annotated[
        float,
        typer.Option("--top-p", help="Nucleus sampling top-p"),
    ] = 0.9,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed for deterministic generations"),
    ] = None,
    no_save: Annotated[
        bool,
        typer.Option("--no-save", help="Disable automatic conversation transcript archiving"),
    ] = False,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Resume conversation from an existing transcript file"),
    ] = None,
    attach_server: Annotated[
        str | None,
        typer.Option(
            "--attach-server",
            help="Connect to an already-running server (e.g. http://127.0.0.1:8080)",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Port for local llama-server engine"),
    ] = None,
    plain: Annotated[
        bool,
        typer.Option("--plain", help="Plain linear output without rich Markdown formatting"),
    ] = False,
) -> None:
    """Open an interactive chat REPL with automatic budgeting and streaming output."""
    console = Console(force_terminal=False, color_system=None) if plain else get_console()
    cfg = load_config()

    # Determine input mode (piped stdin or --prompt)
    is_piped = not sys.stdin.isatty()
    single_prompt = prompt or (sys.stdin.read().strip() if is_piped else None)

    # 1. ATTACH-SERVER WORKFLOW
    if attach_server:
        from ramllm.cli.chat.client import ChatClient, ChatMessage
        from ramllm.cli.chat.repl import start_chat_session

        server_url = attach_server.rstrip("/")
        client = ChatClient(base_url=server_url)
        if not client.check_health():
            print_error(
                f"Could not connect to server at {server_url}. Ensure the server is running."
            )
            raise typer.Exit(code=1)

        model_display = model or "attached-server"

        if single_prompt:
            # Single prompt non-interactive execution
            msgs = []
            if system:
                msgs.append(ChatMessage(role="system", content=system))
            msgs.append(ChatMessage(role="user", content=single_prompt))
            resp_text, _ = client.complete_chat(msgs, temperature=temp, top_p=top_p, seed=seed)
            console.print(resp_text)
            return

        session_model_name = model_display
        start_chat_session(
            model_name=session_model_name,
            client=client,
            system_prompt=system,
            temperature=temp,
            top_p=top_p,
            seed=seed,
            plain_mode=plain,
            auto_save_transcripts=not no_save,
            is_attached=True,
        )
        return

    # 2. LOCAL ENGINE LAUNCH WORKFLOW
    # Determine model target
    target_model_str = model or cfg.default_model
    if not target_model_str:
        # Auto-recommend model from curated index
        hw_quick = probe_hardware(quick=True)
        curated_matches = search_curated_models()
        fitting = [
            m
            for m in curated_matches
            if estimate_model_fit(m, hw_quick)[0] in ("COMFORTABLE", "CONSTRAINED")
        ]
        if fitting:
            target_model_str = fitting[0].id
            console.print(
                f"[dim]No model specified. Using recommended model for your machine: '{target_model_str}'[/dim]"
            )
        else:
            target_model_str = "qwen-2.5-7b"

    # Resolve model file
    try:
        resolved_path = resolve_model_reference(target_model_str, auto_download=True)
    except Exception as e:
        print_error(f"Failed to resolve model '{target_model_str}': {e}")
        raise typer.Exit(code=1) from e

    model_info = analyze_gguf(resolved_path)
    hw = probe_hardware(quick=False)

    # Parse parameters
    ctx_val = int(ctx) if ctx and ctx.isdigit() else 4096
    mode_val = mode or cfg.mode
    mem_preset = memory or cfg.memory_usage

    budgets = resolve_budgets(hw, preset=mem_preset)
    pref, allow_kv_quant, _ = translate_mode(mode_val, context_str=str(ctx_val))
    req = UserRequest(
        model=resolved_path.name,
        min_context=ctx_val,
        target_context=ctx_val,
        ram_budget_bytes=budgets.ram_bytes,
        vram_budget_bytes=budgets.vram_bytes_per_gpu,
        preference=pref,
        allow_kv_quant=allow_kv_quant,
    )

    plans = create_execution_plans(model_info, hw, req)
    chosen_plan = plans[0]

    if chosen_plan.verdict == "INFEASIBLE":
        err = InsufficientMemoryError(
            needed_bytes=chosen_plan.memory.ram_peak_bytes,
            available_bytes=hw.ram_available_bytes,
        )
        render_error(err, console=console)
        raise typer.Exit(code=err.exit_code)

    if chosen_plan.verdict in ("CONSTRAINED", "BATCH_ONLY") and not single_prompt:
        speed_est = f"~{chosen_plan.speed.decode_tok_s_low} tok/s"
        console.print(
            f"[yellow][!] Notice: {chosen_plan.verdict} verdict. Speed will be constrained ({speed_est}).[/yellow]"
        )
        if sys.stdin.isatty():
            confirmed = typer.confirm("Launch anyway?", default=True)
            if not confirmed:
                console.print(
                    "[dim]Aborted by user. Run `ramllm models search --fits` for better options.[/dim]"
                )
                raise typer.Exit(code=0)

    # Detect or install backend engine
    adapter = LlamaCppAdapter()
    install = adapter.detect()
    if not install:
        console.print("[yellow]! llama.cpp backend engine is not currently installed.[/yellow]")
        if sys.stdin.isatty():
            do_install = typer.confirm("Install managed engine now (~40 MB)?", default=True)
            if do_install:
                install = install_backend()
            else:
                print_error("Engine required to run chat. Install with `ramllm backend install`.")
                raise typer.Exit(code=1)
        else:
            print_error("llama-server binary not found. Run `ramllm backend install`.")
            raise typer.Exit(code=1)

    listen_port = port or _find_free_port(8085)

    def _on_retry(attempt: int, next_plan: Any, cause: str) -> None:
        console.print(
            f"[yellow][!] Launch attempt {attempt} failed ({cause}). Adapting plan...[/yellow]"
        )

    try:
        with console.status("[bold cyan]Launching execution engine and verifying buffers..."):
            proc, active_plan, _ = launch_with_recovery(
                adapter=adapter,
                plan=chosen_plan,
                model=model_info,
                hw=hw,
                req=req,
                port=listen_port,
                timeout_s=35.0,
                on_retry=_on_retry,
            )
    except Exception as e:
        print_error(f"Failed to launch engine: {e}")
        raise typer.Exit(code=1) from e

    governor = MemoryGovernor(proc, hw)
    governor.start()

    from ramllm.cli.chat.client import ChatClient

    client = ChatClient(base_url=f"http://127.0.0.1:{listen_port}")

    # Handle single prompt / piped execution
    if single_prompt:
        try:
            from ramllm.cli.chat.client import ChatMessage

            msgs = []
            if system:
                msgs.append(ChatMessage(role="system", content=system))
            msgs.append(ChatMessage(role="user", content=single_prompt))
            resp_text, _ = client.complete_chat(msgs, temperature=temp, top_p=top_p, seed=seed)
            console.print(resp_text)
        finally:
            governor.stop()
            adapter.stop(proc)
        return

    # Interactive REPL session
    try:
        from ramllm.cli.chat.repl import ChatSession
        from ramllm.cli.chat.transcripts import load_transcript

        loaded_messages = None
        if resume:
            try:
                loaded_messages = load_transcript(resume)
                console.print(
                    f"[green]{icons.ok} Resumed {len(loaded_messages)} messages from {resume}.[/green]"
                )
            except Exception as e:
                console.print(f"[yellow]Could not resume transcript: {e}[/yellow]")

        session = ChatSession(
            model_name=resolved_path.stem,
            client=client,
            model_path=resolved_path,
            adapter=adapter,
            backend_proc=proc,
            governor=governor,
            plan=active_plan,
            model_info=model_info,
            hw=hw,
            req=req,
            system_prompt=system,
            temperature=temp,
            top_p=top_p,
            seed=seed,
            plain_mode=plain,
            auto_save_transcripts=not no_save,
            port=listen_port,
            is_attached=False,
            console=console,
        )
        if loaded_messages:
            session.messages = loaded_messages

        from ramllm.cli.chat.repl import run_repl

        run_repl(session)
    finally:
        governor.stop()
        adapter.stop(proc)
