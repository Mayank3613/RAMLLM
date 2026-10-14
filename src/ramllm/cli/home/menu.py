"""Interactive terminal home menu for RAMLLM (launched when called with no arguments in a TTY)."""

from rich import box
from rich.console import Console
from rich.panel import Panel

from ramllm.cli.ui.console import get_console
from ramllm.core.profiler import probe_hardware
from ramllm.services.backend_manager import discover_backend
from ramllm.services.config_service import load_config
from ramllm.services.model_service import get_curated_model


def run_home_menu(console: Console | None = None) -> None:
    """Render interactive home menu with live hardware facts and quick navigation."""
    c = console or get_console()

    # Gather live facts
    cfg = load_config()
    backend = discover_backend()
    hw = probe_hardware(quick=True)

    model_label = "No default model"
    if cfg.default_model:
        curated = get_curated_model(cfg.default_model)
        model_label = curated.name if curated else cfg.default_model

    engine_label = f"llama.cpp ({backend.install.name})" if backend else "Engine missing"

    ram_gb = hw.ram_total_bytes / (1024**3)
    vram_gb = sum(g.vram_total_bytes for g in hw.gpus) / (1024**3) if hw.gpus else 0.0
    mem_label = f"{ram_gb:.0f} GB RAM" + (f" / {vram_gb:.0f} GB GPU" if vram_gb > 0 else "")

    header_text = (
        f"[bold cyan]RAMLLM[/bold cyan]  ·  "
        f"[green]{model_label}[/green]  ·  "
        f"[white]Engine: {engine_label}[/white]  ·  "
        f"[magenta]Memory: {mem_label}[/magenta]"
    )

    suggested_step = (
        "Start chatting: `ramllm chat`"
        if cfg.default_model
        else "Next step: Find or download a model (`ramllm models search`)"
    )

    c.print(Panel(f"{header_text}\n[dim]{suggested_step}[/dim]", box=box.ROUNDED))

    menu_options = [
        ("chat", "Chat with a model"),
        ("search", "Find or download a model"),
        ("my_models", "My models (catalog)"),
        ("plan", "Preview what a model will do (plan)"),
        ("serve", "Start an API server (OpenAI-compatible)"),
        ("bench", "Test my speed (benchmark)"),
        ("profile", "Check my computer (hardware)"),
        ("settings", "Settings (configuration)"),
        ("doctor", "Help & diagnose (doctor)"),
        ("quit", "Quit"),
    ]

    try:
        import questionary

        choices = [label for _, label in menu_options]
        selected_label = questionary.select(
            "What would you like to do?",
            choices=choices,
            style=questionary.Style([("highlighted", "fg:#00ffff bold")]),
        ).ask()

        if not selected_label or selected_label == "Quit":
            c.print("[dim]Goodbye![/dim]")
            return

        selected_key = next(
            (k for k, opt_label in menu_options if opt_label == selected_label), "quit"
        )
    except Exception:
        # Fallback to standard input
        c.print("\n[bold]What would you like to do?[/bold]")
        for idx, (_, label) in enumerate(menu_options, 1):
            c.print(f"  [cyan]{idx}[/cyan]. {label}")
        try:
            val_str = input(f"Select [1-{len(menu_options)}]: ").strip()
            if not val_str:
                return
            idx = int(val_str) - 1
            if 0 <= idx < len(menu_options):
                selected_key = menu_options[idx][0]
            else:
                return
        except (ValueError, EOFError, KeyboardInterrupt):
            return

    # Dispatch selected action
    from ramllm.cli.commands.config_cmd import config_list
    from ramllm.cli.commands.doctor_cmd import doctor_command
    from ramllm.cli.commands.models_cmd import models_list, models_search

    if selected_key == "search":
        models_search()
    elif selected_key == "my_models":
        models_list()
    elif selected_key == "profile":
        from ramllm.cli.app import profile

        profile()
    elif selected_key == "settings":
        config_list()
    elif selected_key == "doctor":
        doctor_command()
    elif selected_key == "bench":
        c.print("[cyan]Run benchmark using:[/cyan] `ramllm bench <model>`")
    elif selected_key == "chat":
        from ramllm.cli.commands.chat_cmd import chat_command

        chat_command()
    elif selected_key == "plan":
        c.print("[cyan]Preview execution using:[/cyan] `ramllm plan <model>`")
    elif selected_key == "serve":
        from ramllm.cli.commands.serve_cmd import serve_command

        serve_command()
    elif selected_key == "quit":
        c.print("[dim]Goodbye![/dim]")
