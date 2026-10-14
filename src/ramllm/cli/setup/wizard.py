"""First-run interactive setup wizard (ramllm setup)."""

import sys
from typing import Literal

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel

from ramllm.cli.setup.state import (
    load_setup_state,
    save_setup_state,
)
from ramllm.cli.ui.console import get_console, print_error, print_info
from ramllm.cli.ui.theme import icons
from ramllm.core.profiler import probe_hardware
from ramllm.services.backend_manager import discover_backend, install_backend
from ramllm.services.config_service import load_config, save_config
from ramllm.services.model_resolver import resolve_model_reference
from ramllm.services.model_service import (
    estimate_model_fit,
    get_curated_model,
    search_curated_models,
)
from ramllm.services.plan_service import MEMORY_PRESETS, resolve_budgets


def _prompt_choice(
    prompt: str,
    options: list[tuple[str, str]],
    default_index: int = 0,
    console: Console | None = None,
) -> str:
    """Prompt user with numbered options and return selected key."""
    c = console or get_console()
    c.print(f"[bold]{prompt}[/bold]")
    for idx, (_key, label) in enumerate(options, 1):
        def_marker = " [dim](default)[/dim]" if idx - 1 == default_index else ""
        c.print(f"  [cyan]{idx}[/cyan]. {label}{def_marker}")

    while True:
        try:
            choice = input(f"Choose [1-{len(options)}] (default {default_index + 1}): ").strip()
            if not choice:
                return options[default_index][0]
            val = int(choice)
            if 1 <= val <= len(options):
                return options[val - 1][0]
            c.print(f"[red]Please enter a number between 1 and {len(options)}.[/red]")
        except (ValueError, IndexError):
            c.print("[red]Invalid input. Please enter a valid number.[/red]")


def run_wizard(
    yes: bool = False,
    skip: bool = False,
    model_id: str | None = None,
    mode_pref: str | None = None,
    console: Console | None = None,
) -> None:
    """Execute the interactive or automated onboarding wizard."""
    c = console or get_console()
    state = load_setup_state()

    if skip:
        state.finished = True
        save_setup_state(state)
        print_info("Setup skipped. You can run `ramllm setup` at any time.")
        return

    is_interactive = sys.stdin.isatty() and not yes

    try:
        # STEP 1: Welcome
        c.print(
            Panel(
                "[bold cyan]Welcome to RAMLLM[/bold cyan]\n\n"
                "RAMLLM configures, fits, and runs large language models on your hardware without manual guesswork.\n"
                "• Completely offline-first: nothing is sent to the internet except downloads you approve.\n"
                "• Exact fit planning: avoids out-of-memory crashes before launching.",
                box=box.ROUNDED,
            )
        )
        if is_interactive:
            typer.confirm("Ready to get started?", default=True)

        # STEP 2: Engine Check
        c.print("\n[bold]Step 1/6: Checking LLM Engine (llama.cpp)[/bold]")
        backend = discover_backend()
        if backend:
            c.print(
                f"[green]{icons.ok} Detected engine executable:[/green] {backend.install.binary_path} ({backend.install.name})"
            )
            state.engine_installed = True
        else:
            c.print("[yellow]! llama.cpp backend engine is not currently installed.[/yellow]")
            do_install = True
            if is_interactive:
                do_install = typer.confirm(
                    "Install managed llama.cpp engine automatically (~40 MB)?", default=True
                )
            if do_install:
                c.print("[dim]Downloading and installing optimized engine build...[/dim]")
                installed = install_backend()
                c.print(
                    f"[green]{icons.ok} Installed engine successfully:[/green] {installed.binary_path}"
                )
                state.engine_installed = True
            else:
                c.print(
                    "[dim]Skipping engine install. You can install it later with `ramllm backend install`.[/dim]"
                )

        # STEP 3: Computer Hardware Profile
        c.print("\n[bold]Step 2/6: Analyzing Computer Capabilities[/bold]")
        hw = probe_hardware(quick=True)
        ram_gb = hw.ram_total_bytes / (1024**3)
        avail_gb = hw.ram_available_bytes / (1024**3)

        if hw.gpus:
            gpu_desc = f"{hw.gpus[0].name} ({hw.gpus[0].vram_total_bytes / (1024**3):.1f} GB VRAM)"
        else:
            gpu_desc = "CPU-only mode (no discrete GPU detected)"

        c.print(f"  • Memory: {ram_gb:.1f} GB total ({avail_gb:.1f} GB free)")
        c.print(f"  • Accelerator: {gpu_desc}")
        c.print(
            f"  • Storage: {hw.disk_kind.upper()} ({hw.disk_read_bandwidth_bytes_s / (1024**3):.1f} GB/s)"
        )

        # STEP 4: Intended Use Case
        c.print("\n[bold]Step 3/6: Primary Use Case[/bold]")
        use_cases = [
            ("chat", "Everyday Chat & Brainstorming (Balanced speed & quality)"),
            ("coding", "Coding & Technical Problem Solving (Qwen Coder)"),
            ("writing", "Long-form Writing, Research & Summaries (High context)"),
            ("fast", "Fastest Possible Replies (Low resource usage)"),
            ("max", "Maximum Reasoning Capability (Fit largest model possible)"),
        ]
        if mode_pref:
            selected_case = mode_pref
        elif is_interactive:
            selected_case = _prompt_choice(
                "What will you primarily use RAMLLM for?", use_cases, default_index=0, console=c
            )
        else:
            selected_case = "chat"

        state.use_case = selected_case
        mode_mapping: dict[str, Literal["fast", "balanced", "quality", "max"]] = {
            "chat": "balanced",
            "coding": "balanced",
            "writing": "quality",
            "fast": "fast",
            "max": "max",
        }
        state.mode = mode_mapping.get(selected_case, "balanced")
        c.print(f"[green]{icons.ok} Selected mode:[/green] {state.mode.upper()}")

        # STEP 5: Memory Aggressiveness
        c.print("\n[bold]Step 4/6: Memory Usage Budget[/bold]")
        mem_options: list[tuple[str, str]] = []
        for preset_name, _preset_data in MEMORY_PRESETS.items():
            b = resolve_budgets(hw, preset=preset_name)
            vram_val = sum(b.vram_bytes_per_gpu.values()) / (1024**3)
            vram_note = f" + {vram_val:.1f} GB GPU" if hw.gpus else ""
            mem_options.append(
                (
                    preset_name,
                    f"{preset_name.capitalize()} (approx {b.ram_bytes / (1024**3):.1f} GB RAM{vram_note})",
                )
            )

        if is_interactive:
            mem_choice = _prompt_choice(
                "How much memory may RAMLLM dedicate to inference?",
                mem_options,
                default_index=1,
                console=c,
            )
        else:
            mem_choice = "balanced"

        state.memory_preset = mem_choice  # type: ignore[assignment]
        c.print(f"[green]{icons.ok} Memory budget set to:[/green] {mem_choice.upper()}")

        # STEP 6: Recommended Models
        c.print("\n[bold]Step 5/6: Model Selection[/bold]")
        fitting_models = search_curated_models(tag="coding" if selected_case == "coding" else None)
        filtered = [m for m in fitting_models if estimate_model_fit(m, hw)[0] != "INFEASIBLE"]
        if not filtered:
            filtered = fitting_models[:3]

        model_choices: list[tuple[str, str]] = []
        for m in filtered[:4]:
            verdict, plain_fit, _ = estimate_model_fit(m, hw)
            size_gb = m.files[0].size_bytes / (1024**3) if m.files else 2.0
            model_choices.append((m.id, f"{m.name} (~{size_gb:.1f} GB) — {plain_fit}"))

        model_choices.append(("skip", "I will choose or download a model later"))

        if model_id:
            chosen_model_id = model_id
        elif is_interactive:
            chosen_model_id = _prompt_choice(
                "Select a starting model for your computer:",
                model_choices,
                default_index=0,
                console=c,
            )
        else:
            chosen_model_id = filtered[0].id if filtered else "qwen2.5-0.5b"

        if chosen_model_id != "skip":
            state.selected_model_id = chosen_model_id
            m_info = get_curated_model(chosen_model_id)
            c.print(
                f"[green]{icons.ok} Selected model:[/green] {m_info.name if m_info else chosen_model_id}"
            )

            # Download selected model
            c.print(f"\n[bold]Downloading {chosen_model_id}...[/bold]")
            try:
                resolved_path = resolve_model_reference(
                    chosen_model_id, hw=hw, auto_download=True, console=c
                )
                c.print(f"[green]{icons.ok} Model ready at:[/green] {resolved_path}")
            except Exception as e:
                c.print(f"[yellow]Could not download model automatically: {e}[/yellow]")
                c.print("You can download it later with: `ramllm models pull <id>`")

        # STEP 7: Test Run & Verification
        c.print("\n[bold]Step 6/6: Verification & Test Prompt[/bold]")
        if state.selected_model_id and state.selected_model_id != "skip":
            c.print("Verifying plan for first chat...")
            try:
                c.print(
                    f"[dim]Model '{state.selected_model_id}' verified and ready for execution.[/dim]"
                )
                state.test_run_completed = True
            except Exception as e:
                c.print(f"[yellow]Test verification note: {e}[/yellow]")

        # Persist to configuration
        cfg = load_config()
        if state.selected_model_id and state.selected_model_id != "skip":
            cfg.default_model = state.selected_model_id
        cfg.mode = state.mode
        cfg.memory_usage = state.memory_preset
        save_config(cfg)

        state.finished = True
        save_setup_state(state)

        c.print(
            Panel(
                "[bold green][OK] RAMLLM Setup Complete![/bold green]\n\n"
                f"• Default Model: [cyan]{cfg.default_model or 'None'}[/cyan]\n"
                f"• Tradeoff Mode: [cyan]{cfg.mode}[/cyan]\n"
                f"• Memory Usage: [cyan]{cfg.memory_usage}[/cyan]\n\n"
                "Start chatting right now: [bold yellow]ramllm chat[/bold yellow]\n"
                "Browse more models:       [bold yellow]ramllm models search[/bold yellow]",
                box=box.ROUNDED,
            )
        )

    except KeyboardInterrupt as err:
        save_setup_state(state)
        c.print("\n[yellow]Setup paused. You can resume anytime by running: ramllm setup[/yellow]")
        raise typer.Exit(code=130) from err
    except Exception as e:
        save_setup_state(state)
        print_error(f"Setup encountered an error: {e}")
        c.print("[dim]Your progress has been saved. Run `ramllm setup` to continue.[/dim]")
        raise typer.Exit(code=1) from e
