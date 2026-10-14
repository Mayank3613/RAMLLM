"""Slash command definitions, handlers, and prompt_toolkit auto-completer."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from rich.table import Table

from ramllm.cli.chat.transcripts import load_transcript, save_transcript
from ramllm.cli.ui.theme import icons

if TYPE_CHECKING:
    from ramllm.cli.chat.repl import ChatSession

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available slash commands and keybindings",
    "/exit": "Quit chat and stop backend engine (also /quit, Ctrl-D)",
    "/quit": "Quit chat and stop backend engine",
    "/clear": "Clear conversation history and start a fresh session",
    "/system": "View or update the system prompt (e.g. `/system You are a coding assistant`)",
    "/model": "Show active model or switch to a new model (`/model <name>`)",
    "/plan": "Display current execution plan, layer placement, and memory buffers",
    "/mem": "Inspect active memory consumption per hardware tier and governor status",
    "/stats": "View generation speed, token counts, and session latency statistics",
    "/ctx": "Change context window limit and replan (`/ctx 8192` or `/ctx auto`)",
    "/temp": "Set sampling temperature (`/temp 0.7`)",
    "/top_p": "Set nucleus sampling top-p (`/top_p 0.9`)",
    "/seed": "Set random sampling seed (`/seed 42`)",
    "/save": "Export conversation transcript to file (`/save` or `/save my_chat.md`)",
    "/load": "Load and resume conversation from transcript (`/load my_chat.md`)",
    "/retry": "Regenerate the last assistant response",
    "/undo": "Remove the last user prompt and assistant response",
    "/copy": "Copy the last assistant response to clipboard",
    "/mode": "Switch execution mode and replan (`/mode fast|balanced|quality|max`)",
    "/budget": "Change memory budget and replan (`/budget 8GB` or `/budget max`)",
    "/serve": "Show local API server endpoint and connection instructions",
    "/toolbar": "Toggle bottom status toolbar (`/toolbar on` or `/toolbar off`)",
}


class SlashCommandCompleter(Completer):
    """Tab-completer for slash commands and parameter choices."""

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        words = text.split(" ")
        if len(words) == 1:
            query = words[0].lower()
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.lower().startswith(query):
                    yield Completion(
                        cmd,
                        start_position=-len(query),
                        display=cmd,
                        display_meta=desc,
                    )
        elif len(words) == 2 and words[0] == "/mode":
            mode_arg = words[1].lower()
            for mode_opt in ["fast", "balanced", "quality", "max"]:
                if mode_opt.startswith(mode_arg):
                    yield Completion(mode_opt, start_position=-len(mode_arg))
        elif len(words) == 2 and words[0] == "/toolbar":
            tb_arg = words[1].lower()
            for tb_opt in ["on", "off"]:
                if tb_opt.startswith(tb_arg):
                    yield Completion(tb_opt, start_position=-len(tb_arg))


def handle_slash_command(cmd_line: str, session: "ChatSession") -> bool:
    """Execute a slash command within the chat session.

    Returns True if the REPL should exit, False to continue the session.
    """
    parts = cmd_line.strip().split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    c = session.console

    if cmd in ("/exit", "/quit"):
        return True

    if cmd == "/help":
        table = Table(title="Available Slash Commands", border_style="cyan")
        table.add_column("Command", style="bold cyan", no_wrap=True)
        table.add_column("Description", style="white")
        for k, v in SLASH_COMMANDS.items():
            table.add_row(k, v)
        c.print(table)
        c.print(
            "[dim]Keybindings: Ctrl-C interrupts generation · Ctrl-D exits · Alt-Enter multi-line[/dim]\n"
        )
        return False

    if cmd == "/clear":
        session.clear_history()
        c.print(f"[green]{icons.ok} Conversation history cleared.[/green]\n")
        return False

    if cmd == "/system":
        if arg:
            session.system_prompt = arg
            session.update_system_message(arg)
            c.print(f"[green]{icons.ok} System prompt updated.[/green]\n")
        else:
            c.print(
                f"[bold]Current System Prompt:[/bold]\n{session.system_prompt or '[dim](none)[/dim]'}\n"
            )
        return False

    if cmd == "/model":
        if arg:
            c.print(f"[dim]Switching model to '{arg}'...[/dim]")
            session.switch_model(arg)
        else:
            c.print(f"[bold]Active Model:[/bold] {session.model_name}")
            if session.plan:
                c.print(f"[dim]Plan ID: {session.plan.id} ({session.plan.verdict})[/dim]\n")
        return False

    if cmd == "/plan":
        if not session.plan:
            c.print("[yellow]No active execution plan available.[/yellow]\n")
            return False

        p = session.plan
        c.print(f"\n[bold cyan]Execution Plan: {p.id}[/bold cyan]")
        c.print(f"  • Verdict: {p.verdict}")
        c.print(f"  • Context window: {p.kv.context} tokens (KV type: {p.kv.type_k})")

        total_gpu_layers = sum(len(layers) for layers in p.placement.gpu_layers.values())
        c.print(f"  • Layers: {total_gpu_layers} on GPU, {len(p.placement.cpu_layers)} on CPU")
        vram_peak_gb = sum(p.memory.vram_peak_bytes.values()) / (1024**3)
        ram_peak_gb = p.memory.ram_peak_bytes / (1024**3)
        c.print(f"  • Memory predicted: {vram_peak_gb:.1f} GB VRAM, {ram_peak_gb:.1f} GB RAM")
        c.print(
            f"  • Expected Speed: {p.speed.decode_tok_s_low}–{p.speed.decode_tok_s_high} tok/s (Bottleneck: {p.speed.bottleneck})\n"
        )
        return False

    if cmd == "/mem":
        if session.governor:
            try:
                metrics = session.governor.get_latest_metrics()
                c.print("\n[bold]Memory Metrics (Live Governor):[/bold]")
                c.print(f"  • RAM RSS: {metrics.ram_rss_bytes / (1024**2):.1f} MB")
                c.print(f"  • Available RAM: {metrics.ram_available_bytes / (1024**3):.1f} GB")
                for gpu_idx, vram_used in metrics.vram_used_bytes.items():
                    c.print(f"  • GPU #{gpu_idx} VRAM used: {vram_used / (1024**2):.1f} MB")
                if metrics.warnings:
                    c.print("[yellow bold]Governor Warnings:[/yellow bold]")
                    for w in metrics.warnings:
                        c.print(f"  [yellow]• {w}[/yellow]")
                c.print()
            except Exception as e:
                c.print(f"[yellow]Could not read governor metrics: {e}[/yellow]\n")
        elif session.plan:
            c.print("\n[bold]Memory Metrics (From Execution Plan):[/bold]")
            vram_gb = sum(session.plan.memory.vram_peak_bytes.values()) / (1024**3)
            ram_gb = session.plan.memory.ram_peak_bytes / (1024**3)
            c.print(f"  • Peak VRAM: {vram_gb:.1f} GB")
            c.print(f"  • Peak RAM: {ram_gb:.1f} GB\n")
        return False

    if cmd == "/stats":
        stats = session.client.stats
        c.print("\n[bold cyan]Session Generation Statistics:[/bold cyan]")
        c.print(f"  • Total conversation turns: {stats.total_turns}")
        c.print(f"  • Total prompt tokens: {stats.total_prompt_tokens}")
        c.print(f"  • Total completion tokens: {stats.total_completion_tokens}")
        c.print(f"  • Total generation time: {stats.total_duration_s:.1f}s")
        c.print(f"  • Average speed: {stats.avg_tokens_per_second} tok/s")
        if stats.last_metrics:
            m = stats.last_metrics
            c.print(
                f"  • Last turn speed: {m.tokens_per_second} tok/s (TTFT: {m.time_to_first_token_s:.2f}s)"
            )
        c.print()
        return False

    if cmd == "/ctx":
        if not arg:
            ctx_len = session.plan.kv.context if session.plan else 4096
            c.print(f"[bold]Current Context Length:[/bold] {ctx_len} tokens")
            c.print("[dim]To change, use: `/ctx 8192` or `/ctx auto`[/dim]\n")
        else:
            try:
                new_ctx = int(arg) if arg != "auto" else 4096
                c.print(f"[dim]Replanning engine with context {new_ctx}...[/dim]")
                session.replan(new_context=new_ctx)
                c.print(f"[green]{icons.ok} Replan successful: context set to {new_ctx}.[/green]\n")
            except Exception as e:
                c.print(f"[red]Failed to change context: {e}[/red]\n")
        return False

    if cmd == "/temp":
        if arg:
            try:
                val = float(arg)
                session.temperature = val
                c.print(f"[green]{icons.ok} Temperature set to {val}.[/green]\n")
            except ValueError:
                c.print("[red]Invalid temperature. Must be a float, e.g. `/temp 0.7`[/red]\n")
        else:
            c.print(f"[bold]Current Temperature:[/bold] {session.temperature}\n")
        return False

    if cmd == "/top_p":
        if arg:
            try:
                val = float(arg)
                session.top_p = val
                c.print(f"[green]{icons.ok} Top-p set to {val}.[/green]\n")
            except ValueError:
                c.print("[red]Invalid top-p. Must be a float, e.g. `/top_p 0.9`[/red]\n")
        else:
            c.print(f"[bold]Current Top-p:[/bold] {session.top_p}\n")
        return False

    if cmd == "/seed":
        if arg:
            try:
                val = int(arg)
                session.seed = val
                c.print(f"[green]{icons.ok} Seed set to {val}.[/green]\n")
            except ValueError:
                c.print("[red]Invalid seed. Must be an integer, e.g. `/seed 42`[/red]\n")
        else:
            c.print(f"[bold]Current Seed:[/bold] {session.seed or 'None (random)'}\n")
        return False

    if cmd == "/save":
        try:
            target_path = Path(arg) if arg else None
            saved_to = save_transcript(
                messages=session.messages,
                model_name=session.model_name,
                plan_id=session.plan.id if session.plan else None,
                dest_path=target_path,
            )
            c.print(f"[green]{icons.ok} Transcript saved to:[/green] {saved_to}\n")
        except Exception as e:
            c.print(f"[red]Failed to save transcript: {e}[/red]\n")
        return False

    if cmd == "/load":
        if not arg:
            c.print("[yellow]Usage: `/load <path_to_transcript>`[/yellow]\n")
        else:
            try:
                loaded_msgs = load_transcript(arg)
                session.messages = loaded_msgs
                c.print(
                    f"[green]{icons.ok} Loaded {len(loaded_msgs)} messages from {arg}.[/green]\n"
                )
            except Exception as e:
                c.print(f"[red]Failed to load transcript: {e}[/red]\n")
        return False

    if cmd == "/retry":
        # Remove last assistant message and re-run last prompt
        if not session.messages:
            c.print("[yellow]No previous messages to retry.[/yellow]\n")
            return False

        if session.messages[-1].role == "assistant":
            session.messages.pop()

        if session.messages and session.messages[-1].role == "user":
            last_prompt = session.messages[-1].content
            # Pop user message since send_prompt will append it again
            session.messages.pop()
            c.print(f"[dim]Retrying last prompt: {last_prompt[:50]}...[/dim]\n")
            session.send_prompt(last_prompt)
        else:
            c.print("[yellow]No user prompt found to retry.[/yellow]\n")
        return False

    if cmd == "/undo":
        if not session.messages:
            c.print("[yellow]Nothing to undo.[/yellow]\n")
            return False

        # If last was assistant, drop it
        if session.messages[-1].role == "assistant":
            session.messages.pop()
        # If preceding was user, drop it
        if session.messages and session.messages[-1].role == "user":
            session.messages.pop()

        c.print(f"[green]{icons.ok} Last conversation turn undone.[/green]\n")
        return False

    if cmd == "/copy":
        last_assistant_msg = next(
            (m.content for m in reversed(session.messages) if m.role == "assistant"), None
        )
        if not last_assistant_msg:
            c.print("[yellow]No assistant response to copy.[/yellow]\n")
            return False

        try:
            import platform
            import subprocess

            os_name = platform.system().lower()
            if os_name == "darwin":
                proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                proc.communicate(last_assistant_msg.encode("utf-8"))
            elif os_name == "windows":
                proc = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                proc.communicate(last_assistant_msg.encode("utf-8"))
            else:
                proc = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
                proc.communicate(last_assistant_msg.encode("utf-8"))
            c.print(f"[green]{icons.ok} Copied last response to clipboard.[/green]\n")
        except Exception:
            c.print("[dim]Clipboard utility not available. Last message content:[/dim]")
            c.print(last_assistant_msg)
            c.print()
        return False

    if cmd == "/mode":
        if arg in ("fast", "balanced", "quality", "max"):
            c.print(f"[dim]Replanning with mode '{arg}'...[/dim]")
            try:
                session.replan(new_mode=arg)
                c.print(
                    f"[green]{icons.ok} Mode changed to {arg.upper()} and engine replanned.[/green]\n"
                )
            except Exception as e:
                c.print(f"[red]Failed to replan: {e}[/red]\n")
        else:
            c.print("[yellow]Usage: `/mode <fast|balanced|quality|max>`[/yellow]\n")
        return False

    if cmd == "/budget":
        if arg:
            c.print(f"[dim]Replanning with budget '{arg}'...[/dim]")
            try:
                session.replan(new_budget=arg)
                c.print(f"[green]{icons.ok} Budget updated and engine replanned.[/green]\n")
            except Exception as e:
                c.print(f"[red]Failed to replan: {e}[/red]\n")
        else:
            c.print("[yellow]Usage: `/budget <e.g. 8GB|conservative|balanced|max>`[/yellow]\n")
        return False

    if cmd == "/serve":
        c.print(
            f"\n[bold green]{icons.ok} Engine running at:[/bold green] {session.client.base_url}"
        )
        c.print(f"  • OpenAI Endpoint: {session.client.api_url}/chat/completions")
        c.print("  • Compatible with Open WebUI, Continue.dev, LangChain, and OpenAI SDK:\n")
        c.print(
            f'    [cyan]client = OpenAI(base_url="{session.client.api_url}", api_key="not-needed")[/cyan]\n'
        )
        return False

    if cmd == "/toolbar":
        if arg == "off":
            session.toolbar_enabled = False
            c.print("[dim]Bottom status toolbar hidden.[/dim]\n")
        elif arg == "on":
            session.toolbar_enabled = True
            c.print("[dim]Bottom status toolbar enabled.[/dim]\n")
        else:
            session.toolbar_enabled = not session.toolbar_enabled
            state_str = "enabled" if session.toolbar_enabled else "hidden"
            c.print(f"[dim]Bottom status toolbar {state_str}.[/dim]\n")
        return False

    c.print(f"[yellow]Unknown command '{cmd}'. Type `/help` for available commands.[/yellow]\n")
    return False
