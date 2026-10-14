"""Interactive chat REPL session, process lifecycle, and prompt_toolkit loop."""

import atexit
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from rich.console import Console

from ramllm.backends.base import BackendAdapter, BackendProcess
from ramllm.cli.chat.client import ChatClient, ChatMessage
from ramllm.cli.chat.history import get_chat_history
from ramllm.cli.chat.render import ChatRenderer
from ramllm.cli.chat.slash import SlashCommandCompleter, handle_slash_command
from ramllm.cli.chat.toolbar import build_bottom_toolbar
from ramllm.cli.chat.transcripts import save_transcript
from ramllm.cli.ui.console import get_console
from ramllm.cli.ui.theme import icons
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.governor.recovery import launch_with_recovery
from ramllm.core.planner import create_execution_plans
from ramllm.core.schemas import ExecutionPlan, HardwareProfile, ModelInfo, UserRequest
from ramllm.services.model_resolver import resolve_model_reference
from ramllm.services.plan_service import resolve_budgets


class ChatSession:
    """State manager for an active RAMLLM chat session."""

    def __init__(
        self,
        model_name: str,
        client: ChatClient,
        model_path: Path | None = None,
        adapter: BackendAdapter | None = None,
        backend_proc: BackendProcess | None = None,
        governor: MemoryGovernor | None = None,
        plan: ExecutionPlan | None = None,
        model_info: ModelInfo | None = None,
        hw: HardwareProfile | None = None,
        req: UserRequest | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int | None = None,
        toolbar_enabled: bool = True,
        plain_mode: bool = False,
        auto_save_transcripts: bool = True,
        port: int = 8085,
        is_attached: bool = False,
        console: Console | None = None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.client = client
        self.adapter = adapter
        self.backend_proc = backend_proc
        self.governor = governor
        self.plan = plan
        self.model_info = model_info
        self.hw = hw
        self.req = req
        self.system_prompt = system_prompt or "You are a helpful, knowledgeable AI assistant."
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.toolbar_enabled = toolbar_enabled
        self.plain_mode = plain_mode
        self.auto_save_transcripts = auto_save_transcripts
        self.port = port
        self.is_attached = is_attached
        self.console = console or get_console()

        self.messages: list[ChatMessage] = []
        if self.system_prompt:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))

        self.renderer = ChatRenderer(console=self.console, plain=self.plain_mode)
        self._is_shutdown = False

    def estimate_total_context_tokens(self) -> int:
        """Estimate current token footprint of conversation history."""
        total_chars = sum(len(m.content) for m in self.messages)
        return max(0, total_chars // 4)

    def clear_history(self) -> None:
        """Clear conversation turns while keeping the active system prompt."""
        self.messages = []
        if self.system_prompt:
            self.messages.append(ChatMessage(role="system", content=self.system_prompt))

    def update_system_message(self, text: str) -> None:
        """Update or insert the system message in the message list."""
        if self.messages and self.messages[0].role == "system":
            self.messages[0].content = text
        else:
            self.messages.insert(0, ChatMessage(role="system", content=text))

    def check_context_window_pressure(self) -> None:
        """Warn user or automatically prune oldest messages if context limit is approaching."""
        ctx_limit = self.plan.kv.context if self.plan else 4096
        current_tokens = self.estimate_total_context_tokens()

        # Warn at 85% utilization
        if current_tokens >= int(ctx_limit * 0.85):
            self.console.print(
                f"[yellow][!] Context window is {int((current_tokens / ctx_limit) * 100)}% full "
                f"({current_tokens}/{ctx_limit} tokens). "
                f"Older messages will be pruned on next turn, or use `/ctx` to expand.[/yellow]\n"
            )

        # Auto-prune oldest turns if exceeding 95% to prevent hard engine crash
        while current_tokens > int(ctx_limit * 0.95) and len(self.messages) > 2:
            # Drop the oldest non-system pair
            start_idx = 1 if (self.messages and self.messages[0].role == "system") else 0
            if len(self.messages) > start_idx + 1:
                del self.messages[start_idx : start_idx + 2]
                current_tokens = self.estimate_total_context_tokens()
            else:
                break

    def replan(
        self,
        new_context: int | None = None,
        new_budget: str | None = None,
        new_mode: str | None = None,
    ) -> None:
        """Replan engine execution and restart backend with updated parameters."""
        if self.is_attached:
            # Talk to server replan endpoint
            url = f"{self.client.base_url}/ramllm/replan"
            import httpx

            body: dict[str, Any] = {}
            if new_context:
                body["context"] = new_context
            resp = httpx.post(url, json=body, timeout=30.0)
            if resp.status_code != 200:
                raise RuntimeError(f"Server replan failed: {resp.text}")
            return

        if not self.adapter or not self.model_info or not self.hw or not self.req:
            raise RuntimeError("Cannot replan: missing hardware profile or adapter.")

        # Shut down current backend and governor
        if self.governor:
            self.governor.stop()
            self.governor = None
        if self.backend_proc:
            self.adapter.stop(self.backend_proc)
            self.backend_proc = None

        # Build updated UserRequest
        ctx = new_context or (self.plan.kv.context if self.plan else 4096)
        if new_budget or new_mode:
            budgets = resolve_budgets(self.hw, preset=new_budget or "balanced")
            ram_limit: int | None = budgets.ram_bytes
            vram_limits: dict[int, int] | None = budgets.vram_bytes_per_gpu
        else:
            ram_limit = self.req.ram_budget_bytes if self.req else None
            vram_limits = self.req.vram_budget_bytes if self.req else None

        req = UserRequest(
            model=self.model_name,
            min_context=ctx,
            target_context=ctx,
            ram_budget_bytes=ram_limit,
            vram_budget_bytes=vram_limits,
        )

        plans = create_execution_plans(self.model_info, self.hw, req)
        chosen = plans[0]
        if chosen.verdict == "INFEASIBLE":
            raise ValueError(f"Requested configuration is infeasible: {'; '.join(chosen.warnings)}")

        proc, final_plan, _ = launch_with_recovery(
            adapter=self.adapter,
            plan=chosen,
            model=self.model_info,
            hw=self.hw,
            req=req,
            port=self.port,
            timeout_s=30.0,
        )

        self.backend_proc = proc
        self.plan = final_plan
        self.req = req
        self.governor = MemoryGovernor(proc, self.hw)
        self.governor.start()

    def switch_model(self, new_model_str: str) -> None:
        """Switch to another model while preserving conversation history."""
        resolved_path = resolve_model_reference(new_model_str, auto_download=True)
        if not self.adapter or not self.hw:
            raise RuntimeError("Cannot switch model without active hardware profile.")

        # Shut down current backend
        if self.governor:
            self.governor.stop()
            self.governor = None
        if self.backend_proc:
            self.adapter.stop(self.backend_proc)
            self.backend_proc = None

        from ramllm.core.analyzer.gguf import analyze_gguf

        new_info = analyze_gguf(resolved_path)
        budgets = resolve_budgets(self.hw, preset="balanced")
        req = UserRequest(
            model=resolved_path.name,
            min_context=4096,
            target_context=4096,
            ram_budget_bytes=budgets.ram_bytes,
            vram_budget_bytes=budgets.vram_bytes_per_gpu,
        )

        plans = create_execution_plans(new_info, self.hw, req)
        chosen = plans[0]
        proc, final_plan, _ = launch_with_recovery(
            adapter=self.adapter,
            plan=chosen,
            model=new_info,
            hw=self.hw,
            req=req,
            port=self.port,
            timeout_s=30.0,
        )

        self.model_name = resolved_path.stem
        self.model_path = resolved_path
        self.model_info = new_info
        self.backend_proc = proc
        self.plan = final_plan
        self.req = req
        self.governor = MemoryGovernor(proc, self.hw)
        self.governor.start()
        self.console.print(f"[green]{icons.ok} Switched model to {self.model_name}.[/green]\n")

    def send_prompt(self, user_text: str) -> str:
        """Send a user message, stream the assistant's reply, and update history."""
        self.messages.append(ChatMessage(role="user", content=user_text))
        self.check_context_window_pressure()

        stream = self.client.stream_chat(
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )

        reply, interrupted = self.renderer.render_stream(stream)
        self.messages.append(ChatMessage(role="assistant", content=reply, interrupted=interrupted))
        return reply

    def shutdown(self) -> None:
        """Cleanly stop governor, terminate backend engine process, and save transcript."""
        if self._is_shutdown:
            return
        self._is_shutdown = True

        if self.governor:
            try:
                self.governor.stop()
            except Exception:
                pass
            self.governor = None

        if self.backend_proc and self.adapter:
            try:
                self.adapter.stop(self.backend_proc)
            except Exception:
                pass
            self.backend_proc = None

        # Auto-save transcript if enabled and there are exchanges
        non_system_turns = [m for m in self.messages if m.role != "system"]
        if self.auto_save_transcripts and non_system_turns:
            try:
                saved_path = save_transcript(
                    messages=self.messages,
                    model_name=self.model_name,
                    plan_id=self.plan.id if self.plan else None,
                )
                self.console.print(f"\n[dim]Transcript archived to {saved_path}[/dim]")
            except Exception:
                pass


def run_repl(session: ChatSession) -> None:
    """Main interactive REPL loop powered by prompt_toolkit."""
    # Ensure clean shutdown on exit or signals
    atexit.register(session.shutdown)

    def _sig_handler(sig: int, frame: FrameType | None) -> None:
        session.shutdown()
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, AttributeError):
        pass

    c = session.console

    # Startup summary (2–3 lines)
    model_name = session.model_name
    quant_tag = ""
    if session.plan and session.plan.kv.type_k:
        quant_tag = f" ({session.plan.id})"

    c.print(f"[bold cyan]Ready: {model_name}{quant_tag}[/bold cyan]")
    if session.plan:
        p = session.plan
        gpu_layers = sum(len(layers) for layers in p.placement.gpu_layers.values())
        cpu_layers = len(p.placement.cpu_layers)
        placement = (
            f"GPU ({gpu_layers} layers)"
            if cpu_layers == 0
            else f"Hybrid ({gpu_layers} GPU / {cpu_layers} CPU)"
        )
        vram_gb = sum(p.memory.vram_peak_bytes.values()) / (1024**3)
        ram_gb = p.memory.ram_peak_bytes / (1024**3)
        speed = f"≈ {p.speed.decode_tok_s_low}–{p.speed.decode_tok_s_high} tok/s"
        c.print(
            f"[dim]Running on: {placement} · memory {vram_gb:.1f}G GPU, {ram_gb:.1f}G RAM · speed {speed}[/dim]"
        )
    c.print("[dim]Type /help for commands · Ctrl-C stops a reply · Ctrl-D or /exit to quit[/dim]\n")

    history = get_chat_history()
    completer = SlashCommandCompleter()

    prompt_style = Style.from_dict(
        {
            "prompt": "ansicyan bold",
            "bottom-toolbar": "bg:#222222 #cccccc",
        }
    )

    pt_session: PromptSession[str] = PromptSession(
        history=history,
        completer=completer,
        bottom_toolbar=lambda: build_bottom_toolbar(session),
        style=prompt_style,
        enable_history_search=True,
    )

    consecutive_ctrl_c = 0

    while True:
        try:
            user_input = pt_session.prompt(HTML("<prompt>&gt; </prompt>")).strip()
            consecutive_ctrl_c = 0

            if not user_input:
                continue

            # Multi-line fence handling: if line starts with """ or '''
            if user_input.startswith(('"""', "'''")):
                fence = user_input[:3]
                lines = [user_input[3:]]
                if not user_input.endswith(fence) or len(user_input) == 3:
                    while True:
                        try:
                            next_line = pt_session.prompt("... ")
                            if next_line.endswith(fence):
                                lines.append(next_line[:-3])
                                break
                            lines.append(next_line)
                        except (KeyboardInterrupt, EOFError):
                            break
                    user_input = "\n".join(lines).strip()

            # Slash command dispatch
            if user_input.startswith("/"):
                should_exit = handle_slash_command(user_input, session)
                if should_exit:
                    break
                continue

            # Standard conversation prompt
            session.send_prompt(user_input)
            c.print()

        except KeyboardInterrupt:
            consecutive_ctrl_c += 1
            if consecutive_ctrl_c == 1:
                c.print("\n[dim]Press Ctrl-C again or Ctrl-D (/exit) to quit.[/dim]")
            else:
                c.print("\n[dim]Goodbye![/dim]")
                break
        except EOFError:
            c.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            c.print(f"\n[red]Error:[/red] {e}\n")

    session.shutdown()


def start_chat_session(
    model_name: str,
    client: ChatClient,
    model_path: Path | None = None,
    adapter: BackendAdapter | None = None,
    backend_proc: BackendProcess | None = None,
    governor: MemoryGovernor | None = None,
    plan: ExecutionPlan | None = None,
    model_info: ModelInfo | None = None,
    hw: HardwareProfile | None = None,
    req: UserRequest | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = None,
    toolbar_enabled: bool = True,
    plain_mode: bool = False,
    auto_save_transcripts: bool = True,
    port: int = 8085,
    is_attached: bool = False,
) -> None:
    """Create a ChatSession and launch the interactive REPL."""
    session = ChatSession(
        model_name=model_name,
        client=client,
        model_path=model_path,
        adapter=adapter,
        backend_proc=backend_proc,
        governor=governor,
        plan=plan,
        model_info=model_info,
        hw=hw,
        req=req,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        toolbar_enabled=toolbar_enabled,
        plain_mode=plain_mode,
        auto_save_transcripts=auto_save_transcripts,
        port=port,
        is_attached=is_attached,
    )
    run_repl(session)
