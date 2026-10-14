"""Bottom status toolbar renderer for prompt_toolkit interactive chat session."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ramllm.cli.chat.repl import ChatSession


def format_tokens(n: int) -> str:
    """Format token count into compact human-readable representation."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def build_bottom_toolbar(session: "ChatSession") -> str:
    """Format one-line status string for prompt_toolkit bottom toolbar."""
    if not session.toolbar_enabled:
        return ""

    parts: list[str] = []

    # 1. Model display
    model_display = session.model_name
    if len(model_display) > 20:
        model_display = model_display[:18] + "…"
    parts.append(model_display)

    # 2. Context usage (estimated vs configured limit)
    ctx_limit = session.plan.kv.context if session.plan else 4096
    used_tokens = session.estimate_total_context_tokens()
    parts.append(f"ctx {format_tokens(used_tokens)}/{format_tokens(ctx_limit)}")

    # 3. Memory stats (from active governor or plan)
    if session.governor:
        try:
            metrics = session.governor.get_latest_metrics()
            vram_gb = sum(metrics.vram_used_bytes.values()) / (1024**3)
            ram_gb = metrics.ram_rss_bytes / (1024**3)
            if vram_gb > 0:
                parts.append(f"mem {vram_gb:.1f}G GPU · {ram_gb:.1f}G RAM")
            else:
                parts.append(f"mem {ram_gb:.1f}G RAM")
        except Exception:
            pass
    elif session.plan:
        vram_est = sum(session.plan.memory.vram_peak_bytes.values()) / (1024**3)
        ram_est = session.plan.memory.ram_peak_bytes / (1024**3)
        if vram_est > 0:
            parts.append(f"plan {vram_est:.1f}G GPU · {ram_est:.1f}G RAM")
        else:
            parts.append(f"plan {ram_est:.1f}G RAM")

    # 4. Generation Speed from last turn
    if session.client.last_metrics and session.client.last_metrics.tokens_per_second > 0:
        parts.append(f"last: {session.client.last_metrics.tokens_per_second:.1f} tok/s")

    return " │ ".join(parts)
