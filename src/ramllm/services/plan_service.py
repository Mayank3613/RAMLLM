"""Planning service: budget resolution, mode translation, and plain-English summaries."""

from dataclasses import dataclass
from typing import Literal

from ramllm.cli.ui.theme import icons
from ramllm.core.schemas import ExecutionPlan, HardwareProfile
from ramllm.util.parsing import parse_bytes, parse_context

MEMORY_PRESETS = {
    "conservative": {"fraction": 0.50, "ram_reserve_gb": 2.0, "vram_reserve_mb": 512},
    "balanced": {"fraction": 0.70, "ram_reserve_gb": 1.5, "vram_reserve_mb": 256},
    "max": {"fraction": 0.85, "ram_reserve_gb": 1.0, "vram_reserve_mb": 128},
}

MODE_DESCRIPTIONS = {
    "fast": "Quickest replies, may trim quality",
    "balanced": "Good mix of speed and quality",
    "quality": "Best answers, may be slower",
    "max": "Biggest model that fits, even if slow",
}


@dataclass
class ResolvedBudget:
    ram_bytes: int
    vram_bytes_per_gpu: dict[int, int]
    explanation: str


def resolve_budgets(
    hw: HardwareProfile,
    ram_str: str | None = None,
    vram_str: str | None = None,
    memory_total_str: str | None = None,
    preset: str = "balanced",
    gpus_str: str | None = None,
) -> ResolvedBudget:
    """Resolve RAM and VRAM budgets either from explicit flags or automatic hardware heuristics."""
    preset_info = MEMORY_PRESETS.get(preset, MEMORY_PRESETS["balanced"])
    fraction = preset_info["fraction"]
    ram_reserve = int(preset_info["ram_reserve_gb"] * (1024**3))
    vram_reserve = int(preset_info["vram_reserve_mb"] * (1024**2))

    gpu_indices: list[int] = [g.index for g in hw.gpus] if hw.gpus else [0]
    if gpus_str:
        gpu_indices = [int(x.strip()) for x in gpus_str.split(",") if x.strip()]

    # Explicit --memory flag (total combined limit)
    if memory_total_str:
        total_limit = parse_bytes(memory_total_str) or int(8 * 1024**3)
        vram_dict: dict[int, int] = {}
        allocated_vram = 0

        if hw.gpus:
            for g in hw.gpus:
                if g.index in gpu_indices:
                    # Allocate up to free VRAM or total limit
                    avail_vram = max(g.vram_free_bytes - vram_reserve, 0)
                    take_vram = min(avail_vram, total_limit - allocated_vram)
                    if take_vram > 0:
                        vram_dict[g.index] = take_vram
                        allocated_vram += take_vram

        remaining_ram = max(total_limit - allocated_vram, 512 * 1024**2)
        ram_budget = min(remaining_ram, max(hw.ram_available_bytes - ram_reserve, 512 * 1024**2))

        ram_gb = ram_budget / (1024**3)
        vram_gb = sum(vram_dict.values()) / (1024**3)
        explanation = f"Using up to {ram_gb:.1f} GB RAM and {vram_gb:.1f} GB GPU memory (from --memory {memory_total_str})"
        return ResolvedBudget(
            ram_bytes=ram_budget, vram_bytes_per_gpu=vram_dict, explanation=explanation
        )

    # Explicit --ram / --vram
    has_explicit = bool(ram_str or vram_str)
    vram_dict = {}

    if vram_str:
        if "," in vram_str:
            parts = [parse_bytes(p.strip()) for p in vram_str.split(",")]
            for i, idx in enumerate(gpu_indices):
                if i < len(parts) and parts[i] is not None:
                    vram_dict[idx] = parts[i]  # type: ignore[assignment]
        else:
            single = parse_bytes(vram_str)
            if single is not None:
                for idx in gpu_indices:
                    vram_dict[idx] = single
    elif not has_explicit and hw.gpus:
        # Automatic VRAM budget
        for g in hw.gpus:
            if g.index in gpu_indices:
                auto_vram = max(int(g.vram_free_bytes * fraction - vram_reserve), 256 * 1024**2)
                vram_dict[g.index] = auto_vram

    if ram_str:
        ram_budget = parse_bytes(ram_str) or int(8 * 1024**3)
    else:
        # Automatic RAM budget
        ram_budget = max(int(hw.ram_available_bytes * fraction - ram_reserve), 512 * 1024**2)

    ram_gb = ram_budget / (1024**3)
    ram_total_gb = hw.ram_total_bytes / (1024**3)
    vram_gb = sum(vram_dict.values()) / (1024**3)
    gpu_total_vram_gb = sum(g.vram_total_bytes for g in hw.gpus) / (1024**3) if hw.gpus else 0.0

    if has_explicit:
        explanation = (
            f"Using requested budget: {ram_gb:.1f} GB RAM, {vram_gb:.1f} GB graphics memory."
        )
    else:
        vram_msg = (
            f" and {vram_gb:.1f} GB of your {gpu_total_vram_gb:.1f} GB graphics memory"
            if hw.gpus
            else ""
        )
        explanation = f"Using up to {ram_gb:.1f} GB of your {ram_total_gb:.1f} GB memory{vram_msg}."

    return ResolvedBudget(
        ram_bytes=ram_budget, vram_bytes_per_gpu=vram_dict, explanation=explanation
    )


def translate_mode(
    mode: str,
    context_str: str | None = None,
    no_kv_quant: bool = False,
) -> tuple[Literal["fastest", "balanced", "fit_at_any_cost"], bool, int]:
    """Translate user mode ('fast', 'balanced', 'quality', 'max') into planner parameters.

    Returns: (preference, allow_kv_quant, context_length)
    """
    m = mode.lower()
    ctx_len = parse_context(context_str) if context_str else 4096

    if m == "fast":
        return ("fastest", not no_kv_quant, ctx_len)
    if m == "quality":
        return ("balanced", False, ctx_len)
    if m == "max":
        return ("fit_at_any_cost", not no_kv_quant, ctx_len)
    # Default: balanced
    return ("balanced", not no_kv_quant, ctx_len)


def format_plain_summary(plan: ExecutionPlan, model_display_name: str, hw: HardwareProfile) -> str:
    """Format a 4-line plain-English plan summary."""
    words_low = max(int(plan.speed.decode_tok_s_low * 0.75), 0)
    words_high = max(int(plan.speed.decode_tok_s_high * 0.75), 1)

    speed_adjective = "fast"
    if words_low < 2:
        speed_adjective = "very slow"
    elif words_low < 8:
        speed_adjective = "moderate"

    verdict_descriptions = {
        "COMFORTABLE": "will run well on your computer",
        "CONSTRAINED": "works, but slowly on your computer",
        "BATCH_ONLY": "will run very slowly (best for background/batch jobs)",
        "INFEASIBLE": "won't fit in your available memory",
    }
    verdict_desc = verdict_descriptions.get(plan.verdict, plan.verdict)

    vram_peak_gb = sum(plan.memory.vram_peak_bytes.values()) / (1024**3)
    ram_peak_gb = plan.memory.ram_peak_bytes / (1024**3)
    gpu_total_gb = sum(g.vram_total_bytes for g in hw.gpus) / (1024**3) if hw.gpus else 0.0

    # Where it runs
    gpu_layer_count = (
        sum(len(layers) for layers in plan.placement.gpu_layers.values())
        if plan.placement.gpu_layers
        else 0
    )
    cpu_layer_count = len(plan.placement.cpu_layers) if plan.placement.cpu_layers else 0

    if gpu_layer_count > 0 and cpu_layer_count == 0:
        placement_text = "the whole model fits on your GPU"
    elif gpu_layer_count > 0 and cpu_layer_count > 0:
        placement_text = f"partially on GPU ({gpu_layer_count} layers) and rest in system RAM ({cpu_layer_count} layers)"
    elif plan.placement.expert_placement == "cpu" or "cmoe" in plan.id:
        placement_text = "attention in GPU memory, active experts routed through system RAM"
    else:
        placement_text = "runs on CPU and system RAM"

    mem_desc = f"{vram_peak_gb:.1f} GB of your {gpu_total_gb:.1f} GB graphics memory, {ram_peak_gb:.1f} GB of RAM"

    status_icon = (
        icons.ok
        if plan.verdict == "COMFORTABLE"
        else (icons.warn if plan.verdict == "CONSTRAINED" else icons.fail)
    )

    lines = [
        f"{status_icon} {model_display_name} {verdict_desc}",
        f"  {icons.bullet} Speed: {speed_adjective} — about {words_low}–{words_high} words per second   [{plan.verdict}]",
        f"  {icons.bullet} Memory: {mem_desc}",
        f"  {icons.bullet} Where it runs: {placement_text}",
    ]

    if plan.warnings:
        lines.append(f"  {icons.bullet} Note: {plan.warnings[0]}")
    elif plan.verdict == "COMFORTABLE" and vram_peak_gb < gpu_total_gb * 0.7:
        lines.append(
            f"  {icons.bullet} Tip: ask for a longer memory with --context 8k or 16k — it still fits comfortably."
        )

    return "\n".join(lines)
