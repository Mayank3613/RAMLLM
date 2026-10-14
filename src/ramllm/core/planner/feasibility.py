"""Safety margins, budget resolution, feasibility verdicts, and remedies."""

from typing import Literal

from ramllm.core.schemas import HardwareProfile, ModelInfo, UserRequest


def calculate_safety_margins(
    hw: HardwareProfile,
    req: UserRequest,
) -> tuple[int, dict[int, int]]:
    """Compute usable host RAM and usable VRAM per GPU after subtracting safety margins.

    Returns:
        tuple[usable_ram_bytes, dict[gpu_index, usable_vram_bytes]]
    """
    # 1. Host RAM margin
    if req.ram_budget_bytes is not None:
        usable_ram = req.ram_budget_bytes
    else:
        # Default safety: max(1.5 GB, 10% of total RAM)
        reserve_ram = max(int(1.5 * 1024**3), int(hw.ram_total_bytes * 0.10))
        usable_ram = max(0, hw.ram_available_bytes - reserve_ram)

    # 2. VRAM margins per GPU
    usable_vram: dict[int, int] = {}
    if not req.use_gpu or not hw.gpus:
        return usable_ram, usable_vram

    allowed_indices = req.gpu_indices or [g.index for g in hw.gpus]

    for g in hw.gpus:
        if g.index not in allowed_indices:
            continue

        if req.vram_budget_bytes and g.index in req.vram_budget_bytes:
            usable_vram[g.index] = req.vram_budget_bytes[g.index]
        else:
            if g.unified_memory:
                # Apple Silicon / Unified Memory: reserve max(3 GB, 20% of unified memory)
                reserve_vram = max(int(3.0 * 1024**3), int(g.vram_total_bytes * 0.20))
                usable_vram[g.index] = max(0, g.vram_free_bytes - reserve_vram)
            else:
                # Discrete GPU: reserve max(0.5 GB, 5% of total VRAM)
                reserve_vram = max(int(0.5 * 1024**3), int(g.vram_total_bytes * 0.05))
                usable_vram[g.index] = max(0, g.vram_free_bytes - reserve_vram)

    return usable_ram, usable_vram


def classify_verdict(
    decode_tok_s_low: float,
    decode_tok_s_high: float,
    has_disk_reads: bool,
    fits_budget: bool,
) -> Literal["COMFORTABLE", "CONSTRAINED", "BATCH_ONLY", "INFEASIBLE"]:
    """Classify execution plan into normative feasibility verdict."""
    if not fits_budget or decode_tok_s_high < 0.05:
        return "INFEASIBLE"

    if has_disk_reads:
        return "BATCH_ONLY" if decode_tok_s_low < 1.0 else "CONSTRAINED"

    if decode_tok_s_high >= 8.0 and decode_tok_s_low >= 4.0:
        return "COMFORTABLE"
    elif decode_tok_s_high >= 1.0:
        return "CONSTRAINED"
    else:
        return "BATCH_ONLY"


def generate_remedies(
    model: ModelInfo,
    req: UserRequest,
    usable_ram: int,
    usable_vram: dict[int, int],
    min_required_bytes: int,
) -> list[str]:
    """Generate concrete, actionable remedies when a request is infeasible."""
    remedies: list[str] = []
    total_budget = usable_ram + sum(usable_vram.values())
    deficit_bytes = min_required_bytes - total_budget

    if deficit_bytes > 0:
        deficit_gb = deficit_bytes / (1024**3)
        remedies.append(
            f"Increase RAM/VRAM budget by at least {deficit_gb:.1f} GB to hold minimum weights."
        )

    current_ctx = req.target_context or req.min_context
    if current_ctx > 2048:
        remedies.append(
            f"Lower context length from {current_ctx} to 2048 to reduce KV cache footprint."
        )

    if not req.allow_kv_quant:
        remedies.append("Enable KV cache quantization (--allow-kv-quant) to save up to 50% KV RAM.")

    if not model.is_moe and model.n_params > 14_000_000_000:
        remedies.append(
            "Consider a Mixture-of-Experts (MoE) model (e.g. Qwen-MoE or Mixtral) which activates fewer weights per token."
        )

    return remedies
