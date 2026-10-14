"""Planner engine: searches candidate configurations, ranks them, and handles remedies."""

from ramllm.core.planner.candidates import generate_candidate_plans
from ramllm.core.planner.feasibility import (
    calculate_safety_margins,
    generate_remedies,
)
from ramllm.core.schemas import (
    ExecutionPlan,
    HardwareProfile,
    KvConfig,
    MemoryEstimate,
    ModelInfo,
    Placement,
    SpeedEstimate,
    UserRequest,
)


def _score_plan(plan: ExecutionPlan, preference: str) -> float:
    """Score plan based on user performance preference."""
    if plan.verdict == "INFEASIBLE":
        return -1000.0

    speed = plan.speed.decode_tok_s_low

    if preference == "fastest":
        return speed

    elif preference == "fit_at_any_cost":
        # Prioritize feasibility and higher context / quality, even if slower
        verdict_weights = {
            "COMFORTABLE": 100.0,
            "CONSTRAINED": 80.0,
            "BATCH_ONLY": 40.0,
            "INFEASIBLE": 0.0,
        }
        return verdict_weights.get(plan.verdict, 0.0) + speed * 0.1

    else:  # balanced
        # Penalize aggressive KV quantization slightly for quality
        kv_penalty = 0.0
        if plan.kv.type_k == "q4_0":
            kv_penalty = 2.0
        elif plan.kv.type_k == "q8_0":
            kv_penalty = 0.5

        # Reward higher speed and non-disk execution
        disk_penalty = 15.0 if plan.memory.disk_read_bytes_per_token > 0 else 0.0
        return max(0.01, speed - kv_penalty - disk_penalty)


def create_execution_plans(
    model: ModelInfo,
    hw: HardwareProfile,
    req: UserRequest,
) -> list[ExecutionPlan]:
    """Analyze hardware and model to produce a ranked list of viable ExecutionPlans.

    If no plan fits without crashing, returns a single INFEASIBLE plan containing
    explicit remedy recommendations.
    """
    usable_ram, usable_vram = calculate_safety_margins(hw, req)

    candidates = generate_candidate_plans(
        model=model,
        hw=hw,
        req=req,
        usable_ram=usable_ram,
        usable_vram=usable_vram,
    )

    # Filter out INFEASIBLE candidates unless none remain
    viable = [c for c in candidates if c.verdict != "INFEASIBLE"]

    if not viable:
        # Generate specific remedies
        remedies = generate_remedies(
            model=model,
            req=req,
            usable_ram=usable_ram,
            usable_vram=usable_vram,
            min_required_bytes=model.total_weight_bytes,
        )
        infeasible_plan = ExecutionPlan(
            id="infeasible_budget_exceeded",
            model_variant=model.quant,
            backend="llamacpp",
            placement=Placement(
                gpu_layers={},
                cpu_layers=list(range(model.n_layers)),
                disk_streamed_layers=[],
            ),
            kv=KvConfig(
                location="cpu",
                type_k="q4_0" if req.allow_kv_quant else "f16",
                type_v="q4_0" if req.allow_kv_quant else "f16",
                context=req.target_context or req.min_context,
            ),
            memory=MemoryEstimate(
                vram_peak_bytes={},
                ram_peak_bytes=model.total_weight_bytes,
            ),
            speed=SpeedEstimate(
                decode_tok_s_low=0.0,
                decode_tok_s_high=0.0,
                prefill_tok_s_low=0.0,
                prefill_tok_s_high=0.0,
                bottleneck="ram_bw",
            ),
            verdict="INFEASIBLE",
            warnings=remedies,
            backend_args=[],
            explanation=[
                f"Model weights ({model.total_weight_bytes / (1024**3):.1f} GB) exceed usable RAM+VRAM "
                f"({(usable_ram + sum(usable_vram.values())) / (1024**3):.1f} GB) under current constraints."
            ],
        )
        return [infeasible_plan]

    # Rank viable candidates
    ranked = sorted(viable, key=lambda p: _score_plan(p, req.preference), reverse=True)
    return ranked
