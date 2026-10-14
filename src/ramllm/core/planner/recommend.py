"""Model recommendation engine over local catalog and hardware constraints (Milestone 4)."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.planner.planner import create_execution_plans
from ramllm.core.schemas import ExecutionPlan, HardwareProfile, UserRequest
from ramllm.registry.catalog import ModelCatalogEntry, list_models


class ModelRecommendation(BaseModel):
    """Ranked model recommendation matching hardware budgets and performance requirements."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    arch: str
    n_params: int
    quant: str
    is_moe: bool
    best_plan: ExecutionPlan
    score: float
    reasons: list[str]


def _quant_quality_bonus(quant: str) -> float:
    q = quant.upper()
    if "Q8" in q or "F16" in q:
        return 15.0
    if "Q6" in q:
        return 12.0
    if "Q5" in q:
        return 10.0
    if "Q4" in q:
        return 8.0
    if "Q3" in q:
        return 4.0
    return 1.0


def recommend_models(
    hw: HardwareProfile,
    req: UserRequest,
    catalog: list[ModelCatalogEntry] | None = None,
) -> list[ModelRecommendation]:
    """Evaluate models in catalog and recommend the highest quality feasible models for hardware.

    Honesty & Acceptance Rule: Returns ONLY non-INFEASIBLE models for the requested hardware profile.
    """
    entries = catalog if catalog is not None else list_models()
    recommendations: list[ModelRecommendation] = []

    for entry in entries:
        p = Path(entry.path)
        if not p.exists():
            continue

        try:
            model_info = analyze_gguf(p)
        except Exception:
            continue

        # Generate candidate execution plans
        plans = create_execution_plans(model_info, hw, req)
        if not plans:
            continue

        best_plan = plans[0]

        # Filter out INFEASIBLE models
        if best_plan.verdict == "INFEASIBLE":
            continue

        # Score formulation
        reasons: list[str] = []
        score = 0.0

        # 1. Feasibility tier
        if best_plan.verdict == "COMFORTABLE":
            score += 100.0
            reasons.append("Comfortable execution headroom without host thrashing.")
        elif best_plan.verdict == "CONSTRAINED":
            score += 60.0
            reasons.append("Constrained execution: memory margins are tight.")
        elif best_plan.verdict == "BATCH_ONLY":
            score += 20.0
            reasons.append("Batch-only throughput due to tier bandwidth limits.")

        # 2. Model capacity (larger param count = higher intelligence)
        param_billions = model_info.n_params / 1e9
        score += min(60.0, param_billions * 4.0)

        # 3. Quantization quality
        score += _quant_quality_bonus(model_info.quant)

        # 4. Decode speed
        speed_bonus = min(40.0, best_plan.speed.decode_tok_s_low * 1.5)
        score += speed_bonus

        # 5. Architecture specific advantages
        if model_info.is_moe:
            reasons.append(
                f"MoE architecture: only ~{int(param_billions * 0.15)}B parameters active per token."
            )
        if len(best_plan.placement.gpu_layers) > 1:
            reasons.append(
                f"Multi-GPU pipeline split across {len(best_plan.placement.gpu_layers)} accelerators."
            )
        elif (
            sum(len(layer_list) for layer_list in best_plan.placement.gpu_layers.values())
            == model_info.n_layers
        ):
            reasons.append("100% GPU VRAM residency for maximum throughput.")

        recommendations.append(
            ModelRecommendation(
                name=entry.name,
                path=entry.path,
                arch=model_info.arch,
                n_params=model_info.n_params,
                quant=model_info.quant,
                is_moe=model_info.is_moe,
                best_plan=best_plan,
                score=round(score, 1),
                reasons=reasons,
            )
        )

    # Sort descending by score
    recommendations.sort(key=lambda r: r.score, reverse=True)
    return recommendations
