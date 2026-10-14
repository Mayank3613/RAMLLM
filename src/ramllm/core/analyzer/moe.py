"""Mixture-of-Experts (MoE) topology analysis and working set calculations."""

import re

# Regex patterns identifying expert tensors in GGUF
EXPERT_PATTERN = re.compile(r"blk\.\d+\.ffn_.*_exps\.")
SHARED_EXPERT_PATTERN = re.compile(r"blk\.\d+\.ffn_.*_shexp\.")


def is_expert_tensor(name: str) -> bool:
    """Check if tensor name corresponds to a routed expert."""
    return bool(EXPERT_PATTERN.search(name))


def is_shared_expert_tensor(name: str) -> bool:
    """Check if tensor name corresponds to a shared expert."""
    return bool(SHARED_EXPERT_PATTERN.search(name))


def calculate_moe_working_set_bytes(
    non_layer_bytes: int,
    per_layer_bytes: list[int],
    expert_bytes_per_layer: list[int] | None,
    shared_expert_bytes_per_layer: list[int] | None,
    n_experts: int | None,
    n_experts_used: int | None,
) -> int:
    """Compute working set bytes that must be read per generated token for MoE models.

    Dense layers + Attention + Shared experts + (n_experts_used / n_experts) * Routed experts.
    """
    if (
        not expert_bytes_per_layer
        or not n_experts
        or not n_experts_used
        or n_experts <= 0
        or n_experts_used <= 0
    ):
        # Dense model: all weights are read
        return non_layer_bytes + sum(per_layer_bytes)

    active_expert_ratio = n_experts_used / n_experts
    working_set = non_layer_bytes

    for i, layer_total in enumerate(per_layer_bytes):
        exp_bytes = expert_bytes_per_layer[i] if i < len(expert_bytes_per_layer) else 0
        non_expert_layer_bytes = max(0, layer_total - exp_bytes)
        active_expert_bytes = int(exp_bytes * active_expert_ratio)
        working_set += non_expert_layer_bytes + active_expert_bytes

    return working_set
