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


def calculate_moe_coverage_fraction(
    n_experts: int,
    n_experts_used: int,
    num_tokens: int = 1,
) -> float:
    """Compute expected fraction of unique routed experts activated across num_tokens.

    For num_tokens = 1 (single-stream decode): exactly n_experts_used / n_experts.
    For num_tokens > 1 (prefill, batch > 1, or mmap/disk paging window):
        P(expert activated) = 1 - (1 - n_experts_used / n_experts) ** num_tokens
    """
    if n_experts <= 0 or n_experts_used <= 0:
        return 1.0
    if n_experts_used >= n_experts:
        return 1.0
    if num_tokens <= 1:
        return n_experts_used / n_experts

    prob_not_selected_single = 1.0 - (float(n_experts_used) / float(n_experts))
    prob_never_selected = prob_not_selected_single**num_tokens
    coverage = 1.0 - prob_never_selected
    return min(1.0, max(float(n_experts_used) / float(n_experts), coverage))


def calculate_moe_working_set_bytes(
    non_layer_bytes: int,
    per_layer_bytes: list[int],
    expert_bytes_per_layer: list[int] | None,
    shared_expert_bytes_per_layer: list[int] | None,
    n_experts: int | None,
    n_experts_used: int | None,
    num_tokens: int = 1,
) -> int:
    """Compute working set bytes that must be read for MoE models across num_tokens.

    Dense layers + Attention + Shared experts + coverage_fraction * Routed experts.
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

    active_expert_ratio = calculate_moe_coverage_fraction(
        n_experts=n_experts,
        n_experts_used=n_experts_used,
        num_tokens=num_tokens,
    )
    working_set = non_layer_bytes

    for i, layer_total in enumerate(per_layer_bytes):
        exp_bytes = expert_bytes_per_layer[i] if i < len(expert_bytes_per_layer) else 0
        non_expert_layer_bytes = max(0, layer_total - exp_bytes)
        active_expert_bytes = int(exp_bytes * active_expert_ratio)
        working_set += non_expert_layer_bytes + active_expert_bytes

    return working_set
