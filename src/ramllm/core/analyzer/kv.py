"""Key-Value attention cache memory formulas and compute overhead estimation."""

from typing import Literal

# Byte sizes per element including block overhead for quantization
KV_TYPE_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,  # 32 values (32 bytes) + 2-byte scale per block -> 34/32
    "q4_0": 0.5625,  # 32 values (16 bytes) + 2-byte scale per block -> 18/32
}

KV_TYPE_BITS: dict[str, float] = {
    "f32": 32.0,
    "f16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q4_0": 4.5,
}


def kv_cache_bytes(
    n_layers: int,
    context_length: int,
    n_kv_heads: int,
    head_dim: int,
    type_k: Literal["f16", "q8_0", "q4_0"] = "f16",
    type_v: Literal["f16", "q8_0", "q4_0"] = "f16",
    n_slots: int = 1,
    sliding_window_size: int | None = None,
    kv_lora_rank: int | None = None,
    is_hybrid_ssm: bool = False,
    ssm_d_state: int | None = None,
) -> int:
    """Calculate the exact total memory footprint of the KV cache in bytes.

    Formula:
        kv_bytes = n_slots * (L * C * H_kv * D * (bits_K + bits_V) / 8)

    Handles:
    - Multiplied by parallel server slots (n_slots, -np)
    - Sliding-window attention layers (capped at min(context_length, sliding_window_size))
    - MLA-style compressed KV (DeepSeek-V2/V3 low-rank dimension)
    - Hybrid / SSM layers (recurrent state buffer)
    """
    bits_k = KV_TYPE_BITS.get(type_k, 16.0)
    bits_v = KV_TYPE_BITS.get(type_v, 16.0)

    eff_context = context_length
    if sliding_window_size and sliding_window_size > 0:
        eff_context = min(context_length, sliding_window_size)

    if is_hybrid_ssm:
        d_state = ssm_d_state or 16
        ssm_bytes_per_layer = float(eff_context * d_state * 4)
        total_ssm = n_layers * ssm_bytes_per_layer * n_slots
        return int(total_ssm)

    if kv_lora_rank and kv_lora_rank > 0:
        bytes_per_token = (kv_lora_rank * (bits_k + bits_v)) / 16.0
        total_bytes = n_layers * eff_context * bytes_per_token * n_slots
        return int(total_bytes)

    elements_per_token = n_kv_heads * head_dim
    bytes_per_token = (elements_per_token * (bits_k + bits_v)) / 8.0
    total_bytes = n_layers * eff_context * bytes_per_token * n_slots
    return int(total_bytes)


def estimate_compute_buffer_bytes(
    context_length: int,
    batch_size: int = 512,
    hidden_size: int = 4096,
    intermediate_size: int | None = None,
    ubatch_size: int = 512,
) -> int:
    """Estimate scratch and compute buffer memory in bytes.

    Scales with ubatch size, context length, and largest intermediate activation tensors.
    """
    base_overhead = 350 * 1024 * 1024

    inter_dim = intermediate_size or int(hidden_size * 2.7)
    activation_scratch = ubatch_size * inter_dim * 4
    attn_scratch = min(context_length, 8192) * ubatch_size * 4

    total = base_overhead + activation_scratch + attn_scratch
    clamped = max(400 * 1024 * 1024, min(total, int(2.5 * 1024**3)))
    return int(clamped)
