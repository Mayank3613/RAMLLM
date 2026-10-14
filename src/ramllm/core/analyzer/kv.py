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


def kv_cache_bytes(
    n_layers: int,
    context_length: int,
    n_kv_heads: int,
    head_dim: int,
    type_k: Literal["f16", "q8_0", "q4_0"] = "f16",
    type_v: Literal["f16", "q8_0", "q4_0"] = "f16",
) -> int:
    """Calculate the exact total memory footprint of the KV cache in bytes.

    Formula:
        kv_bytes = n_layers * context_length * n_kv_heads * head_dim * (bytes(type_k) + bytes(type_v))
    """
    b_k = KV_TYPE_BYTES.get(type_k, 2.0)
    b_v = KV_TYPE_BYTES.get(type_v, 2.0)
    total_elements_per_layer = context_length * n_kv_heads * head_dim
    bytes_per_layer = total_elements_per_layer * (b_k + b_v)
    return int(n_layers * bytes_per_layer)


def estimate_compute_buffer_bytes(
    context_length: int,
    batch_size: int = 512,
    hidden_size: int = 4096,
) -> int:
    """Estimate scratch and compute buffer memory in bytes.

    Scales conservatively with batch size and context length, bounded between 0.3 GB and 1.5 GB.
    """
    # Base runtime context overhead (allocator structures, CUDA/Metal handle slack)
    base_overhead = 350 * 1024 * 1024  # 350 MB

    # Scratch graph activations: scales with ubatch * hidden_size * 4 (float32 activations)
    activation_scratch = batch_size * hidden_size * 16

    # Context attention scratch: scales with batch * context
    attn_scratch = min(context_length, 8192) * batch_size * 4

    total = base_overhead + activation_scratch + attn_scratch
    # Bound between 400 MB and 1.5 GB
    clamped = max(400 * 1024 * 1024, min(total, int(1.5 * 1024**3)))
    return int(clamped)
