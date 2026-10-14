"""Unit tests for Model Analyzer, KV formulas, and MoE analysis."""

from pathlib import Path

import pytest
from tests.fixtures.make_gguf import create_tiny_dense_gguf, create_tiny_moe_gguf

from ramllm.core.analyzer import (
    analyze_gguf,
    calculate_moe_working_set_bytes,
    estimate_compute_buffer_bytes,
    kv_cache_bytes,
)


def test_kv_cache_bytes_calculation() -> None:
    # 32 layers, 4096 context, 8 kv heads, 128 head dim
    # f16 = 2 bytes per element
    f16_bytes = kv_cache_bytes(32, 4096, 8, 128, type_k="f16", type_v="f16")
    # 32 * 4096 * 8 * 128 * 4 = 536,870,912 bytes = 512 MB
    assert f16_bytes == 536_870_912

    # q8_0 = 1.0625 bytes per element
    q8_bytes = kv_cache_bytes(32, 4096, 8, 128, type_k="q8_0", type_v="q8_0")
    assert q8_bytes < f16_bytes
    assert abs(q8_bytes / f16_bytes - (1.0625 / 2.0)) < 0.01

    # q4_0 = 0.5625 bytes per element
    q4_bytes = kv_cache_bytes(32, 4096, 8, 128, type_k="q4_0", type_v="q4_0")
    assert q4_bytes < q8_bytes
    assert abs(q4_bytes / f16_bytes - (0.5625 / 2.0)) < 0.01


def test_estimate_compute_buffer_bytes() -> None:
    buf = estimate_compute_buffer_bytes(context_length=4096, batch_size=512, hidden_size=4096)
    assert 400 * 1024 * 1024 <= buf <= 1.5 * 1024**3


def test_moe_working_set_calculation() -> None:
    # 100 MB non-layer, 4 layers of 100 MB each (total model = 500 MB)
    # Each layer has 80 MB expert weights and 20 MB attention
    # 8 experts total, 2 active per token (25% active)
    non_layer = 100_000_000
    per_layer = [100_000_000] * 4
    expert_bytes = [80_000_000] * 4

    working_set = calculate_moe_working_set_bytes(
        non_layer_bytes=non_layer,
        per_layer_bytes=per_layer,
        expert_bytes_per_layer=expert_bytes,
        shared_expert_bytes_per_layer=None,
        n_experts=8,
        n_experts_used=2,
    )
    # Expected: 100 + 4 * (20 + 80 * 0.25) = 100 + 4 * (40) = 260 MB
    assert working_set == 260_000_000
    assert working_set < sum(per_layer) + non_layer


def test_analyze_tiny_dense_gguf(tmp_path: Path) -> None:
    path = tmp_path / "tiny_dense.gguf"
    create_tiny_dense_gguf(path)

    info = analyze_gguf(path)
    assert info.arch == "llama"
    assert info.n_layers == 4
    assert info.hidden_size == 128
    assert info.n_heads == 8
    assert info.n_kv_heads == 8
    assert info.context_length_max == 4096
    assert not info.is_moe
    assert len(info.per_layer_bytes) == 4
    assert sum(info.per_layer_bytes) + info.non_layer_bytes == info.total_weight_bytes


def test_analyze_tiny_moe_gguf(tmp_path: Path) -> None:
    path = tmp_path / "tiny_moe.gguf"
    create_tiny_moe_gguf(path)

    info = analyze_gguf(path)
    assert info.arch == "qwen3moe"
    assert info.n_layers == 2
    assert info.is_moe
    assert info.n_experts == 4
    assert info.n_experts_used == 2
    assert info.expert_bytes_per_layer is not None
    assert len(info.expert_bytes_per_layer) == 2
    assert info.expert_bytes_per_layer[0] > 0


@pytest.mark.slow
def test_analyze_real_qwen_moe() -> None:
    p = Path("/Users/cassain/Models/qwen3-coder/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf")
    if not p.exists():
        pytest.skip("Local model not found")
    info = analyze_gguf(p)
    assert info.is_moe
    assert info.n_experts is not None and info.n_experts > 0
    assert info.n_experts_used is not None and info.n_experts_used > 0
    assert info.total_weight_bytes > 10 * 1024**3
