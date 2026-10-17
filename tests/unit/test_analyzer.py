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


def test_kv_cache_formula_exact_regression() -> None:
    """Prove rewritten formula L*C*H_kv*D*(bits_K+bits_V)/8 gives identical results to prior implementation."""

    def _old_kv_cache_bytes(n_layers, context_length, n_kv_heads, head_dim, type_k, type_v):
        kv_type_bytes = {
            "f32": 4.0,
            "f16": 2.0,
            "bf16": 2.0,
            "q8_0": 1.0625,
            "q4_0": 0.5625,
        }
        b_k = kv_type_bytes[type_k]
        b_v = kv_type_bytes[type_v]
        total_elements_per_layer = context_length * n_kv_heads * head_dim
        bytes_per_layer = total_elements_per_layer * (b_k + b_v)
        return int(n_layers * bytes_per_layer)

    test_configs = [
        (32, 4096, 8, 128),
        (28, 8192, 4, 128),
        (48, 2048, 8, 64),
        (16, 16384, 2, 128),
    ]
    quant_pairs = [
        ("f16", "f16"),
        ("q8_0", "q8_0"),
        ("q4_0", "q4_0"),
        ("q8_0", "f16"),
        ("q4_0", "f16"),
        ("q8_0", "q4_0"),
    ]

    for n_layers, ctx, n_kv, head_dim in test_configs:
        for tk, tv in quant_pairs:
            old_val = _old_kv_cache_bytes(n_layers, ctx, n_kv, head_dim, tk, tv)
            new_val = kv_cache_bytes(n_layers, ctx, n_kv, head_dim, type_k=tk, type_v=tv)
            assert new_val == old_val, f"Mismatch for {tk}/{tv}: new={new_val}, old={old_val}"


def test_kv_cache_slots_and_advanced_architectures() -> None:
    # 1. Parallel slots (-np 4)
    base = kv_cache_bytes(32, 4096, 8, 128, type_k="f16", type_v="f16", n_slots=1)
    slots4 = kv_cache_bytes(32, 4096, 8, 128, type_k="f16", type_v="f16", n_slots=4)
    assert slots4 == base * 4

    # 2. Sliding window: context is 8192, but window is 2048
    sw_bytes = kv_cache_bytes(
        32, 8192, 8, 128, type_k="f16", type_v="f16", sliding_window_size=2048
    )
    expected_sw = kv_cache_bytes(32, 2048, 8, 128, type_k="f16", type_v="f16")
    assert sw_bytes == expected_sw

    # 3. MLA compressed KV: dimension is rank 512 instead of 2 * 8 * 128 = 2048
    mla_bytes = kv_cache_bytes(32, 4096, 8, 128, type_k="f16", type_v="f16", kv_lora_rank=512)
    # MLA uses 512 elements per token vs standard 1024 elements (8 * 128)
    assert mla_bytes < base
    assert mla_bytes == base // 4

    # 4. SSM / Mamba recurrent state
    ssm_bytes = kv_cache_bytes(32, 4096, 8, 128, is_hybrid_ssm=True, ssm_d_state=16)
    assert ssm_bytes > 0


def test_moe_coverage_regimes() -> None:
    from ramllm.core.analyzer.moe import calculate_moe_coverage_fraction

    # 8 experts total, 2 used per token (25% active)
    # 1. Single stream decode (T = 1): exactly 2 / 8 = 0.25
    cov_decode = calculate_moe_coverage_fraction(n_experts=8, n_experts_used=2, num_tokens=1)
    assert cov_decode == 0.25

    # 2. Short batch (T = 4): 1 - (1 - 2/8)^4 = 1 - (0.75)^4 = 1 - 0.316 = 0.684
    cov_batch = calculate_moe_coverage_fraction(n_experts=8, n_experts_used=2, num_tokens=4)
    assert 0.65 < cov_batch < 0.72

    # 3. Prefill (T = 512): nearly 1.0 (all experts accessed across 512 prompt tokens)
    cov_prefill = calculate_moe_coverage_fraction(n_experts=8, n_experts_used=2, num_tokens=512)
    assert cov_prefill > 0.999

    # Working set test for prefill
    non_layer = 100_000_000
    per_layer = [100_000_000] * 4
    expert_bytes = [80_000_000] * 4

    ws_decode = calculate_moe_working_set_bytes(
        non_layer, per_layer, expert_bytes, None, n_experts=8, n_experts_used=2, num_tokens=1
    )
    ws_prefill = calculate_moe_working_set_bytes(
        non_layer, per_layer, expert_bytes, None, n_experts=8, n_experts_used=2, num_tokens=512
    )

    assert ws_decode == 260_000_000
    # During prefill, all 80MB expert weights per layer must be loaded
    assert ws_prefill > ws_decode
    assert ws_prefill == non_layer + sum(per_layer)  # 500 MB total


def test_compute_buffer_scaling() -> None:
    # Base with ubatch=512
    buf512 = estimate_compute_buffer_bytes(4096, hidden_size=4096, ubatch_size=512)
    # Scaled with ubatch=2048
    buf2048 = estimate_compute_buffer_bytes(4096, hidden_size=4096, ubatch_size=2048)
    assert buf2048 > buf512


def test_analyze_gguf_security_guards(tmp_path: Path) -> None:
    """Verify strict security validation against malicious, non-existent, and malformed files."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        analyze_gguf(tmp_path / "does_not_exist.gguf")

    # 2. File too small (<16 bytes)
    too_small = tmp_path / "too_small.gguf"
    too_small.write_bytes(b"GGUF123")
    with pytest.raises(ValueError, match="too small"):
        analyze_gguf(too_small)

    # 3. Invalid magic bytes
    bad_magic = tmp_path / "bad_magic.gguf"
    bad_magic.write_bytes(b"NOTG" + b"0" * 20)
    with pytest.raises(ValueError, match="Invalid magic header"):
        analyze_gguf(bad_magic)

    # 4. Target is a directory
    dir_path = tmp_path / "a_directory.gguf"
    dir_path.mkdir()
    with pytest.raises(ValueError, match="not a regular file"):
        analyze_gguf(dir_path)
