"""Unit tests for plan service, budget resolution, modes, and plain-English summaries."""

from pathlib import Path

from tests.fixtures.make_gguf import create_tiny_dense_gguf

from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.planner import create_execution_plans
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import UserRequest
from ramllm.services.plan_service import (
    format_plain_summary,
    resolve_budgets,
    translate_mode,
)


def test_resolve_budgets_auto() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    b = resolve_budgets(hw, preset="balanced")
    assert b.ram_bytes > 0
    assert 0 in b.vram_bytes_per_gpu
    assert b.vram_bytes_per_gpu[0] > 0
    assert "Using up to" in b.explanation


def test_resolve_budgets_memory_flag() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    b = resolve_budgets(hw, memory_total_str="10GB")
    total_gb = (b.ram_bytes + sum(b.vram_bytes_per_gpu.values())) / (1024**3)
    assert 8.0 <= total_gb <= 10.5
    assert "--memory 10GB" in b.explanation


def test_resolve_budgets_explicit() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    b = resolve_budgets(hw, ram_str="6GB")
    assert b.ram_bytes == 6 * 1024**3
    assert not b.vram_bytes_per_gpu


def test_translate_mode() -> None:
    pref, kv, ctx = translate_mode("fast", context_str="8k")
    assert pref == "fastest"
    assert kv is True
    assert ctx == 8192

    pref_q, kv_q, _ = translate_mode("quality")
    assert pref_q == "balanced"
    assert kv_q is False

    pref_m, kv_m, _ = translate_mode("max")
    assert pref_m == "fit_at_any_cost"


def test_format_plain_summary_on_synthetic_profiles(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny_summary_test.gguf"
    create_tiny_dense_gguf(model_path)
    model_info = analyze_gguf(model_path)

    profiles = [
        "tests/fixtures/hardware/laptop_16gb_gpu_6gb.json",
        "tests/fixtures/hardware/laptop_8gb_no_gpu.json",
        "tests/fixtures/hardware/apple_m_16gb_unified.json",
    ]

    for prof_path in profiles:
        hw = load_synthetic_profile(prof_path)
        budget = resolve_budgets(hw, preset="balanced")
        req = UserRequest(
            model=str(model_path),
            ram_budget_bytes=budget.ram_bytes,
            vram_budget_bytes=budget.vram_bytes_per_gpu,
            target_context=2048,
        )
        plans = create_execution_plans(model_info, hw, req)
        assert len(plans) > 0
        summary = format_plain_summary(plans[0], "TestModel-7B", hw)

        assert "TestModel-7B" in summary
        assert "Speed:" in summary
        assert "words per second" in summary
        assert "Memory:" in summary
        assert "Where it runs:" in summary
