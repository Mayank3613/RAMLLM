"""Unit tests for scripts/validate_predictions.py aggregation logic."""

import sys
from pathlib import Path

# Add scripts directory to path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "scripts"))

from validate_predictions import (  # type: ignore[import-not-found]  # noqa: E402
    compute_metrics_for_subset,
    generate_markdown_report,
)

from ramllm.telemetry.calibration import BenchmarkRecord  # noqa: E402


def test_validate_predictions_empty() -> None:
    rep = generate_markdown_report([])
    assert "No empirical benchmark records found" in rep


def test_validate_predictions_metrics() -> None:
    rec1 = BenchmarkRecord(
        schema_version=2,
        timestamp="2026-10-17T10:00:00",
        model_path="/models/llama3.gguf",
        plan_id="p1",
        arch="llama",
        quant="q4_k_m",
        strategy="full_gpu",
        predicted_decode_low=10.0,
        predicted_decode_high=20.0,
        measured_decode_tok_s=15.0,  # mid = 15, error = 0%
        predicted_prefill_low=50.0,
        predicted_prefill_high=70.0,
        measured_prefill_tok_s=60.0,  # mid = 60, error = 0%
        peak_ram_rss_bytes=1000,
        decode_within_predicted=True,
        prefill_within_predicted=True,
    )
    rec2 = BenchmarkRecord(
        schema_version=2,
        timestamp="2026-10-17T10:05:00",
        model_path="/models/qwen2.gguf",
        plan_id="p2",
        arch="qwen2",
        quant="q8_0",
        strategy="split",
        predicted_decode_low=10.0,
        predicted_decode_high=20.0,
        measured_decode_tok_s=30.0,  # mid = 15, error = 100%
        predicted_prefill_low=50.0,
        predicted_prefill_high=70.0,
        measured_prefill_tok_s=30.0,  # mid = 60, error = 50%
        peak_ram_rss_bytes=1000,
        decode_within_predicted=False,
        prefill_within_predicted=False,
    )

    metrics = compute_metrics_for_subset([rec1, rec2])
    assert metrics["count"] == 2
    assert metrics["decode_in_range_pct"] == 50.0
    assert metrics["prefill_in_range_pct"] == 50.0
    assert metrics["decode_mape"] == 50.0  # (0 + 100) / 2
    assert metrics["prefill_mape"] == 25.0  # (0 + 50) / 2

    report = generate_markdown_report([rec1, rec2])
    assert "# RAMLLM Cost Model Validation Report" in report
    assert "Total Benchmarks Analyzed: **2**" in report
    assert "`llama`" in report
    assert "`qwen2`" in report
    assert "`full_gpu`" in report
    assert "`split`" in report
