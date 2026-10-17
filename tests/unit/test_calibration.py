"""Unit tests for telemetry benchmark logging, calibration calculations, and schema migration."""

import json
from pathlib import Path

import pytest

from ramllm.telemetry.calibration import (
    BenchmarkRecord,
    append_benchmark_record,
    compute_record_metrics,
    get_calibration_factor,
    load_benchmark_records,
    migrate_legacy_record,
)


def test_calibration_factor_calculation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "bench.jsonl"
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_LOG_FILE", log_file)
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_DIR", tmp_path)

    # Empty log defaults to 1.0
    assert get_calibration_factor() == 1.0

    # Add record where measured was 20 tok/s vs predicted ~10 tok/s -> ratio = 2.0
    rec = BenchmarkRecord(
        timestamp="2026-10-14T10:00:00",
        model_path="/models/test.gguf",
        plan_id="plan-1",
        predicted_decode_low=8.0,
        predicted_decode_high=12.0,
        measured_decode_tok_s=20.0,
        predicted_prefill_low=50.0,
        predicted_prefill_high=70.0,
        measured_prefill_tok_s=60.0,
        peak_ram_rss_bytes=4 * 1024**3,
        ttft_s=0.2,
    )
    append_benchmark_record(rec)

    factor = get_calibration_factor()
    assert 1.9 <= factor <= 2.0


def test_compute_record_metrics() -> None:
    # Within range
    err, within = compute_record_metrics(10.0, 20.0, 15.0)
    assert within is True
    assert err == 0.0

    # Above range (mid = 10, measured = 15 -> +50%)
    err, within = compute_record_metrics(8.0, 12.0, 15.0)
    assert within is False
    assert err == 50.0

    # Below range (mid = 10, measured = 5 -> -50%)
    err, within = compute_record_metrics(8.0, 12.0, 5.0)
    assert within is False
    assert err == -50.0


def test_legacy_record_migration() -> None:
    legacy_data = {
        "timestamp": "2026-10-01T12:00:00",
        "model_path": "/path/to/llama-3-8b.gguf",
        "plan_id": "partial_offload_16L_f16",
        "predicted_decode_low": 10.0,
        "predicted_decode_high": 20.0,
        "measured_decode_tok_s": 15.0,
        "predicted_prefill_low": 80.0,
        "predicted_prefill_high": 120.0,
        "measured_prefill_tok_s": 100.0,
        "peak_ram_rss_bytes": 8000000000,
        "ttft_s": 0.15,
    }

    record = migrate_legacy_record(legacy_data)
    assert record.schema_version == 2
    assert record.arch == "unknown"
    assert record.quant == "unknown"
    assert record.strategy == "split"  # Inferred from partial_offload in plan_id
    assert record.decode_within_predicted is True
    assert record.decode_error_pct == 0.0
    assert record.prefill_within_predicted is True
    assert record.prefill_error_pct == 0.0


def test_load_benchmark_records_mixed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "bench.jsonl"
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_LOG_FILE", log_file)
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_DIR", tmp_path)

    # Write a v1 record and a v2 record
    v1_line = (
        json.dumps(
            {
                "timestamp": "2026-10-01T12:00:00",
                "model_path": "/models/v1.gguf",
                "plan_id": "full_gpu_f16",
                "predicted_decode_low": 20.0,
                "predicted_decode_high": 30.0,
                "measured_decode_tok_s": 25.0,
                "predicted_prefill_low": 100.0,
                "predicted_prefill_high": 150.0,
                "measured_prefill_tok_s": 125.0,
                "peak_ram_rss_bytes": 4000000,
            }
        )
        + "\n"
    )

    v2_rec = BenchmarkRecord(
        schema_version=2,
        timestamp="2026-10-02T12:00:00",
        model_path="/models/v2.gguf",
        plan_id="cpu_only_q8_0",
        arch="qwen2",
        quant="q8_0",
        backend_version="0.6.0",
        strategy="cpu_only",
        predicted_decode_low=5.0,
        predicted_decode_high=7.0,
        measured_decode_tok_s=6.0,
        predicted_prefill_low=20.0,
        predicted_prefill_high=30.0,
        measured_prefill_tok_s=25.0,
        peak_ram_rss_bytes=8000000,
        decode_within_predicted=True,
        decode_error_pct=0.0,
    )

    log_file.write_text(v1_line + v2_rec.model_dump_json() + "\n", encoding="utf-8")

    records = load_benchmark_records(log_file)
    assert len(records) == 2
    assert records[0].schema_version == 2
    assert records[0].strategy == "full_gpu"
    assert records[1].arch == "qwen2"
    assert records[1].strategy == "cpu_only"


def test_hierarchical_fallback_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_file = tmp_path / "bench.jsonl"
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_LOG_FILE", log_file)
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_DIR", tmp_path)

    # 1. Exact match record: llama, q4_k_m, b3800, split -> ratio = 15 / 10 = 1.5
    r_exact = BenchmarkRecord(
        schema_version=2,
        timestamp="2026-10-10T10:00:00",
        model_path="/models/llama3-exact.gguf",
        plan_id="split_plan",
        arch="llama",
        quant="q4_k_m",
        backend_version="b3800",
        strategy="split",
        predicted_decode_low=8.0,
        predicted_decode_high=12.0,  # mid = 10
        measured_decode_tok_s=15.0,  # ratio = 1.5
        predicted_prefill_low=50.0,
        predicted_prefill_high=50.0,
        measured_prefill_tok_s=100.0,  # ratio = 2.0
        peak_ram_rss_bytes=1000,
    )
    append_benchmark_record(r_exact)

    # 2. General arch record: qwen2, q8_0, b3800, cpu_only -> ratio = 6 / 10 = 0.6
    r_qwen = BenchmarkRecord(
        schema_version=2,
        timestamp="2026-10-10T10:05:00",
        model_path="/models/qwen2.gguf",
        plan_id="cpu_plan",
        arch="qwen2",
        quant="q8_0",
        backend_version="b3800",
        strategy="cpu_only",
        predicted_decode_low=8.0,
        predicted_decode_high=12.0,  # mid = 10
        measured_decode_tok_s=6.0,  # ratio = 0.6
        predicted_prefill_low=50.0,
        predicted_prefill_high=50.0,
        measured_prefill_tok_s=40.0,
        peak_ram_rss_bytes=1000,
    )
    append_benchmark_record(r_qwen)

    # Tier 1 Exact match for llama
    f_exact = get_calibration_factor(
        arch="llama",
        quant="q4_k_m",
        backend_version="b3800",
        strategy="split",
    )
    assert abs(f_exact - 1.5) < 0.05

    # Tier 3 Arch + Strategy match
    f_arch_strat = get_calibration_factor(
        arch="llama",
        quant="q8_0",  # Different quant, so falls back to arch+strategy
        backend_version="b3900",
        strategy="split",
    )
    assert abs(f_arch_strat - 1.5) < 0.05

    # Tier 5 Strategy match for cpu_only (matches qwen record)
    f_strat = get_calibration_factor(
        arch="mistral",  # Unseen arch
        strategy="cpu_only",
    )
    assert abs(f_strat - 0.6) < 0.05

    # Tier 6 Arch match for qwen2
    f_arch = get_calibration_factor(
        arch="qwen2",
    )
    assert abs(f_arch - 0.6) < 0.05

    # Tier 7 Global average for unseen arch and strategy
    f_global = get_calibration_factor(
        arch="unknown_model",
        strategy="unknown_strat",
    )
    # Average of 1.5 and 0.6 = 1.05
    assert abs(f_global - 1.05) < 0.05

    # Prefill phase calibration factor
    f_pref = get_calibration_factor(
        arch="llama",
        quant="q4_k_m",
        backend_version="b3800",
        strategy="split",
        phase="prefill",
    )
    assert abs(f_pref - 2.0) < 0.05
