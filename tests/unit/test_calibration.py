"""Unit tests for telemetry benchmark logging and calibration calculations."""

from pathlib import Path

import pytest

from ramllm.telemetry.calibration import (
    BenchmarkRecord,
    append_benchmark_record,
    get_calibration_factor,
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
