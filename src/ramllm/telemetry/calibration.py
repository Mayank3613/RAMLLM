"""Calibration system for fine-tuning speed predictions from empirical benchmarks."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

BENCH_DIR = Path.home() / ".local" / "share" / "ramllm"
BENCH_LOG_FILE = BENCH_DIR / "bench.jsonl"


class BenchmarkRecord(BaseModel):
    """A single empirical benchmark run record for cost model calibration."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    model_path: str
    plan_id: str
    predicted_decode_low: float
    predicted_decode_high: float
    measured_decode_tok_s: float
    predicted_prefill_low: float
    predicted_prefill_high: float
    measured_prefill_tok_s: float
    peak_ram_rss_bytes: int
    ttft_s: float = Field(default=0.0)


def append_benchmark_record(record: BenchmarkRecord) -> None:
    """Append a benchmark result to the local bench.jsonl log."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json() + "\n"
    with open(BENCH_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def get_calibration_factor(model_path: str | None = None) -> float:
    """Calculate moving average ratio of measured speed vs predicted speed.

    Returns a multiplier (default 1.0, clamped between 0.5 and 2.0) used to
    calibrate decode speed estimates on this machine.
    """
    if not BENCH_LOG_FILE.exists():
        return 1.0

    ratios: list[float] = []
    try:
        with open(BENCH_LOG_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                rec = BenchmarkRecord.model_validate(data)

                if model_path and rec.model_path != model_path:
                    continue

                pred_mid = (rec.predicted_decode_low + rec.predicted_decode_high) / 2.0
                if pred_mid > 0 and rec.measured_decode_tok_s > 0:
                    ratio = rec.measured_decode_tok_s / pred_mid
                    ratios.append(ratio)
    except Exception:
        return 1.0

    if not ratios:
        return 1.0

    # Moving average over recent benchmarks (up to last 10)
    recent = ratios[-10:]
    avg_ratio = sum(recent) / len(recent)
    # Clamp to avoid extreme distortions
    return max(0.5, min(avg_ratio, 2.0))
