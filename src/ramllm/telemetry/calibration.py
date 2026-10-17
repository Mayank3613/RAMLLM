"""Calibration system for fine-tuning speed predictions from empirical benchmarks."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BENCH_DIR = Path.home() / ".local" / "share" / "ramllm"
BENCH_LOG_FILE = BENCH_DIR / "bench.jsonl"


class BenchmarkRecord(BaseModel):
    """A single empirical benchmark run record for cost model calibration."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(default=2)
    timestamp: str
    model_path: str
    plan_id: str
    arch: str = Field(default="unknown")
    quant: str = Field(default="unknown")
    backend_version: str = Field(default="unknown")
    strategy: str = Field(default="unknown")
    predicted_decode_low: float
    predicted_decode_high: float
    measured_decode_tok_s: float
    predicted_prefill_low: float
    predicted_prefill_high: float
    measured_prefill_tok_s: float
    peak_ram_rss_bytes: int
    ttft_s: float = Field(default=0.0)
    decode_error_pct: float | None = Field(default=None)
    prefill_error_pct: float | None = Field(default=None)
    decode_within_predicted: bool | None = Field(default=None)
    prefill_within_predicted: bool | None = Field(default=None)


def compute_record_metrics(
    pred_low: float, pred_high: float, measured: float
) -> tuple[float | None, bool]:
    """Compute error percentage against prediction midpoint and within-range boolean."""
    mid = (pred_low + pred_high) / 2.0
    within = pred_low <= measured <= pred_high
    error_pct: float | None = None
    if mid > 0:
        error_pct = round(((measured - mid) / mid) * 100.0, 2)
    return error_pct, within


def migrate_legacy_record(data: dict[str, Any]) -> BenchmarkRecord:
    """Migrate a v1 or unversioned benchmark record dictionary to schema v2."""
    raw = dict(data)
    raw["schema_version"] = 2
    raw.setdefault("arch", "unknown")
    raw.setdefault("quant", "unknown")
    raw.setdefault("backend_version", "unknown")

    plan_id = str(raw.get("plan_id", ""))
    if "strategy" not in raw or not raw["strategy"] or raw["strategy"] == "unknown":
        if "full_gpu" in plan_id:
            raw["strategy"] = "full_gpu"
        elif "cpu_only" in plan_id:
            raw["strategy"] = "cpu_only"
        elif "partial" in plan_id or "split" in plan_id:
            raw["strategy"] = "split"
        elif "disk" in plan_id:
            raw["strategy"] = "disk_stream"
        elif "moe" in plan_id:
            raw["strategy"] = "moe_cpu_experts"
        else:
            raw["strategy"] = "unknown"

    pred_dec_low = float(raw.get("predicted_decode_low", 0.0))
    pred_dec_high = float(raw.get("predicted_decode_high", 0.0))
    meas_dec = float(raw.get("measured_decode_tok_s", 0.0))
    dec_err, dec_within = compute_record_metrics(pred_dec_low, pred_dec_high, meas_dec)
    if raw.get("decode_error_pct") is None:
        raw["decode_error_pct"] = dec_err
    if raw.get("decode_within_predicted") is None:
        raw["decode_within_predicted"] = dec_within

    pred_pref_low = float(raw.get("predicted_prefill_low", 0.0))
    pred_pref_high = float(raw.get("predicted_prefill_high", 0.0))
    meas_pref = float(raw.get("measured_prefill_tok_s", 0.0))
    pref_err, pref_within = compute_record_metrics(pred_pref_low, pred_pref_high, meas_pref)
    if raw.get("prefill_error_pct") is None:
        raw["prefill_error_pct"] = pref_err
    if raw.get("prefill_within_predicted") is None:
        raw["prefill_within_predicted"] = pref_within

    return BenchmarkRecord.model_validate(raw)


def load_benchmark_records(log_path: Path | None = None) -> list[BenchmarkRecord]:
    """Load and migrate all benchmark records from the JSONL log file."""
    p = log_path or BENCH_LOG_FILE
    if not p.exists():
        return []

    records: list[BenchmarkRecord] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    if data.get("schema_version") == 2:
                        records.append(BenchmarkRecord.model_validate(data))
                    else:
                        records.append(migrate_legacy_record(data))
                except Exception:
                    continue
    except Exception:
        return []
    return records


def append_benchmark_record(record: BenchmarkRecord, log_path: Path | None = None) -> None:
    """Append a benchmark result to the local bench.jsonl log."""
    target_file = log_path or BENCH_LOG_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)
    line = f"{record.model_dump_json()}\n"
    with open(target_file, "a", encoding="utf-8") as f:
        f.write(line)


def get_calibration_factor(
    model_path: str | None = None,
    arch: str | None = None,
    quant: str | None = None,
    backend_version: str | None = None,
    strategy: str | None = None,
    phase: Literal["decode", "prefill"] = "decode",
    log_path: Path | None = None,
) -> float:
    """Calculate moving average ratio of measured speed vs predicted speed.

    Uses hierarchical fallback lookup:
    1. Exact match: (arch, quant, backend_version, strategy, phase)
    2. Architecture + Quant + Strategy match: (arch, quant, strategy, phase)
    3. Architecture + Strategy: (arch, strategy, phase)
    4. Exact model path match (if model_path is provided)
    5. Strategy match: (strategy, phase)
    6. Architecture match: (arch, phase)
    7. Global moving average across all records for the phase
    8. Uncalibrated default 1.0

    Returns a multiplier (default 1.0, clamped between 0.5 and 2.0).
    """
    records = load_benchmark_records(log_path)
    if not records:
        return 1.0

    def get_ratio(r: BenchmarkRecord) -> float | None:
        if phase == "prefill":
            pred_mid = (r.predicted_prefill_low + r.predicted_prefill_high) / 2.0
            measured = r.measured_prefill_tok_s
        else:
            pred_mid = (r.predicted_decode_low + r.predicted_decode_high) / 2.0
            measured = r.measured_decode_tok_s
        if pred_mid > 0 and measured > 0:
            return measured / pred_mid
        return None

    def avg_ratios(matching_records: list[BenchmarkRecord]) -> float | None:
        ratios = [get_ratio(r) for r in matching_records]
        valid = [r for r in ratios if r is not None]
        if not valid:
            return None
        recent = valid[-10:]
        return sum(recent) / len(recent)

    # 1. Exact match: arch, quant, backend_version, strategy
    if arch and quant and backend_version and strategy:
        matched = [
            r
            for r in records
            if r.arch == arch
            and r.quant == quant
            and r.backend_version == backend_version
            and r.strategy == strategy
        ]
        avg = avg_ratios(matched)
        if avg is not None:
            return max(0.5, min(avg, 2.0))

    # 2. Architecture + Quant + Strategy
    if arch and quant and strategy:
        matched = [
            r for r in records if r.arch == arch and r.quant == quant and r.strategy == strategy
        ]
        avg = avg_ratios(matched)
        if avg is not None:
            return max(0.5, min(avg, 2.0))

    # 3. Architecture + Strategy
    if arch and strategy:
        matched = [r for r in records if r.arch == arch and r.strategy == strategy]
        avg = avg_ratios(matched)
        if avg is not None:
            return max(0.5, min(avg, 2.0))

    # 4. Exact model path match
    if model_path:
        matched = [r for r in records if r.model_path == model_path]
        avg = avg_ratios(matched)
        if avg is not None:
            return max(0.5, min(avg, 2.0))

    # 5. Strategy match
    if strategy:
        matched = [r for r in records if r.strategy == strategy]
        avg = avg_ratios(matched)
        if avg is not None:
            return max(0.5, min(avg, 2.0))

    # 6. Architecture match
    if arch:
        matched = [r for r in records if r.arch == arch]
        avg = avg_ratios(matched)
        if avg is not None:
            return max(0.5, min(avg, 2.0))

    # 7. Global moving average across all records for phase
    avg = avg_ratios(records)
    if avg is not None:
        return max(0.5, min(avg, 2.0))

    # 8. Uncalibrated fallback
    return 1.0
