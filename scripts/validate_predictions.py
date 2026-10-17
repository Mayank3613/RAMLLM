#!/usr/bin/env python3
"""Aggregate empirical benchmark logs and validate cost model prediction accuracy."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure ramllm is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from ramllm.telemetry.calibration import (  # noqa: E402
    BENCH_LOG_FILE,
    BenchmarkRecord,
    load_benchmark_records,
)


def compute_metrics_for_subset(records: list[BenchmarkRecord]) -> dict[str, Any]:
    """Compute in-range % and MAPE for decode and prefill over a subset of records."""
    n = len(records)
    if n == 0:
        return {
            "count": 0,
            "decode_in_range_pct": 0.0,
            "decode_mape": 0.0,
            "prefill_in_range_pct": 0.0,
            "prefill_mape": 0.0,
        }

    dec_in_range = sum(1 for r in records if r.decode_within_predicted)
    pref_in_range = sum(1 for r in records if r.prefill_within_predicted)

    dec_mapes: list[float] = []
    pref_mapes: list[float] = []

    for r in records:
        dec_mid = (r.predicted_decode_low + r.predicted_decode_high) / 2.0
        if dec_mid > 0:
            dec_mapes.append(abs(r.measured_decode_tok_s - dec_mid) / dec_mid * 100.0)

        pref_mid = (r.predicted_prefill_low + r.predicted_prefill_high) / 2.0
        if pref_mid > 0:
            pref_mapes.append(abs(r.measured_prefill_tok_s - pref_mid) / pref_mid * 100.0)

    dec_mape = (sum(dec_mapes) / len(dec_mapes)) if dec_mapes else 0.0
    pref_mape = (sum(pref_mapes) / len(pref_mapes)) if pref_mapes else 0.0

    return {
        "count": n,
        "decode_in_range_pct": round((dec_in_range / n) * 100.0, 1),
        "decode_mape": round(dec_mape, 1),
        "prefill_in_range_pct": round((pref_in_range / n) * 100.0, 1),
        "prefill_mape": round(pref_mape, 1),
    }


def generate_markdown_report(records: list[BenchmarkRecord]) -> str:
    """Generate a GitHub-flavored Markdown validation report table."""
    if not records:
        return (
            "# RAMLLM Cost Model Validation Report\n\n"
            "No empirical benchmark records found. Run `ramllm bench <model.gguf>` to populate data.\n"
        )

    overall = compute_metrics_for_subset(records)

    by_arch: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    by_strategy: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    by_quant: dict[str, list[BenchmarkRecord]] = defaultdict(list)

    for r in records:
        by_arch[r.arch].append(r)
        by_strategy[r.strategy].append(r)
        by_quant[r.quant].append(r)

    md: list[str] = [
        "# RAMLLM Cost Model Validation Report",
        "",
        f"Total Benchmarks Analyzed: **{len(records)}**",
        "",
        "## Overall Prediction Accuracy",
        "",
        "| Metric | In-Range % | Mean Absolute % Error (MAPE) |",
        "| :--- | :---: | :---: |",
        f"| **Decode Speed (tok/s)** | {overall['decode_in_range_pct']}% | {overall['decode_mape']}% |",
        f"| **Prefill Speed (tok/s)** | {overall['prefill_in_range_pct']}% | {overall['prefill_mape']}% |",
        "",
        "## Breakdown by Architecture",
        "",
        "| Architecture | Runs | Decode In-Range % | Decode MAPE | Prefill In-Range % | Prefill MAPE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for arch, group in sorted(by_arch.items()):
        m = compute_metrics_for_subset(group)
        md.append(
            f"| `{arch}` | {m['count']} | {m['decode_in_range_pct']}% | {m['decode_mape']}% | "
            f"{m['prefill_in_range_pct']}% | {m['prefill_mape']}% |"
        )

    md.extend(
        [
            "",
            "## Breakdown by Placement Strategy",
            "",
            "| Strategy | Runs | Decode In-Range % | Decode MAPE | Prefill In-Range % | Prefill MAPE |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for strat, group in sorted(by_strategy.items()):
        m = compute_metrics_for_subset(group)
        md.append(
            f"| `{strat}` | {m['count']} | {m['decode_in_range_pct']}% | {m['decode_mape']}% | "
            f"{m['prefill_in_range_pct']}% | {m['prefill_mape']}% |"
        )

    md.extend(
        [
            "",
            "## Breakdown by Quantization",
            "",
            "| Quantization | Runs | Decode In-Range % | Decode MAPE | Prefill In-Range % | Prefill MAPE |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for quant, group in sorted(by_quant.items()):
        m = compute_metrics_for_subset(group)
        md.append(
            f"| `{quant}` | {m['count']} | {m['decode_in_range_pct']}% | {m['decode_mape']}% | "
            f"{m['prefill_in_range_pct']}% | {m['prefill_mape']}% |"
        )

    md.append("")
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark records and validate cost model predictions."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=BENCH_LOG_FILE,
        help="Path to bench.jsonl log file",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory containing exported report.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write output Markdown file (default: print to stdout)",
    )
    args = parser.parse_args()

    records = load_benchmark_records(args.log)

    if args.export_dir and args.export_dir.is_dir():
        for json_file in args.export_dir.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    if "benchmark" in data:
                        records.append(BenchmarkRecord.model_validate(data["benchmark"]))
                    elif "measured_decode_tok_s" in data:
                        records.append(BenchmarkRecord.model_validate(data))
            except Exception:
                continue

    report = generate_markdown_report(records)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Validation report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
