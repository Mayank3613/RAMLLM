"""Benchmark execution runner and calibration harness."""

import time
from datetime import datetime
from pathlib import Path

import httpx

from ramllm.backends import LlamaCppAdapter
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.profiler import probe_hardware
from ramllm.core.schemas import ExecutionPlan, UserRequest
from ramllm.telemetry.calibration import BenchmarkRecord, append_benchmark_record

BENCH_PROMPTS = [
    "Why is the sky blue? Answer in 25 words.",
    "Write a concise python function to calculate the fibonacci series up to N.",
]


def run_benchmark(
    model_path: Path | str,
    plan_id: str | None = None,
    port: int = 8085,
    max_tokens: int = 64,
) -> BenchmarkRecord:
    """Run empirical benchmark on model using llama.cpp, measure speed/memory, and record result."""
    p = Path(model_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    from ramllm.core.planner import create_execution_plans

    hw = probe_hardware(quick=True)
    info = analyze_gguf(p)
    req = UserRequest(model=str(p), min_context=2048, target_context=2048)
    plans = create_execution_plans(info, hw, req)

    # Pick plan
    chosen_plan: ExecutionPlan | None = None
    if plan_id:
        for pl in plans:
            if pl.id == plan_id:
                chosen_plan = pl
                break
    if not chosen_plan:
        chosen_plan = plans[0]

    if chosen_plan.verdict == "INFEASIBLE":
        raise RuntimeError("Cannot benchmark an INFEASIBLE plan.")

    adapter = LlamaCppAdapter()
    install = adapter.detect()
    if not install:
        raise RuntimeError("No llama-server backend found.")

    proc = adapter.launch(chosen_plan, port=port, timeout_s=40.0)
    governor = MemoryGovernor(proc, hw, sample_interval_s=0.1)
    governor.start()

    total_tokens = 0
    total_eval_time = 0.0
    first_token_latency = 0.0
    peak_rss = 0

    try:
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        for prompt in BENCH_PROMPTS:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": True,
            }

            t0 = time.perf_counter()
            t_first = 0.0
            token_count = 0

            with httpx.stream("POST", url, json=payload, timeout=60.0) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        if t_first == 0.0:
                            t_first = time.perf_counter()
                        token_count += 1

            t_end = time.perf_counter()
            if t_first > 0:
                first_token_latency = max(first_token_latency, t_first - t0)
                eval_time = max(t_end - t_first, 1e-3)
                total_eval_time += eval_time
                total_tokens += token_count

            # Sample governor RSS
            metrics = governor.get_latest_metrics()
            peak_rss = max(peak_rss, metrics.ram_rss_bytes)

        decode_tok_s = (total_tokens / total_eval_time) if total_eval_time > 0 else 0.0
        prefill_tok_s = (1.0 / first_token_latency * 10) if first_token_latency > 0 else 0.0

        record = BenchmarkRecord(
            timestamp=datetime.now().isoformat(),
            model_path=str(p),
            plan_id=chosen_plan.id,
            predicted_decode_low=chosen_plan.speed.decode_tok_s_low,
            predicted_decode_high=chosen_plan.speed.decode_tok_s_high,
            measured_decode_tok_s=round(decode_tok_s, 2),
            predicted_prefill_low=chosen_plan.speed.prefill_tok_s_low,
            predicted_prefill_high=chosen_plan.speed.prefill_tok_s_high,
            measured_prefill_tok_s=round(prefill_tok_s, 2),
            peak_ram_rss_bytes=peak_rss,
            ttft_s=round(first_token_latency, 3),
        )

        append_benchmark_record(record)
        return record

    finally:
        governor.stop()
        adapter.stop(proc)
