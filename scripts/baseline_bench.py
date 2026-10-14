#!/usr/bin/env python3
"""
Research spike benchmark script for RAMLLM Milestone 0.
Tests llama-cli across offload levels, MoE CPU offload, and KV quantization.
"""

import re
import subprocess
import sys
import time


def run_bench(model_path: str, ngl: int, extra_args: list[str]) -> dict:
    cmd = [
        "llama-cli",
        "-m",
        model_path,
        "-p",
        "Why is the sky blue? Answer in 20 words.",
        "-n",
        "20",
        "-ngl",
        str(ngl),
        *extra_args,
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    # Parse llama-cli output for timings
    out = res.stdout + res.stderr
    prefill_tok_s = None
    decode_tok_s = None

    match_prompt = re.search(r"prompt eval time = .*?([\d\.]+) t/s", out)
    if match_prompt:
        prefill_tok_s = float(match_prompt.group(1))

    match_eval = re.search(r"eval time = .*?([\d\.]+) t/s", out)
    if match_eval:
        decode_tok_s = float(match_eval.group(1))

    return {
        "ngl": ngl,
        "extra_args": " ".join(extra_args),
        "prefill_tok_s": prefill_tok_s,
        "decode_tok_s": decode_tok_s,
        "wall_time_s": elapsed,
        "success": res.returncode == 0,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python baseline_bench.py <path-to-gguf>")
        sys.exit(1)
    model = sys.argv[1]
    print(f"Benchmarking {model}")
