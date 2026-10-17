# Empirical Validation & Calibration Guide

RAMLLM utilizes an analytical cost model to predict prefill and decode throughput ($T_{\\text{tok/s}}$) and memory footprints before launching models. To ensure real-world accuracy across diverse architectures, quantization formats, and execution backends, RAMLLM provides an empirical benchmarking and multi-tiered calibration system.

---

## 1. Cost Model Prediction Bounds

The cost model yields throughput intervals $[L, H]$ for each candidate execution plan:
- **$L$ (Pessimistic / Lower Bound):** Accounts for memory contention, thermal throttling, minimum PCIe bus efficiency (60%), and synchronization latency between CPU and GPU boundaries.
- **$H$ (Optimistic / Upper Bound):** Assumes saturated peak memory bandwidth (85% STREAM efficiency), zero bus stalls, and optimal kernel dispatch.

### Accuracy Metrics
Prediction quality is quantified using two primary metrics:
1. **In-Range Coverage Rate (%):**
   $$\\text{Coverage} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(L_i \\le M_i \\le H_i) \\times 100\\%$$
   where $M_i$ is the empirical measured throughput.
2. **Mean Absolute Percentage Error (MAPE):**
   $$\\text{MAPE} = \\frac{1}{N} \\sum_{i=1}^N \\frac{|M_i - \\text{Mid}_i|}{\\text{Mid}_i} \\times 100\\%$$
   where $\\text{Mid}_i = \\frac{L_i + H_i}{2}$.

---

## 2. Running Benchmarks

Run empirical benchmarks using `ramllm bench`:

```bash
# Standard benchmark on default plan
ramllm bench /path/to/model.gguf

# Benchmark a specific execution plan
ramllm bench /path/to/model.gguf --plan-id partial_offload_16L_f16_gpu

# Benchmark and export a structured validation report for community submission
ramllm bench /path/to/model.gguf --export report.json
```

### Benchmark Output

The CLI prints a rich comparison table displaying predicted intervals, measured throughput, error percentages, and in-range verdicts:

```text
╭──────────────────────────────── Benchmark Results ────────────────────────────────╮
│ Metric                      Predicted Range   Measured       Error %   In Range?  │
├───────────────────────────────────────────────────────────────────────────────────┤
│ Decode Speed                ~8.5–12.0 tok/s   10.4 tok/s       +1.5%     ✓ Yes    │
│ Prefill Speed               ~65.0–90.0 tok/s  78.2 tok/s       +0.9%     ✓ Yes    │
│ Time to First Token (TTFT)  -                 0.185 s              -         -    │
│ Peak RAM RSS                -                 5120.4 MB            -         -    │
╰───────────────────────────────────────────────────────────────────────────────────╯
Saved to bench.jsonl. Active machine calibration factor: 1.02x
Report exported to report.json
```

---

## 3. Stratified Calibration Architecture

Empirical runs are logged locally to `~/.local/share/ramllm/bench.jsonl` using Schema v2. Unlike naive global scalars, RAMLLM employs a **hierarchical fallback lookup** to calibrate cost model predictions:

1. **Tier 1 (Exact Match):** `(arch, quant, backend_version, strategy, phase)`  
   Matches identical model architecture, quantization block type, llama.cpp release, and offload placement strategy.
2. **Tier 2 (Arch + Quant + Strategy):** `(arch, quant, strategy, phase)`  
   Matches architecture, quant, and strategy across backend versions.
3. **Tier 3 (Arch + Strategy):** `(arch, strategy, phase)`  
   Matches architecture and placement strategy across different quantization formats.
4. **Tier 4 (Exact Model Path):** `(model_path, phase)`  
   Matches previous runs of this specific model file.
5. **Tier 5 (Strategy Fallback):** `(strategy, phase)`  
   Matches global placement strategy (e.g. `split` CPU/GPU offload).
6. **Tier 6 (Architecture Fallback):** `(arch, phase)`  
   Matches general architecture (e.g. `llama`, `qwen2`, `mistral`).
7. **Tier 7 (Global Moving Average):**  
   Moving average over the last 10 benchmark runs on the machine.
8. **Tier 8 (Default):**  
   Multiplier of `1.0` (uncalibrated analytical estimate).

All multipliers are safety-clamped within $[0.5\\times, 2.0\\times]$ to prevent feedback divergence.

---

## 4. Validating Cost Model Predictions

Aggregate and evaluate your local benchmark results using `scripts/validate_predictions.py`:

```bash
# Print Markdown validation table to stdout
python scripts/validate_predictions.py

# Save validation report to file
python scripts/validate_predictions.py --output docs/validation_results.md

# Aggregate an external directory of community export reports
python scripts/validate_predictions.py --export-dir ./community_reports --output validation_report.md
```

### Example Validation Report Output

```markdown
# RAMLLM Cost Model Validation Report

Total Benchmarks Analyzed: **24**

## Overall Prediction Accuracy

| Metric | In-Range % | Mean Absolute % Error (MAPE) |
| :--- | :---: | :---: |
| **Decode Speed (tok/s)** | 87.5% | 7.8% |
| **Prefill Speed (tok/s)** | 91.7% | 6.4% |

## Breakdown by Architecture

| Architecture | Runs | Decode In-Range % | Decode MAPE | Prefill In-Range % | Prefill MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `llama` | 14 | 92.9% | 5.8% | 92.9% | 6.1% |
| `qwen2` | 10 | 80.0% | 10.6% | 90.0% | 6.8% |

## Breakdown by Placement Strategy

| Strategy | Runs | Decode In-Range % | Decode MAPE | Prefill In-Range % | Prefill MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `full_gpu` | 8 | 100.0% | 4.2% | 100.0% | 3.9% |
| `split` | 12 | 83.3% | 9.4% | 83.3% | 7.8% |
| `cpu_only` | 4 | 75.0% | 10.8% | 100.0% | 7.5% |
```

---

## 5. Community Submissions

To contribute benchmark results for your hardware configuration:
1. Run `ramllm bench <model.gguf> --export my_hardware_report.json`.
2. Open a PR or submit an issue in GitHub Discussions attaching `my_hardware_report.json` and specifying your operating system, CPU, and GPU models.
