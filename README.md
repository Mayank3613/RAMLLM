# RAMLLM — Run Bigger Models on Smaller Machines

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.11 | 3.12](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Backend: llama.cpp](https://img.shields.io/badge/Backend-llama.cpp-orange.svg)](docs/backend-notes.md)
[![Platform: macOS | Linux | Windows](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](docs/architecture.md)
[![Tests: Passing](https://img.shields.io/badge/Tests-62%20Passing-brightgreen.svg)](tests/)

> **Tell RAMLLM how much VRAM, RAM, and patience you have.**
> It places the hottest weights on your GPU, warm ones in RAM, cold ones on disk, tells you exactly how fast that will run, and enforces the budget while it runs.

---

## Table of Contents

- [The Problem](#the-problem)
- [The RAMLLM Solution](#the-ramllm-solution)
- [Key Features](#key-features)
- [Prerequisites & Installation](#prerequisites--installation)
  - [1. Backend Prerequisite: llama.cpp](#1-backend-prerequisite-llamacpp)
  - [2. Install RAMLLM](#2-install-ramllm)
  - [3. Verify Setup](#3-verify-setup)
- [Quickstart Guide](#quickstart-guide)
  - [Step 1: Check System Diagnostics](#step-1-check-system-diagnostics)
  - [Step 2: Profile Hardware Bandwidth](#step-2-profile-hardware-bandwidth)
  - [Step 3: Register or Pull Models](#step-3-register-or-pull-models)
  - [Step 4: Discover Feasible Models](#step-4-discover-feasible-models)
  - [Step 5: Plan Model Execution](#step-5-plan-model-execution)
  - [Step 6: Run Inference Interactively](#step-6-run-inference-interactively)
  - [Step 7: Serve OpenAI-Compatible API](#step-7-serve-openai-compatible-api)
  - [Step 8: Benchmark & Auto-Calibrate](#step-8-benchmark--auto-calibrate)
- [CLI Command Reference](#cli-command-reference)
- [Architecture & Cost Model](#architecture--cost-model)
  - [Analytical Cost Model](#analytical-cost-model)
  - [Feasibility Verdicts](#feasibility-verdicts)
- [OpenAI-Compatible HTTP API](#openai-compatible-http-api)
  - [Endpoints](#endpoints)
  - [Dynamic Runtime Replanning](#dynamic-runtime-replanning)
- [Development & Testing](#development--testing)
  - [Synthetic Profiles (Headless / CI)](#synthetic-profiles-headless--ci)
  - [Running the Test Suite](#running-the-test-suite)
- [Documentation Links](#documentation-links)
- [License](#license)

---

## The Problem

Local inference tools make open-weight LLMs accessible, but memory remains a brutal barrier:
- **Catastrophic OOM Crashes:** Models requiring tens of gigabytes fail immediately on 8 GB or 16 GB machines.
- **OS Thrashing & Freezes:** Uncoordinated virtual memory swapping locks up the entire operating system.
- **Trial-and-Error Guesswork:** Users guess how many layers to offload (`-ngl`), which quantization fits, whether KV quantization is safe, or if Mixture-of-Experts (MoE) CPU offloading is viable.
- **Dishonest Predictions:** Existing tools launch configurations that drop to 0.05 tokens/sec without prior warning.

---

## The RAMLLM Solution

RAMLLM acts as a **memory-budgeted, hardware-adaptive inference manager**:

1. **Profiles Hardware:** Measures CPU SIMD instructions, RAM throughput, GPU VRAM & PCIe/unified memory bandwidth, and storage drive read rates.
2. **Analyzes GGUF Topology:** Inspects GGUF metadata, tensor layouts, dense vs. MoE expert distribution, and KV cache footprints without loading weight arrays into memory.
3. **Explores Execution Plans:** Evaluates full GPU residency, partial GPU offload, MoE attention-on-GPU / experts-on-CPU offload, quantized KV cache, and disk-spilled execution.
4. **Predicts Speed & Bottlenecks:** Estimates decode and prefill speed ranges with an explicit **feasibility verdict** (`COMFORTABLE`, `CONSTRAINED`, `BATCH_ONLY`, `INFEASIBLE`) and a **named bottleneck**.
5. **Enforces Memory Budgets:** Monitors process RSS and VRAM live via a runtime **Memory Governor** with automated fallback recovery if memory spikes occur.

```
                 ┌────────────────────────────────────────────┐
 Interfaces      │  CLI (typer)   HTTP API (fastapi)   Python lib │
                 └───────────────┬────────────────────────────┘
                                 │  UserRequest (model, budgets, ctx, prefs)
                 ┌───────────────▼────────────────────────────┐
 Brain           │  Planner  ◄── HardwareProfile ◄── Profiler    │
                 │     ▲  ◄── ModelInfo ◄── ModelAnalyzer        │
                 └───────┬────────────────────────────────────┘
                         │  ExecutionPlan
                 ┌───────▼────────────────────────────────────┐
 Runtime         │  Session Manager + Memory Governor (A/B/C) │
                 └───────┬────────────────────────────────────┘
                         │  BackendAdapter interface
                 ┌───────▼───────┐ ┌───────────┐ ┌─────────────┐
 Engines         │ llama.cpp     │ │ MLX (next)│ │ Custom      │
                 └───────────────┘ └───────────┘ │ Streaming   │
                                                 └─────────────┘
```

---

## Key Features

- 🏎️ **Hardware-Aware Planner:** Generates ranked placement plans tailored to your exact RAM, VRAM, and context window budgets.
- 🧠 **MoE Expert Partitioning:** Keeps attention and shared experts in high-speed VRAM while routing sparse expert activations through system RAM (`-cmoe`).
- 🛡️ **Memory Governor with Auto-Recovery:** Actively monitors allocations and automatically adapts to smaller plans if an unexpected spike or OOM occurs.
- ⚡ **OpenAI-Compatible API Server:** High-performance FastAPI server with streaming SSE support (`/v1/chat/completions`), `/ramllm/metrics`, and zero-downtime dynamic replanning (`/ramllm/replan`).
- 📦 **Offline-First Model Catalog:** Add, scan, verify SHA-256 hashes, recommend models, and pull weights with resumable downloads.
- 📊 **Empirical Benchmarking & Auto-Calibration:** Runs standardized prompt suites to measure actual throughput and calibrate future predictions for your specific hardware.
- 🧪 **Deterministic CI Testing:** Includes synthetic hardware profiles and fixture GGUF generators for running tests without physical GPUs or huge weights.

---

## Prerequisites & Installation

### 1. Backend Prerequisite: llama.cpp

RAMLLM delegates tensor execution to battle-tested backends (starting with `llama.cpp`). Ensure `llama-server` and `llama-cli` are available on your system `PATH`:

#### macOS (Homebrew)
```bash
brew install llama.cpp
```

#### Linux (Ubuntu / Debian)
Download the prebuilt release binaries from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) or build with CUDA/ROCm/Vulkan support:
```bash
# Clone and build with CMake
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON  # or -DGGML_VULKAN=ON, etc.
cmake --build build --config Release -j --target llama-server llama-cli

# Add binaries to PATH
export PATH="$PWD/build/bin:$PATH"
```

#### Windows
Download the latest prebuilt `llama-bXXXX-bin-win-*.zip` from [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) and add the directory to your system `PATH`.

---

### 2. Install RAMLLM

RAMLLM requires **Python 3.11 or 3.12**.

#### Option A: Using `pip` and `venv` (Standard)
```bash
# Clone repository
git clone https://github.com/your-org/ramllm.git
cd ramllm

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# (Optional) Install development & test dependencies
pip install -e ".[dev]"
```

#### Option B: Using `uv` (Fast)
```bash
git clone https://github.com/your-org/ramllm.git
cd ramllm

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

### 3. Verify Setup

Run the diagnostic command to check detected backends, accelerators, and OS memory enforcement mechanisms:

```bash
ramllm doctor
```

Example output:
```text
╭─────────────────────────────────────────────────────────────╮
│ RAMLLM Diagnostic Doctor                                    │
╰─────────────────────────────────────────────────────────────╯
✔ Backend: llama.cpp found at `/opt/homebrew/bin/llama-server`
  Version: version: 0.6.0 (build 11429, commit d81235049)
  Capabilities: GPU Offload, KV Quantization, MoE CPU Offload, Flash Attention
! OS Enforcement: macOS memory pressure dispatch supported (soft limits & active monitoring).
```

---

## Quickstart Guide

### Step 1: Check System Diagnostics
```bash
ramllm doctor
```

### Step 2: Profile Hardware Bandwidth
Profile your CPU cores, RAM bandwidth (measured via multi-threaded memcpy), storage drive read speed, and GPU VRAM:
```bash
ramllm profile
```
To force a re-run of bandwidth micro-benchmarks:
```bash
ramllm profile --refresh
```

### Step 3: Register or Pull Models

#### Pull a model from Hugging Face or URL
Download models with resumable streaming and automatic catalog registration:
```bash
# Hugging Face shorthand
ramllm models pull Qwen/Qwen2.5-0.5B-GGUF:qwen2.5-0.5b-q4_k_m.gguf

# Or direct URL with optional SHA256 verification
ramllm models pull https://example.com/models/model.gguf --sha256 <expected-hash>
```

#### Register existing local models
```bash
# Add a single model file
ramllm models add /path/to/model.gguf

# Or recursively scan a folder of GGUF models
ramllm models scan ~/models/
```

#### List models in catalog
```bash
ramllm models list
```

### Step 4: Discover Feasible Models
Find which registered models fit comfortably within your hardware constraints:
```bash
ramllm recommend --vram 6GB --ram 16GB --context 8k --prefer balanced
```

### Step 5: Plan Model Execution
Examine candidate placement strategies, predicted speeds, bottlenecks, and feasibility before launching:
```bash
ramllm plan /path/to/model.gguf --vram 6GB --ram 16GB --context 8k --explain
```

Example plan output:
```text
                  Execution Plans for mistral-7b.Q4_K_M.gguf (4.4 GB)
╭──────┬──────────────┬───────────┬──────────┬──────────────┬───────────┬─────────────╮
│ Rank │ Strategy     │ VRAM Peak │ RAM Peak │ Decode Speed │ Bottleneck│ Verdict     │
├──────┼──────────────┼───────────┼──────────┼──────────────┼───────────┼─────────────┤
│ #1   │ full_gpu_... │ 4.8 GB    │ 1.2 GB   │ ~42.1–51.3 t │ vram_bw   │ COMFORTABLE │
│ #2   │ partial_g... │ 3.1 GB    │ 2.9 GB   │ ~18.5–24.0 t │ ram_bw    │ COMFORTABLE │
│ #3   │ cpu_only_... │ 0.0 GB    │ 5.8 GB   │ ~8.2–11.4 t  │ ram_bw    │ CONSTRAINED │
╰──────┴──────────────┴───────────┴──────────┴──────────────┴───────────┴─────────────╯

Plan #1 Reasoning & Breakdown:
  • Full GPU Residency: All 32 layers placed in VRAM (4.8 GB peak).
  • KV Cache: 8192 tokens (q8_0/q8_0) placed in GPU VRAM.
  • Primary Bottleneck: GPU VRAM memory bus bandwidth.
```

### Step 6: Run Inference Interactively
Launch the model with budget enforcement, buffer validation, and automated recovery:
```bash
ramllm run /path/to/model.gguf --vram 6GB --ram 16GB --prompt "Explain quantum computing in 2 sentences."
```

Enable live memory governor tracking:
```bash
ramllm run /path/to/model.gguf --vram 6GB --ram 16GB --live
```

### Step 7: Serve OpenAI-Compatible API
Start a server compatible with OpenAI SDKs, LangChain, LlamaIndex, or Open WebUI:
```bash
ramllm serve /path/to/model.gguf --vram 6GB --ram 16GB --port 8080
```

Test it with `curl`:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello! What model are you?"}],
    "temperature": 0.7
  }'
```

### Step 8: Benchmark & Auto-Calibrate
Measure empirical decode and prefill speed, time to first token (TTFT), and update the machine calibration factor:
```bash
ramllm bench /path/to/model.gguf
```

---

## CLI Command Reference

| Command | Arguments / Options | Description |
| :--- | :--- | :--- |
| **`ramllm doctor`** | None | Diagnose installed backend engines, capabilities, and OS memory enforcement mechanisms. |
| **`ramllm profile`** | `[--refresh]` `[--json]` `[--hw-json <path>]` | Benchmark CPU, RAM bandwidth, GPU VRAM/bandwidth, and storage read throughput. |
| **`ramllm plan`** | `<model>` `[--vram <budget>]` `[--ram <budget>]` `[--gpus <0,1>]` `[--context <ctx>]` `[--prefer <fastest\|balanced\|fit_at_any_cost>]` `[--cpu-only]` `[--no-kv-quant]` `[--explain]` `[--json]` | Rank execution strategies, calculate per-tier memory peaks, and forecast decode/prefill speed. |
| **`ramllm run`** | `<model>` `[--vram <budget>]` `[--ram <budget>]` `[--gpus <0,1>]` `[--context <ctx>]` `[--prompt, -p <text>]` `[--port <8080>]` `[--live]` `[--auto-retry/--no-auto-retry]` | Launch model with memory governor monitoring, buffer verification, and interactive prompt. |
| **`ramllm serve`** | `<model>` `[--vram <budget>]` `[--ram <budget>]` `[--context <ctx>]` `[--port <8080>]` `[--host <127.0.0.1>]` `[--auto-retry]` | Start an OpenAI-compatible HTTP API server with live memory metrics and runtime replanning. |
| **`ramllm recommend`**| `[--vram <budget>]` `[--ram <budget>]` `[--context <ctx>]` `[--prefer <pref>]` `[--cpu-only]` `[--json]` | Recommend best-quality feasible models from local catalog matching your hardware constraints. |
| **`ramllm bench`** | `<model>` `[--plan-id <id>]` `[--port <8085>]` | Run empirical benchmark suite, verify predicted vs actual speed, and update machine calibration. |
| **`ramllm models add`** | `<model_path>` | Register a local GGUF file into the catalog. |
| **`ramllm models list`** | None | Display all registered models with architecture, quantization, parameter count, and size. |
| **`ramllm models scan`** | `<directory>` | Recursively scan directory and register all discovered GGUF models. |
| **`ramllm models remove`** | `<name>` | Remove model entry from catalog. |
| **`ramllm models verify`** | `[<model>]` | Verify SHA-256 integrity hash of one or all cataloged models. |
| **`ramllm models pull`** | `<target>` `[--dest <dir>]` `[--sha256 <hash>]` | Download model via Hugging Face shorthand (`repo:file.gguf`) or direct URL with resumable streaming. |

---

## Architecture & Cost Model

### Analytical Cost Model

Autoregressive token generation (decode) is fundamentally **memory-bandwidth bound**. For each generated token, the inference engine must traverse all active weights and the accumulated KV cache:

$$\text{Time per token } (t_{\text{token}}) \approx \sum_{\text{tier } i} \frac{\text{Bytes Read from tier } i}{\text{Effective Bandwidth of tier } i} + t_{\text{overhead}}$$

$$\text{Decode Tokens per Second} = \frac{1}{t_{\text{token}}}$$

#### Working Set per Model Type:
- **Dense Models:** Traverses all model weights per token across GPU, CPU, and disk tiers.
- **Mixture of Experts (MoE):** Traverses non-expert weights, shared experts, and only the active routed experts:
  $$\text{Working Set} = \text{NonExpertBytes} + \text{SharedExpertBytes} + \left(\frac{N_{\text{active}}}{N_{\text{total}}}\right) \times \text{RoutedExpertBytes}$$

#### KV Cache Footprint:
$$\text{KV Bytes}(L, C, H_{\text{kv}}, D, T_k, T_v) = 2 \times L \times C \times H_{\text{kv}} \times D \times \left(\frac{\text{bits}(T_k) + \text{bits}(T_v)}{16}\right)$$

Supported KV cache quantization types include **F16** (2.0 bytes/element), **Q8_0** (1.0625 bytes/element), and **Q4_0** (0.5625 bytes/element).

---

### Feasibility Verdicts

Every plan generated by RAMLLM receives an unambiguous feasibility verdict:

| Verdict | Criteria | Actionable Meaning |
| :--- | :--- | :--- |
| **`COMFORTABLE`** | Zero disk paging, predicted decode $\ge 8$ tok/s | Smooth, real-time interactive conversation. |
| **`CONSTRAINED`** | Fits within RAM+VRAM, predicted decode $1 \text{ to } 8$ tok/s | Usable with minor latency; consider smaller context or quant. |
| **`BATCH_ONLY`** | Requires disk paging or predicted decode $0.05 \text{ to } 1$ tok/s | Suitable only for background batch tasks; not interactive. |
| **`INFEASIBLE`** | Exceeds total available memory budget or $< 0.05$ tok/s | Execution rejected; actionable remedies provided. |

---

## OpenAI-Compatible HTTP API

When running `ramllm serve`, the server exposes standard OpenAI endpoints along with RAMLLM-specific telemetry and control endpoints:

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/healthz` | Health check endpoint. |
| `GET` | `/v1/models` | OpenAI-compatible model listing. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions (supports streaming SSE). |
| `POST` | `/v1/completions` | OpenAI-compatible raw completions. |
| `GET` | `/ramllm/plan` | Returns the currently executing `ExecutionPlan`. |
| `GET` | `/ramllm/metrics` | Real-time memory consumption, RSS, VRAM, and governor warnings. |
| `POST` | `/ramllm/replan` | Dynamically adjust budgets or context at runtime without restarting. |

### Dynamic Runtime Replanning

Update memory budgets on the fly via `POST /ramllm/replan`:

```bash
curl -X POST http://127.0.0.1:8080/ramllm/replan \
  -H "Content-Type: application/json" \
  -d '{
    "vram_budget_bytes": {"0": 4294967296},
    "ram_budget_bytes": 12884901888,
    "context": 8192
  }'
```

---

## Development & Testing

### Synthetic Profiles (Headless / CI)

RAMLLM includes synthetic hardware profiles in `tests/fixtures/hardware/` to test execution plans and recommendations on headless machines without a physical GPU:

```bash
# Profile using synthetic hardware profile
ramllm profile --hw-json tests/fixtures/hardware/laptop_16gb_gpu_6gb.json

# Plan against synthetic hardware
ramllm plan /path/to/model.gguf --hw-json tests/fixtures/hardware/desktop_32gb_gpu_24gb.json
```

Available synthetic profiles:
- `apple_m_16gb_unified.json`
- `apple_m_32gb_unified.json`
- `desktop_32gb_gpu_24gb.json`
- `dual_gpu_workstation.json`
- `laptop_16gb_gpu_6gb.json`
- `laptop_8gb_no_gpu.json`
- `mismatched_dual_gpu.json`

### Running the Test Suite

```bash
# Run all unit and integration tests
pytest

# Run tests with coverage or filtering
pytest tests/unit/test_planner.py
pytest tests/unit/test_recommend.py
pytest tests/unit/test_chaos_recovery.py

# Linting with Ruff
ruff check .
ruff format --check .

# Type checking with Mypy
mypy src/ramllm/core
```

---

## Documentation Links

- [Architecture & Design](docs/architecture.md) — Comprehensive architecture and system components.
- [Cost Model Specification](docs/cost-model.md) — Mathematical formulation for memory bandwidth, working sets, and prefill/decode latencies.
- [Backend Compatibility Notes](docs/backend-notes.md) — Supported llama.cpp flags, version adaptations, and server endpoints.
- [Contributing Guidelines](CONTRIBUTING.md) — Code quality standards, typing, and pull request procedures.

---

## License

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.
