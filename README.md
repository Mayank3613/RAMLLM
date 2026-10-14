# RAMLLM — Run Bigger Models on Smaller Machines

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.11 | 3.12 | 3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![Backend: llama.cpp](https://img.shields.io/badge/Backend-llama.cpp-orange.svg)](docs/backend-notes.md)
[![Platform: macOS | Linux | Windows](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey.svg)](docs/architecture.md)
[![CI](https://github.com/Mayank3613/RAMLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/Mayank3613/RAMLLM/actions/workflows/ci.yml)

**RAMLLM runs bigger language models on smaller computers without crashing or freezing your operating system.** It profiles your hardware topology, calculates safe split-layer and quantized execution plans, and manages inference engines with live memory governors.

---

## The Three-Command Promise

Get from zero to chatting with a local LLM in less than two minutes:

```bash
# 1. Detect hardware, install native engine, and download a verified model
ramllm setup

# 2. Start an interactive, memory-safe conversation
ramllm chat

# 3. Monitor live GPU/RAM usage, tok/s, and governor in another terminal
ramllm dashboard
```

```
┌────────────────────────────────────────────────────────────────────────┐
│ RAMLLM Setup & Chat Experience                                         │
├────────────────────────────────────────────────────────────────────────┤
│ > ramllm setup                                                         │
│   ✔ Hardware profiled: Apple M2 (16 GB Unified RAM, 100 GB/s bandwidth)│
│   ✔ Native engine detected: llama.cpp (Metal acceleration enabled)     │
│   ✔ Recommended starter model: Qwen2.5 7B (Q4_K_M)                     │
│   ✔ Testing token generation... OK (32.4 tok/s)                        │
│                                                                        │
│ > ramllm chat qwen2.5-7b                                               │
│   ╭─ Plan Summary ───────────────────────────────────────────────────╮ │
│   │ ✔ Qwen2.5 7B (Q4_K_M) will run well on your computer             │ │
│   │   • Speed: fast — about 24–34 words per second [COMFORTABLE]     │ │
│   │   • Memory: 4.8 GB allocated, leaving 11.2 GB safe headroom      │ │
│   ╰──────────────────────────────────────────────────────────────────╯ │
│                                                                        │
│   > Explain quantum entanglement in two simple sentences.              │
│   Quantum entanglement is a physical phenomenon where two particles    │
│   become deeply interconnected such that the state of one instantly    │
│   dictates the state of the other, regardless of distance.             │
│                                                                        │
│ [qwen2.5-7b-q4_k_m | 240/4096 ctx (6%) | RAM: 4.8 GB | 31.8 tok/s]     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 1-Minute Quick Install

### macOS & Linux
```bash
curl -fsSL https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.sh | bash
```

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.ps1 | iex
```

### Or using Python Package Managers
```bash
# Recommended: Install directly from GitHub with uv tool
uv tool install git+https://github.com/Mayank3613/RAMLLM.git

# Or with pipx
pipx install git+https://github.com/Mayank3613/RAMLLM.git

# Or from a local repository clone
uv tool install . --force
```

---

## "What Fits My Computer?"

RAMLLM replaces blind guessing with verified hardware tiers:

| Your Hardware | Recommended Models | Speed (approx) | Experience |
|---|---|---|---|
| **8 GB Mac / Thin PC**<br>(8 GB RAM, Intel/AMD/Apple) | **Llama-3.2 3B**, **Qwen2.5 3B**, SmolLM2 1.7B | ~25–40 tok/s<br>(~18–30 wps) | **Runs Great** (`COMFORTABLE`)<br>Leaves 3–4 GB safe headroom for OS & browser |
| **16 GB Mac or PC**<br>(16 GB RAM + 6–8 GB GPU) | **Qwen2.5 7B**, **Llama-3.1 8B**, Mistral 7B | ~20–35 tok/s<br>(~15–25 wps) | **Runs Great** (`COMFORTABLE`)<br>Full model in fast memory, rich context |
| **24 GB – 32 GB RAM**<br>(12–16 GB GPU VRAM) | **DeepSeek-R1-Distill-Qwen 14B**, Qwen2.5 14B, Gemma-2 9B | ~15–25 tok/s<br>(~11–18 wps) | **Smooth & Smart** (`COMFORTABLE`)<br>Advanced reasoning & coding |
| **36 GB – 64 GB Mac/PC**<br>(Unified or multi-GPU) | **Qwen2.5 32B**, DeepSeek-R1 32B, **Mixtral 8x7B** (MoE) | ~10–20 tok/s<br>(~7–15 wps) | **Powerhouse** (`COMFORTABLE` / `CONSTRAINED`)<br>MoE attention in GPU, experts in RAM |
| **64 GB+ Workstation**<br>(Home server, DDR5) | **Llama-3.3 70B**, Qwen2.5 72B | ~3–8 tok/s<br>(~2–6 wps) | **Maximum Scale** (`CONSTRAINED` / `BATCH_ONLY`)<br>Runs massive frontier open weights locally |

Want exact personalized suggestions for your machine? Run:
```bash
ramllm recommend
```

---

## Key Features

- 🏎️ **Hardware-Aware Planner:** Evaluates GGUF layers, tensor types, and KV cache footprints without loading weight arrays.
- 🧠 **MoE Expert Partitioning:** Keeps attention and shared experts in high-speed VRAM while streaming sparse expert activations from system RAM (`-cmoe`).
- 🛡️ **Memory Governor with Auto-Recovery:** Actively samples memory every second and automatically adapts to smaller plans if memory pressure occurs.
- 💬 **Interactive Chat REPL:** Rich markdown streaming, in-session `/mode` and `/replan` commands, transcript saving, and clipboard integration.
- 📊 **Live Textual TUI Dashboard:** Monitor active background engines, VRAM/RAM allocations, tokens/second, and governor events in real time.
- 🌐 **OpenAI-Compatible API Server:** High-performance background serving with streaming SSE (`/v1/chat/completions`), `/ramllm/metrics`, and zero-downtime dynamic replanning (`/ramllm/replan`).
- 📦 **Verified Model Registry:** Curated catalog of real Hugging Face models with verified SHA-256 integrity checks and resumable downloads.
- 🔌 **Integrations Ready:** Drop-in compatibility with Open WebUI, Cursor, VS Code Continue, LangChain, and OpenAI SDKs.

---

## Common Commands & Cheatsheet

```bash
# Launch interactive terminal home menu
ramllm

# Chat with your default model or a specific one
ramllm chat
ramllm chat qwen2.5-7b --mode fast

# Serve an OpenAI API in the background
ramllm serve qwen2.5-7b --detach --port 8080

# Manage running engines
ramllm ps                          # List active engines
ramllm logs <id> -f                # Follow engine logs
ramllm replan <id> --memory 8GB    # Dynamically change memory limit
ramllm stop <id>                   # Stop background engine

# Curated Models
ramllm recommend                   # See what fits best
ramllm models search coder         # Search verified models
ramllm models pull qwen2.5-7b      # Download with progress bar & hash check
ramllm models du                   # Inspect disk space per model

# Health & System
ramllm doctor --fix                # Diagnose and fix issues
ramllm dashboard                   # Live full-screen TUI monitor
ramllm tour                        # 30-second tour of concepts
ramllm clean                       # Free disk space (partials, caches)
```

---

## Frequently Asked Questions (FAQ)

### How does RAMLLM compare to Ollama or LM Studio?
Standard tools assume models either fit entirely in VRAM or default to painfully slow pure-CPU execution. When memory runs out, they crash or lock up your OS. RAMLLM:
1. Calculates exact layer split mathematical boundaries before launch.
2. Supports KV-cache quantization (`q8_0`/`q4_0`) to save 50–75% of context memory.
3. Protects your operating system with an active runtime Memory Governor.
4. Provides full transparency into predicted speed (tok/s) and bottlenecks.

### Does RAMLLM require an NVIDIA GPU?
**No.** RAMLLM natively auto-detects and supports:
- **Apple Silicon** (Metal unified memory acceleration)
- **NVIDIA GPUs** (CUDA acceleration)
- **Vulkan** (AMD / Intel GPUs on Linux and Windows)
- **CPU-only fallback** with optimized AVX2 / AVX-512 SIMD vector extensions

### What happens if my computer runs low on memory?
RAMLLM's Memory Governor continuously samples memory headroom. If external apps (like browser tabs or IDEs) consume memory, RAMLLM automatically intervenes, downscaling context or adapting placement strategies before your system can freeze.

### Can I use my own GGUF model files?
**Yes.** Register any existing local GGUF file with:
```bash
ramllm models add /path/to/model.gguf
# Or scan an entire folder:
ramllm models scan ~/Downloads/models
```

### Is RAMLLM offline-first?
**Yes.** RAMLLM never makes unsolicited network calls, collects zero telemetry, and performs all hardware profiling and model planning completely offline on your local machine.

---

## Full Documentation

Explore detailed guides in the [`docs/`](docs/) directory:
- 📖 [Getting Started Guide](docs/getting-started.md)
- 🖥️ [CLI Command Reference](docs/cli.md)
- 💡 [Core Concepts (Tiers, Quants, MoE, Verdicts)](docs/concepts.md)
- 📋 [One-Page Cheatsheet](docs/cheatsheet.md)
- 🛠️ [Troubleshooting (Error Codes RAM-E001 – RAM-E015)](docs/troubleshooting.md)
- 🍳 [Recipes & Integrations (Open WebUI, LangChain, SDK)](docs/recipes.md)
- 🏗️ [Architecture & Placement Engine](docs/architecture.md)
- 📐 [Analytical Cost Model](docs/cost-model.md)

---

## Technical Deep-Dive & Architecture

For developers and researchers interested in RAMLLM's internal placement algorithms and cost models:

### Analytical Cost Model
The planner evaluates execution strategies using empirical memory bandwidth equations:

$$\text{Time}_{\text{token}} = \max\left( \frac{\text{Weights}_{\text{VRAM}}}{\text{BW}_{\text{VRAM}}} + \frac{\text{Weights}_{\text{RAM}}}{\text{BW}_{\text{RAM}}}, \frac{\text{ComputeFlops}}{\text{PeakFlops}} \right)$$

Where:
- $\text{BW}_{\text{VRAM}}$ is measured GPU memory bandwidth (e.g., 800 GB/s on RTX 4090, 100–400 GB/s on Apple Silicon).
- $\text{BW}_{\text{RAM}}$ is physical system DDR throughput (e.g., 45–90 GB/s).
- Attention layers and KV matrices are prioritized on the lowest-latency tier.

### Feasibility Verdicts
- `COMFORTABLE`: Full model fits within high-speed headroom with $>15\%$ reserve buffer.
- `CONSTRAINED`: Fits within budget via partial layer offloading or quantized KV cache.
- `BATCH_ONLY`: Heavy CPU offload resulting in 1–5 tok/s; suited for non-interactive workloads.
- `INFEASIBLE`: Exceeds physical RAM + VRAM limits. Blocked to protect OS stability.

---

## Development & Testing

```bash
# Clone and setup development environment
git clone https://github.com/Mayank3613/RAMLLM.git
cd RAMLLM
uv sync --all-extras

# Run full test suite
uv run pytest

# Run linting and type checking
uv run ruff check .
uv run mypy src tests
```

---

## License

RAMLLM is open-source software licensed under the [Apache License 2.0](LICENSE).
