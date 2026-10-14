# RAMLLM Architecture & Design

## 1. System Overview

RAMLLM is an open-source, RAM-adaptive and GPU-aware inference manager. It decouples the user-facing interface from the underlying execution runtime, allowing users to run large language models on memory-constrained hardware with predictable, budget-enforced execution.

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
 Runtime         │  Session Manager + Memory Governor            │
                 └───────┬────────────────────────────────────┘
                         │  BackendAdapter interface
                 ┌───────▼───────┐ ┌───────────┐ ┌─────────────┐
 Engines         │ llama.cpp     │ │ MLX (later)│ │ custom      │
                 └───────────────┘ └───────────┘ │ streaming   │
                                                 └─────────────┘
```

## 2. Core Modules

### 2.1 Hardware Profiler (`core/profiler/`)
- Detects physical/logical CPU cores, SIMD instruction sets (AVX2, AVX-512, ARM NEON).
- Measures available RAM and swap headroom.
- Probes GPU accelerators:
  - Apple Silicon unified memory (detects GPU core count, system RAM sharing).
  - NVIDIA GPUs via NVML (VRAM free/total, PCIe generation/lanes).
  - AMD ROCm and Vulkan best-effort detection.
- Benchmarks memory bandwidth:
  - RAM throughput (memcpy micro-benchmark across threads).
  - Disk read throughput (direct/uncached sequential read benchmark on model storage volume).
  - Host-to-Device (H2D) PCIe bandwidth.
- Fails soft: missing optional libraries never crash the system; a synthetic profile loader (`--hw-json`) enables CI testing on headless runners.

### 2.2 Model Analyzer (`core/analyzer/`)
- Inspects GGUF files by parsing metadata and tensor descriptors without loading multi-gigabyte weight arrays into memory.
- Extracts model architecture, context window limit, parameter count, layer count, attention heads, and quantization format.
- Computes per-layer weight distribution and non-layer weights (embeddings, final norm, output head).
- Analyzes Mixture-of-Experts (MoE) parameters: number of experts, active experts per token, expert layer byte footprints vs shared expert bytes.
- Calculates exact KV cache memory demand across context lengths and quantization types (F16, Q8_0, Q4_0).

### 2.3 Execution Planner (`core/planner/`)
- Search space generator evaluates:
  1. Full GPU residency.
  2. Partial offload (GPU layers + CPU layers).
  3. MoE-specific placement (Attention + Shared experts on GPU, routed experts on CPU/RAM).
  4. Disk-spilled execution (batched layer streaming or mmap paging).
  5. Pure CPU fallback.
- Enforces strict safety margins: reserves OS buffer (1.5 GB / 10% RAM, 0.5 GB / 5% VRAM, 3 GB / 20% on Apple Silicon).
- Evaluates candidate plans with the memory-bandwidth cost model.
- Assigns feasibility verdicts: `COMFORTABLE`, `CONSTRAINED`, `BATCH_ONLY`, `INFEASIBLE`.
- Generates human-readable explanations and explicit remedies for infeasible configurations.

### 2.4 Backend Adapters (`backends/`)
- Abstract interface (`BackendAdapter`) with capability flags (`partial_gpu_offload`, `per_tensor_placement`, `kv_quantization`, `mmap_control`, etc.).
- Primary adapter: `llama.cpp` (`llama-server` process management).
  - Discovers binary and introspects capabilities via `--help`.
  - Translates `ExecutionPlan` into verified flags (`-ngl`, `-c`, `-ctk`, `-ctv`, `-cmoe`, `-ncmoe`, `-lm`).
  - Monitored process lifetime with health checking and buffer size extraction from stdout/stderr.
- Mock adapter: deterministic test harness for unit, integration, and CI testing.

### 2.5 Memory Governor (`core/governor/`)
- Stage A: Continuous sampling of process RSS, system free memory, and VRAM; checks actual allocations against plan predictions.
- Stage B: Operating system level limits (Linux cgroups v2, Windows Job Objects).
- Stage C: Reactive adaptation to memory pressure and out-of-memory errors (auto-retry with smaller plan).
- Stage D (Future): Fine-grained weight eviction (`madvise` / custom streaming).
