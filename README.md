# RAMLLM — Run bigger models on smaller machines

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)

> **Tell RAMLLM how much VRAM, RAM, and patience you have.**
> It places the hottest weights on your GPU, warm ones in RAM, cold ones on disk, tells you exactly how fast that will run, and enforces the budget while it runs.

---

## The Problem
Local inference tools make open-weight LLMs accessible, but memory remains a barrier:
- Models requiring tens of gigabytes fail on 8 GB or 16 GB laptops.
- Uncoordinated OS swapping causes catastrophic thrashing and freezes.
- Users have to guess how many layers to offload (`-ngl`), which quantization fits, or whether MoE offload is viable.

## The RAMLLM Solution
RAMLLM acts as a **memory-budgeted, hardware-adaptive inference manager**:
1. **Profiles** CPU, RAM bandwidth, GPU VRAM, PCIe transfer rates, and disk read bandwidth.
2. **Analyzes** GGUF model topologies (dense, MoE, attention heads, KV cache footprint).
3. **Plans** optimal execution strategies (full GPU, partial offload, MoE attention-on-GPU/experts-on-CPU, CPU-only).
4. **Predicts** decode and prefill speed ranges with an explicit **feasibility verdict** and **named bottleneck**.
5. **Executes** via robust backends (starting with `llama.cpp`) with per-tier memory governor monitoring.

---

## Quickstart

```bash
# Profile your system's memory and compute bandwidth
ramllm profile

# Inspect your system and check dependencies
ramllm doctor

# Plan execution for a model given your memory budget
ramllm plan /path/to/model.gguf --vram 6GB --ram 12GB --context 8k

# Run with honest predictions and live memory tracking
ramllm run /path/to/model.gguf --vram 6GB --ram 12GB
```

## Architecture

See [docs/architecture.md](docs/architecture.md) and [docs/cost-model.md](docs/cost-model.md).

## License

Apache-2.0
