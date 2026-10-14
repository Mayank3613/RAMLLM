# RAMLLM Baseline Benchmark Report (Milestone 0 Research Spike)

## 1. Environment
- **Platform**: Apple M5 (10-core CPU, 24 GB Unified Memory)
- **OS**: macOS Darwin arm64
- **Backend**: `llama.cpp` version 0.6.0 (build 11429, commit d81235049) built with AppleClang 21.0.0.21000334 (Metal enabled)
- **Test Model**: `Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf` (File size: ~14 GB, MoE architecture: 30B total parameters, ~3B active parameters per token)

## 2. Experimental Results

| Run Configuration | Offload Strategy | Prompt Eval (Prefill) | Generation (Decode) | Memory Observations |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Only (`-ngl 0`)** | Pure CPU host threads (6 threads) | 4.6 t/s | 7.6 t/s | ~14.2 GB RSS in host RAM |
| **Full Metal Offload (`-ngl 99`)** | All layers to Metal GPU | 34.7 t/s | 44.8 t/s | ~14.2 GB resident in Apple Silicon unified memory |
| **MoE CPU Offload (`-ngl 99 -cmoe`)** | Attention/dense in GPU, routed experts in CPU | *Measuring...* | *Measuring...* | Keeps expert weights in host CPU buffers |
| **KV Quantization (`-ctk q8_0 -ctv q8_0`)** | Quantized KV cache | *Scheduled* | *Scheduled* | KV cache reduced by ~47% |

## 3. Key Observations & Architectural Lessons

1. **MoE Working Set Efficiency**:
   Even though `Qwen3-Coder-30B` is a 30B parameter model (14 GB weight file), on pure CPU it generated tokens at **7.6 t/s**. This is because only ~3B parameters are activated per token, verifying the core RAMLLM principle: **MoE working set $\ll$ total model footprint**.

2. **Flag Modernization**:
   `llama.cpp` build 11429 replaces `--no-mmap` with `-lm, --load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`. The RAMLLM backend adapter must use feature-detection rather than hardcoding legacy flags.

3. **Memory Margins on Unified Memory**:
   On a 24 GB unified memory machine, a 14 GB model fits in unified memory comfortably without disk paging. However, when context expands to 16k–32k or concurrent applications run, budget enforcement is essential to avoid memory compression latency.
