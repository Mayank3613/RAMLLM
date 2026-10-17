# RAMLLM Cost Model Specification

## 1. Principles

Autoregressive token generation (decode) in large language models is governed by a balance of **memory bandwidth** and **compute capacity**. For each generated token, the inference engine must traverse active weights, load the accumulated KV cache, transfer activations across tier boundaries, and execute matrix-vector multiplications.

RAMLLM models both bandwidth and compute bottlenecks:

$$\\mathrm{Time}_{\\mathrm{token}} \\approx \\max(\\mathrm{Time}_{\\mathrm{bandwidth}}, \\mathrm{Time}_{\\mathrm{compute}}) + \\mathrm{Time}_{\\mathrm{overhead}}$$

$$\\mathrm{DecodeTokensPerSecond} = \\frac{1}{\\mathrm{Time}_{\\mathrm{token}}}$$

---

## 2. Working Set Formulation

### 2.1 Dense Models
For dense architectures (e.g. LLaMA, Mistral, Qwen Dense):
$$\\mathrm{WeightBytesPerToken} = \\mathrm{TotalWeightBytes}$$

$$\\mathrm{Bytes}_{\\mathrm{GPU}} = \\sum_{l \\in \\mathrm{gpu\\_layers}} \\mathrm{LayerBytes}[l] + \\mathrm{NonLayerBytes}_{\\mathrm{GPU}}$$
$$\\mathrm{Bytes}_{\\mathrm{CPU}} = \\sum_{l \\in \\mathrm{cpu\\_layers}} \\mathrm{LayerBytes}[l] + \\mathrm{NonLayerBytes}_{\\mathrm{CPU}}$$
$$\\mathrm{Bytes}_{\\mathrm{Disk}} = \\sum_{l \\in \\mathrm{disk\\_layers}} \\mathrm{LayerBytes}[l]$$

### 2.2 Mixture of Experts (MoE) Models & Coverage Probability
In MoE architectures (e.g. Mixtral, Qwen-MoE, DeepSeek), only $k$ out of $E$ routed experts are activated per token ($k = N_{\\mathrm{active}}, E = N_{\\mathrm{total}}$).

For single-stream decode ($T = 1$ token):
$$\\mathrm{Coverage}(1) = \\frac{N_{\\mathrm{active}}}{N_{\\mathrm{total}}}$$

For prefill ($T \\gg 1$ tokens), batch decode ($B > 1$), or mmap/disk-backed expert paging:
The probability that any given expert is activated at least once across $T$ tokens is:
$$\\mathrm{Coverage}(T) = 1 - \\left(1 - \\frac{N_{\\mathrm{active}}}{N_{\\mathrm{total}}}\\right)^T$$

$$\\mathrm{WorkingSet}(T) = \\mathrm{NonExpertBytes} + \\mathrm{SharedExpertBytes} + \\mathrm{Coverage}(T) \\times \\mathrm{RoutedExpertBytes}$$

When attention/shared weights reside in VRAM and routed experts reside in CPU RAM:
$$\\mathrm{Bytes}_{\\mathrm{GPU}} = \\mathrm{NonExpertBytes} + \\mathrm{SharedExpertBytes}$$
$$\\mathrm{Bytes}_{\\mathrm{CPU}} = \\mathrm{Coverage}(T) \\times \\mathrm{RoutedExpertBytes}$$

---

## 3. KV Cache Footprint & Quantization Rules

### 3.1 Footprint Formulation
$$\\mathrm{KVBytes} = \\frac{L \\times C_{\\mathrm{eff}} \\times H_{\\mathrm{kv}} \\times D \\times (\\mathrm{bits}_K + \\mathrm{bits}_V)}{8} \\times N_{\\mathrm{slots}}$$

Where:
- $L$: number of layers
- $C_{\\mathrm{eff}}$: effective context length (tokens). For sliding-window attention layers with window $W$, $C_{\\mathrm{eff}} = \\min(C, W)$.
- $H_{\\mathrm{kv}}$: number of Key-Value heads (accounting for Grouped-Query Attention)
- $D$: dimension per head
- $N_{\\mathrm{slots}}$: number of parallel inference server slots (`-np`)
- $\\mathrm{bits}_K, \\mathrm{bits}_V$: bits per value including block scale overhead:
  - F16 / BF16: 16.0 bits (2.00 bytes/element)
  - Q8_0: 8.5 bits (1.0625 bytes/element: 32 bytes + 2-byte scale per 32-element block)
  - Q4_0: 4.5 bits (0.5625 bytes/element: 16 bytes + 2-byte scale per 32-element block)

For DeepSeek Multi-head Latent Attention (MLA), the KV projection is compressed to rank dimension $D_{\\mathrm{lora}}$:
$$\\mathrm{KVBytes}_{\\mathrm{MLA}} = \\frac{L \\times C_{\\mathrm{eff}} \\times D_{\\mathrm{lora}} \\times (\\mathrm{bits}_K + \\mathrm{bits}_V)}{16} \\times N_{\\mathrm{slots}}$$

For Hybrid / SSM layers (e.g. Mamba), state memory replaces the attention history:
$$\\mathrm{StateBytes}_{\\mathrm{SSM}} = L_{\\mathrm{ssm}} \\times C_{\\mathrm{eff}} \\times d_{\\mathrm{state}} \\times 4 \\times N_{\\mathrm{slots}}$$

### 3.2 Flash Attention Requirement for V-Cache Quantization
Quantized V-cache (`q8_0` or `q4_0`) requires Flash Attention (`-fa`) in `llama.cpp`. On execution targets where Flash Attention is unsupported (e.g. CPU KV cache), RAMLLM automatically downgrades $V$-cache to `f16` while preserving quantized $K$-cache, surfacing an explicit notice in the plan.

---

## 4. Compute Buffers & Scratch Memory

Memory peaks include scratch buffers for intermediate activations and attention graphs:
$$\\mathrm{ComputeBuffer} = \\mathrm{BaseOverhead} + (\\mathrm{ubatch} \\times \\mathrm{dim}_{\\mathrm{inter}} \\times 4) + (\\min(C, 8192) \\times \\mathrm{ubatch} \\times 4)$$

Where:
- $\\mathrm{ubatch}$: physical micro-batch size (configurable via `--ubatch`, default 512).
- $\\mathrm{dim}_{\\mathrm{inter}}$: largest intermediate tensor dimension (typically $2.7 \\times$ to $4.0 \\times$ hidden size).
- $\\mathrm{BaseOverhead}$: backend runtime runtime allocator handles and CUDA/Metal graph context (~350 MB).

---

## 5. Hardware Bandwidth, Compute Capacity & Efficiencies

RAMLLM measures both read-only STREAM RAM bandwidth and CPU effective GFLOPs:

| Tier | Measurement Kernel | Realistic Efficiency Range | Notes |
| :--- | :--- | :--- | :--- |
| **VRAM** | `vram_bandwidth_bytes_s` | 60% – 85% | Discrete GPU GDDR6 / HBM bus |
| **Unified RAM** | `ram_bandwidth_bytes_s` | 65% – 85% | Apple Silicon unified memory (read STREAM) |
| **Host RAM (CPU)** | `ram_bandwidth_bytes_s` | 50% – 78% | Multi-channel DDR4 / DDR5 read STREAM |
| **PCIe (H2D)** | `h2d_bandwidth_bytes_s` | 70% – 90% | Pinned DMA host-to-device transfers |
| **NVMe / SSD** | `disk_read_bandwidth_bytes_s` | 60% – 90% | OS page cache & sequential queue depth |

### 5.1 Per-Boundary Overhead
When execution transitions across tiers (GPU $\\leftrightarrow$ CPU), each boundary incurs synchronization latency and an activation transfer:
$$\\mathrm{Time}_{\\mathrm{overhead}} = \\mathrm{BaseOverhead} + N_{\\mathrm{boundaries}} \\times \\left(\\mathrm{Time}_{\\mathrm{sync}} + \\frac{\\mathrm{batch} \\times \\mathrm{hidden} \\times 2}{\\mathrm{Bandwidth}_{\\mathrm{PCIe}} \\times \\mathrm{Efficiency}_{\\mathrm{PCIe}}}\\right)$$

For MoE models with expert-on-CPU offloading (`-cmoe`), each MoE layer performs GPU $\\leftrightarrow$ CPU synchronization and activation routing.

### 5.2 CPU Decode Compute Bound
$$\\mathrm{Time}_{\\mathrm{compute, CPU}} = \\frac{2 \\times \\mathrm{ActiveParams}_{\\mathrm{CPU}}}{\\mathrm{cpu\\_effective\\_gflops} \\times 10^9}$$

$$\\mathrm{Time}_{\\mathrm{CPU}} = \\max(\\mathrm{Time}_{\\mathrm{bandwidth, CPU}}, \\mathrm{Time}_{\\mathrm{compute, CPU}})$$

---

## 6. Prefill Estimation

Prompt prefill processes $N$ tokens in parallel and is primarily **compute-bound**:
$$\\mathrm{FLOPs}_{\\mathrm{prefill}} \\approx 2 \\times N_{\\mathrm{active\\_params}} \\times N_{\\mathrm{tokens}}$$
$$\\mathrm{Time}_{\\mathrm{prefill}} \\approx \\frac{\\mathrm{FLOPs}_{\\mathrm{prefill}}}{\\mathrm{EffectiveComputeTFLOPs}} + \\frac{\\mathrm{TotalWeightsLoaded}}{\\mathrm{TierBandwidth}}$$

---

## 7. Feasibility Classification & Slowness Tolerance

Feasibility is evaluated against configurable slowness thresholds (`SlownessTolerance`):

| Verdict | Default Condition | Actionable Guidance |
| :--- | :--- | :--- |
| **`COMFORTABLE`** | Zero disk reads, predicted decode $\\ge 8.0$ tok/s, decode low $\\ge 4.0$ tok/s | Ready for real-time interactive chat |
| **`CONSTRAINED`** | Fits within RAM+VRAM, predicted decode $\\ge \\mathrm{min\\_tps}$ (default 1.0 tok/s) | Interactive with slight latency |
| **`BATCH_ONLY`** | Requires disk paging/streaming, or predicted decode between 0.05 and $\\mathrm{min\\_tps}$ | Background / document processing only |
| **`INFEASIBLE`** | Exceeds total available budget or $< 0.05$ tok/s | Suggest reduction in context, quant, or model |

The threshold $\\mathrm{min\\_tps}$ is configurable via the `--min-tps` flag in `plan`, `run`, `serve`, and `recommend`.
