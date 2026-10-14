# RAMLLM Cost Model Specification

## 1. Principles

Autoregressive token generation (decode) in large language models is strictly **memory-bandwidth bound**. For each generated token, the inference engine must traverse all active weights and the accumulated KV cache.

$$\text{Time per token } (t_{\text{token}}) \approx \sum_{\text{tier } i} \frac{\text{Bytes Read from tier } i}{\text{Effective Bandwidth of tier } i} + t_{\text{overhead}}$$

$$\text{Decode Tokens per Second} = \frac{1}{t_{\text{token}}}$$

## 2. Working Set Formulation

### 2.1 Dense Models
For dense architectures (e.g. LLaMA, Mistral, Qwen Dense):
$$\text{Weight Bytes per Token} = \text{Total Weight Bytes}$$

$$\text{Bytes}_{\text{GPU}} = \sum_{l \in \text{gpu\_layers}} \text{LayerBytes}[l] + \text{NonLayerBytes}_{\text{GPU}}$$
$$\text{Bytes}_{\text{CPU}} = \sum_{l \in \text{cpu\_layers}} \text{LayerBytes}[l] + \text{NonLayerBytes}_{\text{CPU}}$$
$$\text{Bytes}_{\text{Disk}} = \sum_{l \in \text{disk\_layers}} \text{LayerBytes}[l]$$

### 2.2 Mixture of Experts (MoE) Models
In MoE architectures (e.g. Mixtral 8x7B, Qwen-MoE, DeepSeek), only a subset of experts are activated per token:

$$\text{Working Set} = \text{NonExpertBytes} + \text{SharedExpertBytes} + \left(\frac{N_{\text{active}}}{N_{\text{total}}}\right) \times \text{RoutedExpertBytes}$$

When attention/shared weights reside in VRAM and routed experts reside in CPU RAM:
$$\text{Bytes}_{\text{GPU}} = \text{NonExpertBytes} + \text{SharedExpertBytes}$$
$$\text{Bytes}_{\text{CPU}} = \left(\frac{N_{\text{active}}}{N_{\text{total}}}\right) \times \text{RoutedExpertBytes}$$

## 3. KV Cache Footprint

$$\text{KV Bytes}(L, C, H_{\text{kv}}, D, T_k, T_v) = 2 \times L \times C \times H_{\text{kv}} \times D \times \left(\frac{\text{bits}(T_k) + \text{bits}(T_v)}{16}\right)$$

Where:
- $L$: number of layers
- $C$: active context length (tokens)
- $H_{\text{kv}}$: number of Key-Value heads (accounting for Grouped-Query Attention)
- $D$: dimension per head
- Type precisions:
  - F16: 2.00 bytes/element
  - Q8_0: 1.0625 bytes/element (32 elements per block + 2-byte scale)
  - Q4_0: 0.5625 bytes/element (32 elements per block + 2-byte scale)

## 4. Hardware Bandwidth & Efficiency Ranges

Because memory controllers, bus contention, and backend implementations do not achieve 100% of theoretical peak, RAMLLM uses empirical efficiency ranges:

| Tier | Measured Peak Variable | Realistic Efficiency Range | Notes |
| :--- | :--- | :--- | :--- |
| **VRAM** | `vram_bandwidth_bytes_s` | 60% – 85% | Discrete GPU memory bus |
| **Unified RAM** | `ram_bandwidth_bytes_s` | 65% – 85% | Apple Silicon unified memory |
| **Host RAM (CPU)** | `ram_bandwidth_bytes_s` | 50% – 80% | Multi-channel DDR4 / DDR5 |
| **PCIe (H2D)** | `h2d_bandwidth_bytes_s` | 70% – 90% | Pinned DMA transfers |
| **NVMe / SSD** | `disk_read_bandwidth_bytes_s` | 60% – 90% | OS page cache & drive queue depth |

$$\text{Time}_{\text{tier}} = \frac{\text{Bytes}_{\text{tier}}}{\text{Bandwidth}_{\text{tier}} \times \text{Efficiency}_{\text{tier}}}$$

## 5. Prefill Estimation

Prompt prefill processes $N$ tokens in parallel and is primarily **compute-bound**:
$$\text{FLOPs}_{\text{prefill}} \approx 2 \times N_{\text{active\_params}} \times N_{\text{tokens}}$$
$$\text{Time}_{\text{prefill}} \approx \frac{\text{FLOPs}_{\text{prefill}}}{\text{Effective Compute TFLOPs}} + \frac{\text{Total Weights Loaded}}{\text{Tier Bandwidth}}$$

## 6. Feasibility Classification

| Verdict | Condition | Actionable Guidance |
| :--- | :--- | :--- |
| **`COMFORTABLE`** | Zero disk reads, predicted decode $\ge 8$ tok/s | Ready for real-time interactive chat |
| **`CONSTRAINED`** | Fits within RAM+VRAM, predicted decode $1 \text{ to } 8$ tok/s | Interactive with slight latency |
| **`BATCH_ONLY`** | Requires disk paging/streaming, or predicted decode $0.05 \text{ to } 1$ tok/s | Background / document processing only |
| **`INFEASIBLE`** | Exceeds total available budget or $< 0.05$ tok/s | Suggest reduction in context, quant, or model |
