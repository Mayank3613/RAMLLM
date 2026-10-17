# Technical Positioning: RAMLLM vs. llama.cpp -fit, Ollama, and LM Studio

Local large language model inference has advanced rapidly, but users continually face a frustrating dilemma: models fail to load, crash with Out-Of-Memory (OOM) errors, or silently run at unusable speeds (e.g., 0.5 tokens/sec).

RAMLLM solves this by operating as an **adaptive planning, forward cost modeling, and runtime memory governance layer** positioned directly above standard inference engines like `llama.cpp`.

---

## Detailed Architectural Comparison

| Capability | llama.cpp (native) | llama.cpp `-fit` | Ollama | LM Studio | **RAMLLM** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | High-performance inference engine | Boot-time parameter downscaling | Easy model packaging & serving | Desktop GUI for casual users | **Hardware-adaptive planning & memory supervision** |
| **Relationship to llama.cpp** | The engine itself | Feature within llama.cpp | Bundled engine fork | Bundled engine | **Supervises official llama.cpp without forking** |
| **Pre-Flight Topology Profiling** | None (loads on call) | VRAM-only probe | Coarse heuristics | Manual slider guess | **STREAM RAM BW, PCIe bandwidth, VRAM, and SIMD FLOPs** |
| **Speed Forecasting** | No | No (speed blind) | No | No | **Analytical interval $[L, H]$ tok/s + named bottleneck** |
| **Slowness Verdicts & Thresholds** | No | No (fits at any cost) | No | No | **Normative verdicts (`COMFORTABLE`, `CONSTRAINED`, `BATCH_ONLY`)** |
| **KV-Cache Optimization** | Manual flags (`-ctk`, `-ctv`) | Defaults to unquantized FP16 | Fixed FP16 | Limited manual dropdown | **Automated KV quantization (`q8_0`, `q4_0`) based on budget** |
| **MoE Expert Partitioning** | Manual flags (`-cmoe`) | No partial expert split | All-or-nothing | Not exposed | **Automated attention-in-VRAM / experts-in-RAM planning** |
| **Runtime Memory Governor** | None (OS kills on OOM) | None (static at launch) | None (OS swap thrashing) | None (crashes on OOM) | **Live 1 Hz sampling with tiered automatic recovery** |
| **Dynamic In-Session Replanning** | Requires restart | Requires restart | Requires restart | Requires restart | **Hot replanning via `/ramllm/replan` API & `/replan` in chat** |
| **Empirical Calibration** | No | No | No | No | **8-tier stratified empirical benchmark calibration** |
| **Cross-Check Verification** | N/A | N/A | No | No | **Live advisory via `ramllm plan --cross-check-fit`** |

---

## Deep Dive: RAMLLM vs. llama.cpp `-fit` (`llama-fit-params`)

Upstream `llama.cpp` introduced the `-fit` option and the companion `llama-fit-params` binary to address the difficulty of setting `-ngl` (number of GPU layers) and `-c` (context length) manually. While `-fit` is a valuable utility, RAMLLM approaches the problem with a fundamentally different design philosophy:

### 1. Speed-Blindness vs. Forward Throughput Modeling
- **llama.cpp `-fit`:** Iteratively tests lower layer counts or context lengths until estimated weight memory fits inside reported free VRAM. However, it has **zero concept of speed**. If fitting a 30B model requires offloading 90% of layers over an 8 GB/s PCIe bus, `-fit` will enthusiastically launch the configuration even if generation drops to an unusable 0.8 tokens per second.
- **RAMLLM:** Calculates memory transfer times ($t_{\text{bandwidth}} = \frac{\text{Bytes}_{\text{VRAM}}}{\text{BW}_{\text{VRAM}}} + \frac{\text{Bytes}_{\text{RAM}}}{\text{BW}_{\text{RAM}}}$), compute times ($t_{\text{compute}}$), and PCIe boundary synchronization penalties ($t_{\text{split}}$). RAMLLM projects explicit throughput bounds ($[L, H]$ tok/s) and flags plans that fall below acceptable human reading speeds.

### 2. Normative Slowness Verdicts & Minimum TPS Policies
- **llama.cpp `-fit`:** Treats all successful allocations as equal.
- **RAMLLM:** Implements explicit normative verdicts:
  - `COMFORTABLE`: Smooth interactive conversational throughput ($\ge 8$ tok/s).
  - `CONSTRAINED`: Usable but noticeably sluggish ($1–8$ tok/s).
  - `BATCH_ONLY`: Sub-interactive speeds; recommended only for overnight or background tasks ($< 1$ tok/s).
  - Users and automation pipelines can enforce strict `--min-tps` thresholds to fail early rather than waste time on unviable configurations.

### 3. KV-Cache Quantization & Context Preservation
- **llama.cpp `-fit`:** Leaves the KV-cache in unquantized 16-bit float format (`f16`). When VRAM is tight, `-fit`'s only recourse is to drastically truncate your context window down to 2,048 or 1,024 tokens.
- **RAMLLM:** Evaluates KV-cache quantization (`q8_0` and `q4_0`). By saving up to 50–75% of KV memory footprint, RAMLLM can fit 8,192 or 16,384 tokens in the same VRAM footprint that forces `-fit` to severely compromise context length.

### 4. Static Boot Allocation vs. Dynamic Runtime Governance
- **llama.cpp `-fit`:** Queries available memory once at engine boot time. If another process (a web browser tab, IDE compilation, or video playback) claims system memory during a generation burst, llama.cpp crashes or locks up the host operating system.
- **RAMLLM:** Launches a real-time Memory Governor sampling host RAM and VRAM at 1 Hz. If available memory drops below safety reserves, RAMLLM intervenes before the OS triggers an OOM kill:
  1. Clamps KV cache context length.
  2. Migrates KV buffers from VRAM to host RAM.
  3. Evicts non-essential scratch allocations.
  4. Triggers clean in-place process recovery without losing server session state.

### 5. Live Cross-Checking via `--cross-check-fit`
RAMLLM embraces upstream tools rather than hiding them. You can directly inspect how RAMLLM's analytical plan compares to upstream `llama-fit-params`:

```bash
ramllm plan /path/to/model.gguf --cross-check-fit
```

---

## Deep Dive: RAMLLM vs. Ollama

Ollama is an exceptional developer tool for pulling and running models with minimal friction. However, Ollama's resource allocation strategy is opaque:
- **Ollama's Coarse Heuristic:** If a model's weights and basic context exceed VRAM by even a few megabytes, Ollama often drops execution entirely to CPU without warning, causing generation speed to plunge from 35 tok/s to 0.7 tok/s.
- **OS Swap Thrashing:** Ollama lacks dynamic OS headroom monitoring. On memory-constrained laptops (e.g. 8 GB or 16 GB), running an oversized model can trigger aggressive operating system swap file paging, completely freezing the mouse cursor and system UI.
- **RAMLLM's Approach:** RAMLLM inspects hardware topology *before* downloading models. It tells you the exact predicted speed, warns if your hardware is constrained, and enforces strict memory safety reserves so your computer never locks up.

---

## Deep Dive: RAMLLM vs. LM Studio

LM Studio provides a polished desktop GUI for experimenting with LLMs:
- **Manual Guess-and-Check:** LM Studio requires users to manually manipulate GPU offload sliders (`-ngl`) and trial-and-error context sizes. If a user selects 2 layers too many, the backend crashes with a CUDA/Metal error.
- **GUI Overhead:** LM Studio's Electron desktop environment consumes several hundred megabytes of RAM that could otherwise be allocated to model weights or larger KV contexts.
- **RAMLLM's Approach:** RAMLLM is headless-first, CLI-driven, and automation-friendly. It mathematically calculates the optimal split point instantly, requires zero manual slider adjustments, and consumes negligible system overhead.
