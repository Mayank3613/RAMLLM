# RAMLLM Core Concepts

RAMLLM removes the guesswork from running local language models. This document explains the underlying concepts in plain language.

---

## 1. Memory Tiers: How Your Computer Stores AI Models

When an LLM runs, it requires memory for three distinct components:
1. **Model Weights:** The neural network parameters (the "brain" of the model).
2. **Context & KV Cache:** The working memory of the ongoing conversation.
3. **Compute Activations:** Temporary scratch buffers needed while calculating the next token.

Your computer provides memory in different tiers with radically different speeds:

```
┌────────────────────────────────────────────────────────┐
│ GPU VRAM (Unified / Dedicated)  [Fast]   150–1,000+ GB/s│
├────────────────────────────────────────────────────────┤
│ System RAM (DDR4 / DDR5)        [Medium] 40–90 GB/s    │
├────────────────────────────────────────────────────────┤
│ NVMe SSD / Disk (Swap)          [Slow]   1–7 GB/s      │
└────────────────────────────────────────────────────────┘
```

### Tier 1: GPU Video Memory (VRAM)
- **What it is:** Ultra-high-speed memory directly connected to GPU cores (or Apple Silicon Unified Memory).
- **Why it matters:** Generates tokens at maximum speed (25–100+ tokens/sec).
- **RAMLLM's rule:** Prioritizes placing as many model layers and attention matrices into VRAM as will fit safely.

### Tier 2: System Memory (RAM)
- **What it is:** Your motherboard's main DDR4 or DDR5 memory pool.
- **Why it matters:** Slower than VRAM, but often 2x to 4x larger. Running offloaded layers here keeps models usable when they exceed GPU memory.
- **RAMLLM's rule:** Splits larger models across GPU and CPU so you can run 14B or 32B models on modest hardware.

### Tier 3: Disk & Swap
- **What it is:** Storage used by the operating system when physical RAM runs out.
- **Why it matters:** Swap is orders of magnitude slower than RAM. If an AI engine causes your system to page to swap, your mouse will stutter and your system may freeze.
- **RAMLLM's rule:** **Never allows active model weights or KV cache into swap.** RAMLLM calculates strict headroom buffers to keep your system responsive.

---

## 2. Quantization: Compressing Models Without Losing Intelligence

Original model weights are trained in 16-bit floating point numbers (`FP16`). A 7-billion parameter model in FP16 weighs approximately 14 GB.

**Quantization** converts these numbers into smaller formats (8-bit, 5-bit, 4-bit, or even 2-bit integers):

| Quantization | Size per Param | 7B Model Size | Quality Retention | Plain English Recommendation |
|---|---|---|---|---|
| `FP16` | 16 bits (2 bytes) | ~14.0 GB | 100% | Rarely needed for inference |
| `Q8_0` | 8 bits (1 byte) | ~7.5 GB | 99.8% | Near-lossless, great for code & math |
| `Q5_K_M` | 5 bits (~0.65 B) | ~5.1 GB | 98.5% | High quality sweet spot |
| `Q4_K_M` | 4 bits (~0.55 B) | ~4.4 GB | 97.0% | **Standard Default**: Best balance of speed, size, and smarts |
| `Q3_K_M` | 3 bits (~0.45 B) | ~3.5 GB | 92.0% | Use when constrained on memory |
| `IQ2_M` | 2 bits (~0.30 B) | ~2.5 GB | 82.0% | Extreme fit mode for small hardware |

---

## 3. KV Cache & KV Cache Quantization

When you chat with an LLM, every previous message in the conversation must be remembered and processed to produce the next word. This saved history is stored in the **KV (Key-Value) Cache**.

- A 4,096 token context window in FP16 can take **1 GB to 2 GB** of memory just for the chat history!
- An 8k or 16k context window can easily exceed the size of the model itself.

### KV Cache Quantization (`q8_0` & `q4_0`)
RAMLLM supports quantizing the KV cache:
- **`q8_0` cache:** Cuts KV memory usage by 50% with zero noticeable impact on conversational reasoning.
- **`q4_0` cache:** Cuts KV memory usage by 75%, allowing massive context windows on modest GPUs.

---

## 4. Mixture of Experts (MoE)

Models like **Mixtral 8x7B** or **Qwen1.5-MoE** use Mixture-of-Experts architecture:
- Total parameters: 47 Billion.
- Active parameters per token: Only 13 Billion.

RAMLLM's placement engine specifically analyzes MoE models:
- It keeps the shared non-expert layers and the most frequently activated routing layers in fast GPU memory.
- It safely streams expert layers from system RAM, letting you run a 47B model with the responsiveness of a much smaller model.

---

## 5. Execution Plan Verdicts

Whenever you plan, run, or chat with a model, RAMLLM analyzes all possible placement strategies and assigns a clear verdict:

### `COMFORTABLE` (Runs smoothly)
- The model and full context fit entirely in GPU VRAM or unified memory with comfortable headroom.
- Generates at full native hardware speed (typically 25–60+ words/sec).

### `CONSTRAINED` (Works reliably)
- The model fits, but requires layer splitting between GPU and RAM, or requires a quantized KV cache (`q8_0`/`q4_0`).
- Speed is slightly reduced (typically 8–20 words/sec), but completely usable for real-time chat.

### `BATCH_ONLY` (Fine for overnight / offline jobs)
- The model exceeds GPU memory and runs mostly in CPU system RAM.
- Speed is 1–4 words/sec. Too slow for interactive typing, but excellent for background document summarization, batch translation, or unattended tasks.

### `INFEASIBLE` (Won't fit)
- The model's minimum memory footprint exceeds your total physical RAM and VRAM combined.
- RAMLLM blocks launch to protect your computer from crashing or freezing, and recommends smaller models or alternative quantizations.

---

## 6. The Memory Governor & Chaos Recovery

Unlike ordinary inference tools that crash when unexpected system memory pressure occurs, RAMLLM features an active **Memory Governor**:
1. **Pre-flight verification:** Allocates safety buffers and verifies RSS limits before committing to execution.
2. **Live monitoring:** Samples system RAM, GPU free buffers, and process memory every second.
3. **Adaptive Auto-Recovery:** If unexpected memory spikes occur, RAMLLM catches them and automatically switches to an optimized fallback plan (reducing context or cache quantization) without dropping your session.
