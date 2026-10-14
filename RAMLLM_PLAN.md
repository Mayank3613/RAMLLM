# RAMLLM — Project Specification & Implementation Plan

> **Tagline:** Run bigger models on smaller machines.
> **One-line pitch:** Tell RAMLLM how much VRAM, RAM, and patience you have. It places the hottest weights on your GPU, warm ones in RAM, cold ones on disk, tells you exactly how fast that will run, and enforces the budget while it runs.

---

## 0. Instructions for the AI coding agent

Read this whole document before writing code. Then follow these rules:

1. **Work milestone by milestone** (Section 11). Do not start a milestone until the previous one meets its acceptance criteria. Commit after each task with a clear message.
2. **Do not build a new inference engine early.** RAMLLM is first an *orchestrator and planner* on top of existing engines (llama.cpp first). Custom kernels or streaming code are only allowed in Milestone 7+.
3. **Verify, don't assume.** llama.cpp flags and APIs change often. Before using any flag listed in this document, run `llama-server --help` / `llama-cli --help` on the installed build and confirm it exists. If a flag differs, adapt and note it in `docs/backend-notes.md`.
4. **Be honest in output.** The tool must never claim a model "runs fine" when the estimate says otherwise. Every plan prints a feasibility verdict and the named bottleneck.
5. **Everything is testable without a GPU or a large model.** Use mocks, fixture GGUF headers, and synthetic hardware profiles so CI can run on a plain runner. Real-model tests are marked `@pytest.mark.slow` / `@pytest.mark.gpu` and skipped by default.
6. **Offline-first.** No network call may happen unless the user runs an explicit download command. No telemetry is sent anywhere unless the user opts in.
7. **Code quality:** typed Python (mypy strict on `core/`), `ruff` for lint/format, docstrings on public functions, small modules, no global mutable state.
8. **When a decision is ambiguous**, choose the simplest option, document it in `docs/decisions/NNNN-title.md` (short ADR), and continue.

---

## 1. Product summary

RAMLLM is an open-source, RAM-adaptive and GPU-aware inference manager. It separates the **user interface** from the **inference engine**. The user specifies:

- the model,
- a **memory budget per tier** (VRAM, RAM, optionally disk cache),
- context length requirements,
- a performance preference and quality floor.

RAMLLM then:

1. **Profiles** the hardware (including GPU, VRAM, PCIe, disk bandwidth).
2. **Analyzes** the model (size, layers, MoE structure, KV cost).
3. **Plans** an execution strategy (quantization, CPU/GPU placement, KV placement, MoE expert placement, layer streaming) and predicts speed and memory per tier.
4. **Runs** the model through a backend, while a **memory governor** monitors and enforces the budget.
5. **Reports** honestly: feasibility verdict, bottleneck, estimated vs measured speed.

**Central promise:** make larger models accessible on constrained hardware. **Not** a promise that every model runs at useful speed in every budget. When something is infeasible or very slow, say so and suggest alternatives.

---

## 2. Goals and non-goals

### Goals
- G1. A `plan` command that, from hardware + model + constraints, outputs a ranked list of execution plans with estimated decode/prefill speed and per-tier peak memory.
- G2. A `run` / `serve` command that executes the chosen plan via a backend and reports actual memory and speed.
- G3. GPU-aware placement: full GPU residency, partial layer offload, KV cache placement, multi-GPU split, MoE expert placement.
- G4. Budget enforcement and verification: report peak usage per tier; hard-enforce where the OS allows, soft-enforce and warn otherwise.
- G5. Cross-platform: Linux, macOS (including Apple Silicon unified memory), Windows.
- G6. Backend-agnostic design with capability flags; llama.cpp first.
- G7. Offline-first, open-source (Apache-2.0).

### Non-goals (for v1)
- Training or fine-tuning.
- A new tensor/kernel library.
- Guaranteeing interactive speed for dense models far larger than RAM+VRAM.
- Cloud or distributed inference.
- A polished GUI (comes after the CLI/server are solid).

---

## 3. Key terms

| Term | Meaning |
|---|---|
| **Tier** | A memory level: `vram`, `ram`, `disk`. |
| **Plan** | A concrete execution configuration: quantization, per-layer/tensor placement, KV placement/precision, context size, backend flags. |
| **Working set** | Bytes that must be read per generated token. Dense: ~all weights. MoE: non-expert weights + active experts. |
| **Feasibility verdict** | `COMFORTABLE`, `CONSTRAINED`, `BATCH_ONLY`, `INFEASIBLE`. |
| **Governor** | Runtime component that monitors memory per tier and reacts (shrink cache, reduce offload, warn). |
| **Capability flags** | What a backend can do (partial offload, per-tensor placement, quantized KV, etc.). The planner only proposes plans the backend can execute. |
