# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-17

### Added
- **Hardware-Aware Cost Model Correctness (Phase 1):**
  - Read-only STREAM memory benchmark replacing memcpy-bound bandwidth estimation.
  - SIMD GFLOPs compute bound for CPU decode tokens ($t_{\text{token}} = \max(t_{\text{bandwidth}}, t_{\text{compute}}) + t_{\text{overhead}}$).
  - Explicit PCIe transfer and synchronization latency per CPU↔GPU split boundary.
  - Unified KV-cache formula with sliding window, DeepSeek MLA, and hybrid SSM/Mamba state memory support.
  - Flash Attention prerequisite enforcement for V-cache quantization.
  - Configurable micro-batch sizing (`--ubatch`) scaling compute activation buffers.
  - Analytical MoE working set coverage modeling $1 - (1 - k/E)^T$ across prefill and decode regimes.
  - Configurable slowness verdicts (`--min-tps`) supporting `COMFORTABLE`, `CONSTRAINED`, and `BATCH_ONLY`.
- **Calibration & Validation Pipeline (Phase 2):**
  - Multi-keyed calibration schema v2 keyed by `(arch, quant, backend_version, strategy, phase)`.
  - 8-tier hierarchical fallback lookup for empirical calibration factors.
  - Accuracy table reporting in `ramllm bench` with `--export <report.json>`.
  - Community validation aggregation tool (`scripts/validate_predictions.py`) and documentation (`docs/validation.md`).
  - Plan calibration confidence tags (`[HIGH]`, `[MEDIUM]`, `[LOW]`, `[UNCALIBRATED]`) in plan explainers.
- **Upstream `-fit` Positioning & Advisory (Phase 3):**
  - Introspection of llama.cpp `-fit` support and `llama-fit-params` binary availability.
  - Advisory `--cross-check-fit` CLI flag comparing RAMLLM analytical plans against `-fit` heuristic offload.
  - Comprehensive positioning guide (`docs/positioning.md`) and feature comparison matrix in `README.md`.
- **Governor Honesty & Dynamic Replanning (Phase 4):**
  - Explicit OS enforcement declarations across Linux (cgroups v2), Windows (Job Objects), and macOS (soft polling & tiered recovery).
  - Fixed RAM OOM auto-recovery to preserve GPU layer offload and relieve host RAM via KV quantization and context shrinking.
  - MoE Active vs. Total Weights clarification in plan summaries and explainers.
  - Verified dynamic replanning endpoint (`/ramllm/replan`) in API server.
- **Security & Robustness Hardening (Phase 5):**
  - Bearer token and `X-API-Key` authentication for all protected API server endpoints (`--api-key`, `RAMLLM_API_KEY`).
  - Loopback binding enforcement refusing public network interfaces (`0.0.0.0`) without an API key unless `--insecure` is supplied.
  - Configurable CORS middleware (`--cors-origins`) with preflight OPTIONS validation.
  - GGUF parser bounds checking (layer count, tensor count, dimensions, metadata length) and header validation.
  - Download pipeline hardening: destination filename path traversal prevention, preflight disk space checks ($\ge 1.15\times$ size), redirect limits, HTTPS→HTTP downgrade rejection, and masked `HF_TOKEN` support.
  - Backend process isolation via dedicated process groups and recursive child process tree cleanup on exit.
- **Tests & Continuous Integration (Phase 6):**
  - 4 Hypothesis property tests for planner: monotonicity, hardware memory bounds, placement completeness, and determinism.
  - Synthetic lightweight GGUF fixture generator script (`scripts/create_fixture_gguf.py`).
  - Weekly upstream llama.cpp compatibility workflow (`.github/workflows/weekly_llamacpp.yml`).
  - Upstream compatibility matrix in `docs/backend-notes.md`.
  - Cross-platform CI coverage workflow across Python 3.11 and 3.12 with `pytest-cov`.

## [0.1.1] - 2024-10-14

### Added
- Initial public release of RAMLLM.
- Hardware profiler with CPU, RAM, and GPU detection.
- Model analyzer reading GGUF headers and tensor shapes without weight loading.
- Budget-aware execution planner mapping models to llama.cpp flags.
- Real-time Memory Governor with multi-tiered pressure mitigation.
- Interactive terminal chat REPL and Textual TUI live dashboard.
- OpenAI-compatible REST API server with streaming SSE.
- Offline-first verified model registry and catalog.
