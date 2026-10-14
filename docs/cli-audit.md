# RAMLLM CLI & Architecture Audit (Milestone 0)

Date: October 2026  
Status: Baseline Established  
Environment Tested: macOS 15+ (Apple Silicon M5, Metal Unified Memory 24 GB RAM / 18 GB VRAM), Python 3.11, 3.12, 3.13

---

## 1. Executive Summary

RAMLLM's core engine and planning intelligence are functional and tested:
- Hardware profiling (`probe_hardware`, synthetic profiles) accurately measures CPU SIMD, RAM bandwidth, GPU VRAM & unified memory, and disk read rates.
- GGUF analyzer (`analyze_gguf`) reads headers, tensor topologies, and MoE structures without loading full tensor arrays into memory.
- Execution planner (`create_execution_plans`) generates valid, ranked plans with memory footprints and speed predictions.
- Memory governor (`MemoryGovernor`) and recovery engine (`launch_with_recovery`) can launch `llama-server`, observe process RSS, and recover gracefully on mock backends.
- 62 tests pass under pytest across Python 3.11, 3.12, and 3.13.

However, the existing CLI in `src/ramllm/cli/main.py` is a monolithic 731-line single-shot script tailored for developers who already know memory terms and have `llama-server` manually installed on their system path. There is no interactive chat REPL, no automated engine installer/manager, no configuration file, no background process tracking, and no curated model index.

---

## 2. Command-by-Command Audit

| Command | Status | Implementation Details | Gaps vs UX Plan |
|---|---|---|---|
| `ramllm` (no args) | Works (basic) | Shows Typer help text (`no_args_is_help=True`). | Needs interactive Home Menu when in a TTY (Milestone 4). Help should only show when piped or non-TTY. |
| `ramllm profile` | Works | Probes CPU, RAM, Disk, and GPU; renders Rich table or JSON (`--json`). Supports `--hw-json` for synthetic fixtures. | Needs plain-English summary, cleaner units, and integration with the auto-budget system. |
| `ramllm doctor` | Partial | Detects `llama-server` on PATH, inspects version & capabilities, checks OS memory enforcement (macOS dispatch / Linux cgroups / Windows Job Objects). | Lacks `--fix` mode to auto-remediate problems (e.g. installing missing backend, clearing stale state). Lacks severity levels (OK / Notice / Problem). |
| `ramllm plan <model>` | Works | Computes candidate placements across VRAM/RAM/Disk, ranks them, prints a technical Rich table with VRAM peak, RAM peak, decode speed, bottleneck, verdict. | Requires manual `--vram`/`--ram` or uses probed values. Dumps a technical table first instead of a plain-English 2–3 line summary. Lacks model reference resolution (only takes local file paths). |
| `ramllm run <model>` | Works | Generates plans, launches `llama-server` via `launch_with_recovery`, executes a single prompt (`--prompt`), prints the response, then kills the server. | Not an interactive chat REPL. Only runs a single prompt. Does not maintain persistent conversation history or streaming output. |
| `ramllm serve <model>` | Partial | Creates FastAPI server with `/v1/chat/completions`, `/ramllm/metrics`, and `/ramllm/replan`. Runs foreground `uvicorn.run`. | No background mode (`--detach`). No process registry (`ps`, `stop`, `logs`). Requires specifying a model file directly. |
| `ramllm bench <model>` | Works | Launches server, runs standard prompt suite (`BENCH_PROMPTS`), records TTFT, decode speed, peak RAM, updates `bench.jsonl` calibration. | Requires existing model file path. |
| `ramllm recommend` | Works | Recommends models from the local catalog that fit available budgets. | Only inspects locally added models in `~/.local/share/ramllm/models.json`. Does not search an offline curated index. Does not offer to download & chat with #1 pick. |
| `ramllm models add <path>` | Works | Parses GGUF and registers in `~/.local/share/ramllm/models.json`. | Hard-coded path in home directory; needs `platformdirs` and `RAMLLM_MODELS_DIR`. |
| `ramllm models list` | Works | Displays registered models in a table. | Missing "Fits?" column based on current machine hardware. |
| `ramllm models scan <dir>` | Works | Recursively scans for `.gguf` files and registers them. | Functional. |
| `ramllm models remove <name>` | Works | Unregisters entry from `models.json`. | Does not optionally delete model file or show disk space reclaimed. |
| `ramllm models verify [model]` | Works | Computes SHA256 of model file and validates against registered hash. | Functional. |
| `ramllm models pull <target>` | Partial | Basic HF chunked downloader with SHA-256 verification and resume. | Target must be `org/repo:file.gguf` or direct URL. No short model names (`qwen2.5-7b`), no auto-quant selection, no disk-space preflight check. |

---

## 3. Missing Commands Required by Target UX

The following commands are completely absent from the current codebase and must be implemented:
1. `ramllm setup`: First-run interactive wizard.
2. `ramllm chat [model]`: Interactive streaming chat REPL with multi-line input, slash commands, live toolbar, and governor recovery.
3. `ramllm ps`: List running RAMLLM engine and server instances.
4. `ramllm stop [id|--all]`: Stop running engine/server instances.
5. `ramllm logs [id] [-f]`: Stream/view engine process logs.
6. `ramllm status`: One-screen summary of engine, models, active instances, and memory status.
7. `ramllm dashboard`: Live Textual TUI for monitoring memory tiers, speed, sparklines, and runtime replanning.
8. `ramllm backend install|update|list|use|remove`: Manage `llama.cpp` binaries automatically across macOS, Linux, and Windows.
9. `ramllm config get|set|list|edit|path|reset`: TOML configuration manager using `platformdirs`.
10. `ramllm models search [query]`: Search curated offline model index with real-time hardware fit analysis.
11. `ramllm models info <model>`: Display quants, sizes, and hardware fit verdicts for a model.
12. `ramllm models du`: Disk usage summary for local models.
13. `ramllm clean`: Remove partial downloads and temporary caches.
14. `ramllm tour`: Short interactive overview of memory tiers and feasibility verdicts.
15. `ramllm update`: Check for RAMLLM CLI updates and print installation upgrade commands.
16. `ramllm completion install|show`: Shell completion setup.
17. `ramllm explain <term>`: Plain-language glossary lookup for technical terms and error codes.

---

## 4. Code & Architecture Defects Identified

1. **Monolithic CLI File:** `src/ramllm/cli/main.py` contains all command definitions, flag parsing, table rendering, and process execution in a single 731-line file.
2. **Hardcoded Storage Paths:** `src/ramllm/registry/catalog.py` hard-codes `Path.home() / ".local" / "share" / "ramllm"` which does not follow platform standards on Windows (`%LOCALAPPDATA%`) or macOS (`~/Library/Application Support`).
3. **Mypy Type Error:** In `src/ramllm/cli/main.py:63`, dictionary comprehension failed type narrowing for `parts[i]` (resolved in Milestone 0).
4. **README Defects:**
   - Git clone URL placeholder: `https://github.com/your-org/ramllm.git` instead of `https://github.com/Mayank3613/RAMLLM.git`.
   - Static test badge: `[![Tests: Passing](https://img.shields.io/badge/Tests-62%20Passing-brightgreen.svg)]` instead of dynamic GitHub Actions CI workflow badge.
   - Python support statement: README says "Python 3.11 or 3.12", whereas Python 3.13 is fully supported and verified.
5. **No Service Layer:** Business logic for planning, downloading, resolving models, and starting processes is tightly coupled with CLI presentation. A clean `services/` layer is necessary to share logic between CLI commands, the home menu, the chat REPL, and the TUI dashboard.

---

## 5. Architectural Adjustments & Roadmap

To support the UX plan while maintaining full backward compatibility:
- Introduce `src/ramllm/services/`:
  - `backend_manager.py`: discovery, release assets, downloads, extraction, verification.
  - `config_service.py`: reading/writing TOML config with defaults and validation.
  - `model_service.py`: curated index, model reference resolution, pull, disk usage.
  - `plan_service.py`: automatic memory budget resolution, mode translation, plain-English formatting.
  - `session_service.py`: process tracking, state registry, detached servers, clean shutdown.
- Refactor `src/ramllm/cli/`:
  - `app.py`: Typer app with grouped help panels, global flags, typo suggestions.
  - `ui/`: `console.py`, `theme.py`, `tables.py`, `progress.py`, `errors.py`, `glossary.py`, `prompts.py`.
  - `commands/`: modular command modules (`plan_cmd.py`, `chat_cmd.py`, `models_cmd.py`, `backend_cmd.py`, etc.).
  - `setup/`: wizard step engine and resumable state.
  - `home/`: interactive terminal menu.
  - `chat/`: streaming prompt_toolkit REPL with slash commands.
  - `tui/`: Textual live dashboard.
- Backwards compatibility: Existing entry point `ramllm.cli.main:app` will import and expose the modular app so existing scripts and tests continue to work without breaking changes.
