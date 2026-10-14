# RAMLLM — CLI & User-Friendliness Implementation Plan

**Repo:** https://github.com/Mayank3613/RAMLLM (existing, ~12 commits, Python, Typer/Rich, FastAPI, llama.cpp backend)
**Purpose of this document:** instructions for an AI coding agent (Antigravity) to turn RAMLLM from "a toolkit that configures and plans model execution" into **a complete, friendly, terminal-first product that a non-expert can install and use in minutes.**

---

## 0. Read this first (context, assumptions, rules)

### 0.1 What exists today (from the repo README)
The README describes a Typer-based command set already: `doctor`, `profile`, `plan`, `run`, `serve`, `recommend`, `bench`, `models add|list|scan|remove|verify|pull`, an OpenAI-compatible FastAPI server (`/v1/chat/completions`, `/ramllm/metrics`, `/ramllm/replan`), a planner, a memory governor, synthetic hardware profiles for tests, and a mock backend/fixtures. Install today means: install llama.cpp manually, `git clone`, create a venv, `pip install -e .`, Python 3.11/3.12 only.

### 0.2 The problem we are solving
1. **There is no real terminal experience.** The existing commands are flag-heavy, single-shot utilities (mostly "configure and plan"). There is no guided flow, no interactive chat, no menu, no way to manage running models, no live view. A user cannot do "everything" from one friendly terminal app.
2. **Setup is tedious** and aimed at intermediate developers: manual llama.cpp install, git clone, venv, version-pinned Python, placeholder clone URL in the README (`your-org/ramllm`), users must know what `--vram`, `--ram`, quantization, KV cache, "MoE" mean.

### 0.3 Rules for the agent
1. **Start with an audit (Milestone 0).** I (the author of this plan) could only read the README, not the source files. Run `ramllm --help`, read `src/ramllm/cli/`, and write `docs/cli-audit.md` listing what each command really does, what is stubbed, and what differs from this plan. **Adapt module names/paths in this plan to the real layout.** Do not rewrite things that already work; extend them.
2. **Backward compatibility:** every existing command and flag keeps working. New behaviour is additive. If a flag must change, keep the old one as a hidden deprecated alias for at least one release.
3. **Beginner-first, expert-friendly.** Defaults are automatic and safe. Every expert flag still exists. Anything interactive must have a non-interactive flag equivalent (`--yes`, `--json`, `--plain`).
4. **Offline-first stays true.** Nothing touches the network without the user's explicit consent in that moment (or an explicit command). Telemetry stays off.
5. **Never invent external facts.** Model repos/filenames, llama.cpp release asset names, checksums, and URLs must be verified (fetched and checked) when you write them, and the code must handle them changing. Do not hard-code guesses.
6. **Every milestone ships with tests and docs** and must satisfy its acceptance criteria before the next one starts. Use the existing mock backend and synthetic hardware profiles so CI needs no GPU or big model.
7. **Plain language everywhere.** Technical terms (VRAM, quantization, KV cache, MoE, layers) must be explained at first use or behind a `?`/`--details` view. Never show a raw traceback unless `--debug`.
8. **Cross-platform is a requirement, not a bonus:** Windows (PowerShell/Windows Terminal + legacy console fallback), macOS (Intel + Apple Silicon), Linux. Test paths, subprocess handling, signals, ANSI/Unicode fallback.
9. When a decision is ambiguous, pick the simplest option, record it in `docs/decisions/NNNN-title.md`, and continue.

---

## 1. Product vision (target experience)

> **Install in one line. Run `ramllm`. Answer a few questions. Be chatting with a model that fits your computer in minutes — without ever typing a flag.**

### 1.1 The three-command promise (must be true at the end)
```
# 1. install (no Python knowledge needed)
curl -LsSf https://<project-domain-or-raw-github>/install.sh | sh        # macOS/Linux
irm https://<project-domain-or-raw-github>/install.ps1 | iex              # Windows PowerShell

# 2. set up (guided; installs the engine, checks hardware, picks + downloads a model)
ramllm setup

# 3. talk
ramllm chat
```
Also supported: `pipx install ramllm` / `uv tool install ramllm` / Homebrew / winget or scoop (Milestone 8).

### 1.2 User personas
| Persona | Needs | Served by |
|---|---|---|
| **Newcomer** ("I just want a local ChatGPT-like thing") | No jargon, no flags, no manual installs | `ramllm` home menu, `setup` wizard, `chat`, auto budgets |
| **Hobbyist** | Pick models, understand what fits, run an API for Open WebUI etc. | `models`, `recommend`, `serve --detach`, `dashboard` |
| **Developer/power user** | Exact control, scripting, JSON, CI | Existing flags, `--json`, `--yes`, `run` one-shot/stdin, `config`, env vars |

### 1.3 Design principles
1. **One obvious next step** after every command ("Next: `ramllm chat`").
2. **Defaults over questions; questions over flags.** Auto-detect first, ask only when a real choice exists.
3. **Show, don't dump:** a plain-English summary first; tables/details on demand.
4. **Honest and calm:** keep the project's honesty rule (verdict + bottleneck), phrased kindly ("This will work but slowly — about 2 words per second").
5. **Safe by default:** confirm big downloads, check disk space, never freeze the user's machine (leave memory headroom).
6. **Discoverable:** grouped help, typo suggestions, tab-completion, a menu for people who don't know commands.

---

## 2. Feature parity: everything doable from the terminal

Every user task maps to a command AND an entry in the interactive home menu.

| Task | Command | Interactive entry |
|---|---|---|
| First-time setup | `ramllm setup` | auto-launched on first run |
| Install / update / choose engine (llama.cpp) | `ramllm backend install|update|list|use|remove` | Settings → Engine |
| Fix problems | `ramllm doctor [--fix]` | Help → Diagnose |
| Measure my computer | `ramllm profile [--refresh]` | "Check my computer" |
| Find a model that fits | `ramllm recommend` / `ramllm models search` | "Find a model" |
| Download / add / remove / verify / scan models | `ramllm models pull|add|remove|verify|scan|list|du` | "My models" |
| See what will happen before running | `ramllm plan <model>` | "Preview a model" |
| Chat interactively | `ramllm chat [model]` | "Chat" |
| One-shot / scripting / pipes | `ramllm run <model> -p "…"` or stdin | — |
| Start an API server (foreground/background) | `ramllm serve [--detach]` | "Start API server" |
| See / stop running models | `ramllm ps`, `ramllm stop`, `ramllm logs` | "Running now" |
| Live memory/speed monitor | `ramllm dashboard` | "Live monitor" |
| Change memory limits at runtime | `/replan` in chat, dashboard keybind, `ramllm replan` | in chat/dashboard |
| Benchmark & calibrate | `ramllm bench` | "Test my speed" |
| Settings | `ramllm config get|set|list|edit|path|reset` | "Settings" |
| Update RAMLLM | `ramllm update` (explicit) | Settings → Updates |
| Clean up disk | `ramllm clean`, `ramllm models du` | "My models" → Storage |
| Shell completion | `ramllm completion install` | offered in wizard |

---

## 3. Final CLI map

Keep all existing commands. Add the ones marked **NEW**. Group them in `--help` using Typer `rich_help_panel`.

```
ramllm                       NEW  Home: first-run wizard OR interactive menu (TTY); help (non-TTY)

GET STARTED
  setup                      NEW  Guided first-run wizard
  doctor [--fix]             (extend) diagnose and optionally auto-fix
  tour                       NEW  30-second explanation of tiers/verdicts (optional, short)

CHAT & RUN
  chat [model]               NEW  Interactive chat REPL (auto-plan, auto-launch, auto-stop)
  run <model> [-p TEXT]      (keep) one-shot / stdin / scripting; back-compatible
  serve [model] [--detach]   (extend) OpenAI-compatible API; background mode
  ps                         NEW  List running RAMLLM engines/servers
  stop [id|--all]            NEW  Stop them
  logs [id] [-f]             NEW  Show/follow engine logs
  status                     NEW  One-screen summary: engine, models, running, memory
  dashboard                  NEW  Live TUI (memory per tier, tok/s, warnings, controls)

MODELS
  models list|add|scan|remove|verify|pull   (keep)
  models search [query]      NEW  Search curated index (offline) [+ optional online]
  models info <model>        NEW  Size, quants, what fits, license, source
  models du                  NEW  Disk usage per model
  recommend                  (extend) friendlier; offers "pull and chat" at the end
  plan <model>               (extend) plain-English summary first; --details for table
  bench <model>              (keep)

SYSTEM
  profile                    (keep)
  backend install|update|list|use|remove    NEW  Manage llama.cpp builds
  config get|set|list|edit|path|reset       NEW
  update                     NEW  Upgrade RAMLLM itself (explicit, never automatic)
  clean                      NEW  Remove partial downloads/caches (with confirmation)
  completion install|show    NEW
  version / --version        (keep/extend) prints RAMLLM + backend versions
```

**Global flags** (on every command): `--yes/-y` (assume yes), `--plain` (no color/emoji/boxes), `--json` (where data is returned), `--quiet/-q`, `--debug`, `--no-input` (fail instead of prompting), `--config PATH`.
**Env vars:** `RAMLLM_HOME`, `RAMLLM_CONFIG`, `RAMLLM_MODELS_DIR`, `RAMLLM_BACKEND_PATH`, `NO_COLOR`, `HF_TOKEN`.
**Precedence:** CLI flag > env var > config file > built-in default.
**TTY rule:** prompts and TUIs only when stdin and stdout are TTYs and `--no-input` is not set; otherwise behave deterministically and print actionable errors.

---

## 4. Milestones

### Milestone 0 — Audit and baseline (small, do first)
**Tasks**
1. Run the full existing suite (`pytest`, `ruff`, `mypy`) and record results. Fix failing tests/lint first if trivial.
2. Walk through every documented command on at least one real OS using the mock backend and (if available) a real llama.cpp build with a tiny GGUF. Write `docs/cli-audit.md`: command → works/partial/broken → notes.
3. Fix README defects: placeholder clone URL (`https://github.com/your-org/ramllm.git` → `https://github.com/Mayank3613/RAMLLM.git`), static "62 passing" badge (replace with a real CI badge), Python-version claims.
4. Check Python support: try 3.13; widen `requires-python` if dependencies allow, otherwise document why not.
5. Establish a **UX test harness** (see Section 9) and a `docs/ux-principles.md` containing Section 1.3.
**Acceptance:** `docs/cli-audit.md` merged; README has no placeholders; CI green; list of gaps vs. this plan appended to the audit.

---

### Milestone 1 — Zero-friction install and engine management
**Goal:** remove "install llama.cpp + clone + venv" from the user's life.

**1A. Packaging and installers**
- Publish to PyPI as `ramllm` (check name availability first; if taken, choose and document an alternative such as `ramllm-cli`, keeping the `ramllm` console script). Use PyPI **trusted publishing** from GitHub Actions on tagged releases.
- Console entry point `ramllm` (and `python -m ramllm`).
- `scripts/install.sh` (macOS/Linux) and `scripts/install.ps1` (Windows):
  - Detect `uv`; if missing, install it via its official installer (explain and ask before doing so unless `-y`). `uv` can provision a compatible Python itself, so users need **no** Python setup.
  - Run `uv tool install ramllm` (fallback: `pipx install ramllm`, then `python -m pip install --user`).
  - Ensure the tool bin dir is on PATH (guide the user if not; never silently edit shell profiles without asking).
  - End by printing: "Installed. Run `ramllm setup`."
  - Scripts must be idempotent, support `--version X`, `--uninstall`, and be linted (`shellcheck`, PSScriptAnalyzer).
- README top section becomes the three-command promise; the old clone/venv instructions move to "Developer install" in `docs/development.md`.

**1B. Backend (llama.cpp) manager** — `ramllm backend …`
- **Discovery order:** `RAMLLM_BACKEND_PATH` / config → RAMLLM-managed install → `PATH`. Report which one is used.
- **`backend install`** (explicit, asks consent, shows download size):
  1. Detect OS, CPU arch, CPU features (AVX2/AVX-512/NEON), and GPU/API availability (NVIDIA CUDA, Apple Metal, Vulkan, AMD ROCm) using the existing profiler.
  2. Query the llama.cpp GitHub Releases API for the latest release, **inspect the real asset names at runtime**, and choose the best asset via a scoring function (prefer CUDA build matching driver/CUDA version when NVIDIA is present, else Vulkan on any GPU, Metal on macOS, else CPU with the best supported instruction set). **Do not hard-code asset filenames**; keep the selection logic in one module with unit tests using recorded release-JSON fixtures for each OS.
  3. On Windows CUDA builds, also fetch the matching CUDA runtime archive if the release splits it out; otherwise fall back to Vulkan/CPU and explain why.
  4. Download with progress and resume, verify checksum/digest when the release provides one, extract to `<data_dir>/backends/llama.cpp/<release-tag>/`, mark executables (`chmod +x`), and on macOS remove the quarantine attribute for downloaded binaries when needed (explain what you're doing).
  5. Run a smoke test: `llama-server --version` and a capability probe (`--help` parse) and print "Engine ready: llama.cpp <tag> (GPU: CUDA/Metal/Vulkan/CPU)".
  6. Record in `<data_dir>/backends/registry.json`.
- **Package-manager fallback** when download fails/offline: suggest `brew install llama.cpp` (macOS), `winget install llama.cpp` if available, or offline: `ramllm backend use /path/to/llama-server`. (Verify each suggestion is currently valid before printing it.)
- `backend list` (all discovered + active), `backend use <path|tag>`, `backend update` (latest, keeps old for rollback), `backend remove <tag>`.
- Capability detection (already in the adapter) must adapt to flag differences across versions. Unknown version → warn, don't crash.

**1C. `doctor --fix`**
- Extend `doctor` with severity levels (OK / Notice / Problem) and a **fix suggestion for each problem**. With `--fix` (or interactive "Fix now? [Y/n]"), perform safe fixes: install backend, create directories, repair catalog, remove stale pid/state files, clear partial downloads, re-run profile.
- `doctor` output ends with a one-line verdict: "Everything looks good. Try `ramllm chat`."

**Acceptance (M1)**
- On a clean VM/container for each OS in CI (or documented manual runs for macOS/Windows), the installer + `ramllm backend install` produces a working `llama-server` without the user installing anything else.
- Backend asset-selector unit tests pass for fixtures: linux-x64 (CPU, CUDA, Vulkan), macOS-arm64, macOS-x64, windows-x64 (CPU, CUDA, Vulkan).
- Offline run of `ramllm backend install` fails with a clear message and manual alternatives, no traceback.

---

### Milestone 2 — Config, automatic budgets, modes, friendly output and errors
**Goal:** users stop needing `--vram/--ram/--context` and stop seeing jargon.

**2A. Config system** (`ramllm config`)
- File: `<config_dir>/config.toml` (use `platformdirs`; `tomllib` to read, `tomli-w` to write; comments preserved when possible).
- Keys (validate and document each; unknown keys warn): `default_model`, `mode` (fast|balanced|quality|max), `memory_usage` (conservative|balanced|max or a percentage), `context` (auto or number), `models_dir`, `backend_path`, `port`, `host`, `color` (auto|always|never), `unicode` (auto|ascii), `confirm_downloads` (true), `auto_retry` (true), `hf_token_env` (name of env var; never store tokens in plain text by default), `telemetry` (false).
- Commands: `config list`, `config get K`, `config set K V` (validates), `config edit` (opens `$EDITOR` / notepad), `config path`, `config reset [K]`.

**2B. Automatic budgets**
- If `--vram`/`--ram` are not given, compute: `ram_budget = available_ram × memory_usage_fraction − reserve`, `vram_budget = free_vram × fraction − reserve` per GPU. Presets: conservative ≈ 50%, balanced ≈ 70%, max ≈ 85% (tune using the existing margin rules; never exceed what the planner considers safe).
- Show the resolved budget in plain words on every `plan/run/chat/serve`: "Using up to 11 GB of your 16 GB memory and 5 GB of your 6 GB graphics memory."
- New simple flag `--memory 12GB` (total RAMLLM may use across RAM+VRAM where sensible) alongside the existing `--ram/--vram`.

**2C. Modes (replace "preference" jargon)**
| Mode | Maps to | Plain description |
|---|---|---|
| `fast` | `preference=fastest`, allow KV quant, moderate context | "Quickest replies, may trim quality" |
| `balanced` (default) | `balanced` | "Good mix of speed and quality" |
| `quality` | favor higher quants, no KV quant | "Best answers, may be slower" |
| `max` | `fit_at_any_cost` | "Biggest model that fits, even if slow" (always warns) |
Old `--prefer` stays as an alias.

**2D. Output redesign** (shared `cli/ui/` package: `console.py`, `theme.py`, `tables.py`, `progress.py`, `errors.py`, `prompts.py`)
- **Plain-English plan summary first**, details after (`--details`, `--explain` as today):
  ```
  ✔ Qwen2.5 7B (Q4) will run well on your computer
    • Speed: fast — about 25–35 words per second   [COMFORTABLE]
    • Memory: 4.8 GB of your 6 GB graphics memory, 1.2 GB of RAM
    • Where it runs: the whole model fits on your GPU
    • Tip: ask for a longer memory (context) with --context 16k — it still fits.
    More detail: ramllm plan qwen2.5-7b --details
  ```
- Verdict language: COMFORTABLE → "runs well", CONSTRAINED → "works, but slowly", BATCH_ONLY → "very slow — fine for overnight jobs, not chat", INFEASIBLE → "won't fit" + specific fixes. Keep the original verdict code visible for experts.
- Tokens/sec shown as both tok/s and approx words/sec (≈ 0.75 words/token; label as approximate).
- Consistent icons with ASCII fallback (✔ ✘ ! → vs `[ok] [x] [!] ->`) chosen from terminal capability + `--plain`.
- Progress UI: Rich progress bars with ETA for downloads; spinners with a text label for profiling/launching ("Measuring memory speed… 3s").
- **Glossary hints:** a one-line explanation the first time a term appears, and `ramllm explain <term>` (vram, ram, quantization, kv-cache, context, moe, layers, verdicts, bottleneck) in 2–4 sentences each.
- **Typo help:** custom Click group that suggests close matches ("No such command 'chta'. Did you mean 'chat'?") using `difflib`.
- Grouped help panels (Section 3) and examples at the bottom of every command's `--help`.

**2E. Friendly error system** (`cli/ui/errors.py`)
- Define `RamllmError(code, title, cause, fix, doc_anchor)`; catch at the top-level CLI entry; print:
  ```
  ✘ Not enough memory for this model (RAM-E012)
    Why:  The model needs 41 GB but you have 16 GB RAM + 6 GB graphics memory available.
    Try:  • Pick a smaller model:  ramllm recommend
          • Or a lower-quality version:  ramllm plan <model> --mode max
    More: https://…/troubleshooting#ram-e012     (offline copy: ramllm explain RAM-E012)
  ```
- Catalog at least: backend missing/outdated, download failed/disk full/checksum mismatch/gated model (needs HF token), model file corrupt, port in use, out-of-memory at load, GPU driver problem, permission errors, unsupported GGUF architecture, config invalid.
- Exit codes: 0 ok, 1 generic, 2 usage, 3 not-feasible, 4 backend problem, 5 network/download, 6 interrupted. Document them.
- `--debug` shows traceback and writes `<data_dir>/logs/ramllm.log`.

**Acceptance (M2)**
- `ramllm plan <model>` and `ramllm run <model>` work with **no budget flags**; a snapshot test asserts the plain-English summary on 3 synthetic hardware profiles.
- Every catalogued error has a unit test that renders the expected text and exit code.
- Existing tests still pass; old flags still work.

---

### Milestone 3 — Model experience (find, name, download, manage)
**Goal:** users never need to know file paths or Hugging Face syntax.

**3A. Curated model index (offline-first)**
- Ship `src/ramllm/data/models_index.json` (versioned, schema-validated). Schema per entry:
  ```json
  {
    "id": "short-friendly-id",
    "name": "Display name",
    "family": "…", "params_b": 7.6, "architecture": "dense|moe",
    "active_params_b": null,
    "tags": ["chat","coding","small","multilingual"],
    "license": "…",
    "files": [
      {"quant":"Q4_K_M","size_bytes":0,"hf_repo":"…","hf_file":"…","sha256":"…"}
    ],
    "notes": "one line in plain language",
    "recommended_min_memory_gb": 8
  }
  ```
- **The agent must populate this by actually checking real Hugging Face repos/files** (existence, filenames, sizes, SHA-256 from the repo metadata), pick ~15–25 popular, permissively usable GGUF models across sizes (≈0.5B up to a large MoE) and use cases (general chat, coding, small/fast, long context, a vision-less default), and record `last_verified` dates. **Do not invent entries.** Add a validation script `scripts/verify_index.py` (network, run by maintainers/CI on a schedule) that flags dead links and checksum drift.
- Optional explicit `ramllm models update-index` to fetch a newer index from the repo (asks consent, verifies signature/hash).

**3B. Model reference resolution** (`models/resolver.py`)
`<model>` may be, in order: an existing file path → a catalog name → an index id (`qwen2.5-7b`, optionally `@Q5_K_M`) → HF shorthand `repo:file.gguf` → fuzzy search. If it's in the index but not downloaded, **offer to download it** (size, disk-free check, ETA, "Continue? [Y/n]") and then continue the original command. Ambiguity → numbered chooser.

**3C. Commands**
- `models search [query] [--tag coding] [--fits]`: searchable table with a **"Fits?"** column computed with the real planner against the current hardware (✔ runs well / ~ slow / ✘ too big), sorted by best fit. Interactive: arrow-key selection → actions (Info / Download / Chat / Plan).
- `models info <model>`: plain description, available quants with size and per-quant verdict on this machine, license, source.
- `models pull`: add `--quant` auto-pick (best quant that is COMFORTABLE/CONSTRAINED for this machine when unspecified), progress with resume, checksum verify, disk-space pre-check (require size + 10% free), cleanup of partials on failure, gated-model handling (`HF_TOKEN`; clear instructions on 401/403), concurrent-pull lock.
- `models list`: add columns **Fits?**, **Last used**, **Disk**; `--json`.
- `models du` and `clean`: show space used; delete unused/partials with confirmation.
- `recommend`: end with "Download and chat with #1? [Y/n]" and a one-line reason per pick ("Biggest model that runs at chat speed on your machine").
- Default `models_dir` per OS (via `platformdirs`), configurable; first-run wizard asks only if the default drive has <20 GB free.

**Acceptance (M3)**
- `ramllm chat qwen…` (any index id) with nothing downloaded → prompts → downloads (tested against a local HTTP server fixture with resume) → proceeds.
- `models search --fits` ordering is deterministic and tested on 4 synthetic profiles.
- Index schema validated in CI; `verify_index.py` runs in a scheduled workflow.

---

### Milestone 4 — First-run wizard and home menu
**4A. `ramllm setup` wizard** (also auto-offered the first time any interactive command is run; `--skip` to dismiss; re-runnable)

Steps (each skippable, each with a one-line "why"):
1. **Welcome:** one sentence on what RAMLLM does + "This takes about N minutes. Nothing is sent to the internet except downloads you approve."
2. **Engine:** if no usable backend → "RAMLLM needs a small engine program (llama.cpp, ~X MB). Install it now? [Y/n]" → `backend install`.
3. **Check your computer** (profile with spinner). Plain readout: "16 GB memory · NVIDIA GPU with 6 GB · fast SSD." Mention anything unusual (very low free RAM, slow disk, no GPU).
4. **What will you use it for?** (arrow-select): General chat · Coding help · Writing & summaries · Fastest possible · Biggest/smartest that fits → sets `mode` and index tag filters.
5. **How much memory may RAMLLM use?** Conservative / Balanced (recommended) / Maximum, with a one-line explanation and the resulting GB figures.
6. **Pick a model:** top 3 recommended from the index with plain labels (Fast / Smart / Biggest that works), size, and speed estimate; option "I already have a model file" (scan common folders such as `~/models`, LM Studio/Ollama GGUF dirs if present, *only with permission*) and "Show more".
7. **Download** with progress + verification.
8. **Test run:** launch, send "Say hello in one sentence", show the answer and the measured speed. Offer to run a quick calibration (`bench`) now or later.
9. **Finish:** write `config.toml` (default model, mode, memory usage); offer shell completion install; print "Done! Start chatting: `ramllm chat`".
- Non-interactive: `ramllm setup --yes --model <id> --mode balanced` performs everything with defaults for scripts.
- Interrupt-safe: Ctrl-C at any step shows what is already done and how to resume; partial progress persists (`setup_state.json`).

**4B. Home menu** (`ramllm` with no arguments, TTY)
```
 RAMLLM  ·  Qwen2.5 7B ready  ·  Engine: llama.cpp (CUDA)  ·  Memory: 16 GB RAM / 6 GB GPU

  ▸ Chat with a model
    Find or download a model
    My models
    Preview what a model will do (plan)
    Start an API server (for other apps)
    Running now
    Live monitor
    Test my speed (benchmark)
    Check my computer
    Settings
    Help & diagnose
    Quit
```
- Arrow keys + Enter, `/` to search, `?` for help, number shortcuts; implemented with `prompt_toolkit`/`questionary`. Each entry calls the same service functions as the CLI commands (no duplicated logic).
- Header shows live facts (default model, engine, memory) and a "Next suggested step" line (e.g., "No model yet → Find or download a model").
- Non-TTY: `ramllm` prints grouped help.

**Acceptance (M4)**
- Scripted pexpect/pty test drives the wizard end-to-end with the **mock backend** and a fake download server, ending with a valid config and a successful test prompt.
- Wizard never crashes on: no GPU, offline, low disk, Ctrl-C, unsupported terminal (falls back to `--plain` prompts).
- Manual test: a person who has never used a terminal-based tool follows the README and reaches a first chat reply without help (see Section 9).

---

### Milestone 5 — Interactive chat (`ramllm chat`)
**One command does everything:** resolve model → (offer download) → auto-plan within the user's budget/mode → launch engine → open chat → on exit, stop the engine cleanly.

**Startup summary (2–3 lines, not a wall of text):**
```
Loading Qwen2.5 7B (Q4)… done in 4.2s
Running on: GPU (all 32 layers) · memory 4.8/6 GB GPU, 1.2/16 GB RAM · speed ≈ 30 words/sec
Type /help for commands · Ctrl-C stops a reply · Ctrl-D or /exit to quit
```
If the plan verdict is CONSTRAINED/BATCH_ONLY, explain and offer alternatives before launching ("This model will be slow (~1 word/sec). Use it anyway? [y/N] / pick a smaller one").

**REPL features** (use `prompt_toolkit` for input, `rich` for output; talk to the engine's OpenAI-compatible endpoint with `httpx` streaming):
- Streaming token output rendered as Markdown (Rich `Live`); code blocks syntax-highlighted; graceful degradation with `--plain`.
- Persistent history (up/down, search with Ctrl-R), multi-line input (Alt/Esc+Enter, or `"""` fences), paste support.
- **Ctrl-C** interrupts the current generation (keeps the partial answer in context, marked); second Ctrl-C at an empty prompt hints how to exit; **Ctrl-D**/`/exit` quits.
- **Bottom status toolbar:** `model · ctx 1.2k/8k · mem 5.1/6 GB GPU · last: 28 tok/s` (live from governor metrics; hide with `/toolbar off`).
- **Slash commands** (tab-completable, with `/help` listing):
  `/help` `/exit` `/clear` (new conversation) `/system <text>` (set/show system prompt) `/model [name]` (switch; relaunches with a new plan) `/plan` (show current plan summary) `/mem` (current memory per tier + governor warnings) `/stats` (speed, tokens, time) `/ctx <n|auto>` (change context via replan) `/temp <x>` `/top_p <x>` `/seed <n>` `/save [file]` `/load <file>` (transcripts as Markdown/JSON) `/retry` (regenerate last reply) `/undo` (drop last exchange) `/copy` (copy last reply to clipboard if available) `/mode <fast|balanced|quality|max>` (replans) `/budget <size>` (replan with a new memory limit) `/serve` (also expose this running engine as an API on a port) `/quit`.
- **Context management:** track usage (use the engine's token counts/`usage` or `/tokenize` if available — verify); at ~85% warn; offer "trim oldest messages / summarize and continue / increase context (if it fits)". Never silently truncate.
- **Governor integration:** if memory pressure or OOM triggers an automatic smaller plan, show a clear one-line notice ("Memory got tight, so I reduced the context to 4k and restarted. Your conversation is kept.") and **replay the conversation state** into the new engine (re-send message history) rather than losing it.
- **Transcripts:** auto-save each chat to `<data_dir>/chats/<date>-<model>.md` (opt-out in config); `ramllm chat --resume [file]`.
- **Piped/non-interactive:** `echo "…" | ramllm chat -` or `ramllm run` for single answers; if stdin isn't a TTY, do not open the REPL.
- Options: `--system "…"`, `--ctx`, `--mode`, `--memory`, `--temp`, `--no-save`, `--attach-server` (connect to an already-running `serve` instead of launching a new engine).
- Clean shutdown on exit/crash/SIGTERM/closing the terminal: no orphaned `llama-server` processes (process-group/Job Object handling on Windows; test it).

**Acceptance (M5)**
- pty-based test using the mock backend: send prompt → streamed reply → `/stats` → `/replan`-style `/ctx 4k` → conversation preserved → `/save` → exit with zero orphan processes.
- Chat works on Windows Terminal and legacy console (degraded gracefully), macOS Terminal/iTerm, common Linux terminals.
- Ctrl-C during generation never kills the app or the engine.

---

### Milestone 6 — Process management and friendlier `serve`
**Goal:** run models in the background and manage them like a product, not like scripts.

- **State registry:** `<data_dir>/run/<id>.json` per running engine: `id` (short human-readable, e.g. `qwen-7b-1`), pid(s), port, host, model, plan summary, started_at, log path, mode (chat/serve). Liveness checked via `psutil` + health endpoint; stale entries auto-cleaned.
- `serve --detach` (background; POSIX: new session; Windows: `DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP`), logs to file, waits until healthy, then prints connection help:
  ```
  ✔ Server running (id: qwen-7b-1) at http://127.0.0.1:8080/v1
    Use with OpenAI SDK:  base_url="http://127.0.0.1:8080/v1", api_key="not-needed"
    Open WebUI / LangChain / etc.: use the same base URL.
    Stop:  ramllm stop qwen-7b-1      Logs:  ramllm logs qwen-7b-1 -f      Monitor:  ramllm dashboard
  ```
- **Auto port selection** (default 8080, else next free), localhost-only by default; `--host 0.0.0.0` requires explicit confirmation and an API key (`--api-key` or generated and printed once) — warn about exposure.
- `ps` (table: id, model, port, uptime, memory per tier, state), `stop [id|--all]` (graceful then forced, tree-kill including children), `logs [id] [-f] [-n 100]`, `status` (engine + models + running + memory in one screen), `replan <id>` (calls `/ramllm/replan`).
- `serve` with no model uses `default_model` from config, or opens the model picker.
- Startup health wait with a visible spinner and timeout with actionable error (show the last log lines on failure).
- Concurrency rule: warn when starting a second engine would exceed the memory budget (use the planner with the *remaining* budget).

**Acceptance (M6):** detach → ps → logs → stop lifecycle works on all 3 OSes with the mock backend; no orphan processes; port conflicts handled; killing the terminal doesn't kill a detached server, while closing a foreground `chat` does clean up.

---

### Milestone 7 — Live dashboard (TUI)
**Goal:** a calm, readable live view for the memory governor and speed.

- Build with `textual` (include as a normal dependency to avoid "install extra" friction; verify size and Windows behaviour).
- Reads `/ramllm/metrics` and `/ramllm/plan` from a chosen running engine (auto-attach to the only/last one; selector if several); can also launch-and-watch (`ramllm dashboard <model>`).
- Panels:
  - **Memory:** bars for GPU memory (per GPU), RAM, and disk-read rate, with budget lines and color + text status (not color-only).
  - **Speed:** current and average tok/s, time-to-first-token, sparkline of recent speed.
  - **Plan:** strategy, layers on GPU/CPU, KV cache type/size, context, verdict, bottleneck in plain words.
  - **Governor events:** timeline of warnings/auto-recoveries ("14:02 memory tight → reduced context to 4k").
  - **Requests:** active/queued/recent requests for server mode.
- Keybindings (shown in footer): `r` replan (dialog: new memory limit / context / mode), `p` switch plan, `l` logs, `s` stop engine, `d` details/plain mode, `?` help, `q` quit.
- Graceful fallback if terminal too small/unsupported: `ramllm status --watch` text refresh.

**Acceptance (M7):** snapshot tests with Textual's testing pilot against recorded metric streams from the mock backend; replan dialog round-trips to the API; no crashes on terminal resize.

---

### Milestone 8 — Distribution and release automation
- **GitHub Actions release pipeline:** on tag → build sdist/wheel → publish to PyPI (trusted publishing) → build standalone executables with PyInstaller (or Nuitka) for Windows/macOS (x64+arm64)/Linux → generate SHA-256 checksums → create GitHub Release with notes. Smoke-test each artifact (`ramllm --version`, `ramllm doctor --plain`) in a clean runner.
- **Homebrew tap** (`brew install <tap>/ramllm`), **winget/scoop manifests** (submit after first stable release), optional `.deb/.rpm` later.
- `ramllm update`: checks for newer versions **only when run explicitly**; chooses the right upgrade path based on how it was installed (uv/pipx/pip/brew/standalone) and prints the exact command or runs it with consent.
- `ramllm completion install` (bash/zsh/fish/PowerShell) + offered in wizard.
- Uninstall docs: `ramllm clean --all`, plus how to remove the tool, backends, models (with sizes), config.
- macOS Gatekeeper/Windows SmartScreen notes for standalone binaries; document signing as future work.

**Acceptance (M8):** tagged pre-release produces installable artifacts on all OSes; installer scripts succeed on clean runners; `ramllm update` correct for each install method (mocked).

---

### Milestone 9 — Documentation, demos, polish, and real-user testing
- **README rewrite:** top = what it is in 2 sentences + the three-command promise + a short terminal recording (asciinema/GIF) of setup → chat; then "What fits my computer?" explainer, FAQ, link to full docs. Keep the existing technical sections (cost model, API) lower down.
- `docs/getting-started.md` (per OS with screenshots/recordings), `docs/cli.md` (**auto-generated from Typer** in CI so it can't drift), `docs/troubleshooting.md` (indexed by error code), `docs/concepts.md` (tiers, quantization, context, MoE, verdicts in plain language), `docs/cheatsheet.md` (one page), `docs/recipes.md` (use with Open WebUI, OpenAI SDK, LangChain; running in background; low-RAM tips).
- **Usability study (required):** 5 people with different backgrounds (non-developer, student, hobbyist) install from scratch on their own machines with only the README. Record time to first chat reply, every confusion point, and every error. Fix the top issues. Repeat with 3 new people.
- Accessibility pass: `--plain` works everywhere; no information conveyed by color alone; screen-reader-friendly linear output mode.
- Performance pass: CLI cold start for `--help` < 300 ms (lazy-import heavy modules: textual, fastapi, psutil probes, planner).
- Security pass: path traversal in archive extraction, checksum enforcement, no shell injection in subprocess calls (always argument lists), safe temp files, localhost-only defaults, no tokens in logs.

---

## 5. Architecture changes (adapt to the real layout after M0)

```
src/ramllm/
├── cli/
│   ├── app.py                 # Typer root, grouped help, global flags, typo-suggest Group, TTY detection
│   ├── commands/              # one module per command (thin: parse flags → call services)
│   ├── ui/                    # console.py theme.py tables.py progress.py prompts.py errors.py glossary.py
│   ├── setup/                 # wizard.py, state.py (resumable)
│   ├── home/                  # menu.py
│   ├── chat/                  # repl.py slash.py render.py history.py transcripts.py client.py
│   └── tui/                   # dashboard.py (textual)
├── services/                  # NEW: business logic shared by CLI, menu, TUI, API (no I/O formatting)
│   ├── backend_manager.py     # discovery, install, update, asset selection
│   ├── model_service.py       # resolve, search, pull, info, disk usage
│   ├── plan_service.py        # auto-budget + mode → planner call → human summary data
│   ├── session_service.py     # start/stop engines, registry, detach, replay on replan
│   └── config_service.py
├── data/models_index.json
└── (existing) core/ backends/ api/ registry/ telemetry/
```
Rule: **commands, menu, chat slash commands, and TUI all call `services/`.** No business logic in presentation code, so behaviour is identical everywhere and testable without a terminal.

**New dependencies** (verify current versions, pin sensible lower bounds): `prompt_toolkit`, `questionary` (or prompt_toolkit-only), `textual`, `httpx`, `platformdirs`, `tomli-w`, `psutil` (already), `rich`/`typer` (already). Keep import-time cheap via lazy imports.

---

## 6. Interaction details and copy guidelines

- **Tone:** friendly, short, concrete, no blame. Prefer "Try:" over "You must".
- **Numbers:** show GB with one decimal; speed as "about 25–35 words/sec (≈ 33–47 tok/s)".
- **Confirmations:** `[Y/n]` capital = default. Destructive actions default to No.
- **Never ask what you can detect.** Ask only about real choices.
- **Always end with a next step.**
- **Time expectations:** any step >2 s shows a spinner with a label; any download shows size, speed, ETA.
- **Windows legacy console:** detect unsupported ANSI/Unicode and switch to ASCII theme automatically.
- **Internationalization:** keep user-facing strings in one module per area (`strings.py`) so translation is possible later; don't translate in v1.

---

## 7. Edge cases the agent must handle

- No internet / flaky internet / captive portals / proxy (`HTTPS_PROXY`) / corporate TLS interception (clear message + `--insecure` is **not** offered; document certificate env vars).
- Low disk space during download; download interrupted; corrupted partial files; checksum mismatch; model path with spaces or non-ASCII characters (Windows).
- Two RAMLLM processes starting simultaneously (file locks for catalog, downloads, registry).
- Very low-RAM machines (4–8 GB): wizard recommends the small models and says plainly what to expect.
- No GPU / old GPU / driver missing / GPU in use by another program (free VRAM shrinks) / laptop on battery (note thermal/perf) / integrated GPU / Apple unified memory.
- Antivirus quarantining the downloaded engine on Windows (detect missing exe after extraction; explain).
- Model architecture unsupported by the installed engine version → suggest `backend update`.
- Existing llama.cpp on PATH is too old/new → warn with capability differences; offer managed install.
- Terminal resize, piping, redirecting, CI environments (`CI=true` → non-interactive defaults), `TERM=dumb`.
- Running from inside a VS Code/Jupyter terminal or SSH (no clipboard, limited colors).

---

## 8. Telemetry, privacy, and safety
- Default: **no telemetry.** Local-only logs and benchmark history. If an opt-in export is ever added, it must be a file the user can inspect first.
- Network calls only for: backend install/update, `models pull/update-index`, `update`, optional HF search — each preceded by consent unless the command itself is the explicit action.
- Redact tokens and home paths in logs shared via `ramllm doctor --report` (a command that writes a redacted diagnostics file the user can attach to a bug report).
- Chat transcripts are stored locally only; document where, and provide `ramllm clean --chats`.

---

## 9. Testing strategy (UX-specific, in addition to existing tests)

1. **Unit:** services (`backend_manager` asset selection, `resolver`, budget calculation, mode mapping, error rendering, config precedence).
2. **CLI contract tests:** Typer `CliRunner` for every command; `--help` snapshot tests; `--json` schema tests; exit codes.
3. **Interactive tests:** `pexpect` (POSIX) / `wexpect` or `winpty` (Windows) scripted sessions for wizard, menu, and chat using the **mock backend** and a local fake HTTP server for downloads (with resume and checksum cases).
4. **Snapshot tests of rendered output** (Rich `Console(record=True)`) for plan summaries, error blocks, tables — in both rich and `--plain` modes.
5. **Process hygiene tests:** after every lifecycle test assert no stray `llama-server`/mock processes remain.
6. **Fresh-machine E2E (slow, nightly):** clean container/VM per OS → run installer → `ramllm setup --yes --model <tiny model>` → `ramllm run -p "hi"` → `ramllm serve --detach` → `curl` → `ramllm stop`. Use a very small GGUF for CI and record timings.
7. **Installer tests:** `shellcheck`, PSScriptAnalyzer, dry-run mode, idempotency (run twice).
8. **Cold-start benchmark** in CI (`--help` latency budget).
9. **Manual usability study** (Milestone 9) with a written protocol and results in `docs/usability/`.

---

## 10. Definition of done and success metrics

- A new user on a clean Windows, macOS, or Linux machine can go from "nothing installed" to **a first chat reply in ≤ 10 minutes of active time** (excluding model download time), using only the README and without Python/llama.cpp knowledge.
- ≥ 4 of 5 first-time testers succeed unaided; every failure is turned into a fixed bug or a clearer message.
- Every task in Section 2's table is possible from the terminal, via both a command and the menu.
- Zero raw tracebacks in normal failure scenarios; each error has a code, cause, and fix.
- No orphaned engine processes in any lifecycle test.
- `ramllm --help` renders in < 300 ms warm; docs auto-generated and in sync.
- All existing commands/flags continue to work; existing test suite stays green.

---

## 11. Suggested order of work (summary)

1. **M0** Audit + README fixes
2. **M1** One-line install + engine manager + `doctor --fix`
3. **M2** Config, auto-budgets, modes, friendly output/errors
4. **M3** Model index, name resolution, search/pull/manage
5. **M4** `setup` wizard + home menu
6. **M5** `chat` REPL ← the headline feature
7. **M6** `serve --detach`, `ps`, `stop`, `logs`, `status`
8. **M7** `dashboard` TUI
9. **M8** Release automation, Homebrew/winget/standalone binaries, `update`
10. **M9** Docs, demos, usability testing, polish
