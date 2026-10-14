# RAMLLM CLI Reference

> Auto-generated from Typer CLI specifications. Do not edit manually.
> Run `uv run python scripts/generate_cli_docs.py` to regenerate.

## Overview

```bash
ramllm [OPTIONS] COMMAND [ARGS]...
```

### Global Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--version`, `-v` | Flag | `False` | Show RAMLLM and backend version |
| `--plain` | Flag | `False` | Disable rich formatting, emojis, and colors |
| `--debug` | Flag | `False` | Enable verbose debug logging and stack traces |
| `--help` | Flag | | Show global help and exit |

---

## GET STARTED

### `ramllm doctor`

Diagnose backend engines, hardware topology, catalogs, and OS memory enforcement levels.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--fix` | boolean | `False` | Automatically attempt safe repairs for identified issues |
| `--yes`, `-y` | boolean | `False` | Assume yes to confirmation prompts |

---

### `ramllm setup`

Run the interactive onboarding wizard to configure RAMLLM and set up models.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--yes`, `-y` | boolean | `False` | Run non-interactively with defaults |
| `--skip` | boolean | `False` | Mark setup as completed without changes |
| `--model` | str | none | Pre-select starting model ID |
| `--mode` | str | none | Pre-select inference mode (fast|balanced|quality|max) |

---

### `ramllm status`

Print a comprehensive one-screen status dashboard for RAMLLM on this system.

---

### `ramllm tour`

Take a 30-second tour of memory tiers, verdicts, and key RAMLLM commands.

---

## CHAT & RUN

### `ramllm chat`

Open an interactive chat REPL with automatic budgeting and streaming output.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `model` | str | none | Model name, local file path, or Hugging Face ID (e.g. 'qwen-2.5-7b', 'llama-3.2-3b') |
| `-p`, `--prompt` | str | none | Single prompt for non-interactive execution |
| `--system` | str | none | Custom system instruction prompt |
| `--ctx` | str | none | Context window limit (e.g. 4096, 8192) |
| `--mode` | str | none | Execution mode ('fast', 'balanced', 'quality', 'max') |
| `--memory` | str | none | Memory budget preset ('conservative', 'balanced', 'max') or bytes ('8GB') |
| `--temp` | float | `0.7` | Sampling temperature |
| `--top-p` | float | `0.9` | Nucleus sampling top-p |
| `--seed` | int | none | Random seed for deterministic generations |
| `--no-save` | boolean | `False` | Disable automatic conversation transcript archiving |
| `--resume` | path | none | Resume conversation from an existing transcript file |
| `--attach-server` | str | none | Connect to an already-running server (e.g. http://127.0.0.1:8080) |
| `--port` | int | none | Port for local llama-server engine |
| `--plain` | boolean | `False` | Plain linear output without rich Markdown formatting |

---

### `ramllm run`

Run model with budget enforcement, buffer validation, and execution.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `model` | path | *required* | Path to GGUF model |
| `--vram` | str | none | VRAM budget (e.g. 6GB) |
| `--ram` | str | none | RAM budget (e.g. 12GB) |
| `--memory` | str | none | Total memory budget across RAM and VRAM (e.g. 12GB) |
| `--gpus` | str | none | Comma-separated GPU indices to use (e.g. 0,1) |
| `--context` | str | `4096` | Target context window |
| `--mode` | choice | none | Performance and quality tradeoff mode |
| `--prompt`, `-p` | str | none | Single prompt generation |
| `--port` | int | `8080` | Port for llama-server |
| `--live` | boolean | `False` | Monitor memory live with governor |
| `--auto-retry` | boolean | `True` | Automatically adapt to smaller plans on failure or OOM |
| `--max-retries` | int | `3` | Maximum auto-retry recovery attempts |
| `--hw-json` | path | none | Path to synthetic hardware JSON profile |
| `--dry-run` | boolean | `False` | Verify execution plan and parameters without starting engine |

---

### `ramllm serve`

Start OpenAI-compatible API server with active memory governor.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `model` | str | none | Path or name of GGUF model (uses config default if omitted) |
| `--vram` | str | none | VRAM budget (e.g. 6GB) |
| `--ram` | str | none | RAM budget (e.g. 12GB) |
| `--gpus` | str | none | Comma-separated GPU indices to use (e.g. 0,1) |
| `--context` | str | `4096` | Target context window length |
| `--port` | int | `8080` | Port to listen (auto-selected if taken) |
| `--host` | str | `127.0.0.1` | Host interface to bind |
| `-d`, `--detach` | boolean | `False` | Run server in the background as a detached service |
| `--api-key` | str | none | API key required for completions (recommended on 0.0.0.0) |
| `--auto-retry` | boolean | `True` | Automatically adapt to smaller plans on failure or OOM |
| `--max-retries` | int | `3` | Maximum auto-retry recovery attempts |
| `--engine-id` | str | none | Internal explicit engine ID for daemon |

---

## MODELS

### `ramllm bench`

Run empirical benchmark on model and update calibration factor.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `model` | path | *required* | Path to GGUF model |
| `--plan-id` | str | none | Benchmark specific plan |
| `--port` | int | `8085` | Port for llama-server |

---

### `ramllm plan`

Plan model execution strategy given hardware constraints and memory budgets.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `model` | path | *required* | Path to GGUF model |
| `--vram` | str | none | VRAM budget (e.g. 6GB or 8GB,6GB) |
| `--ram` | str | none | RAM budget (e.g. 12GB) |
| `--memory` | str | none | Total memory budget across RAM and VRAM (e.g. 12GB) |
| `--gpus` | str | none | Comma-separated GPU indices to use (e.g. 0,1) |
| `--context` | str | `4096` | Target context window (e.g. 4096, 8k) |
| `--mode` | choice | none | Performance and quality tradeoff mode |
| `--prefer` | choice | none |  |
| `--cpu-only` | boolean | `False` | Disable GPU offloading |
| `--no-kv-quant` | boolean | `False` | Disallow quantized KV cache |
| `--details` | boolean | `False` | Display full candidate comparison table |
| `--explain` | boolean | `False` | Print detailed reasoning |
| `--json` | boolean | `False` | Output JSON |
| `--hw-json` | path | none | Synthetic hardware profile |

---

### `ramllm recommend`

Recommend the best-quality models that fit within hardware budgets (offline catalog).

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--vram` | str | none | VRAM budget (e.g. 6GB or 8GB,6GB) |
| `--ram` | str | none | RAM budget (e.g. 12GB) |
| `--memory` | str | none | Total memory budget across RAM and VRAM (e.g. 12GB) |
| `--gpus` | str | none | Comma-separated GPU indices to use (e.g. 0,1) |
| `--context` | str | `4096` | Target context window (e.g. 4096, 8k) |
| `--mode` | choice | none | Performance and quality tradeoff mode |
| `--prefer` | choice | none |  |
| `--cpu-only` | boolean | `False` | Disable GPU offloading |
| `--no-kv-quant` | boolean | `False` | Disallow quantized KV cache |
| `--json` | boolean | `False` | Output JSON |
| `--hw-json` | path | none | Synthetic hardware profile |

---

## SYSTEM

### `ramllm clean`

Free disk space by removing cached models, managed engines, or logs.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--all`, `-a` | boolean | `False` | Clean all cached models, managed engines, chats, and logs |
| `--models` | boolean | `False` | Delete downloaded model files |
| `--backends` | boolean | `False` | Delete managed engine binaries |
| `--chats` | boolean | `False` | Delete conversation transcript files |
| `--logs` | boolean | `False` | Delete log files |
| `--partials` | boolean | `False` | Delete only incomplete/partial downloads (.part, .tmp) |
| `--dry-run` | boolean | `False` | Show files and disk space to reclaim without deleting |
| `-y`, `--yes` | boolean | `False` | Skip confirmation prompt |

---

### `ramllm dashboard`

Open a live visual TUI dashboard monitoring memory headroom, governor, and speed.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `target` | str | none | ID, port, or base URL of running engine to monitor (auto-detected if omitted) |
| `--watch`, `-w` | boolean | `False` | Run in linear terminal text watch mode without full TUI |

---

### `ramllm explain`

Explain technical inference terms or error codes in plain English.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `term` | str | none | Term or error code to explain (e.g. vram, quantization, RAM-E012) |

---

### `ramllm logs`

View or stream stdout/stderr log output of a running or recent engine.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `target` | str | none | ID of the running engine to view logs for |
| `-n`, `--lines` | int | `50` | Number of recent log lines to display |
| `-f`, `--follow` | boolean | `False` | Stream log updates continuously |

---

### `ramllm profile`

Profile hardware topology, memory headroom, and compute/memory bandwidth.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--refresh` | boolean | `False` | Re-run bandwidth benchmarks |
| `--json` | boolean | `False` | Output raw JSON |
| `--hw-json` | path | none | Load synthetic hardware profile |

---

### `ramllm ps`

List all running RAMLLM engine instances and resource usage.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--json` | boolean | `False` | Output raw JSON array |

---

### `ramllm replan`

Dynamically replan an active background server without losing state.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `target` | str | *required* | ID or port of running engine to replan |
| `--ctx` | int | none | New context window length |
| `--ram` | str | none | New RAM budget (e.g. 8GB) |
| `--vram` | str | none | New VRAM budget (e.g. 6GB) |

---

### `ramllm stop`

Stop running RAMLLM engine instances cleanly.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `target` | str | none | ID or PID of the running engine to stop |
| `--all`, `-a` | boolean | `False` | Stop all currently running engines |
| `--force`, `-f` | boolean | `False` | Force immediate kill without graceful wait |

---

### `ramllm update`

Check for newer RAMLLM releases and update using your installation method.

| Option / Argument | Type | Default | Description |
|---|---|---|---|
| `--check` | boolean | `False` | Check for available updates without applying them |
| `-y`, `--yes` | boolean | `False` | Automatically proceed with upgrade if available |

---

### `ramllm version`

Show RAMLLM and backend version details.

---

## COMMANDS

### `ramllm backend`

Manage, install, update, and switch llama.cpp engine builds.

#### Subcommands

- **`ramllm backend install`**: Download and set up prebuilt llama.cpp engine optimized for your hardware.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `--tag` | str | none | Specific release tag to install |
  | `--yes`, `-y` | boolean | `False` | Automatically accept download prompt |

- **`ramllm backend list`**: List all managed and active llama.cpp engine builds.
- **`ramllm backend remove`**: Remove a managed engine installation from storage.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `tag` | str | *required* | Tag of managed installation to remove |

- **`ramllm backend update`**: Fetch and install the newest available llama.cpp release.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `--yes`, `-y` | boolean | `False` | Assume yes to download prompts |

- **`ramllm backend use`**: Switch active engine build to a managed tag or custom file path.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `target` | str | *required* | Tag name or absolute path to llama-server binary |


---

### `ramllm completion`

Install and manage shell auto-completion for tab completion.

#### Subcommands

- **`ramllm completion install`**: Install tab auto-completion for your shell.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `shell` | str | none | Target shell ('bash', 'zsh', 'fish', 'powershell') |

- **`ramllm completion show`**: Print the raw shell completion script to stdout.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `shell` | str | none | Shell type ('bash', 'zsh', 'fish') |


---

### `ramllm config`

Manage persistent RAMLLM configuration (settings).

#### Subcommands

- **`ramllm config edit`**: Open config.toml in system text editor.
- **`ramllm config get`**: Print the value of a configuration setting.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `key` | str | *required* | Configuration setting key to inspect |

- **`ramllm config list`**: Display all current configuration settings.
- **`ramllm config path`**: Print the absolute filesystem path of config.toml.
- **`ramllm config reset`**: Reset all settings or a specific setting to factory default.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `key` | str | none | Optional specific key to reset to default |

- **`ramllm config set`**: Update a configuration setting value.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `key` | str | *required* | Configuration setting key |
  | `value` | str | *required* | New value for setting |


---

### `ramllm models`

Search, download, inspect, and manage local models.

#### Subcommands

- **`ramllm models add`**: Register a local GGUF model into the catalog.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `model_path` | path | *required* | Path to .gguf model file |

- **`ramllm models clean`**: Clean up incomplete downloads and temporary files.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `--yes`, `-y` | boolean | `False` | Do not ask for confirmation |

- **`ramllm models du`**: Show disk space usage of stored models and available drive capacity.
- **`ramllm models info`**: Inspect model details, available quantizations, and compatibility.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `model` | str | *required* | Model friendly ID, catalog name, or path |

- **`ramllm models list`**: List all registered models in the local catalog.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `--json` | boolean | `False` | Output models as JSON |

- **`ramllm models pull`**: Download a model with automatic quant selection, resume support, and checksum verification.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `target` | str | *required* | Model friendly ID (e.g. qwen2.5-7b) or Hugging Face repo:file.gguf |
  | `--quant`, `-q` | str | none | Specific quantization (e.g. Q4_K_M, Q5_K_M) |
  | `--dest` | path | none | Destination directory |
  | `--sha256` | str | none | Expected SHA256 checksum |

- **`ramllm models remove`**: Remove a model entry from the local catalog.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `name` | str | *required* | Model name, path, or stem to remove |

- **`ramllm models scan`**: Scan directory recursively for GGUF models.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `directory` | path | *required* | Directory containing models |

- **`ramllm models search`**: Search the curated model catalog with hardware fit predictions.

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `query` | str | none | Search query for name, family, or description |
  | `--tag`, `-t` | str | none | Filter by tag (e.g. chat, coding, small) |
  | `--fits` | boolean | `False` | Only show models that fit your computer's memory |

- **`ramllm models verify`**: Compute and verify SHA-256 integrity hash of model file(s).

  | Option / Argument | Type | Default | Description |
  |---|---|---|---|
  | `model` | str | none | Model name or file path to verify (or all if omitted) |


---
