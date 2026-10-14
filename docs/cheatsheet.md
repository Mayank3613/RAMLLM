# RAMLLM CLI Cheatsheet

A quick one-page reference for common RAMLLM commands, options, and shortcuts.

---

## Getting Started

```bash
ramllm                     # Interactive home menu (TTY) or setup wizard
ramllm setup               # Guided hardware detection and initial model download
ramllm doctor              # Run comprehensive health diagnostics
ramllm doctor --fix        # Automatically resolve missing backends or permissions
ramllm tour                # 30-second tour of memory tiers and plan verdicts
ramllm explain <term>      # Plain-English glossary (vram, ram, quant, kv, moe...)
```

---

## Interactive Chat REPL (`ramllm chat`)

```bash
ramllm chat                # Launch chat with your default configured model
ramllm chat qwen2.5-7b     # Chat with a specific model
ramllm chat --mode fast    # Prioritize rapid response speed
ramllm chat --context 8k   # Request an 8,192 token context window
ramllm chat --plain        # Simple text mode (no colors, markdown, or status bar)
```

### In-Chat Slash Commands
| Command | Action |
|---|---|
| `/help` | List all available slash commands |
| `/mode [fast\|balanced\|quality\|max]` | Switch performance and memory tradeoff |
| `/replan [limit]` | Adjust memory limits dynamically |
| `/stats` | View tokens/sec, TTFT, and process memory |
| `/context` | Show token usage and context breakdown |
| `/compact` | Summarize and prune earlier history |
| `/save [name]` | Save transcript to `<data_dir>/chats/` |
| `/export [path]` | Export transcript to Markdown or JSON |
| `/copy` | Copy last assistant response to clipboard |
| `/dashboard` | Link to the live terminal dashboard |
| `/clear` | Clear screen and reset current conversation |
| `/exit`, `/quit` | Gracefully exit chat and stop engine |

---

## API Server & Background Processes

```bash
ramllm serve <model>              # Start OpenAI-compatible API server on port 8080
ramllm serve <model> --detach     # Run API server in background (daemon)
ramllm serve <model> --port 9000  # Run on custom port
ramllm ps                         # List all running RAMLLM engines
ramllm logs <id> -f               # Follow streaming logs of background engine
ramllm replan <id> --memory 8GB   # Update memory budget of running engine
ramllm stop <id>                  # Stop background engine gracefully
ramllm stop --all                 # Terminate all running engines
ramllm dashboard                  # Full-screen live TUI monitoring all engines
```

---

## Models Management

```bash
ramllm recommend                  # See ranked models that fit your computer best
ramllm models search [query]      # Search curated index of verified models
ramllm models info <model>        # View parameter count, quants, and license
ramllm models pull <model>        # Download model with progress bar & SHA-256 check
ramllm models list                # List locally downloaded and registered models
ramllm models du                  # Display disk usage broken down per model
ramllm models remove <model>      # Delete model file from disk
ramllm models scan <dir>          # Scan and register local GGUF files in directory
```

---

## Memory Budgets & Modes

```bash
# Performance Tradeoff Modes
ramllm plan <model> --mode fast       # Quickest replies, quantized KV cache
ramllm plan <model> --mode balanced   # Default: balanced quality and speed
ramllm plan <model> --mode quality    # Highest quality quantizations, uncompressed KV
ramllm plan <model> --mode max        # Squeeze maximum possible model into RAM+VRAM

# Explicit Budget Overrides
ramllm plan <model> --memory 12GB     # Limit total RAM + VRAM usage to 12 GB
ramllm plan <model> --vram 6GB        # Limit GPU VRAM allocation to 6 GB
ramllm plan <model> --ram 16GB        # Limit system RAM allocation to 16 GB
ramllm plan <model> --cpu-only        # Force pure CPU execution
```

---

## Maintenance & Configuration

```bash
ramllm config list                # View all configured settings
ramllm config set <key> <val>     # Update setting (e.g. `mode`, `default_model`)
ramllm config edit                # Open config file in default $EDITOR
ramllm config reset               # Reset configuration to defaults
ramllm clean                      # Free space by removing partial downloads and logs
ramllm clean --all                # Clean cached models, backends, and logs
ramllm update --check             # Check PyPI for newer versions of RAMLLM
ramllm update                     # Upgrade RAMLLM using your install method
ramllm completion install         # Install shell tab-completion for your shell
```
