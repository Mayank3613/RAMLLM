# Getting Started with RAMLLM

Welcome to **RAMLLM** — a terminal-first runtime and planner that lets you run bigger, more capable Large Language Models on smaller computers without freezing your operating system or triggering out-of-memory crashes.

---

## 1. Quick Installation

Install RAMLLM in seconds with a single terminal command:

### macOS & Linux
```bash
curl -fsSL https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.sh | bash
```

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.ps1 | iex
```

### Or with Package Managers
```bash
# Using uv (fastest & recommended)
uv tool install ramllm

# Using Homebrew (macOS / Linux)
brew install Mayank3613/tap/ramllm

# Using pipx
pipx install ramllm
```

---

## 2. Guided Setup Wizard (`ramllm setup`)

If this is your first time using RAMLLM, run the guided onboarding wizard:

```bash
ramllm setup
```

The wizard will:
1. **Analyze your hardware**: Detect your CPU cores, unified memory, GPU VRAM, and RAM bandwidth.
2. **Install the engine backend**: Check for `llama-server` or automatically install a pre-compiled native build for your platform (Metal on macOS, CUDA on NVIDIA Linux/Windows, Vulkan, or optimized CPU AVX2).
3. **Select your default mode**: Choose between `fast`, `balanced` (default), `quality`, or `max`.
4. **Recommend & download a starter model**: Select from curated models verified to fit comfortably in your detected memory headroom.
5. **Run a smoke test**: Perform an instant token generation test to ensure flawless execution.
6. **Configure shell auto-completion**: Add tab-completion for your active shell (Zsh, Bash, Fish, or PowerShell).

---

## 3. The Three-Command Workflow

RAMLLM makes local AI simple and reliable:

```bash
# 1. Setup your machine & download a starter model
ramllm setup

# 2. Start an interactive, memory-safe chat
ramllm chat

# 3. Open a live full-screen dashboard in another terminal
ramllm dashboard
```

---

## 4. Interactive Chat REPL (`ramllm chat`)

Launch an interactive conversation with streamed responses and real-time status bars:

```bash
# Chat with your default configured model
ramllm chat

# Or specify a model and performance mode
ramllm chat qwen2.5-7b --mode fast
```

### Live Status Toolbar
At the bottom of your chat window, RAMLLM displays:
- **Model**: Active model name and quantization format.
- **Context**: Used tokens vs total allocated window (e.g., `1,280 / 4,096 ctx (31%)`).
- **Memory**: Active process RAM + VRAM usage.
- **Speed**: Last prompt evaluation and token generation speed.

### In-Chat Slash Commands
Type `/` in the chat prompt for tab-completable shortcuts:
- `/help` — List all available slash commands.
- `/mode [fast|balanced|quality|max]` — Switch performance tradeoffs on the fly.
- `/replan [limit]` — Change memory allocations dynamically without dropping context.
- `/stats` — Show generation speed (tok/s, words/sec), TTFT, and memory breakdown.
- `/compact` — Prune or summarize earlier conversation history when memory is tight.
- `/save [name]` / `/export [path]` — Save conversation transcripts as Markdown or JSON.
- `/copy` — Copy the last model reply directly to your system clipboard.
- `/dashboard` — Launch or link the live terminal dashboard.
- `/exit` or `/quit` — Gracefully stop the backend engine and return to your shell.

---

## 5. Starting an OpenAI-Compatible API Server (`ramllm serve`)

Integrate RAMLLM with external apps (Open WebUI, cursor, VS Code Continue, LangChain, or Python scripts):

```bash
# Start foreground server
ramllm serve llama-3.2-3b --port 8080

# Or run detached in the background
ramllm serve qwen2.5-7b --detach --port 8080
```

When started, RAMLLM prints a connection help panel:
- API Base URL: `http://127.0.0.1:8080/v1`
- Chat Completions: `POST http://127.0.0.1:8080/v1/chat/completions`
- Replan Endpoint: `POST http://127.0.0.1:8080/ramllm/replan`
- Live Metrics: `GET http://127.0.0.1:8080/ramllm/metrics`

---

## 6. Managing Background Engines

Track and control background models without digging for PIDs:

```bash
# List all active RAMLLM engines
ramllm ps

# View live streaming logs
ramllm logs <id> -f

# Replan memory limits of a running engine
ramllm replan <id> --memory 8GB

# Stop a specific engine or all engines
ramllm stop <id>
ramllm stop --all
```

---

## 7. Next Steps

- Explore the [CLI Reference](cli.md) for full parameter and flag details.
- Learn about memory tiers and verdicts in [Concepts](concepts.md).
- Integrate with your tools using [Recipes & Integrations](recipes.md).
- Check the [Cheatsheet](cheatsheet.md) for quick terminal lookups.
- Need help? Browse the [Troubleshooting Guide](troubleshooting.md) indexed by error codes.
