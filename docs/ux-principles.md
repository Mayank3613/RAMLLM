# RAMLLM UX Principles

These design principles guide every user-facing surface in RAMLLM: the CLI, the interactive chat REPL, the home menu, and the live dashboard.

---

## Core Principles

### 1. One Obvious Next Step
After every command, show the user what to do next. Never leave them with a dead prompt wondering what the command did or what to type now.
*Example:* After `setup`, print: `Next: ramllm chat`. After `plan`, show: `Next: ramllm run <model>` or `ramllm chat <model>`.

### 2. Defaults Over Questions; Questions Over Flags
Auto-detect everything possible (OS, CPU features, RAM, GPUs, VRAM, disk bandwidth).
If a sensible default exists (balanced mode, automatic memory budgets, recommended quant), use it.
Only prompt the user when a genuine human choice exists (e.g. "What will you use this model for?", "Do you want to download 4.2 GB?").

### 3. Show, Don't Dump
Always present a plain-English summary first (2–4 lines of human-readable conclusions).
Place extensive tables, raw metrics, and technical breakdowns behind `--details`, `--explain`, or separate inspect commands.

### 4. Honest and Calm
Uphold RAMLLM's core promise: **never lie about performance or fit**.
Pair every feasibility verdict (`COMFORTABLE`, `CONSTRAINED`, `BATCH_ONLY`, `INFEASIBLE`) and named bottleneck with kindly phrased, plain-language guidance:
- `COMFORTABLE` → "Runs well (fast — about 25–35 words per second)"
- `CONSTRAINED` → "Works, but slowly (about 2–4 words per second)"
- `BATCH_ONLY` → "Very slow (~0.5 words per second) — fine for overnight jobs, not interactive chat"
- `INFEASIBLE` → "Won't fit in your available memory" + concrete remedies (smaller model, lower quantization)

### 5. Safe by Default
- Pre-check disk space before downloads (require model size + 10% safety margin).
- Never thrash or freeze the user's computer: automatic memory budgets always reserve headroom for the OS and applications.
- Always require confirmation before downloading gigabytes of data or deleting files, unless `--yes` is passed.
- No network calls or telemetry without explicit user action or consent.

### 6. Discoverable
- Group commands logically in `--help` using Rich panels.
- Provide fuzzy typo suggestions when a user mistypes a command (e.g., `chta` → `chat`).
- Provide tab-completion for models, options, and commands.
- Provide a guided interactive home menu (`ramllm`) for users who don't remember command names.
