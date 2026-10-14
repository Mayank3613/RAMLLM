# ADR 0001: Adapting to Modern llama.cpp Flag Syntax

## Context
The RAMLLM project specification refers to standard `llama.cpp` flags including `--no-mmap` and `--mlock`. When introspecting `llama-server` and `llama-cli` on installed build 11429 (Darwin arm64), these legacy flags have been unified under:
`-lm, --load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`

## Decision
RAMLLM's `llama.cpp` backend adapter will:
1. Probe `--help` during discovery to detect whether `--load-mode` or `--no-mmap` is available.
2. If `--load-mode` is present, map `mmap_mode=False` to `--load-mode none` and `mlock=True` to `--load-mode mlock`.
3. If `--no-mmap` is present on older builds, fallback to the legacy flag `--no-mmap`.

## Consequences
- Ensures out-of-the-box compatibility with current and future `llama.cpp` builds without breaking compatibility with legacy versions.
