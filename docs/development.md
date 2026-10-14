# Developer Guide & Local Setup

This document describes how to set up RAMLLM for local development, testing, and contribution.

---

## 1. Prerequisites

- Python 3.11, 3.12, or 3.13
- Git
- `uv` (recommended) or `pip` / `venv`

---

## 2. Setting Up Development Environment

### Using `uv` (Fastest)

```bash
# Clone the repository
git clone https://github.com/Mayank3613/RAMLLM.git
cd RAMLLM

# Install all dependencies including dev tools
uv sync --all-extras
```

### Using standard `venv` and `pip`

```bash
# Clone the repository
git clone https://github.com/Mayank3613/RAMLLM.git
cd RAMLLM

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## 3. Running Tests and Linters

```bash
# Run unit & integration tests
uv run pytest

# Run linting checks
uv run ruff check .

# Run auto-formatting check
uv run ruff format --check .

# Run static type checking
uv run mypy src tests
```

---

## 4. Synthetic Fixtures for CI

RAMLLM includes synthetic hardware topologies in `tests/fixtures/hardware/` and a synthetic GGUF generator in `tests/fixtures/make_gguf.py`.
This enables headless testing of the planner, governor, and catalog without needing large model weights or physical GPUs.
