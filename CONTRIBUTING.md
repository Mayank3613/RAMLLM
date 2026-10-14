# Contributing to RAMLLM

Thank you for your interest in contributing to RAMLLM!

## Development Setup

RAMLLM uses Python 3.11+ and `uv` for lightning-fast virtual environment management.

1. Clone repository:
   ```bash
   git clone https://github.com/your-org/ramllm.git
   cd ramllm
   ```

2. Create virtual environment and install dev dependencies:
   ```bash
   uv venv --python 3.12 .venv
   uv pip install -e ".[dev]"
   ```

3. Run linting and type checking:
   ```bash
   uv run ruff check .
   uv run mypy src/ramllm/core
   ```

4. Run tests:
   ```bash
   uv run pytest
   ```

## Code Quality Standards
- Strict type hints on all core modules (`src/ramllm/core/`).
- Document all public functions with docstrings.
- Small, focused modules without global mutable state.
- Offline-first: no silent network calls or telemetry.
- When making architectural or ambiguous decisions, create an ADR in `docs/decisions/`.
