# ADR 0002: Modular CLI and Service Layer Architecture

## Context
The initial implementation placed all CLI logic into `src/ramllm/cli/main.py`. As we expand RAMLLM with an interactive first-run setup wizard, interactive chat REPL, home menu, and Textual live dashboard, embedding business logic directly in CLI command handlers would lead to severe code duplication, untestable presentation code, and fragile process handling.

Furthermore, commands must support multiple interfaces (rich TTY interactive, non-TTY piped, `--plain`, and `--json`) and maintain 100% backward compatibility with all existing command flags and test suites.

## Decision
1. **Introduce a Decoupled `services/` Package:**
   - Business logic will reside in `src/ramllm/services/` (`backend_manager.py`, `config_service.py`, `model_service.py`, `plan_service.py`, `session_service.py`).
   - Services return structured data (Pydantic models and domain objects) and perform no terminal I/O or ANSI styling.
2. **Modularize the CLI into `src/ramllm/cli/`:**
   - `cli/app.py`: Central Typer application with grouped help panels (`rich_help_panel`), typo suggestions via `difflib`, and global flag handling.
   - `cli/ui/`: Presentation helpers for consoles, color themes, ASCII fallbacks, error boxes, progress bars, and glossary lookups.
   - `cli/commands/`: Individual modules for commands (`plan_cmd.py`, `chat_cmd.py`, `models_cmd.py`, etc.) that parse flags, invoke the respective service, and render results via `cli/ui/`.
   - `cli/setup/`: Step engine for the guided onboarding wizard.
   - `cli/home/`: Interactive terminal menu powered by `prompt_toolkit`.
   - `cli/chat/`: Streaming chat REPL powered by `prompt_toolkit` and `rich`.
   - `cli/tui/`: Full-terminal live dashboard powered by `textual`.
3. **Preserve Compatibility:**
   - `src/ramllm/cli/main.py` will expose `app` by re-exporting the modular Typer application from `src/ramllm/cli/app.py`.
   - All existing flags, options, subcommands, and exit codes remain supported.

## Consequences
- Clean separation of concerns: Business logic is unit-testable without spawning terminal emulators.
- The interactive menu, CLI commands, chat REPL, and TUI all share identical execution rules and state.
- Existing tests and scripts using `ramllm.cli.main:app` continue to function without modification.
