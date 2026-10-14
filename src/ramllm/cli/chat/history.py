"""Persistent input history management for prompt_toolkit REPL."""

from prompt_toolkit.history import FileHistory, History, InMemoryHistory

from ramllm.services.paths import get_user_data_dir


def get_chat_history() -> History:
    """Return persistent FileHistory for prompt_toolkit, falling back to InMemoryHistory if unavailable."""
    try:
        history_dir = get_user_data_dir() / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "chat_history.txt"
        return FileHistory(str(history_file))
    except Exception:
        return InMemoryHistory()
