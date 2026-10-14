"""Setup wizard state persistence for resumable onboarding."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ramllm.services.paths import get_ramllm_home


class SetupState(BaseModel):
    """Tracks wizard progress across restarts or interruptions."""

    completed_steps: list[str] = Field(default_factory=list)
    engine_installed: bool = False
    use_case: str | None = None
    mode: Literal["fast", "balanced", "quality", "max"] = "balanced"
    memory_preset: Literal["conservative", "balanced", "max"] = "balanced"
    selected_model_id: str | None = None
    test_run_completed: bool = False
    finished: bool = False


def get_setup_state_file() -> Path:
    """Return path to setup_state.json."""
    return get_ramllm_home() / "setup_state.json"


def load_setup_state() -> SetupState:
    """Load persistent setup state or return a fresh state."""
    p = get_setup_state_file()
    if not p.exists():
        return SetupState()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return SetupState.model_validate(data)
    except Exception:
        return SetupState()


def save_setup_state(state: SetupState) -> None:
    """Save setup state to disk."""
    get_ramllm_home().mkdir(parents=True, exist_ok=True)
    p = get_setup_state_file()
    p.write_text(state.model_dump_json(indent=2), encoding="utf-8")


def clear_setup_state() -> None:
    """Remove setup state file."""
    p = get_setup_state_file()
    p.unlink(missing_ok=True)


def is_setup_completed() -> bool:
    """Check if the setup wizard has been successfully completed."""
    state = load_setup_state()
    return state.finished
