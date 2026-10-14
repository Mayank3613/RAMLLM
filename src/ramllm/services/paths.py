"""Centralized filesystem paths management for RAMLLM."""

import os
from pathlib import Path

import platformdirs


def get_ramllm_home() -> Path:
    """Return the base directory for RAMLLM data and state."""
    env_home = os.environ.get("RAMLLM_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return Path(platformdirs.user_data_dir("ramllm", appauthor=False))


# Alias for backward compatibility
get_user_data_dir = get_ramllm_home


def get_config_dir() -> Path:
    """Return the configuration directory."""
    env_config = os.environ.get("RAMLLM_CONFIG")
    if env_config:
        return Path(env_config).expanduser().resolve().parent
    return Path(platformdirs.user_config_dir("ramllm", appauthor=False))


def get_config_file() -> Path:
    """Return the path to config.toml."""
    env_config = os.environ.get("RAMLLM_CONFIG")
    if env_config:
        return Path(env_config).expanduser().resolve()
    return get_config_dir() / "config.toml"


def get_models_dir() -> Path:
    """Return the directory where downloaded models are stored."""
    env_models = os.environ.get("RAMLLM_MODELS_DIR")
    if env_models:
        return Path(env_models).expanduser().resolve()
    return get_ramllm_home() / "models"


def get_backends_dir() -> Path:
    """Return the directory where managed engine builds are stored."""
    return get_ramllm_home() / "backends"


def get_chats_dir() -> Path:
    """Return the directory where chat transcripts are archived."""
    return get_ramllm_home() / "chats"


def get_run_dir() -> Path:
    """Return the directory containing runtime state files and PIDs."""
    return get_ramllm_home() / "run"


def get_logs_dir() -> Path:
    """Return the directory where system and engine logs are stored."""
    return get_ramllm_home() / "logs"


def ensure_directories() -> None:
    """Ensure essential RAMLLM directories exist."""
    get_ramllm_home().mkdir(parents=True, exist_ok=True)
    get_config_dir().mkdir(parents=True, exist_ok=True)
    get_models_dir().mkdir(parents=True, exist_ok=True)
    get_backends_dir().mkdir(parents=True, exist_ok=True)
    get_chats_dir().mkdir(parents=True, exist_ok=True)
    get_run_dir().mkdir(parents=True, exist_ok=True)
    get_logs_dir().mkdir(parents=True, exist_ok=True)
