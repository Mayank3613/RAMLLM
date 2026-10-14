"""Configuration management service using TOML storage with validation."""

import tomllib
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, Field, ValidationError

from ramllm.services.paths import get_config_dir, get_config_file


class ConfigModel(BaseModel):
    """Configuration schema for RAMLLM."""

    default_model: str | None = Field(default=None, description="Default model ID or path")
    mode: Literal["fast", "balanced", "quality", "max"] = Field(
        default="balanced", description="Performance/quality tradeoff mode"
    )
    memory_usage: Literal["conservative", "balanced", "max"] = Field(
        default="balanced", description="Memory allocation aggressiveness"
    )
    context: str = Field(default="auto", description="Default context window (e.g. auto, 4096, 8k)")
    models_dir: str | None = Field(default=None, description="Custom directory for model downloads")
    backend_path: str | None = Field(
        default=None, description="Custom path to llama-server executable"
    )
    port: int = Field(default=8080, ge=1024, le=65535, description="Default API server port")
    host: str = Field(default="127.0.0.1", description="Default API server host interface")
    color: Literal["auto", "always", "never"] = Field(
        default="auto", description="Terminal color styling"
    )
    unicode: Literal["auto", "ascii"] = Field(default="auto", description="Unicode glyph support")
    confirm_downloads: bool = Field(default=True, description="Prompt before large file downloads")
    auto_retry: bool = Field(
        default=True, description="Automatically adapt to smaller plans on failure"
    )
    hf_token_env: str = Field(
        default="HF_TOKEN", description="Environment variable holding HF token"
    )
    telemetry: bool = Field(default=False, description="Telemetry disabled by default")


RamllmConfig = ConfigModel


def get_default_config() -> ConfigModel:
    return ConfigModel()


def load_config() -> ConfigModel:
    """Load configuration from config.toml, or return defaults if absent/invalid."""
    cfg_file = get_config_file()
    if not cfg_file.exists():
        return get_default_config()

    try:
        with open(cfg_file, "rb") as f:
            data = tomllib.load(f)
        return ConfigModel.model_validate(data)
    except Exception:
        return get_default_config()


def save_config(config: ConfigModel) -> None:
    """Persist configuration to config.toml."""
    get_config_dir().mkdir(parents=True, exist_ok=True)
    cfg_file = get_config_file()
    data = config.model_dump(exclude_none=True)
    with open(cfg_file, "wb") as f:
        tomli_w.dump(data, f)


def get_config_value(key: str) -> Any:
    """Retrieve a specific configuration value."""
    cfg = load_config()
    if not hasattr(cfg, key):
        raise KeyError(f"Unknown configuration key: '{key}'")
    return getattr(cfg, key)


def set_config_value(key: str, raw_value: str) -> Any:
    """Set and validate a configuration key value."""
    cfg = load_config()
    if not hasattr(cfg, key):
        valid_keys = ", ".join(ConfigModel.model_fields.keys())
        raise KeyError(f"Unknown configuration key '{key}'. Available keys: {valid_keys}")

    # Parse and coerce types
    val_to_set: Any = raw_value
    if raw_value.lower() in ("true", "1", "yes"):
        val_to_set = True
    elif raw_value.lower() in ("false", "0", "no"):
        val_to_set = False
    elif raw_value.lower() in ("none", "null"):
        val_to_set = None
    elif raw_value.isdigit():
        val_to_set = int(raw_value)

    data = cfg.model_dump()
    data[key] = val_to_set

    try:
        validated = ConfigModel.model_validate(data)
        save_config(validated)
        return getattr(validated, key)
    except ValidationError as e:
        raise ValueError(
            f"Invalid value '{raw_value}' for key '{key}': {e.errors()[0]['msg']}"
        ) from e


def reset_config(key: str | None = None) -> None:
    """Reset all keys or a single key to its default."""
    defaults = get_default_config()
    if key is None:
        save_config(defaults)
        return

    cfg = load_config()
    if not hasattr(cfg, key):
        raise KeyError(f"Unknown configuration key: '{key}'")

    data = cfg.model_dump()
    data[key] = getattr(defaults, key)
    save_config(ConfigModel.model_validate(data))
