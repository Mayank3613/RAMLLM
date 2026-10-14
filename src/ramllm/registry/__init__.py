"""Model catalog, registry, and download verification."""

from ramllm.registry.catalog import (
    ModelCatalogEntry,
    add_model,
    compute_file_sha256,
    get_model,
    list_models,
    remove_model,
    scan_directory,
    verify_model,
)
from ramllm.registry.download import pull_model, resolve_model_url

__all__ = [
    "ModelCatalogEntry",
    "add_model",
    "compute_file_sha256",
    "get_model",
    "list_models",
    "pull_model",
    "remove_model",
    "resolve_model_url",
    "scan_directory",
    "verify_model",
]
