"""Local model catalog and registry storage."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from ramllm.core.analyzer.gguf import analyze_gguf

REGISTRY_DIR = Path.home() / ".local" / "share" / "ramllm"
REGISTRY_FILE = REGISTRY_DIR / "models.json"


class ModelCatalogEntry(BaseModel):
    """Entry representing a locally discovered or added model."""

    path: str
    name: str
    arch: str
    n_params: int
    n_layers: int
    quant: str
    total_weight_bytes: int
    is_moe: bool
    added_at: datetime


def _load_catalog() -> list[ModelCatalogEntry]:
    if not REGISTRY_FILE.exists():
        return []
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        return [ModelCatalogEntry.model_validate(e) for e in data]
    except Exception:
        return []


def _save_catalog(entries: list[ModelCatalogEntry]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    data = [e.model_dump(mode="json") for e in entries]
    REGISTRY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def add_model(file_path: Path | str) -> ModelCatalogEntry:
    """Analyze a local GGUF file and add it to the catalog."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    info = analyze_gguf(p)
    entry = ModelCatalogEntry(
        path=str(p),
        name=p.name,
        arch=info.arch,
        n_params=info.n_params,
        n_layers=info.n_layers,
        quant=info.quant,
        total_weight_bytes=info.total_weight_bytes,
        is_moe=info.is_moe,
        added_at=datetime.now(),
    )

    catalog = _load_catalog()
    # Remove existing entry with same path
    catalog = [e for e in catalog if e.path != entry.path]
    catalog.append(entry)
    _save_catalog(catalog)
    return entry


def list_models() -> list[ModelCatalogEntry]:
    """Retrieve all models currently in the catalog."""
    return _load_catalog()


def scan_directory(dir_path: Path | str) -> list[ModelCatalogEntry]:
    """Scan directory recursively for .gguf files and register them."""
    root = Path(dir_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    added: list[ModelCatalogEntry] = []
    for f in root.rglob("*.gguf"):
        try:
            entry = add_model(f)
            added.append(entry)
        except Exception:
            pass
    return added
