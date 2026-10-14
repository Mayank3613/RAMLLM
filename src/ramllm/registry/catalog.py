"""Local model catalog, registry storage, and checksum verification."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from ramllm.core.analyzer.gguf import analyze_gguf

REGISTRY_DIR = Path.home() / ".local" / "share" / "ramllm"
REGISTRY_FILE = REGISTRY_DIR / "models.json"


class ModelCatalogEntry(BaseModel):
    """Entry representing a locally registered model with metadata and integrity hash."""

    path: str
    name: str
    arch: str
    n_params: int
    n_layers: int
    quant: str
    total_weight_bytes: int
    is_moe: bool
    added_at: datetime
    sha256: str | None = None
    source_url: str | None = None


def compute_file_sha256(file_path: Path | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute SHA-256 hash incrementally without loading large model files into RAM."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


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


def add_model(
    file_path: Path | str,
    sha256: str | None = None,
    source_url: str | None = None,
) -> ModelCatalogEntry:
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
        sha256=sha256,
        source_url=source_url,
    )

    catalog = _load_catalog()
    catalog = [e for e in catalog if e.path != entry.path]
    catalog.append(entry)
    _save_catalog(catalog)
    return entry


def get_model(query: str) -> ModelCatalogEntry | None:
    """Find a model in the catalog by exact path, filename, or case-insensitive name."""
    catalog = _load_catalog()
    q_norm = query.strip().lower()
    for e in catalog:
        if e.path == query or e.name == query:
            return e
        if e.name.lower() == q_norm or Path(e.path).stem.lower() == q_norm:
            return e
    return None


def remove_model(query: str) -> bool:
    """Remove a model entry from the catalog. Returns True if found and removed."""
    catalog = _load_catalog()
    initial_len = len(catalog)
    q_norm = query.strip().lower()
    filtered = [
        e
        for e in catalog
        if e.path != query
        and e.name != query
        and e.name.lower() != q_norm
        and Path(e.path).stem.lower() != q_norm
    ]
    if len(filtered) < initial_len:
        _save_catalog(filtered)
        return True
    return False


def verify_model(query: str, update_catalog: bool = True) -> tuple[bool, str, str]:
    """Verify SHA-256 integrity of a model file against catalog record.

    Returns:
        (is_valid_or_matched, sha256_hash, status_message)
    """
    entry = get_model(query)
    model_path = Path(entry.path) if entry else Path(query).resolve()

    if not model_path.exists():
        return False, "", f"Model file not found: {model_path}"

    computed_hash = compute_file_sha256(model_path)

    if entry and entry.sha256:
        if computed_hash.lower() == entry.sha256.lower():
            return True, computed_hash, "Checksum verified: matches recorded catalog hash."
        else:
            return (
                False,
                computed_hash,
                f"Checksum mismatch: expected {entry.sha256}, got {computed_hash}.",
            )

    # If no prior checksum was recorded, update catalog with verified hash
    if entry and update_catalog:
        catalog = _load_catalog()
        for idx, e in enumerate(catalog):
            if e.path == entry.path:
                catalog[idx] = e.model_copy(update={"sha256": computed_hash})
                break
        _save_catalog(catalog)
        return True, computed_hash, "Computed and registered new SHA-256 hash in catalog."

    return True, computed_hash, "Checksum computed successfully."


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
