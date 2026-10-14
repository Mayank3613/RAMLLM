"""Model service: curated model index, fit estimation, and disk management."""

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ramllm.core.schemas import HardwareProfile
from ramllm.services.paths import get_models_dir


class IndexedFile(BaseModel):
    """File entry within a curated model."""

    quant: str
    size_bytes: int
    hf_repo: str
    hf_file: str
    sha256: str = ""


class CuratedModel(BaseModel):
    """Curated model entry in models_index.json."""

    id: str
    name: str
    family: str
    params_b: float
    architecture: Literal["dense", "moe"] = "dense"
    active_params_b: float | None = None
    tags: list[str] = Field(default_factory=list)
    license: str = ""
    files: list[IndexedFile] = Field(default_factory=list)
    notes: str = ""
    recommended_min_memory_gb: int = 4
    last_verified: str | None = None


class ModelsIndex(BaseModel):
    """Full curated models index document."""

    version: str = "1.0.0"
    updated_at: str = ""
    models: list[CuratedModel] = Field(default_factory=list)


def get_index_path() -> Path:
    """Return path to packaged models_index.json."""
    return Path(__file__).resolve().parent.parent / "data" / "models_index.json"


def load_curated_index() -> ModelsIndex:
    """Load and parse the curated models index from disk."""
    p = get_index_path()
    if not p.exists():
        return ModelsIndex()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ModelsIndex.model_validate(data)
    except Exception:
        return ModelsIndex()


def get_curated_model(model_id: str) -> CuratedModel | None:
    """Retrieve a curated model by its friendly ID."""
    idx = load_curated_index()
    mid = model_id.lower().strip()
    for m in idx.models:
        if m.id.lower() == mid:
            return m
    return None


def search_curated_models(query: str | None = None, tag: str | None = None) -> list[CuratedModel]:
    """Search curated models by substring and/or tag."""
    idx = load_curated_index()
    results = idx.models

    if tag:
        t = tag.lower().strip()
        results = [m for m in results if any(t in existing_tag.lower() for existing_tag in m.tags)]

    if query:
        q = query.lower().strip()
        results = [
            m
            for m in results
            if q in m.id.lower()
            or q in m.name.lower()
            or q in m.family.lower()
            or q in m.notes.lower()
            or any(q in existing_tag.lower() for existing_tag in m.tags)
        ]

    return results


def estimate_model_fit(
    model: CuratedModel,
    hw: HardwareProfile,
    quant: str = "Q4_K_M",
    context_length: int = 4096,
) -> tuple[str, str, str]:
    """Estimate whether a model fits current hardware.

    Returns:
        (verdict_code, plain_verdict, color)
        e.g. ("COMFORTABLE", "runs well", "green")
    """
    # Find matching file to get exact size if available
    matching_file = next((f for f in model.files if f.quant.upper() == quant.upper()), None)
    if matching_file and matching_file.size_bytes > 0:
        weight_bytes = matching_file.size_bytes
    else:
        # Approximate 0.55 bytes per parameter for Q4, 0.68 for Q5, 1.05 for Q8
        q = quant.upper()
        bytes_per_param = 1.05 if "8" in q else (0.70 if "5" in q else 0.55)
        weight_bytes = int(model.params_b * 1e9 * bytes_per_param)

    # Context KV cache estimate (approx 0.5 MB per 1k context per billion active params)
    active_b = model.active_params_b if model.active_params_b else model.params_b
    kv_bytes = int(context_length * active_b * 512)
    working_overhead = 512 * 1024 * 1024  # 512 MB runtime compute buffers

    total_needed_bytes = weight_bytes + kv_bytes + working_overhead

    # Available system memory
    total_vram = sum(g.vram_total_bytes for g in hw.gpus) if hw.gpus else 0
    total_ram = hw.ram_available_bytes

    # Hardware with GPU
    if hw.gpus and total_vram > 0:
        if total_needed_bytes <= total_vram * 0.90:
            return ("COMFORTABLE", "runs well on GPU", "green")
        if total_needed_bytes <= (total_vram + total_ram) * 0.80:
            return ("CONSTRAINED", "works, but slowly (split)", "yellow")
        return ("INFEASIBLE", "too big for available memory", "red")

    # CPU/Unified memory
    if total_needed_bytes <= total_ram * 0.75:
        return ("COMFORTABLE", "runs well", "green")
    if total_needed_bytes <= total_ram * 0.92:
        return ("CONSTRAINED", "works, but slowly", "yellow")
    return ("INFEASIBLE", "too big for available memory", "red")


def pick_best_quant(model: CuratedModel, hw: HardwareProfile) -> IndexedFile:
    """Select the best quantization file that fits in available hardware."""
    if not model.files:
        raise ValueError(f"Model '{model.id}' has no files in index.")

    preferred_order = ["Q5_K_M", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q8_0"]
    sorted_files = sorted(
        model.files,
        key=lambda f: preferred_order.index(f.quant) if f.quant in preferred_order else 99,
    )

    for f in sorted_files:
        verdict, _, _ = estimate_model_fit(model, hw, quant=f.quant)
        if verdict in ("COMFORTABLE", "CONSTRAINED"):
            return f

    # If none fit comfortably, return the smallest file
    return min(model.files, key=lambda f: f.size_bytes if f.size_bytes > 0 else 10**12)


def get_models_disk_usage() -> dict[str, Any]:
    """Inspect the models directory and calculate disk usage statistics."""
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    model_files: list[dict[str, Any]] = []
    partial_files: list[dict[str, Any]] = []

    total_model_bytes = 0
    total_partial_bytes = 0

    for item in models_dir.iterdir():
        if item.is_file():
            size = item.stat().st_size
            if item.name.endswith(".part") or item.name.endswith(".tmp"):
                partial_files.append({"name": item.name, "path": str(item), "size_bytes": size})
                total_partial_bytes += size
            elif item.name.endswith(".gguf"):
                model_files.append({"name": item.name, "path": str(item), "size_bytes": size})
                total_model_bytes += size

    disk_usage = shutil.disk_usage(models_dir)

    return {
        "models_dir": str(models_dir),
        "model_files": model_files,
        "partial_files": partial_files,
        "total_model_bytes": total_model_bytes,
        "total_partial_bytes": total_partial_bytes,
        "total_used_bytes": total_model_bytes + total_partial_bytes,
        "free_disk_bytes": disk_usage.free,
        "total_disk_bytes": disk_usage.total,
    }


def clean_models_dir(remove_partials: bool = True) -> tuple[int, int]:
    """Remove partial download files (.part, .tmp). Returns (count_removed, bytes_reclaimed)."""
    usage = get_models_disk_usage()
    removed_count = 0
    bytes_reclaimed = 0

    if remove_partials:
        for p in usage["partial_files"]:
            try:
                Path(p["path"]).unlink(missing_ok=True)
                removed_count += 1
                bytes_reclaimed += p["size_bytes"]
            except OSError:
                pass

    return removed_count, bytes_reclaimed
