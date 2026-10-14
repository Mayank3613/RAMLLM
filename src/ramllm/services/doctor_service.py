"""Diagnostic doctor service and automated remediation routines."""

import json
from dataclasses import dataclass
from pathlib import Path

import psutil

from ramllm.core.governor.enforcement import get_current_enforcement_status
from ramllm.core.profiler import probe_hardware
from ramllm.registry.catalog import list_models, remove_model
from ramllm.services.backend_manager import (
    discover_backend,
    get_fallback_suggestions,
    install_backend,
)
from ramllm.services.paths import (
    ensure_directories,
    get_backends_dir,
    get_config_dir,
    get_models_dir,
    get_ramllm_home,
    get_run_dir,
)


@dataclass
class DiagnosticItem:
    category: str
    title: str
    severity: str  # "ok", "notice", "problem"
    detail: str
    fix_suggestion: str | None = None
    fix_action: str | None = None


def run_diagnostics() -> list[DiagnosticItem]:
    """Run all diagnostic checks across backends, hardware, catalogs, and runtime state."""
    items: list[DiagnosticItem] = []

    # 1. Directories
    missing_dirs = []
    for d in [
        get_ramllm_home(),
        get_config_dir(),
        get_models_dir(),
        get_backends_dir(),
        get_run_dir(),
    ]:
        if not d.exists():
            missing_dirs.append(str(d))
    if missing_dirs:
        items.append(
            DiagnosticItem(
                category="Storage",
                title="Directories",
                severity="notice",
                detail=f"Required directories not created yet: {', '.join(missing_dirs)}",
                fix_suggestion="Create missing RAMLLM system directories",
                fix_action="ensure_directories",
            )
        )
    else:
        items.append(
            DiagnosticItem(
                category="Storage",
                title="Directories",
                severity="ok",
                detail=f"RAMLLM storage initialized at {get_ramllm_home()}",
            )
        )

    # 2. Backend Engine
    disc = discover_backend()
    if disc:
        caps = disc.install.capabilities
        cap_list = []
        if caps.partial_gpu_offload:
            cap_list.append("GPU Offload")
        if caps.kv_quantization:
            cap_list.append("KV Quantization")
        if caps.per_tensor_placement:
            cap_list.append("MoE CPU Offload")
        if caps.flash_attention:
            cap_list.append("Flash Attention")
        cap_str = f" ({', '.join(cap_list)})" if cap_list else ""
        items.append(
            DiagnosticItem(
                category="Backend",
                title="llama.cpp Engine",
                severity="ok",
                detail=f"Found via {disc.source} at `{disc.install.binary_path}`{cap_str}",
            )
        )
    else:
        fallbacks = "\n".join(f"  • {s}" for s in get_fallback_suggestions())
        items.append(
            DiagnosticItem(
                category="Backend",
                title="llama.cpp Engine",
                severity="problem",
                detail="No llama.cpp binary found on PATH or managed directory.",
                fix_suggestion=f"Install engine with `ramllm backend install` or:\n{fallbacks}",
                fix_action="install_backend",
            )
        )

    # 3. OS Memory Enforcement
    enf = get_current_enforcement_status()
    if enf.can_enforce:
        items.append(
            DiagnosticItem(
                category="OS & Enforcement",
                title="Memory Governor Enforcement",
                severity="ok",
                detail=f"{enf.mechanism} active ({enf.detail})",
            )
        )
    elif enf.supported:
        items.append(
            DiagnosticItem(
                category="OS & Enforcement",
                title="Memory Governor Enforcement",
                severity="notice",
                detail=f"{enf.mechanism} supported ({enf.detail})",
            )
        )
    else:
        items.append(
            DiagnosticItem(
                category="OS & Enforcement",
                title="Memory Governor Enforcement",
                severity="notice",
                detail=f"Soft enforcement only ({enf.detail})",
            )
        )

    # 4. Hardware Headroom
    try:
        hw = probe_hardware(quick=True)
        ram_avail_gb = hw.ram_available_bytes / (1024**3)
        if ram_avail_gb < 1.5:
            items.append(
                DiagnosticItem(
                    category="Hardware",
                    title="Available RAM",
                    severity="problem",
                    detail=f"Critically low free RAM: {ram_avail_gb:.1f} GB available.",
                    fix_suggestion="Close memory-heavy applications to free up headroom for model weights",
                )
            )
        elif ram_avail_gb < 3.0:
            items.append(
                DiagnosticItem(
                    category="Hardware",
                    title="Available RAM",
                    severity="notice",
                    detail=f"Low free RAM: {ram_avail_gb:.1f} GB available. Models will rely more on GPU or lower context.",
                )
            )
        else:
            items.append(
                DiagnosticItem(
                    category="Hardware",
                    title="Available RAM",
                    severity="ok",
                    detail=f"{ram_avail_gb:.1f} GB available ({hw.ram_total_bytes / (1024**3):.1f} GB total)",
                )
            )
    except Exception as e:
        items.append(
            DiagnosticItem(
                category="Hardware",
                title="Hardware Prober",
                severity="notice",
                detail=f"Could not probe full hardware stats ({e})",
            )
        )

    # 5. Catalog Integrity
    try:
        entries = list_models()
        missing_models = [e for e in entries if not Path(e.path).exists()]
        if missing_models:
            items.append(
                DiagnosticItem(
                    category="Catalog",
                    title="Model Catalog Entries",
                    severity="problem",
                    detail=f"{len(missing_models)} registered model(s) point to non-existent files.",
                    fix_suggestion="Prune missing models from local catalog",
                    fix_action="repair_catalog",
                )
            )
        else:
            items.append(
                DiagnosticItem(
                    category="Catalog",
                    title="Model Catalog Entries",
                    severity="ok",
                    detail=f"{len(entries)} model(s) registered and verified on disk",
                )
            )
    except Exception:
        pass

    # 6. Stale Runtime State
    run_dir = get_run_dir()
    if run_dir.exists():
        stale_files: list[Path] = []
        for state_file in run_dir.glob("*.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                pid = data.get("pid")
                if pid and not psutil.pid_exists(pid):
                    stale_files.append(state_file)
            except Exception:
                stale_files.append(state_file)
        if stale_files:
            items.append(
                DiagnosticItem(
                    category="Runtime",
                    title="Active Sessions",
                    severity="notice",
                    detail=f"Found {len(stale_files)} stale runtime state files from previous sessions.",
                    fix_suggestion="Clean up orphaned session state files",
                    fix_action="clean_stale_pids",
                )
            )

    return items


def execute_fix(action: str) -> tuple[bool, str]:
    """Execute a specific remediation action."""
    if action == "ensure_directories":
        ensure_directories()
        return True, "Created RAMLLM directories."

    if action == "repair_catalog":
        entries = list_models()
        removed_count = 0
        for e in entries:
            if not Path(e.path).exists():
                remove_model(e.name)
                removed_count += 1
        return True, f"Pruned {removed_count} missing entries from catalog."

    if action == "clean_stale_pids":
        run_dir = get_run_dir()
        count = 0
        if run_dir.exists():
            for state_file in run_dir.glob("*.json"):
                try:
                    data = json.loads(state_file.read_text(encoding="utf-8"))
                    pid = data.get("pid")
                    if not pid or not psutil.pid_exists(pid):
                        state_file.unlink()
                        count += 1
                except Exception:
                    state_file.unlink()
                    count += 1
        return True, f"Removed {count} stale session state files."

    if action == "install_backend":
        try:
            install = install_backend()
            return True, f"Installed llama.cpp engine successfully at {install.binary_path}."
        except Exception as e:
            return False, f"Failed to install backend engine: {e}"

    return False, f"Unknown fix action: {action}"
