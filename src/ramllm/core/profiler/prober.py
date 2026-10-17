"""Hardware profile prober orchestrator, caching, and synthetic loader."""

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Literal

from ramllm.core.profiler.cpu import estimate_cpu_gflops, get_cpu_info
from ramllm.core.profiler.disk import detect_disk_kind, measure_disk_read_bandwidth
from ramllm.core.profiler.gpu_amd import probe_amd_gpus
from ramllm.core.profiler.gpu_apple import probe_apple_gpu
from ramllm.core.profiler.gpu_nvidia import probe_nvidia_gpus
from ramllm.core.profiler.gpu_vulkan import probe_vulkan_gpus
from ramllm.core.profiler.memory import (
    get_memory_info,
    measure_ram_bandwidth,
    measure_ram_bandwidth_memcpy,
)
from ramllm.core.profiler.pcie import estimate_pcie_bandwidth
from ramllm.core.schemas import GpuInfo, HardwareProfile

CACHE_DIR = Path.home() / ".cache" / "ramllm"
CACHE_FILE = CACHE_DIR / "hardware.json"


def _get_os() -> Literal["linux", "macos", "windows"]:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "macos"
    if sys_name == "windows":
        return "windows"
    return "linux"


def load_synthetic_profile(path: Path | str) -> HardwareProfile:
    """Load a hardware profile from a JSON file (e.g. for CI or simulation)."""
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return HardwareProfile.model_validate_json(content)


def probe_hardware(
    refresh: bool = False,
    model_dir: Path | None = None,
    quick: bool = False,
) -> HardwareProfile:
    """Probe system hardware, run micro-benchmarks, and return HardwareProfile.

    If cached profile is available and not expired, re-probes dynamic fields (free RAM/VRAM)
    while preserving benchmarked bandwidths for instant startup.
    """
    ram_total, ram_avail, swap_total = get_memory_info()

    if not refresh and CACHE_FILE.exists():
        try:
            cached_data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            # Update dynamic memory fields
            cached_data["ram_available_bytes"] = ram_avail
            # Re-check free VRAM for dynamic accuracy
            profile = HardwareProfile.model_validate(cached_data)
            return profile
        except Exception:
            pass

    os_type = _get_os()
    cpu_name, phys_cores, log_cores, cpu_feats = get_cpu_info()

    # Compute and memory bandwidths
    cpu_gflops = estimate_cpu_gflops(phys_cores, log_cores, cpu_feats)
    if quick:
        ram_bw = 35.0 * 1024**3
        ram_memcpy_bw: float | None = 30.0 * 1024**3
        disk_bw = 3.0 * 1024**3
    else:
        ram_bw = measure_ram_bandwidth()
        try:
            ram_memcpy_bw = measure_ram_bandwidth_memcpy()
        except Exception:
            ram_memcpy_bw = None
        disk_bw = measure_disk_read_bandwidth(model_dir)

    disk_type = detect_disk_kind(model_dir or Path.home())

    # Probe GPUs
    gpus: list[GpuInfo] = []
    apple_gpu = probe_apple_gpu(ram_total, ram_avail)
    if apple_gpu:
        gpus.append(apple_gpu)
    else:
        gpus.extend(probe_nvidia_gpus())
        if not gpus:
            gpus.extend(probe_amd_gpus())
        if not gpus:
            gpus.extend(probe_vulkan_gpus())

    # Update PCIe estimates for discrete GPUs
    updated_gpus: list[GpuInfo] = []
    for g in gpus:
        if not g.unified_memory and g.h2d_bandwidth_bytes_s is None:
            est_pcie = estimate_pcie_bandwidth(g)
            updated_gpus.append(g.model_copy(update={"h2d_bandwidth_bytes_s": est_pcie}))
        else:
            updated_gpus.append(g)

    profile = HardwareProfile(
        os=os_type,
        cpu_name=cpu_name,
        cpu_cores_physical=phys_cores,
        cpu_cores_logical=log_cores,
        cpu_features=cpu_feats,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        swap_total_bytes=swap_total,
        ram_bandwidth_bytes_s=ram_bw,
        ram_bandwidth_memcpy_bytes_s=ram_memcpy_bw,
        cpu_effective_gflops=cpu_gflops,
        disk_read_bandwidth_bytes_s=disk_bw,
        disk_kind=disk_type,
        gpus=updated_gpus,
        measured_at=datetime.now(),
    )

    # Cache profile
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    except Exception:
        pass

    return profile
