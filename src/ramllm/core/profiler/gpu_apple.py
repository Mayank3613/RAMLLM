"""Apple Silicon Metal GPU prober."""

import platform
import subprocess

from ramllm.core.schemas import GpuInfo


def _lookup_apple_bandwidth(chip_name: str) -> float:
    """Return theoretical/empirical unified memory bandwidth for Apple chips in bytes/s."""
    lower = chip_name.lower()
    if "ultra" in lower:
        return 800.0 * 1024**3
    if "max" in lower:
        return 400.0 * 1024**3
    if "pro" in lower:
        return 200.0 * 1024**3
    if "m5" in lower:
        return 150.0 * 1024**3
    if "m4" in lower:
        return 120.0 * 1024**3
    if "m3" in lower:
        return 100.0 * 1024**3
    if "m2" in lower:
        return 100.0 * 1024**3
    if "m1" in lower:
        return 68.0 * 1024**3
    # Default conservative unified bandwidth
    return 100.0 * 1024**3


def probe_apple_gpu(ram_total_bytes: int, ram_available_bytes: int) -> GpuInfo | None:
    """Detect Apple Silicon GPU and unified memory characteristics.

    Fails soft and returns None if not running on macOS with Apple Silicon.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    gpu_name = "Apple Silicon GPU"
    try:
        brand = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        )
        if brand:
            gpu_name = f"{brand} (Metal)"
    except Exception:
        pass

    bw = _lookup_apple_bandwidth(gpu_name)

    # By default, Apple Silicon allows Metal to allocate up to ~75% of total RAM without sysctl overrides
    vram_total = int(ram_total_bytes * 0.75)
    vram_free = int(min(ram_available_bytes, vram_total))

    return GpuInfo(
        index=0,
        vendor="apple",
        name=gpu_name,
        api_support=["metal"],
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        compute_capability=None,
        unified_memory=True,
        pcie_gen=None,
        pcie_lanes=None,
        h2d_bandwidth_bytes_s=None,
        vram_bandwidth_bytes_s=bw,
    )
