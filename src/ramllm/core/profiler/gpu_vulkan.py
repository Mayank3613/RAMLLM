"""Generic Vulkan GPU prober via vulkaninfo."""

import shutil
import subprocess

from ramllm.core.schemas import GpuInfo


def probe_vulkan_gpus() -> list[GpuInfo]:
    """Probe system for Vulkan-compatible devices using vulkaninfo.

    Fails soft and returns empty list if vulkaninfo is absent or fails.
    """
    gpus: list[GpuInfo] = []
    if not shutil.which("vulkaninfo"):
        return gpus

    try:
        cmd = ["vulkaninfo", "--summary"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        # Scan for device summary in output
        idx = 0
        for line in output.splitlines():
            if "deviceName" in line:
                name = line.split("=")[-1].strip()
                gpus.append(
                    GpuInfo(
                        index=idx,
                        vendor="other",
                        name=name,
                        api_support=["vulkan"],
                        vram_total_bytes=8 * 1024**3,  # Fallback estimate
                        vram_free_bytes=6 * 1024**3,
                        compute_capability=None,
                        unified_memory=False,
                        pcie_gen=None,
                        pcie_lanes=None,
                        h2d_bandwidth_bytes_s=None,
                        vram_bandwidth_bytes_s=250.0 * 1024**3,
                    )
                )
                idx += 1
    except Exception:
        pass

    return gpus
