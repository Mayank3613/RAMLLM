"""AMD GPU prober via rocm-smi or sysfs."""

import shutil
import subprocess

from ramllm.core.schemas import GpuInfo


def probe_amd_gpus() -> list[GpuInfo]:
    """Probe system for AMD Radeon/Instinct GPUs via rocm-smi.

    Fails soft and returns empty list if ROCm is not present.
    """
    gpus: list[GpuInfo] = []
    if not shutil.which("rocm-smi"):
        return gpus

    try:
        cmd = ["rocm-smi", "--showmeminfo", "vram", "--json"]
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        import json

        data = json.loads(output)
        # Parse standard rocm-smi json output
        idx = 0
        for card_key, card_info in data.items():
            if "card" in card_key.lower():
                total = int(card_info.get("VRAM Total Memory (B)", 16 * 1024**3))
                used = int(card_info.get("VRAM Total Used Memory (B)", 0))
                free = max(0, total - used)
                gpus.append(
                    GpuInfo(
                        index=idx,
                        vendor="amd",
                        name="AMD Radeon/Instinct GPU",
                        api_support=["rocm", "vulkan"],
                        vram_total_bytes=total,
                        vram_free_bytes=free,
                        compute_capability=None,
                        unified_memory=False,
                        pcie_gen=None,
                        pcie_lanes=None,
                        h2d_bandwidth_bytes_s=None,
                        vram_bandwidth_bytes_s=500.0 * 1024**3,
                    )
                )
                idx += 1
    except Exception:
        pass

    return gpus
