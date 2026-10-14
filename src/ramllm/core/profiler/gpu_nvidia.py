"""NVIDIA GPU prober via NVML with nvidia-smi fallback."""

import shutil
import subprocess

from ramllm.core.schemas import GpuInfo


def _lookup_nvidia_bandwidth(name: str) -> float:
    """Return typical memory bandwidth for popular NVIDIA GPUs in bytes/s."""
    lower = name.lower()
    if "4090" in lower:
        return 1008.0 * 1024**3
    if "4080" in lower:
        return 716.0 * 1024**3
    if "4070" in lower:
        return 504.0 * 1024**3
    if "3090" in lower:
        return 936.0 * 1024**3
    if "3080" in lower:
        return 760.0 * 1024**3
    if "3070" in lower:
        return 448.0 * 1024**3
    if "3060" in lower:
        return 360.0 * 1024**3
    if "a100" in lower:
        return 1935.0 * 1024**3
    if "h100" in lower:
        return 3350.0 * 1024**3
    if "a5000" in lower or "a6000" in lower:
        return 768.0 * 1024**3
    if "t4" in lower:
        return 320.0 * 1024**3
    # Default conservative GDDR6 bandwidth
    return 300.0 * 1024**3


def probe_nvidia_gpus() -> list[GpuInfo]:
    """Probe system for NVIDIA GPUs using pynvml or nvidia-smi.

    Fails soft and returns an empty list if NVIDIA drivers or GPUs are absent.
    """
    gpus: list[GpuInfo] = []

    # 1. Try pynvml
    try:
        import pynvml  # type: ignore[import-not-found,import-untyped]

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Compute capability
            cc_str: str | None = None
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                cc_str = f"{major}.{minor}"
            except Exception:
                pass

            # PCIe info
            pcie_gen: int | None = None
            pcie_lanes: int | None = None
            try:
                pcie_gen = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
                pcie_lanes = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
            except Exception:
                pass

            gpus.append(
                GpuInfo(
                    index=i,
                    vendor="nvidia",
                    name=name,
                    api_support=["cuda", "vulkan"],
                    vram_total_bytes=int(mem_info.total),
                    vram_free_bytes=int(mem_info.free),
                    compute_capability=cc_str,
                    unified_memory=False,
                    pcie_gen=pcie_gen,
                    pcie_lanes=pcie_lanes,
                    h2d_bandwidth_bytes_s=None,
                    vram_bandwidth_bytes_s=_lookup_nvidia_bandwidth(name),
                )
            )
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        pass

    # 2. Try nvidia-smi fallback
    if shutil.which("nvidia-smi"):
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            for line in output.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    idx = int(parts[0])
                    name = parts[1]
                    total_mb = int(parts[2])
                    free_mb = int(parts[3])
                    gpus.append(
                        GpuInfo(
                            index=idx,
                            vendor="nvidia",
                            name=name,
                            api_support=["cuda", "vulkan"],
                            vram_total_bytes=total_mb * 1024 * 1024,
                            vram_free_bytes=free_mb * 1024 * 1024,
                            compute_capability=None,
                            unified_memory=False,
                            pcie_gen=None,
                            pcie_lanes=None,
                            h2d_bandwidth_bytes_s=None,
                            vram_bandwidth_bytes_s=_lookup_nvidia_bandwidth(name),
                        )
                    )
        except Exception:
            pass

    return gpus
