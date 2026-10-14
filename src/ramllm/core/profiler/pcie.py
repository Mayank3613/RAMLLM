"""PCIe Host-to-Device (H2D) transfer bandwidth estimation and measurement."""

from ramllm.core.schemas import GpuInfo


def estimate_pcie_bandwidth(gpu: GpuInfo) -> float | None:
    """Estimate or measure Host-to-Device transfer bandwidth in bytes/sec.

    Uses PCIe generation and link width if available, with standard 80% bus efficiency.
    Returns None for unified memory architectures (e.g. Apple Silicon).
    """
    if gpu.unified_memory:
        return None

    # Theoretical peak per lane in GB/s for PCIe specs
    lane_bandwidth = {
        1: 0.25 * 1024**3,
        2: 0.50 * 1024**3,
        3: 0.985 * 1024**3,
        4: 1.969 * 1024**3,
        5: 3.938 * 1024**3,
        6: 7.560 * 1024**3,
    }

    gen = gpu.pcie_gen or 4
    lanes = gpu.pcie_lanes or 16

    base_per_lane = lane_bandwidth.get(gen, 1.969 * 1024**3)
    theoretical = base_per_lane * lanes
    effective = theoretical * 0.82  # ~82% effective PCIe throughput after protocol overhead
    return float(effective)
