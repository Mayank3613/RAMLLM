"""Memory bandwidth cost model and speed range estimation."""

from typing import Literal

from ramllm.core.schemas import HardwareProfile, SpeedEstimate


def estimate_speed(
    hw: HardwareProfile,
    bytes_gpu: int,
    bytes_cpu: int,
    bytes_disk: int,
    bytes_xfer_h2d: int,
    kv_read_bytes_gpu: int,
    kv_read_bytes_cpu: int,
    active_params: int,
    batch_size: int = 1,
    calibration_factor: float = 1.0,
) -> SpeedEstimate:
    """Calculate decode and prefill speed ranges based on tier bandwidths and working set.

    Accounts for realistic memory bus efficiencies (low vs high ranges).
    Applies empirical calibration factor if prior benchmark runs are recorded.
    """
    # 1. Bandwidths
    # VRAM bandwidth: from GPU or fallback to 300 GB/s
    bw_vram = 300.0 * 1024**3
    if hw.gpus:
        bw_vram = hw.gpus[0].vram_bandwidth_bytes_s or (
            hw.ram_bandwidth_bytes_s if hw.gpus[0].unified_memory else 300.0 * 1024**3
        )

    bw_ram = max(hw.ram_bandwidth_bytes_s, 10.0 * 1024**3)
    bw_disk = max(hw.disk_read_bandwidth_bytes_s, 50.0 * 1024**2)

    # PCIe bandwidth
    bw_pcie = 15.0 * 1024**3
    if hw.gpus and hw.gpus[0].h2d_bandwidth_bytes_s:
        bw_pcie = hw.gpus[0].h2d_bandwidth_bytes_s

    # 2. Decode Time Modeling
    # Efficiency factors: (low_eff, high_eff)
    eff_vram_low, eff_vram_high = 0.60, 0.85
    eff_ram_low, eff_ram_high = 0.50, 0.78
    eff_disk_low, eff_disk_high = 0.60, 0.90
    eff_pcie_low, eff_pcie_high = 0.70, 0.90

    # Low estimate (conservative speed = pessimistic time)
    t_gpu_pess = (
        (bytes_gpu + kv_read_bytes_gpu) / (bw_vram * eff_vram_low) if bytes_gpu > 0 else 0.0
    )
    t_cpu_pess = (bytes_cpu + kv_read_bytes_cpu) / (bw_ram * eff_ram_low) if bytes_cpu > 0 else 0.0
    t_disk_pess = bytes_disk / (bw_disk * eff_disk_low) if bytes_disk > 0 else 0.0
    t_xfer_pess = bytes_xfer_h2d / (bw_pcie * eff_pcie_low) if bytes_xfer_h2d > 0 else 0.0
    t_overhead = 0.001  # 1ms kernel launch / sampling overhead

    t_token_pess = t_cpu_pess + t_disk_pess + t_gpu_pess + t_xfer_pess + t_overhead
    decode_low = max(0.01, (1.0 / t_token_pess) * calibration_factor)

    # High estimate (optimistic speed = optimistic time)
    t_gpu_opt = (
        (bytes_gpu + kv_read_bytes_gpu) / (bw_vram * eff_vram_high) if bytes_gpu > 0 else 0.0
    )
    t_cpu_opt = (bytes_cpu + kv_read_bytes_cpu) / (bw_ram * eff_ram_high) if bytes_cpu > 0 else 0.0
    t_disk_opt = bytes_disk / (bw_disk * eff_disk_high) if bytes_disk > 0 else 0.0
    t_xfer_opt = bytes_xfer_h2d / (bw_pcie * eff_pcie_high) if bytes_xfer_h2d > 0 else 0.0

    t_token_opt = t_cpu_opt + t_disk_opt + max(t_gpu_opt, t_xfer_opt) + (t_overhead * 0.5)
    decode_high = max(decode_low, (1.0 / t_token_opt) * calibration_factor)

    # 3. Bottleneck Identification
    times = {
        "disk_bw": t_disk_pess,
        "ram_bw": t_cpu_pess,
        "vram_bw": t_gpu_pess,
        "pcie_bw": t_xfer_pess,
    }
    sorted_times = sorted(times.items(), key=lambda kv: kv[1], reverse=True)
    primary_bottleneck: Literal[
        "vram_bw", "ram_bw", "disk_bw", "pcie_bw", "compute", "expert_cache_miss"
    ] = sorted_times[0][0]  # type: ignore[assignment]

    # 4. Prefill Estimation
    # Prompt eval is compute bound (2 * active_params FLOPs / token)
    # Estimate device TFLOPs: GPU ~10-40 TFLOPs, CPU ~0.5-2 TFLOPs
    if bytes_gpu > bytes_cpu:
        g_tflops = 15.0 * 1024**4  # 15 TFLOPs FP16
    else:
        g_tflops = 1.0 * 1024**4  # 1 TFLOPs CPU

    flops_per_token = 2.0 * active_params
    t_prefill_compute = flops_per_token / g_tflops
    prefill_high = max(1.0, min(500.0, 1.0 / max(1e-4, t_prefill_compute)))
    prefill_low = max(0.5, prefill_high * 0.55)

    return SpeedEstimate(
        decode_tok_s_low=round(decode_low, 1),
        decode_tok_s_high=round(decode_high, 1),
        prefill_tok_s_low=round(prefill_low, 1),
        prefill_tok_s_high=round(prefill_high, 1),
        bottleneck=primary_bottleneck,
    )
