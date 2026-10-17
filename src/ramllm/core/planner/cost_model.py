"""Memory bandwidth and compute cost model and speed range estimation."""

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
    vram_bandwidth_override: float | None = None,
    num_boundaries: int = 0,
    hidden_size: int = 4096,
    moe_layers_on_cpu: int = 0,
) -> SpeedEstimate:
    """Calculate decode and prefill speed ranges based on tier bandwidths, compute capacity, and working set.

    Accounts for:
    - Realistic memory bus efficiencies (low vs high ranges).
    - Compute-bound CPU decode: t_cpu = max(bandwidth_time, compute_time).
    - Per-boundary overhead: synchronization latency + activation transfer over PCIe.
    - MoE routed expert CPU offload per-layer synchronization.
    """
    # 1. Bandwidths
    if vram_bandwidth_override is not None:
        bw_vram = vram_bandwidth_override
    elif hw.gpus:
        bw_vram = hw.gpus[0].vram_bandwidth_bytes_s or (
            hw.ram_bandwidth_bytes_s if hw.gpus[0].unified_memory else 300.0 * 1024**3
        )
    else:
        bw_vram = 300.0 * 1024**3

    bw_ram = max(hw.ram_bandwidth_bytes_s, 10.0 * 1024**3)
    bw_disk = max(hw.disk_read_bandwidth_bytes_s, 50.0 * 1024**2)

    # PCIe bandwidth
    bw_pcie = 15.0 * 1024**3
    if hw.gpus and hw.gpus[0].h2d_bandwidth_bytes_s:
        bw_pcie = hw.gpus[0].h2d_bandwidth_bytes_s

    # 2. Decode Time Modeling
    eff_vram_low, eff_vram_high = 0.60, 0.85
    eff_ram_low, eff_ram_high = 0.50, 0.78
    eff_disk_low, eff_disk_high = 0.60, 0.90
    eff_pcie_low, eff_pcie_high = 0.70, 0.90

    # CPU decode compute modeling
    cpu_gflops = hw.cpu_effective_gflops or (hw.cpu_cores_physical * 25.0)
    total_weights = bytes_gpu + bytes_cpu
    if total_weights > 0:
        active_params_cpu = int(active_params * (bytes_cpu / total_weights))
    else:
        active_params_cpu = active_params if bytes_cpu > 0 else 0

    flops_cpu_decode = 2.0 * active_params_cpu
    t_cpu_compute = (
        flops_cpu_decode / (max(cpu_gflops, 1.0) * 1e9) if active_params_cpu > 0 else 0.0
    )

    # Low estimate (conservative speed = pessimistic time)
    t_gpu_pess = (
        (bytes_gpu + kv_read_bytes_gpu) / (bw_vram * eff_vram_low) if bytes_gpu > 0 else 0.0
    )
    t_cpu_bw_pess = (
        (bytes_cpu + kv_read_bytes_cpu) / (bw_ram * eff_ram_low) if bytes_cpu > 0 else 0.0
    )
    t_cpu_pess = max(t_cpu_bw_pess, t_cpu_compute)

    t_disk_pess = bytes_disk / (bw_disk * eff_disk_low) if bytes_disk > 0 else 0.0
    t_xfer_pess = bytes_xfer_h2d / (bw_pcie * eff_pcie_low) if bytes_xfer_h2d > 0 else 0.0

    # Per-boundary overhead (synchronization + PCIe activation transfer)
    effective_boundaries = max(num_boundaries, moe_layers_on_cpu * 2)
    activation_bytes = batch_size * hidden_size * 2  # FP16 activations
    t_act_xfer_pess = (
        (activation_bytes / (bw_pcie * eff_pcie_low)) if effective_boundaries > 0 else 0.0
    )
    t_act_xfer_opt = (
        (activation_bytes / (bw_pcie * eff_pcie_high)) if effective_boundaries > 0 else 0.0
    )
    t_sync_pess = 0.0004
    t_sync_opt = 0.0002

    t_overhead_pess = 0.001 + effective_boundaries * (t_sync_pess + t_act_xfer_pess)
    t_overhead_opt = 0.0005 + effective_boundaries * (t_sync_opt + t_act_xfer_opt)

    t_token_pess = t_cpu_pess + t_disk_pess + t_gpu_pess + t_xfer_pess + t_overhead_pess
    decode_low = max(0.01, (1.0 / t_token_pess) * calibration_factor)

    # High estimate (optimistic speed = optimistic time)
    t_gpu_opt = (
        (bytes_gpu + kv_read_bytes_gpu) / (bw_vram * eff_vram_high) if bytes_gpu > 0 else 0.0
    )
    t_cpu_bw_opt = (
        (bytes_cpu + kv_read_bytes_cpu) / (bw_ram * eff_ram_high) if bytes_cpu > 0 else 0.0
    )
    t_cpu_opt = max(t_cpu_bw_opt, t_cpu_compute * 0.75)

    t_disk_opt = bytes_disk / (bw_disk * eff_disk_high) if bytes_disk > 0 else 0.0
    t_xfer_opt = bytes_xfer_h2d / (bw_pcie * eff_pcie_high) if bytes_xfer_h2d > 0 else 0.0

    t_token_opt = t_cpu_opt + t_disk_opt + max(t_gpu_opt, t_xfer_opt) + t_overhead_opt
    decode_high = max(decode_low, (1.0 / t_token_opt) * calibration_factor)

    # 3. Bottleneck Identification
    times = {
        "disk_bw": t_disk_pess,
        "ram_bw": t_cpu_bw_pess,
        "vram_bw": t_gpu_pess,
        "pcie_bw": t_xfer_pess + (effective_boundaries * t_act_xfer_pess),
    }
    sorted_times = sorted(times.items(), key=lambda kv: kv[1], reverse=True)
    primary_bottleneck: Literal[
        "vram_bw", "ram_bw", "disk_bw", "pcie_bw", "compute", "expert_cache_miss"
    ] = sorted_times[0][0]  # type: ignore[assignment]

    if t_cpu_compute > t_cpu_bw_pess and t_cpu_compute >= sorted_times[0][1]:
        primary_bottleneck = "compute"

    # 4. Prefill Estimation
    if bytes_gpu > bytes_cpu:
        g_tflops = 15.0 * 1024**4  # 15 TFLOPs FP16
    else:
        g_tflops = max(cpu_gflops * 1e9, 1.0 * 1024**4)

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
