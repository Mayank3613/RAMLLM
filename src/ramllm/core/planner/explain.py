"""Human-readable explanation generation for execution plans."""

from ramllm.core.schemas import KvConfig, MemoryEstimate, ModelInfo, Placement, SpeedEstimate


def explain_plan(
    model: ModelInfo,
    placement: Placement,
    kv: KvConfig,
    memory: MemoryEstimate,
    speed: SpeedEstimate,
    total_layers: int,
) -> list[str]:
    """Generate human-readable explanation steps for an execution plan."""
    lines: list[str] = []

    # 1. Layer placement explanation
    gpu_layer_count = sum(len(layers) for layers in placement.gpu_layers.values())
    cpu_layer_count = len(placement.cpu_layers)
    disk_layer_count = len(placement.disk_streamed_layers)

    if gpu_layer_count == total_layers:
        gpu_vram_gb = sum(memory.vram_peak_bytes.values()) / (1024**3)
        lines.append(
            f"Full GPU Residency: All {total_layers} layers placed in VRAM ({gpu_vram_gb:.1f} GB peak)."
        )
    elif gpu_layer_count > 0:
        gpu_vram_gb = sum(memory.vram_peak_bytes.values()) / (1024**3)
        lines.append(
            f"Partial Offload: {gpu_layer_count}/{total_layers} layers on GPU ({gpu_vram_gb:.1f} GB VRAM); "
            f"{cpu_layer_count} layers on CPU ({memory.ram_peak_bytes / (1024**3):.1f} GB RAM)."
        )
    else:
        lines.append(
            f"CPU-Only: All {total_layers} layers placed in host RAM ({memory.ram_peak_bytes / (1024**3):.1f} GB RAM)."
        )

    if disk_layer_count > 0:
        lines.append(
            f"Disk Spill: {disk_layer_count} layers streamed from disk on demand ({memory.disk_read_bytes_per_token / (1024**2):.1f} MB/token)."
        )

    # 2. MoE explanation
    if model.is_moe and placement.expert_placement == "cpu":
        lines.append(
            f"MoE Optimized Placement: Attention & shared weights on GPU; {model.n_experts} routed experts placed in CPU RAM."
        )

    # 3. KV cache explanation
    kv_loc = "GPU VRAM" if kv.location == "gpu" else "Host RAM"
    lines.append(f"KV Cache: {kv.context} tokens ({kv.type_k}/{kv.type_v}) placed in {kv_loc}.")

    # 4. Bottleneck explanation
    bottleneck_descriptions = {
        "vram_bw": "GPU VRAM memory bus bandwidth",
        "ram_bw": "Host CPU system RAM memory bandwidth",
        "disk_bw": "Storage drive read bandwidth (streaming from disk)",
        "pcie_bw": "Host-to-Device PCIe bus bandwidth",
        "compute": "Processing core compute throughput",
        "expert_cache_miss": "Expert routing cache latency",
    }
    desc = bottleneck_descriptions.get(speed.bottleneck, speed.bottleneck)
    lines.append(f"Primary Bottleneck: {desc}.")

    return lines
