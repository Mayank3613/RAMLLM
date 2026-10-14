from typing import Literal

from ramllm.core.analyzer.kv import estimate_compute_buffer_bytes, kv_cache_bytes
from ramllm.core.analyzer.moe import calculate_moe_working_set_bytes
from ramllm.core.planner.candidates import _build_backend_args
from ramllm.core.planner.cost_model import estimate_speed
from ramllm.core.planner.explain import explain_plan
from ramllm.core.planner.feasibility import classify_verdict
from ramllm.core.schemas import (
    ExecutionPlan,
    HardwareProfile,
    KvConfig,
    MemoryEstimate,
    ModelInfo,
    Placement,
    UserRequest,
)
from ramllm.telemetry.calibration import get_calibration_factor


def get_reduced_plan(
    failed_plan: ExecutionPlan,
    model: ModelInfo,
    hw: HardwareProfile,
    req: UserRequest,
    failure_type: str = "gpu_oom",
    attempt: int = 1,
) -> ExecutionPlan | None:
    """Generate a strictly smaller, more conservative ExecutionPlan following a failure.

    Reduction hierarchy (Milestone 5 Stage C):
    1. GPU Driver Loss / Reset -> Fallback immediately to pure CPU execution.
    2. GPU OOM -> Decrease GPU layers by ~25-50% -> quantize KV (f16 -> q8_0 -> q4_0)
       -> shrink context window -> drop to pure CPU fallback.
    3. Host OOM -> Quantize KV to q4_0 -> shrink context window towards min_context.
    """
    total_layers = model.n_layers
    curr_gpu_layers = sum(len(layers) for layers in failed_plan.placement.gpu_layers.values())
    curr_kv_type = failed_plan.kv.type_k
    curr_ctx = failed_plan.kv.context
    curr_kv_loc = failed_plan.kv.location
    curr_moe_expert_placement = failed_plan.placement.expert_placement
    primary_gpu_idx = min(hw.gpus, key=lambda g: g.index).index if hw.gpus else 0
    calib_factor = get_calibration_factor(model.path)

    # Strategy 1: Driver loss / GPU unavailable -> Immediate CPU fallback
    if failure_type in ("driver_loss", "gpu_unavailable"):
        if curr_gpu_layers == 0 and failed_plan.verdict == "INFEASIBLE":
            return None

        target_ctx = curr_ctx
        kv_t: Literal["f16", "q8_0", "q4_0"] = "q8_0" if req.allow_kv_quant else "f16"
        kv_footprint = kv_cache_bytes(
            n_layers=total_layers,
            context_length=target_ctx,
            n_kv_heads=model.n_kv_heads,
            head_dim=model.head_dim,
            type_k=kv_t,
            type_v=kv_t,
        )
        compute_buffer_bytes = estimate_compute_buffer_bytes(
            target_ctx, batch_size=512, hidden_size=model.hidden_size
        )
        total_ram = model.total_weight_bytes + kv_footprint + compute_buffer_bytes

        placement = Placement(
            gpu_layers={},
            cpu_layers=list(range(total_layers)),
            disk_streamed_layers=[],
            expert_placement="cpu" if model.is_moe else None,
        )
        kv_cfg = KvConfig(location="cpu", type_k=kv_t, type_v=kv_t, context=target_ctx)
        mem = MemoryEstimate(
            vram_peak_bytes={},
            ram_peak_bytes=total_ram,
            disk_read_bytes_per_token=0,
        )
        working_set = (
            calculate_moe_working_set_bytes(
                model.non_layer_bytes,
                model.per_layer_bytes,
                model.expert_bytes_per_layer,
                model.shared_expert_bytes_per_layer,
                model.n_experts,
                model.n_experts_used,
            )
            if model.is_moe
            else model.total_weight_bytes
        )
        speed = estimate_speed(
            hw=hw,
            bytes_gpu=0,
            bytes_cpu=working_set,
            bytes_disk=0,
            bytes_xfer_h2d=0,
            kv_read_bytes_gpu=0,
            kv_read_bytes_cpu=kv_footprint,
            active_params=model.n_params,
            calibration_factor=calib_factor,
        )
        verdict = classify_verdict(
            speed.decode_tok_s_low, speed.decode_tok_s_high, False, total_ram <= hw.ram_total_bytes
        )
        args = _build_backend_args(
            model.path,
            ngl=0,
            ctx=target_ctx,
            type_k=kv_t,
            type_v=kv_t,
            no_kv_offload=True,
            cpu_moe=model.is_moe,
        )
        expl = explain_plan(model, placement, kv_cfg, mem, speed, total_layers)
        warnings = [
            "Advisory: GPU acceleration became unavailable (driver reset/device lost). Fallen back to pure CPU execution."
        ]

        return ExecutionPlan(
            id=f"cpu_fallback_v{attempt}",
            model_variant=model.quant,
            backend="llamacpp",
            placement=placement,
            kv=kv_cfg,
            memory=mem,
            speed=speed,
            verdict=verdict,
            warnings=warnings,
            backend_args=args,
            explanation=expl,
        )

    # Strategy 2: GPU OOM or General Load Failure with active GPU layers
    # Progress through reduction steps:
    new_gpu_layers = curr_gpu_layers
    new_kv_type = curr_kv_type
    new_ctx = curr_ctx
    new_kv_loc = curr_kv_loc
    new_expert_placement = curr_moe_expert_placement
    action_notes: list[str] = []

    # Step A: If MoE and experts are still on GPU, offload routed experts to CPU first
    if model.is_moe and new_expert_placement != "cpu" and failure_type == "gpu_oom":
        new_expert_placement = "cpu"
        action_notes.append("Offloaded MoE routed experts to CPU (-cmoe)")

    # Step B: Decrease GPU layers
    elif new_gpu_layers > 0:
        if attempt == 1:
            drop = max(2, int(new_gpu_layers * 0.25))
            new_gpu_layers = max(0, new_gpu_layers - drop)
            action_notes.append(f"Reduced GPU layers from {curr_gpu_layers} to {new_gpu_layers}")
        elif attempt == 2:
            drop = max(2, int(new_gpu_layers * 0.50))
            new_gpu_layers = max(0, new_gpu_layers - drop)
            action_notes.append(f"Further reduced GPU layers to {new_gpu_layers}")
            # Also quantize KV if f16
            if new_kv_type == "f16":
                new_kv_type = "q8_0"
                action_notes.append("Quantized KV cache to q8_0")
        else:
            # Fall back to 0 GPU layers (CPU fallback)
            new_gpu_layers = 0
            action_notes.append("Dropped all GPU layers to pure CPU fallback")
            if new_kv_type in ("f16", "q8_0"):
                new_kv_type = "q4_0"
                action_notes.append("Quantized KV cache to q4_0")
            if new_ctx > req.min_context:
                new_ctx = max(req.min_context, new_ctx // 2)
                action_notes.append(f"Halved context length to {new_ctx}")

    # Step C: If already 0 GPU layers (Host OOM or pure CPU OOM)
    else:
        if new_kv_type == "f16":
            new_kv_type = "q8_0"
            action_notes.append("Quantized KV cache to q8_0")
        elif new_kv_type == "q8_0":
            new_kv_type = "q4_0"
            action_notes.append("Quantized KV cache to q4_0")
        elif new_ctx > req.min_context:
            new_ctx = max(req.min_context, new_ctx // 2)
            action_notes.append(f"Halved context length to {new_ctx}")
        elif new_ctx > 512:
            new_ctx = max(512, new_ctx // 2)
            action_notes.append(f"Shrunk context window below minimum to {new_ctx}")
        else:
            # Cannot reduce further
            return None

    # Check if anything changed
    if (
        new_gpu_layers == curr_gpu_layers
        and new_kv_type == curr_kv_type
        and new_ctx == curr_ctx
        and new_expert_placement == curr_moe_expert_placement
    ):
        return None

    # Reconstruct Placement
    gpu_layers_map: dict[int, list[int]] = {}
    cpu_layers: list[int] = []

    if new_gpu_layers > 0:
        # Distribute across GPUs matching failed plan proportion or primary GPU
        existing_gpus = sorted(failed_plan.placement.gpu_layers.keys())
        if not existing_gpus:
            existing_gpus = [primary_gpu_idx]

        if len(existing_gpus) == 1:
            gpu_idx = existing_gpus[0]
            gpu_layers_map[gpu_idx] = list(range(new_gpu_layers))
            cpu_layers = list(range(new_gpu_layers, total_layers))
        else:
            # Multi-GPU proportional split
            total_failed_ngl = max(1, curr_gpu_layers)
            assigned_count = 0
            for g_idx in existing_gpus:
                prev_count = len(failed_plan.placement.gpu_layers.get(g_idx, []))
                ratio = prev_count / total_failed_ngl
                g_count = int(round(new_gpu_layers * ratio))
                if assigned_count + g_count > new_gpu_layers:
                    g_count = max(0, new_gpu_layers - assigned_count)
                g_layers = list(range(assigned_count, assigned_count + g_count))
                if g_layers:
                    gpu_layers_map[g_idx] = g_layers
                assigned_count += g_count

            # Any remaining up to new_gpu_layers go to last GPU
            if assigned_count < new_gpu_layers and existing_gpus:
                last_gpu = existing_gpus[-1]
                extra = list(range(assigned_count, new_gpu_layers))
                gpu_layers_map.setdefault(last_gpu, []).extend(extra)
                assigned_count = new_gpu_layers

            cpu_layers = list(range(assigned_count, total_layers))
    else:
        cpu_layers = list(range(total_layers))
        new_kv_loc = "cpu"

    placement = Placement(
        gpu_layers=gpu_layers_map,
        cpu_layers=cpu_layers,
        disk_streamed_layers=[],
        expert_placement=new_expert_placement,
    )

    kv_cfg = KvConfig(
        location=new_kv_loc,
        type_k=new_kv_type,  # type: ignore[arg-type]
        type_v=new_kv_type,  # type: ignore[arg-type]
        context=new_ctx,
    )

    # Compute exact memory buffers
    kv_footprint = kv_cache_bytes(
        n_layers=total_layers,
        context_length=new_ctx,
        n_kv_heads=model.n_kv_heads,
        head_dim=model.head_dim,
        type_k=new_kv_type,  # type: ignore[arg-type]
        type_v=new_kv_type,  # type: ignore[arg-type]
    )
    compute_buffer_bytes = estimate_compute_buffer_bytes(
        new_ctx, batch_size=512, hidden_size=model.hidden_size
    )

    vram_peak_map: dict[int, int] = {}
    for g_idx, layers_list in gpu_layers_map.items():
        l_bytes = sum(model.per_layer_bytes[i] for i in layers_list)
        non_l = model.non_layer_bytes if g_idx == primary_gpu_idx else 0
        kv_b = kv_footprint if (new_kv_loc == "gpu" and g_idx == primary_gpu_idx) else 0
        vram_peak_map[g_idx] = l_bytes + non_l + kv_b + compute_buffer_bytes

    cpu_layer_bytes = sum(model.per_layer_bytes[i] for i in cpu_layers)
    ram_peak = (
        cpu_layer_bytes
        + (kv_footprint if new_kv_loc == "cpu" else 0)
        + (model.non_layer_bytes if not gpu_layers_map else 0)
        + int(500 * 1024 * 1024)
    )

    mem = MemoryEstimate(
        vram_peak_bytes=vram_peak_map,
        ram_peak_bytes=ram_peak,
        disk_read_bytes_per_token=0,
    )

    # Compute active speed
    if model.is_moe and model.expert_bytes_per_layer and model.n_experts and model.n_experts_used:
        active_ratio = model.n_experts_used / model.n_experts
        exp_bytes_list = model.expert_bytes_per_layer

        def _layer_working_set(i: int) -> int:
            l_total = model.per_layer_bytes[i]
            exp_bytes = exp_bytes_list[i] if i < len(exp_bytes_list) else 0
            return int((l_total - exp_bytes) + exp_bytes * active_ratio)

        active_gpu_bytes = sum(
            sum(_layer_working_set(i) for i in layers) for layers in gpu_layers_map.values()
        )
        active_cpu_bytes = sum(_layer_working_set(i) for i in cpu_layers)
    else:
        active_gpu_bytes = sum(
            sum(model.per_layer_bytes[i] for i in layers) for layers in gpu_layers_map.values()
        )
        active_cpu_bytes = sum(model.per_layer_bytes[i] for i in cpu_layers)

    speed = estimate_speed(
        hw=hw,
        bytes_gpu=active_gpu_bytes,
        bytes_cpu=active_cpu_bytes,
        bytes_disk=0,
        bytes_xfer_h2d=0,
        kv_read_bytes_gpu=kv_footprint if new_kv_loc == "gpu" else 0,
        kv_read_bytes_cpu=kv_footprint if new_kv_loc == "cpu" else 0,
        active_params=model.n_params,
        calibration_factor=calib_factor,
    )

    fits = ram_peak <= hw.ram_total_bytes
    verdict = classify_verdict(speed.decode_tok_s_low, speed.decode_tok_s_high, False, fits)

    # Build backend args
    multi_gpu_ts: str | None = None
    if len(gpu_layers_map) > 1 and new_gpu_layers > 0:
        max_idx = max(gpu_layers_map.keys())
        fractions = [
            f"{len(gpu_layers_map.get(idx, [])) / new_gpu_layers:.2f}" for idx in range(max_idx + 1)
        ]
        multi_gpu_ts = ",".join(fractions)

    args = _build_backend_args(
        model_path=model.path,
        ngl=new_gpu_layers,
        ctx=new_ctx,
        type_k=new_kv_type,
        type_v=new_kv_type,
        cpu_moe=new_expert_placement == "cpu",
        no_kv_offload=new_kv_loc == "cpu",
        multi_gpu_ts=multi_gpu_ts,
        main_gpu=primary_gpu_idx if len(gpu_layers_map) > 1 else None,
    )

    expl = explain_plan(model, placement, kv_cfg, mem, speed, total_layers)
    warnings = [
        f"Auto-recovery (attempt {attempt}): {', '.join(action_notes)} due to {failure_type}."
    ]

    return ExecutionPlan(
        id=f"{failed_plan.id}_reduced_v{attempt}",
        model_variant=model.quant,
        backend="llamacpp",
        placement=placement,
        kv=kv_cfg,
        memory=mem,
        speed=speed,
        verdict=verdict,
        warnings=warnings,
        backend_args=args,
        explanation=expl,
    )
