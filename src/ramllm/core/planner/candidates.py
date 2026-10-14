"""Candidate plan generation across hardware tiers and placement strategies."""

from typing import Literal

from ramllm.core.analyzer.kv import estimate_compute_buffer_bytes, kv_cache_bytes
from ramllm.core.analyzer.moe import calculate_moe_working_set_bytes
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


def _build_backend_args(
    model_path: str,
    ngl: int,
    ctx: int,
    type_k: str,
    type_v: str,
    cpu_moe: bool = False,
    no_kv_offload: bool = False,
    mmap: bool = True,
) -> list[str]:
    """Generate concrete llama-server / llama-cli CLI arguments."""
    args = [
        "-m",
        model_path,
        "-ngl",
        str(ngl),
        "-c",
        str(ctx),
    ]
    if type_k != "f16":
        args.extend(["-ctk", type_k])
    if type_v != "f16":
        args.extend(["-ctv", type_v])
    if cpu_moe:
        args.append("-cmoe")
    if no_kv_offload:
        args.append("--no-kv-offload")
    if not mmap:
        args.extend(["-lm", "none"])
    return args


def generate_candidate_plans(
    model: ModelInfo,
    hw: HardwareProfile,
    req: UserRequest,
    usable_ram: int,
    usable_vram: dict[int, int],
) -> list[ExecutionPlan]:
    """Generate viable execution candidates matching the user request and hardware constraints."""
    plans: list[ExecutionPlan] = []
    total_layers = model.n_layers
    target_ctx = req.target_context or req.min_context
    compute_buffer_bytes = estimate_compute_buffer_bytes(
        target_ctx, batch_size=512, hidden_size=model.hidden_size
    )

    # Allowed KV types
    kv_types: list[Literal["f16", "q8_0", "q4_0"]] = ["f16"]
    if req.allow_kv_quant:
        kv_types.extend(["q8_0", "q4_0"])

    total_usable_vram = sum(usable_vram.values())
    primary_gpu_idx = min(usable_vram.keys()) if usable_vram else 0
    primary_gpu_vram = usable_vram.get(primary_gpu_idx, 0)

    # Strategy 1: Full GPU Residency
    if req.use_gpu and usable_vram:
        for kv_t in kv_types:
            kv_footprint = kv_cache_bytes(
                n_layers=total_layers,
                context_length=target_ctx,
                n_kv_heads=model.n_kv_heads,
                head_dim=model.head_dim,
                type_k=kv_t,
                type_v=kv_t,
            )
            vram_needed = model.total_weight_bytes + kv_footprint + compute_buffer_bytes

            if vram_needed <= primary_gpu_vram:
                # Full single GPU fit
                placement = Placement(
                    gpu_layers={primary_gpu_idx: list(range(total_layers))},
                    cpu_layers=[],
                    disk_streamed_layers=[],
                )
                kv_cfg = KvConfig(location="gpu", type_k=kv_t, type_v=kv_t, context=target_ctx)
                mem = MemoryEstimate(
                    vram_peak_bytes={primary_gpu_idx: vram_needed},
                    ram_peak_bytes=int(500 * 1024 * 1024),  # Host process RSS baseline
                )
                working_set_gpu = (
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
                    bytes_gpu=working_set_gpu,
                    bytes_cpu=0,
                    bytes_disk=0,
                    bytes_xfer_h2d=0,
                    kv_read_bytes_gpu=kv_footprint,
                    kv_read_bytes_cpu=0,
                    active_params=model.n_params,
                )
                verdict = classify_verdict(
                    speed.decode_tok_s_low, speed.decode_tok_s_high, False, True
                )
                args = _build_backend_args(
                    model.path, ngl=total_layers, ctx=target_ctx, type_k=kv_t, type_v=kv_t
                )
                expl = explain_plan(model, placement, kv_cfg, mem, speed, total_layers)

                plans.append(
                    ExecutionPlan(
                        id=f"full_gpu_{kv_t}",
                        model_variant=model.quant,
                        backend="llamacpp",
                        placement=placement,
                        kv=kv_cfg,
                        memory=mem,
                        speed=speed,
                        verdict=verdict,
                        backend_args=args,
                        explanation=expl,
                    )
                )

    # Strategy 2: Partial Offload (GPU + CPU)
    if req.use_gpu and usable_vram and total_usable_vram > 0:
        for kv_t in kv_types:
            for kv_loc in ["gpu", "cpu"]:
                kv_footprint = kv_cache_bytes(
                    n_layers=total_layers,
                    context_length=target_ctx,
                    n_kv_heads=model.n_kv_heads,
                    head_dim=model.head_dim,
                    type_k=kv_t,  # type: ignore[arg-type]
                    type_v=kv_t,  # type: ignore[arg-type]
                )
                # Compute how many layers fit in VRAM
                available_for_layers = primary_gpu_vram - compute_buffer_bytes
                if kv_loc == "gpu":
                    available_for_layers -= kv_footprint
                available_for_layers -= model.non_layer_bytes

                if available_for_layers <= 0:
                    continue

                # Count how many layers can be placed in VRAM
                accumulated = 0
                gpu_layers: list[int] = []
                for l_idx, l_bytes in enumerate(model.per_layer_bytes):
                    if accumulated + l_bytes <= available_for_layers:
                        accumulated += l_bytes
                        gpu_layers.append(l_idx)
                    else:
                        break

                if not gpu_layers or len(gpu_layers) == total_layers:
                    continue

                cpu_layers = [i for i in range(total_layers) if i not in gpu_layers]
                cpu_weight_bytes = sum(model.per_layer_bytes[i] for i in cpu_layers)
                ram_needed = (
                    cpu_weight_bytes
                    + (kv_footprint if kv_loc == "cpu" else 0)
                    + int(500 * 1024 * 1024)
                )

                fits_budget = ram_needed <= usable_ram
                has_disk_spill = False
                disk_streamed: list[int] = []

                if not fits_budget:
                    if req.allow_streaming:
                        has_disk_spill = True
                        disk_streamed = cpu_layers
                    else:
                        continue

                placement = Placement(
                    gpu_layers={primary_gpu_idx: gpu_layers},
                    cpu_layers=[] if has_disk_spill else cpu_layers,
                    disk_streamed_layers=disk_streamed,
                )
                kv_cfg = KvConfig(
                    location=kv_loc,  # type: ignore[arg-type]
                    type_k=kv_t,
                    type_v=kv_t,
                    context=target_ctx,
                )
                mem = MemoryEstimate(
                    vram_peak_bytes={
                        primary_gpu_idx: int(
                            accumulated
                            + model.non_layer_bytes
                            + (kv_footprint if kv_loc == "gpu" else 0)
                            + compute_buffer_bytes
                        )
                    },
                    ram_peak_bytes=ram_needed if not has_disk_spill else int(1.5 * 1024**3),
                    disk_read_bytes_per_token=cpu_weight_bytes if has_disk_spill else 0,
                )
                speed = estimate_speed(
                    hw=hw,
                    bytes_gpu=accumulated,
                    bytes_cpu=0 if has_disk_spill else cpu_weight_bytes,
                    bytes_disk=cpu_weight_bytes if has_disk_spill else 0,
                    bytes_xfer_h2d=0,
                    kv_read_bytes_gpu=kv_footprint if kv_loc == "gpu" else 0,
                    kv_read_bytes_cpu=kv_footprint if kv_loc == "cpu" else 0,
                    active_params=model.n_params,
                )
                verdict = classify_verdict(
                    speed.decode_tok_s_low, speed.decode_tok_s_high, has_disk_spill, True
                )
                args = _build_backend_args(
                    model.path,
                    ngl=len(gpu_layers),
                    ctx=target_ctx,
                    type_k=kv_t,
                    type_v=kv_t,
                    no_kv_offload=(kv_loc == "cpu"),
                )
                expl = explain_plan(model, placement, kv_cfg, mem, speed, total_layers)

                plans.append(
                    ExecutionPlan(
                        id=f"partial_offload_{len(gpu_layers)}L_{kv_t}_{kv_loc}",
                        model_variant=model.quant,
                        backend="llamacpp",
                        placement=placement,
                        kv=kv_cfg,
                        memory=mem,
                        speed=speed,
                        verdict=verdict,
                        backend_args=args,
                        explanation=expl,
                    )
                )

    # Strategy 3: MoE CPU Offload (Attention + Shared on GPU, Routed Experts on CPU)
    if model.is_moe and req.use_gpu and usable_vram and model.expert_bytes_per_layer:
        # Non-expert weights + shared expert weights on GPU
        shared_bytes = sum(model.shared_expert_bytes_per_layer or [0])
        expert_total_bytes = sum(model.expert_bytes_per_layer)
        non_expert_total = model.total_weight_bytes - expert_total_bytes
        vram_needed_moe = non_expert_total + shared_bytes + compute_buffer_bytes

        if vram_needed_moe <= primary_gpu_vram:
            kv_footprint = kv_cache_bytes(
                n_layers=total_layers,
                context_length=target_ctx,
                n_kv_heads=model.n_kv_heads,
                head_dim=model.head_dim,
                type_k="q8_0" if req.allow_kv_quant else "f16",
                type_v="q8_0" if req.allow_kv_quant else "f16",
            )
            ram_needed = expert_total_bytes + int(500 * 1024 * 1024)

            if ram_needed <= usable_ram:
                placement = Placement(
                    gpu_layers={primary_gpu_idx: list(range(total_layers))},
                    cpu_layers=list(range(total_layers)),
                    expert_placement="cpu",
                )
                kv_cfg = KvConfig(
                    location="gpu",
                    type_k="q8_0" if req.allow_kv_quant else "f16",
                    type_v="q8_0" if req.allow_kv_quant else "f16",
                    context=target_ctx,
                )
                mem = MemoryEstimate(
                    vram_peak_bytes={primary_gpu_idx: vram_needed_moe + kv_footprint},
                    ram_peak_bytes=ram_needed,
                )
                active_ratio = (
                    (model.n_experts_used / model.n_experts)
                    if model.n_experts and model.n_experts_used
                    else 0.25
                )
                active_expert_bytes = int(expert_total_bytes * active_ratio)

                speed = estimate_speed(
                    hw=hw,
                    bytes_gpu=non_expert_total,
                    bytes_cpu=active_expert_bytes,
                    bytes_disk=0,
                    bytes_xfer_h2d=0,
                    kv_read_bytes_gpu=kv_footprint,
                    kv_read_bytes_cpu=0,
                    active_params=model.n_params,
                )
                verdict = classify_verdict(
                    speed.decode_tok_s_low, speed.decode_tok_s_high, False, True
                )
                args = _build_backend_args(
                    model.path,
                    ngl=total_layers,
                    ctx=target_ctx,
                    type_k=kv_cfg.type_k,
                    type_v=kv_cfg.type_v,
                    cpu_moe=True,
                )
                expl = explain_plan(model, placement, kv_cfg, mem, speed, total_layers)

                plans.append(
                    ExecutionPlan(
                        id="moe_cpu_routed_experts",
                        model_variant=model.quant,
                        backend="llamacpp",
                        placement=placement,
                        kv=kv_cfg,
                        memory=mem,
                        speed=speed,
                        verdict=verdict,
                        backend_args=args,
                        explanation=expl,
                    )
                )

    # Strategy 4: Pure CPU Fallback
    for kv_t in kv_types:
        kv_footprint = kv_cache_bytes(
            n_layers=total_layers,
            context_length=target_ctx,
            n_kv_heads=model.n_kv_heads,
            head_dim=model.head_dim,
            type_k=kv_t,
            type_v=kv_t,
        )
        total_ram_needed = model.total_weight_bytes + kv_footprint + compute_buffer_bytes
        fits_ram = total_ram_needed <= usable_ram
        has_disk = False
        disk_streamed_layers: list[int] = []

        if not fits_ram:
            if req.allow_streaming:
                has_disk = True
                disk_streamed_layers = list(range(total_layers))
            else:
                continue

        placement = Placement(
            gpu_layers={},
            cpu_layers=[] if has_disk else list(range(total_layers)),
            disk_streamed_layers=disk_streamed_layers,
        )
        kv_cfg = KvConfig(location="cpu", type_k=kv_t, type_v=kv_t, context=target_ctx)
        mem = MemoryEstimate(
            vram_peak_bytes={},
            ram_peak_bytes=total_ram_needed if not has_disk else int(1.5 * 1024**3),
            disk_read_bytes_per_token=model.total_weight_bytes if has_disk else 0,
        )
        active_weight_bytes = (
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
            bytes_cpu=0 if has_disk else active_weight_bytes,
            bytes_disk=active_weight_bytes if has_disk else 0,
            bytes_xfer_h2d=0,
            kv_read_bytes_gpu=0,
            kv_read_bytes_cpu=kv_footprint,
            active_params=model.n_params,
        )
        verdict = classify_verdict(
            speed.decode_tok_s_low, speed.decode_tok_s_high, has_disk, fits_ram or has_disk
        )
        args = _build_backend_args(
            model.path,
            ngl=0,
            ctx=target_ctx,
            type_k=kv_t,
            type_v=kv_t,
            no_kv_offload=True,
            mmap=not has_disk,
        )
        expl = explain_plan(model, placement, kv_cfg, mem, speed, total_layers)

        plans.append(
            ExecutionPlan(
                id=f"cpu_only_{kv_t}" if not has_disk else f"disk_spill_{kv_t}",
                model_variant=model.quant,
                backend="llamacpp",
                placement=placement,
                kv=kv_cfg,
                memory=mem,
                speed=speed,
                verdict=verdict,
                backend_args=args,
                explanation=expl,
            )
        )

    return plans
