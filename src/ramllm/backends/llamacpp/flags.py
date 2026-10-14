"""Flag translation from ExecutionPlan to llama-server CLI arguments."""

from ramllm.backends.base import BackendInstall
from ramllm.core.schemas import ExecutionPlan


def build_llamacpp_args(
    plan: ExecutionPlan,
    install: BackendInstall | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> list[str]:
    """Map an ExecutionPlan to concrete arguments for llama-server."""
    binary = install.binary_path if install else "llama-server"
    args = [binary, "--host", host, "--port", str(port)]

    # Extract model path from backend_args in plan
    model_path = ""
    for i, a in enumerate(plan.backend_args):
        if a in ("-m", "--model") and i + 1 < len(plan.backend_args):
            model_path = plan.backend_args[i + 1]
            break

    if model_path:
        args.extend(["-m", model_path])

    # GPU layers
    gpu_layer_count = sum(len(layer_list) for layer_list in plan.placement.gpu_layers.values())
    args.extend(["-ngl", str(gpu_layer_count)])

    # Context size
    args.extend(["-c", str(plan.kv.context)])

    # Predictable single-slot memory usage
    args.extend(["-np", "1"])

    # KV quantization
    if plan.kv.type_k != "f16":
        args.extend(["-ctk", plan.kv.type_k])
    if plan.kv.type_v != "f16":
        args.extend(["-ctv", plan.kv.type_v])

    # KV offload to CPU if requested
    if plan.kv.location == "cpu":
        args.append("--no-kv-offload")

    # Multi-GPU split
    if len(plan.placement.gpu_layers) > 1:
        total_offloaded = sum(len(layers) for layers in plan.placement.gpu_layers.values())
        if total_offloaded > 0:
            gpu_indices = sorted(plan.placement.gpu_layers.keys())
            max_gpu_idx = max(gpu_indices)
            fractions = [
                f"{len(plan.placement.gpu_layers.get(idx, [])) / total_offloaded:.2f}"
                for idx in range(max_gpu_idx + 1)
            ]
            main_gpu = gpu_indices[0]
            for i, a in enumerate(plan.backend_args):
                if a in ("-mg", "--main-gpu") and i + 1 < len(plan.backend_args):
                    main_gpu = int(plan.backend_args[i + 1])
                    break
            args.extend(["-sm", "layer", "-ts", ",".join(fractions), "-mg", str(main_gpu)])

    # MoE CPU offload flags
    if plan.placement.expert_placement == "cpu":
        args.append("-cmoe")

    # Partial MoE CPU offload (-ncmoe)
    for i, a in enumerate(plan.backend_args):
        if a in ("-ncmoe", "--n-cpu-moe") and i + 1 < len(plan.backend_args):
            args.extend(["-ncmoe", plan.backend_args[i + 1]])
            break

    # Disk streaming / no-mmap
    if plan.placement.disk_streamed_layers:
        args.extend(["-lm", "none"])

    return args
