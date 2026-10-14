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

    # MoE CPU offload
    if plan.placement.expert_placement == "cpu":
        args.append("-cmoe")

    # Disk streaming / no-mmap
    if plan.placement.disk_streamed_layers:
        args.extend(["-lm", "none"])

    return args
