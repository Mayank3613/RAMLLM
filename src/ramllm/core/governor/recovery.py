from collections.abc import Callable
from enum import StrEnum

from ramllm.backends.base import BackendAdapter, BackendProcess
from ramllm.core.planner.feasibility import generate_remedies
from ramllm.core.planner.reduction import get_reduced_plan
from ramllm.core.schemas import ExecutionPlan, HardwareProfile, ModelInfo, UserRequest


class FailureType(StrEnum):
    """Classified failure category for backend launch or execution crashes."""

    GPU_OOM = "gpu_oom"
    HOST_OOM = "host_oom"
    DRIVER_LOSS = "driver_loss"
    TIMEOUT = "timeout"
    GENERAL_CRASH = "general_crash"


class RecoveryExhaustedError(Exception):
    """Raised when plan reduction attempts cannot recover execution or retries are disabled."""

    def __init__(
        self,
        message: str,
        failure_type: FailureType,
        attempted_plans: list[ExecutionPlan],
        remedies: list[str],
    ):
        super().__init__(message)
        self.message = message
        self.failure_type = failure_type
        self.attempted_plans = attempted_plans
        self.remedies = remedies


def classify_failure(
    exc_or_output: Exception | str, plan: ExecutionPlan | None = None
) -> FailureType:
    """Analyze backend exception or stderr output to determine root failure mode."""
    text = str(exc_or_output).lower()

    # 1. Driver loss / accelerator unavailable
    driver_patterns = [
        "driver reset",
        "device lost",
        "device unavailable",
        "devices are busy or unavailable",
        "cudadeviceunavailable",
        "cudaerrornodevice",
        "metal device failure",
        "driver shutting down",
        "rocm device not found",
    ]
    if any(p in text for p in driver_patterns):
        return FailureType.DRIVER_LOSS

    # 2. GPU OOM
    gpu_has_layers = bool(
        plan and plan.placement.gpu_layers and any(plan.placement.gpu_layers.values())
    )
    gpu_oom_patterns = [
        "cuda out of memory",
        "ggml_cuda_init: failed to allocate",
        "ggml_backend_metal_buffer_init",
        "vk_error_out_of_device_memory",
        "hiperrormemoryallocation",
        "cudaerrormemoryallocation",
        "failed to allocate buffer",
    ]
    if any(p in text for p in gpu_oom_patterns):
        return FailureType.GPU_OOM

    if "out of memory" in text and gpu_has_layers:
        return FailureType.GPU_OOM

    # 3. Host RAM OOM
    host_oom_patterns = [
        "std::bad_alloc",
        "cannot allocate memory",
        "cgroup memory limit",
    ]
    if any(p in text for p in host_oom_patterns):
        return FailureType.HOST_OOM

    if "out of memory" in text and not gpu_has_layers:
        return FailureType.HOST_OOM

    # 4. Timeout
    if isinstance(exc_or_output, TimeoutError) or "timed out" in text or "timeout" in text:
        return FailureType.TIMEOUT

    # Exit code 137 / SIGKILL
    if "code 137" in text or "code -9" in text:
        return FailureType.GPU_OOM if gpu_has_layers else FailureType.HOST_OOM

    return FailureType.GENERAL_CRASH


def format_failure_cause(exc_or_output: Exception | str, failure_type: FailureType) -> str:
    """Format a clean, human-readable description of the failure cause."""
    text = str(exc_or_output)
    match failure_type:
        case FailureType.DRIVER_LOSS:
            return "GPU accelerator or driver became unavailable (driver reset / device loss)."
        case FailureType.GPU_OOM:
            return "GPU memory allocation failed (VRAM Out of Memory)."
        case FailureType.HOST_OOM:
            return "System memory allocation failed (RAM Out of Memory / cgroup limit)."
        case FailureType.TIMEOUT:
            return "Backend server process failed to become ready within timeout window."
        case FailureType.GENERAL_CRASH:
            first_line = text.strip().split("\n")[0] if text else "Process exited with error"
            return f"Backend process crashed unexpectedly: {first_line[:120]}"


def launch_with_recovery(
    adapter: BackendAdapter,
    plan: ExecutionPlan,
    model: ModelInfo,
    hw: HardwareProfile,
    req: UserRequest,
    port: int = 8080,
    timeout_s: float = 30.0,
    max_retries: int = 3,
    auto_retry: bool = True,
    on_retry: Callable[[int, ExecutionPlan, str], None] | None = None,
) -> tuple[BackendProcess, ExecutionPlan, list[str]]:
    """Launch backend engine with automatic reactive adaptation on failure.

    If the initial plan crashes due to OOM or driver loss, systematically generates
    progressively smaller plans (fewer GPU layers, quantized KV, smaller context, CPU fallback)
    and relaunches until execution succeeds or retries are exhausted.
    """
    attempt = 0
    current_plan = plan
    attempted_plans: list[ExecutionPlan] = [current_plan]
    recovery_logs: list[str] = []

    while True:
        try:
            proc = adapter.launch(current_plan, port=port, timeout_s=timeout_s)
            return proc, current_plan, recovery_logs
        except Exception as exc:
            f_type = classify_failure(exc, current_plan)
            cause = format_failure_cause(exc, f_type)
            log_entry = f"Launch attempt {attempt + 1} with '{current_plan.id}' failed: {cause}"
            recovery_logs.append(log_entry)

            # Check if auto-recovery is allowed
            if not auto_retry:
                remedies = generate_remedies(
                    model=model,
                    req=req,
                    usable_ram=hw.ram_available_bytes,
                    usable_vram={g.index: g.vram_total_bytes for g in hw.gpus},
                    min_required_bytes=model.total_weight_bytes,
                )
                raise RecoveryExhaustedError(
                    message=f"Launch failed ({cause}). Auto-retry is disabled (--no-auto-retry).",
                    failure_type=f_type,
                    attempted_plans=attempted_plans,
                    remedies=remedies,
                ) from exc

            if attempt >= max_retries:
                remedies = generate_remedies(
                    model=model,
                    req=req,
                    usable_ram=hw.ram_available_bytes,
                    usable_vram={g.index: g.vram_total_bytes for g in hw.gpus},
                    min_required_bytes=model.total_weight_bytes,
                )
                raise RecoveryExhaustedError(
                    message=f"Failed to launch after {attempt + 1} recovery attempts. Last cause: {cause}",
                    failure_type=f_type,
                    attempted_plans=attempted_plans,
                    remedies=remedies,
                ) from exc

            attempt += 1
            next_plan = get_reduced_plan(
                failed_plan=current_plan,
                model=model,
                hw=hw,
                req=req,
                failure_type=f_type.value,
                attempt=attempt,
            )

            if next_plan is None or next_plan.verdict == "INFEASIBLE":
                remedies = generate_remedies(
                    model=model,
                    req=req,
                    usable_ram=hw.ram_available_bytes,
                    usable_vram={g.index: g.vram_total_bytes for g in hw.gpus},
                    min_required_bytes=model.total_weight_bytes,
                )
                raise RecoveryExhaustedError(
                    message=f"Plan reduction exhausted: no smaller viable plan could fit within constraints. Cause: {cause}",
                    failure_type=f_type,
                    attempted_plans=attempted_plans,
                    remedies=remedies,
                ) from exc

            attempted_plans.append(next_plan)

            if on_retry:
                on_retry(attempt, next_plan, cause)

            current_plan = next_plan
