"""Mock backend adapter for testing, CI, and chaos fault injection."""

from ramllm.backends.base import (
    BackendAdapter,
    BackendInstall,
    BackendProcess,
    Capabilities,
    MemoryReport,
)
from ramllm.core.schemas import ExecutionPlan


class MockBackendAdapter(BackendAdapter):
    """Mock backend simulating process lifecycle, memory reporting, and fault injection."""

    name = "mock"

    def __init__(
        self,
        fail_launches: int = 0,
        fail_error: Exception | None = None,
        fail_if_gpu: bool = False,
        fail_if_gpu_layers_gt: int | None = None,
        fail_if_kv_f16: bool = False,
        fail_if_context_gt: int | None = None,
    ):
        self.fail_launches = fail_launches
        self.fail_error = fail_error
        self.fail_if_gpu = fail_if_gpu
        self.fail_if_gpu_layers_gt = fail_if_gpu_layers_gt
        self.fail_if_kv_f16 = fail_if_kv_f16
        self.fail_if_context_gt = fail_if_context_gt

        self.launch_count: int = 0
        self.launched_plans: list[ExecutionPlan] = []
        self.stopped: bool = False

    def detect(self) -> BackendInstall | None:
        return BackendInstall(
            name="mock",
            binary_path="/usr/local/bin/mock-server",
            version="1.0.0-mock",
            build_info="mock build",
            capabilities=Capabilities(
                partial_gpu_offload=True,
                per_tensor_placement=True,
                kv_on_gpu_or_cpu_choice=True,
                kv_quantization=True,
                multi_gpu_layer_split=True,
                apis=["cuda", "metal"],
            ),
        )

    def capabilities(self, install: BackendInstall) -> Capabilities:
        return install.capabilities

    def build_args(self, plan: ExecutionPlan, port: int = 8080) -> list[str]:
        return ["mock-server", *plan.backend_args, "--port", str(port)]

    def launch(
        self, plan: ExecutionPlan, port: int = 8080, timeout_s: float = 30.0
    ) -> BackendProcess:
        self.launch_count += 1
        self.launched_plans.append(plan)

        gpu_layer_count = sum(len(layers) for layers in plan.placement.gpu_layers.values())

        # Fault Injection: Driver Loss / GPU Unavailable
        if self.fail_if_gpu and gpu_layer_count > 0:
            raise self.fail_error or RuntimeError(
                "CUDA error: all CUDA-capable devices are busy or unavailable (driver reset)"
            )

        # Fault Injection: GPU VRAM limit
        if self.fail_if_gpu_layers_gt is not None and gpu_layer_count > self.fail_if_gpu_layers_gt:
            raise self.fail_error or RuntimeError(
                f"CUDA out of memory: failed to allocate {gpu_layer_count} layers in VRAM"
            )

        # Fault Injection: KV Cache allocation failure
        if self.fail_if_kv_f16 and plan.kv.type_k == "f16":
            raise self.fail_error or RuntimeError(
                "ggml_cuda_init: failed to allocate f16 KV cache buffer"
            )

        # Fault Injection: Context limit (Host OOM)
        if self.fail_if_context_gt is not None and plan.kv.context > self.fail_if_context_gt:
            raise self.fail_error or RuntimeError(
                f"std::bad_alloc: host memory exhausted for context window {plan.kv.context}"
            )

        # Fault Injection: N consecutive failures
        if self.fail_launches > 0:
            self.fail_launches -= 1
            raise self.fail_error or RuntimeError(
                "CUDA out of memory: failed to allocate 8.0 GiB buffer"
            )

        # Simulate initial memory report from plan estimates
        initial_mem = MemoryReport(
            vram_used_bytes=plan.memory.vram_peak_bytes,
            ram_rss_bytes=plan.memory.ram_peak_bytes,
            model_buffer_bytes=int(plan.memory.ram_peak_bytes * 0.8),
            kv_buffer_bytes=int(plan.memory.ram_peak_bytes * 0.15),
            compute_buffer_bytes=int(plan.memory.ram_peak_bytes * 0.05),
        )
        return BackendProcess(
            pid=99999,
            port=port,
            process=None,
            plan=plan,
            initial_memory_report=initial_mem,
        )

    def memory_report(self, proc: BackendProcess) -> MemoryReport:
        return proc.initial_memory_report

    def stop(self, proc: BackendProcess) -> None:
        self.stopped = True
