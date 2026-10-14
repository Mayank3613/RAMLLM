"""Mock backend adapter for testing and CI."""

from ramllm.backends.base import (
    BackendAdapter,
    BackendInstall,
    BackendProcess,
    Capabilities,
    MemoryReport,
)
from ramllm.core.schemas import ExecutionPlan


class MockBackendAdapter(BackendAdapter):
    """Mock backend simulating process lifecycle and memory reporting."""

    name = "mock"

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
        pass
