"""Llama.cpp Backend Adapter implementation."""

import psutil

from ramllm.backends.base import (
    BackendAdapter,
    BackendInstall,
    BackendProcess,
    Capabilities,
    MemoryReport,
)
from ramllm.backends.llamacpp.flags import build_llamacpp_args
from ramllm.backends.llamacpp.install import detect_llamacpp
from ramllm.backends.llamacpp.launcher import launch_llamacpp_server
from ramllm.core.schemas import ExecutionPlan


class LlamaCppAdapter(BackendAdapter):
    """Execution engine adapter for llama.cpp via managed llama-server subprocess."""

    name = "llamacpp"

    def __init__(self, binary_path: str | None = None):
        self.binary_path = binary_path
        self._install: BackendInstall | None = None

    def detect(self) -> BackendInstall | None:
        self._install = detect_llamacpp(self.binary_path)
        return self._install

    def capabilities(self, install: BackendInstall) -> Capabilities:
        return install.capabilities

    def build_args(self, plan: ExecutionPlan, port: int = 8080) -> list[str]:
        if not self._install:
            self.detect()
        return build_llamacpp_args(plan, install=self._install, port=port)

    def launch(
        self, plan: ExecutionPlan, port: int = 8080, timeout_s: float = 30.0
    ) -> BackendProcess:
        cmd = self.build_args(plan, port=port)
        return launch_llamacpp_server(cmd, plan=plan, port=port, timeout_s=timeout_s)

    def memory_report(self, proc: BackendProcess) -> MemoryReport:
        rss = 0
        try:
            p = psutil.Process(proc.pid)
            rss = p.memory_info().rss
            for child in p.children(recursive=True):
                rss += child.memory_info().rss
        except Exception:
            rss = proc.initial_memory_report.ram_rss_bytes

        return MemoryReport(
            vram_used_bytes=proc.plan.memory.vram_peak_bytes,
            ram_rss_bytes=rss,
            model_buffer_bytes=proc.initial_memory_report.model_buffer_bytes,
            kv_buffer_bytes=proc.initial_memory_report.kv_buffer_bytes,
            compute_buffer_bytes=proc.initial_memory_report.compute_buffer_bytes,
        )

    def stop(self, proc: BackendProcess) -> None:
        if proc.process and proc.process.poll() is None:
            proc.process.terminate()
            try:
                proc.process.wait(timeout=3)
            except Exception:
                proc.process.kill()
