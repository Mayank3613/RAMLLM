"""Abstract backend interface, capabilities, and process models."""

import subprocess
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from ramllm.core.schemas import ExecutionPlan


class Capabilities(BaseModel):
    """Features supported by a specific backend installation."""

    model_config = ConfigDict(frozen=True)

    partial_gpu_offload: bool = True
    per_tensor_placement: bool = False
    kv_on_gpu_or_cpu_choice: bool = True
    kv_quantization: bool = True
    multi_gpu_layer_split: bool = True
    multi_gpu_tensor_split: bool = False
    flash_attention: bool = True
    mmap_control: bool = True
    async_weight_streaming: bool = False
    expert_level_eviction: bool = False
    apis: list[str] = Field(default_factory=list)


class BackendInstall(BaseModel):
    """Metadata describing a discovered backend binary on the system."""

    model_config = ConfigDict(frozen=True)

    name: str
    binary_path: str
    version: str | None = None
    build_info: str | None = None
    capabilities: Capabilities = Field(default_factory=Capabilities)


class MemoryReport(BaseModel):
    """Reported peak and current memory usage from runtime process sampling and logs."""

    vram_used_bytes: dict[int, int] = Field(default_factory=dict)
    ram_rss_bytes: int = 0
    model_buffer_bytes: int = 0
    kv_buffer_bytes: int = 0
    compute_buffer_bytes: int = 0


class BackendProcess(BaseModel):
    """Handle to a running backend server process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pid: int
    port: int
    process: subprocess.Popen[str] | None = None
    plan: ExecutionPlan
    initial_memory_report: MemoryReport = Field(default_factory=MemoryReport)


class BackendAdapter(ABC):
    """Abstract interface for all RAMLLM execution engines."""

    name: str

    @abstractmethod
    def detect(self) -> BackendInstall | None:
        """Discover backend binary on system PATH or configured locations."""
        ...

    @abstractmethod
    def capabilities(self, install: BackendInstall) -> Capabilities:
        """Inspect backend capabilities from its binary help output."""
        ...

    @abstractmethod
    def build_args(self, plan: ExecutionPlan, port: int = 8080) -> list[str]:
        """Convert an ExecutionPlan into concrete command-line arguments."""
        ...

    @abstractmethod
    def launch(
        self, plan: ExecutionPlan, port: int = 8080, timeout_s: float = 30.0
    ) -> BackendProcess:
        """Launch the backend process, await healthiness, and extract parsed buffer sizes."""
        ...

    @abstractmethod
    def memory_report(self, proc: BackendProcess) -> MemoryReport:
        """Collect current memory consumption across hardware tiers."""
        ...

    @abstractmethod
    def stop(self, proc: BackendProcess) -> None:
        """Gracefully terminate the backend process."""
        ...
