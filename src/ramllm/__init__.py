"""RAMLLM: RAM-adaptive planning and enforcement layer above llama.cpp."""

from ramllm.backends.llamacpp import LlamaCppAdapter
from ramllm.backends.mock import MockBackendAdapter
from ramllm.core.analyzer.gguf import analyze_gguf
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.planner import create_execution_plans
from ramllm.core.profiler import probe_hardware
from ramllm.core.schemas import ExecutionPlan, HardwareProfile, ModelInfo, UserRequest

__version__ = "0.2.1"

__all__ = [
    "ExecutionPlan",
    "HardwareProfile",
    "LlamaCppAdapter",
    "MemoryGovernor",
    "MockBackendAdapter",
    "ModelInfo",
    "UserRequest",
    "__version__",
    "analyze_gguf",
    "create_execution_plans",
    "probe_hardware",
]
