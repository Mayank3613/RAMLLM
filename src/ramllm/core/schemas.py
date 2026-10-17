"""Normative data models for RAMLLM.

All models use Pydantic v2 with strict validation and immutability where appropriate.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GpuInfo(BaseModel):
    """Information and metrics for a physical or integrated GPU accelerator."""

    model_config = ConfigDict(frozen=True)

    index: int
    vendor: Literal["nvidia", "amd", "apple", "intel", "other"]
    name: str
    api_support: list[Literal["cuda", "rocm", "metal", "vulkan", "sycl"]]
    vram_total_bytes: int
    vram_free_bytes: int
    compute_capability: str | None = None
    unified_memory: bool = False
    pcie_gen: int | None = None
    pcie_lanes: int | None = None
    h2d_bandwidth_bytes_s: float | None = None
    vram_bandwidth_bytes_s: float | None = None


class HardwareProfile(BaseModel):
    """System hardware topology, memory headroom, and measured bandwidth."""

    model_config = ConfigDict(frozen=True)

    os: Literal["linux", "macos", "windows"]
    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_features: list[str]
    ram_total_bytes: int
    ram_available_bytes: int
    swap_total_bytes: int
    ram_bandwidth_bytes_s: float
    ram_bandwidth_memcpy_bytes_s: float | None = None
    cpu_effective_gflops: float | None = None
    disk_read_bandwidth_bytes_s: float
    disk_kind: Literal["nvme", "ssd", "hdd", "unknown"]
    gpus: list[GpuInfo] = Field(default_factory=list)
    measured_at: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """Extracted structural and tensor metadata of a model."""

    model_config = ConfigDict(frozen=True)

    path: str
    arch: str
    n_params: int
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    context_length_max: int
    quant: str
    total_weight_bytes: int
    per_layer_bytes: list[int]
    non_layer_bytes: int
    is_moe: bool = False
    n_experts: int | None = None
    n_experts_used: int | None = None
    expert_bytes_per_layer: list[int] | None = None
    shared_expert_bytes_per_layer: list[int] | None = None
    sliding_window_size: int | None = None
    kv_lora_rank: int | None = None
    is_hybrid_ssm: bool = False
    ssm_d_state: int | None = None


class UserRequest(BaseModel):
    """User-specified execution constraints and performance preferences."""

    model_config = ConfigDict(frozen=True)

    model: str
    vram_budget_bytes: dict[int, int] | None = None
    ram_budget_bytes: int | None = None
    disk_cache_budget_bytes: int | None = None
    min_context: int = 4096
    target_context: int | None = None
    preference: Literal["fastest", "balanced", "fit_at_any_cost"] = "balanced"
    quality_floor: str | None = None
    allow_streaming: bool = True
    allow_kv_quant: bool = True
    use_gpu: bool = True
    gpu_indices: list[int] | None = None
    min_tps: float | None = None
    ubatch_size: int = 512
    n_parallel_slots: int = 1


class Placement(BaseModel):
    """Assignment of model layers and tensors across hardware tiers."""

    model_config = ConfigDict(frozen=True)

    gpu_layers: dict[int, list[int]] = Field(default_factory=dict)
    cpu_layers: list[int] = Field(default_factory=list)
    disk_streamed_layers: list[int] = Field(default_factory=list)
    expert_placement: Literal["gpu", "cpu", "cache"] | None = None
    expert_cache_bytes_vram: int | None = None
    tensor_overrides: list[tuple[str, str]] = Field(default_factory=list)


class KvConfig(BaseModel):
    """Configuration and precision of the Key-Value attention cache."""

    model_config = ConfigDict(frozen=True)

    location: Literal["gpu", "cpu", "split"]
    type_k: Literal["f16", "q8_0", "q4_0"]
    type_v: Literal["f16", "q8_0", "q4_0"]
    context: int
    sliding_window: int | None = None


class MemoryEstimate(BaseModel):
    """Estimated peak memory consumption per hardware tier."""

    model_config = ConfigDict(frozen=True)

    vram_peak_bytes: dict[int, int] = Field(default_factory=dict)
    ram_peak_bytes: int
    pinned_ram_bytes: int = 0
    disk_read_bytes_per_token: int = 0


class SpeedEstimate(BaseModel):
    """Estimated generation speed ranges and the identified system bottleneck."""

    model_config = ConfigDict(frozen=True)

    decode_tok_s_low: float
    decode_tok_s_high: float
    prefill_tok_s_low: float
    prefill_tok_s_high: float
    bottleneck: Literal["vram_bw", "ram_bw", "disk_bw", "pcie_bw", "compute", "expert_cache_miss"]


class ExecutionPlan(BaseModel):
    """A fully concrete execution configuration ready to be scheduled."""

    model_config = ConfigDict(frozen=True)

    id: str
    model_variant: str
    backend: str
    placement: Placement
    kv: KvConfig
    memory: MemoryEstimate
    speed: SpeedEstimate
    verdict: Literal["COMFORTABLE", "CONSTRAINED", "BATCH_ONLY", "INFEASIBLE"]
    warnings: list[str] = Field(default_factory=list)
    backend_args: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
