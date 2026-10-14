"""Unit tests for RAMLLM schemas."""

from datetime import datetime
from ramllm.core.schemas import (
    GpuInfo,
    HardwareProfile,
    ModelInfo,
    UserRequest,
    Placement,
    KvConfig,
    MemoryEstimate,
    SpeedEstimate,
    ExecutionPlan,
)


def test_hardware_profile_creation() -> None:
    gpu = GpuInfo(
        index=0,
        vendor="nvidia",
        name="RTX 4090",
        api_support=["cuda"],
        vram_total_bytes=24 * 1024**3,
        vram_free_bytes=22 * 1024**3,
        vram_bandwidth_bytes_s=1000 * 1024**3,
    )
    profile = HardwareProfile(
        os="linux",
        cpu_name="AMD 7950X",
        cpu_cores_physical=16,
        cpu_cores_logical=32,
        cpu_features=["avx2"],
        ram_total_bytes=32 * 1024**3,
        ram_available_bytes=28 * 1024**3,
        swap_total_bytes=16 * 1024**3,
        ram_bandwidth_bytes_s=50 * 1024**3,
        disk_read_bandwidth_bytes_s=5 * 1024**3,
        disk_kind="nvme",
        gpus=[gpu],
        measured_at=datetime.now(),
    )
    assert profile.os == "linux"
    assert len(profile.gpus) == 1
    assert profile.gpus[0].name == "RTX 4090"


def test_model_info_creation() -> None:
    model = ModelInfo(
        path="/models/test.gguf",
        arch="llama",
        n_params=7_000_000_000,
        n_layers=32,
        hidden_size=4096,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        context_length_max=8192,
        quant="Q4_K_M",
        total_weight_bytes=4_000_000_000,
        per_layer_bytes=[100_000_000] * 32,
        non_layer_bytes=800_000_000,
        is_moe=False,
    )
    assert model.n_layers == 32
    assert not model.is_moe


def test_execution_plan_creation() -> None:
    plan = ExecutionPlan(
        id="plan-1",
        model_variant="Q4_K_M",
        backend="llamacpp",
        placement=Placement(
            gpu_layers={0: list(range(32))},
            cpu_layers=[],
            disk_streamed_layers=[],
        ),
        kv=KvConfig(
            location="gpu",
            type_k="f16",
            type_v="f16",
            context=4096,
        ),
        memory=MemoryEstimate(
            vram_peak_bytes={0: 5_000_000_000},
            ram_peak_bytes=1_000_000_000,
        ),
        speed=SpeedEstimate(
            decode_tok_s_low=25.0,
            decode_tok_s_high=35.0,
            prefill_tok_s_low=120.0,
            prefill_tok_s_high=180.0,
            bottleneck="vram_bw",
        ),
        verdict="COMFORTABLE",
        warnings=[],
        backend_args=["-m", "model.gguf", "-ngl", "32"],
        explanation=["All 32 layers fit into GPU VRAM."],
    )
    assert plan.verdict == "COMFORTABLE"
    assert plan.memory.vram_peak_bytes[0] == 5_000_000_000
