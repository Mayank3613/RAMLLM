"""Unit tests for backend adapters (Mock and llama.cpp)."""

from ramllm.backends import LlamaCppAdapter, MockBackendAdapter
from ramllm.core.schemas import (
    ExecutionPlan,
    KvConfig,
    MemoryEstimate,
    Placement,
    SpeedEstimate,
)


def _make_sample_plan(ngl: int = 24, cpu_moe: bool = False, type_k: str = "f16") -> ExecutionPlan:
    return ExecutionPlan(
        id="sample_plan",
        model_variant="Q4_K_M",
        backend="llamacpp",
        placement=Placement(
            gpu_layers={0: list(range(ngl))},
            cpu_layers=list(range(ngl, 32)),
            expert_placement="cpu" if cpu_moe else None,
        ),
        kv=KvConfig(
            location="gpu",
            type_k=type_k,  # type: ignore[arg-type]
            type_v=type_k,  # type: ignore[arg-type]
            context=4096,
        ),
        memory=MemoryEstimate(
            vram_peak_bytes={0: 6 * 1024**3},
            ram_peak_bytes=4 * 1024**3,
        ),
        speed=SpeedEstimate(
            decode_tok_s_low=12.0,
            decode_tok_s_high=18.0,
            prefill_tok_s_low=60.0,
            prefill_tok_s_high=90.0,
            bottleneck="vram_bw",
        ),
        verdict="COMFORTABLE",
        backend_args=["-m", "/models/test.gguf", "-ngl", str(ngl), "-c", "4096"],
        explanation=["Sample test plan."],
    )


def test_mock_backend_lifecycle() -> None:
    adapter = MockBackendAdapter()
    install = adapter.detect()
    assert install is not None
    assert install.name == "mock"

    plan = _make_sample_plan()
    args = adapter.build_args(plan, port=9090)
    assert "--port" in args
    assert "9090" in args

    proc = adapter.launch(plan, port=9090)
    assert proc.pid == 99999
    assert proc.port == 9090

    report = adapter.memory_report(proc)
    assert report.ram_rss_bytes == plan.memory.ram_peak_bytes

    adapter.stop(proc)


def test_llamacpp_detection_and_flag_building() -> None:
    adapter = LlamaCppAdapter()
    install = adapter.detect()
    # If llama-server is present on system, install should be populated
    if install:
        assert install.name == "llamacpp"
        assert install.capabilities.partial_gpu_offload

    # Test argument building with different plans
    plan1 = _make_sample_plan(ngl=16, cpu_moe=False, type_k="q8_0")
    args1 = adapter.build_args(plan1, port=8088)
    assert "-ngl" in args1
    assert "16" in args1
    assert "-c" in args1
    assert "4096" in args1
    assert "-ctk" in args1
    assert "q8_0" in args1
    assert "-ctv" in args1
    assert "q8_0" in args1

    # Test MoE offload flag
    plan_moe = _make_sample_plan(ngl=32, cpu_moe=True)
    args_moe = adapter.build_args(plan_moe, port=8088)
    assert "-cmoe" in args_moe


def test_backend_fit_capabilities() -> None:
    from ramllm.backends.base import Capabilities

    caps = Capabilities(
        supports_fit=True, has_fit_params=True, fit_params_binary_path="/path/to/llama-fit-params"
    )
    assert caps.supports_fit is True
    assert caps.has_fit_params is True
    assert caps.fit_params_binary_path == "/path/to/llama-fit-params"


def test_detect_llamacpp_with_fit(monkeypatch) -> None:
    import subprocess

    from ramllm.backends.llamacpp.install import detect_llamacpp

    monkeypatch.setattr("shutil.which", lambda cmd: "/bin/" + cmd)

    class DummyProc:
        def __init__(self, stdout: str):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = 0

    def mock_run(cmd, *args, **kwargs):
        if "--version" in cmd:
            return DummyProc("version: 0.6.0 (build 11429)")
        if "--help" in cmd:
            return DummyProc("-ngl, --gpu-layers N\n-fit, --fit [on|off]\n-ctk TYPE\n")
        return DummyProc("")

    monkeypatch.setattr(subprocess, "run", mock_run)

    install = detect_llamacpp("/bin/llama-server")
    assert install is not None
    assert install.capabilities.supports_fit is True
    assert install.capabilities.has_fit_params is True
