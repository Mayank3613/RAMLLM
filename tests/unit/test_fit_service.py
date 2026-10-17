"""Unit tests for upstream llama.cpp -fit advisory cross-check service."""

import subprocess

from ramllm.backends.base import BackendInstall, Capabilities
from ramllm.core.schemas import ExecutionPlan, KvConfig, MemoryEstimate, Placement, SpeedEstimate
from ramllm.services.fit_service import parse_fit_params_output, run_fit_cross_check


def _sample_plan(ngl: int = 24, type_k: str = "q8_0") -> ExecutionPlan:
    return ExecutionPlan(
        id="sample_fit_plan",
        model_variant="Q4_K_M",
        backend="llamacpp",
        placement=Placement(
            gpu_layers={0: list(range(ngl))},
            cpu_layers=list(range(ngl, 32)),
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
            decode_tok_s_low=15.0,
            decode_tok_s_high=20.0,
            prefill_tok_s_low=60.0,
            prefill_tok_s_high=90.0,
            bottleneck="vram_bw",
        ),
        verdict="COMFORTABLE",
    )


def test_parse_fit_params_output() -> None:
    output1 = "main: fitting 20 layers to free device memory: -ngl 20 -c 4096"
    ngl1, ctx1 = parse_fit_params_output(output1)
    assert ngl1 == 20
    assert ctx1 == 4096

    output2 = "fit_params: -ngl 28 --ctx-size 8192"
    ngl2, ctx2 = parse_fit_params_output(output2)
    assert ngl2 == 28
    assert ctx2 == 8192

    output3 = "n_gpu_layers = 16\nctx_size = 2048"
    ngl3, ctx3 = parse_fit_params_output(output3)
    assert ngl3 == 16
    assert ctx3 == 2048


def test_run_fit_cross_check_not_found(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    plan = _sample_plan(ngl=24)
    install = BackendInstall(
        name="llamacpp",
        binary_path="/bin/llama-server",
        capabilities=Capabilities(has_fit_params=False, fit_params_binary_path=None),
    )

    res = run_fit_cross_check("/models/test.gguf", 4096, plan, install=install)
    assert res.available is False
    assert "not found" in (res.error_message or "").lower()
    assert len(res.advisory_notes) > 0


def test_run_fit_cross_check_success(monkeypatch) -> None:
    plan = _sample_plan(ngl=28, type_k="q4_0")
    install = BackendInstall(
        name="llamacpp",
        binary_path="/bin/llama-server",
        capabilities=Capabilities(
            has_fit_params=True, fit_params_binary_path="/bin/llama-fit-params"
        ),
    )

    class DummyProc:
        stdout = "fit_params: -ngl 20 -c 4096"
        stderr = ""
        returncode = 0

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: DummyProc())

    res = run_fit_cross_check("/models/test.gguf", 4096, plan, install=install)
    assert res.available is True
    assert res.fit_gpu_layers == 20
    assert res.fit_context == 4096
    assert res.ramllm_gpu_layers == 28
    assert any("places more layers on GPU" in diff for diff in res.differences)
    assert any("KV-cache quantization" in diff for diff in res.differences)
