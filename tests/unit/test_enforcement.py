"""Unit tests for OS-level memory enforcement (cgroups v2, Job Objects, fallback)."""

from pathlib import Path

from ramllm.backends.base import BackendProcess, MemoryReport
from ramllm.core.governor.enforcement.base import EnforcementManager, EnforcementStatus
from ramllm.core.governor.enforcement.cgroups import CgroupV2EnforcementManager
from ramllm.core.governor.enforcement.darwin import NullEnforcementManager
from ramllm.core.governor.enforcement.factory import (
    get_current_enforcement_status,
    get_enforcement_manager,
)
from ramllm.core.governor.enforcement.windows import WindowsJobObjectEnforcementManager
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.profiler import load_synthetic_profile
from ramllm.core.schemas import (
    ExecutionPlan,
    KvConfig,
    MemoryEstimate,
    Placement,
    SpeedEstimate,
)


def _make_dummy_plan() -> ExecutionPlan:
    return ExecutionPlan(
        id="test_plan",
        model_variant="Q4_K_M",
        backend="llamacpp",
        placement=Placement(gpu_layers={0: [0, 1]}, cpu_layers=[2, 3]),
        kv=KvConfig(location="gpu", type_k="f16", type_v="f16", context=2048),
        memory=MemoryEstimate(
            vram_peak_bytes={0: 2 * 1024**3},
            ram_peak_bytes=4 * 1024**3,
        ),
        speed=SpeedEstimate(
            decode_tok_s_low=10.0,
            decode_tok_s_high=15.0,
            prefill_tok_s_low=50.0,
            prefill_tok_s_high=80.0,
            bottleneck="vram_bw",
        ),
        verdict="COMFORTABLE",
        backend_args=["-m", "dummy.gguf"],
        explanation=["Test explanation"],
    )


class MockEnforcementManager(EnforcementManager):
    """Mock enforcement manager for deterministic compliance testing."""

    def __init__(self, succeed: bool = True):
        self.succeed = succeed
        self.applied_pids: dict[int, int] = {}
        self.cleaned_pids: list[int] = []

    @property
    def name(self) -> str:
        return "mock_enforcement"

    def get_status(self) -> EnforcementStatus:
        return EnforcementStatus(
            platform="mock",
            mechanism="mock",
            supported=True,
            can_enforce=self.succeed,
            detail="Mock enforcement active",
        )

    def apply_limit(self, pid: int, limit_bytes: int) -> bool:
        if self.succeed:
            self.applied_pids[pid] = limit_bytes
            return True
        return False

    def cleanup(self, pid: int | None = None) -> None:
        if pid:
            self.cleaned_pids.append(pid)


def test_cgroup_v2_mock_filesystem(tmp_path: Path) -> None:
    """Test Linux cgroup v2 detection, limit application, and cleanup."""
    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    controllers = cgroup_root / "cgroup.controllers"
    controllers.write_text("cpu memory io\n", encoding="utf-8")

    mgr = CgroupV2EnforcementManager(cgroup_root=cgroup_root)
    status = mgr.get_status()
    assert status.supported
    assert status.can_enforce
    assert "hard limits enabled" in status.detail

    # Test limit application
    test_pid = 12345
    limit = 8 * 1024**3
    success = mgr.apply_limit(test_pid, limit)
    assert success

    proc_dir = cgroup_root / f"ramllm_{test_pid}"
    assert proc_dir.exists()

    # Test cleanup
    mgr.cleanup(test_pid)
    assert not proc_dir.exists()


def test_windows_job_object_status_on_non_windows() -> None:
    """Test Windows Job Objects status report on non-Windows host."""
    mgr = WindowsJobObjectEnforcementManager()
    status = mgr.get_status()
    assert status.platform == "windows"
    # On macOS/Linux host, apply_limit safely returns False
    assert not mgr.apply_limit(99999, 1024**3)


def test_null_enforcement_status() -> None:
    """Test Darwin / Null soft-monitoring manager."""
    mgr = NullEnforcementManager("darwin")
    status = mgr.get_status()
    assert not status.can_enforce
    assert "soft limits" in status.detail
    assert not mgr.apply_limit(111, 1024**3)


def test_factory_returns_enforcement_manager() -> None:
    """Test that factory returns a valid EnforcementManager instance."""
    mgr = get_enforcement_manager()
    assert isinstance(mgr, EnforcementManager)
    status = get_current_enforcement_status()
    assert isinstance(status, EnforcementStatus)
    assert status.platform


def test_governor_compliance_enforcement_integration() -> None:
    """Test that MemoryGovernor correctly applies limit and cleans up with mock manager."""
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    plan = _make_dummy_plan()
    proc = BackendProcess(
        pid=54321,
        port=8080,
        plan=plan,
        initial_memory_report=MemoryReport(
            ram_rss_bytes=1024**3,
            vram_used_bytes={0: 1024**3},
        ),
    )

    mock_mgr = MockEnforcementManager(succeed=True)
    gov = MemoryGovernor(
        process_handle=proc,
        hw=hw,
        sample_interval_s=0.1,
        enforce=True,
        enforcement_manager=mock_mgr,
    )

    gov.start()
    assert gov.enforcement_applied
    assert 54321 in mock_mgr.applied_pids
    assert mock_mgr.applied_pids[54321] == plan.memory.ram_peak_bytes

    latest = gov.get_latest_metrics()
    assert latest.enforcement_applied

    gov.stop()
    assert 54321 in mock_mgr.cleaned_pids
