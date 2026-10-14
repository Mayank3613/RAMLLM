"""Unit tests for HardwareProfiler and synthetic profiles."""

from pathlib import Path

from ramllm.core.profiler import load_synthetic_profile, probe_hardware
from ramllm.core.profiler.pcie import estimate_pcie_bandwidth


def test_synthetic_profiles_loading() -> None:
    fixtures_dir = Path("tests/fixtures/hardware")
    profiles = list(fixtures_dir.glob("*.json"))
    assert len(profiles) >= 6

    for p in profiles:
        prof = load_synthetic_profile(p)
        assert prof.ram_total_bytes > 0
        assert prof.cpu_cores_logical > 0
        assert prof.ram_bandwidth_bytes_s > 0


def test_probe_hardware_live() -> None:
    # Live quick hardware probe should not fail or raise exceptions on any platform
    profile = probe_hardware(refresh=True, quick=True)
    assert profile.os in ("linux", "macos", "windows")
    assert profile.ram_total_bytes > 0
    assert profile.ram_available_bytes > 0
    assert profile.ram_bandwidth_bytes_s > 0
    assert profile.disk_read_bandwidth_bytes_s > 0


def test_pcie_bandwidth_estimation() -> None:
    p = load_synthetic_profile("tests/fixtures/hardware/desktop_32gb_gpu_24gb.json")
    gpu = p.gpus[0]
    bw = estimate_pcie_bandwidth(gpu)
    assert bw is not None
    assert bw > 10 * 1024**3  # PCIe 4.0 x16 should be > 20 GB/s
