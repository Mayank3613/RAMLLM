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


def test_measure_ram_bandwidth_mocked_timer(monkeypatch) -> None:
    """Verify STREAM bandwidth calculation with mocked timer."""
    import time

    from ramllm.core.profiler.memory import measure_ram_bandwidth, measure_ram_bandwidth_memcpy

    # Mock time.perf_counter so each benchmark worker measures exactly 0.01 seconds elapsed
    state = 0

    def mock_perf_counter():
        nonlocal state
        # alternates between 10.00 and 10.01
        if state % 2 == 0:
            val = 10.0
        else:
            val = 10.01
        state += 1
        return val

    monkeypatch.setattr(time, "perf_counter", mock_perf_counter)

    # 1MB chunk, 1 thread, 1 iteration
    # Transferred bytes for STREAM: 1 * 1024 * 1024 * 1 = 1,048,576 bytes
    # Elapsed: 0.01s -> 104,857,600 B/s (~100 MB/s)
    bw_stream = measure_ram_bandwidth(num_threads=1, chunk_mb=1, iterations=1)
    assert bw_stream > 0
    assert abs(bw_stream - 104857600.0) < 1000.0

    # Transferred bytes for memcpy: 1 * 1024 * 1024 * 1 * 2 = 2,097,152 bytes (read + write)
    bw_memcpy = measure_ram_bandwidth_memcpy(num_threads=1, chunk_mb=1, iterations=1)
    assert bw_memcpy > 0
    assert abs(bw_memcpy - 209715200.0) < 1000.0
    # Memcpy counts 2x bytes of STREAM for identical memory size and elapsed time
    assert abs(bw_memcpy / bw_stream - 2.0) < 0.05


def test_estimate_cpu_gflops() -> None:
    from ramllm.core.profiler.cpu import estimate_cpu_gflops

    # 8 cores AVX-512 @ 3.0 GHz: 8 * 3 * 32 * 0.65 = 499.2 GFLOPs
    avx512_gflops = estimate_cpu_gflops(8, 16, ["avx512f", "avx512bw"])
    assert avx512_gflops > 400.0

    # 8 cores AVX2 @ 3.0 GHz: 8 * 3 * 16 * 0.65 = 249.6 GFLOPs
    avx2_gflops = estimate_cpu_gflops(8, 16, ["avx2", "fma"])
    assert 200.0 <= avx2_gflops <= 300.0

    # 8 cores scalar: 8 * 3 * 4 * 0.65 = 62.4 GFLOPs
    scalar_gflops = estimate_cpu_gflops(8, 8, [])
    assert scalar_gflops < avx2_gflops
