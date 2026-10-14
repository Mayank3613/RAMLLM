"""Unit tests for Stage A Memory Governor and TierMetrics sampling."""

import time

from tests.unit.test_backends import _make_sample_plan

from ramllm.backends.mock import MockBackendAdapter
from ramllm.core.governor.monitor import MemoryGovernor
from ramllm.core.profiler import load_synthetic_profile


def test_governor_sampling_and_history() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    plan = _make_sample_plan(ngl=16)

    adapter = MockBackendAdapter()
    proc = adapter.launch(plan, port=9099)

    governor = MemoryGovernor(proc, hw, sample_interval_s=0.05, tolerance=0.10)
    governor.start()

    time.sleep(0.2)
    latest = governor.get_latest_metrics()
    assert latest.ram_rss_bytes > 0
    assert latest.predicted_ram_bytes == plan.memory.ram_peak_bytes
    assert not latest.budget_exceeded

    history = governor.get_history()
    assert len(history) >= 2

    governor.stop()
