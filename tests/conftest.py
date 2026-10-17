"""Pytest session-level configuration and test environment isolation fixtures."""

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_telemetry_bench_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Ensure tests run against an isolated benchmark telemetry file instead of host ~/.local/share."""
    fake_log = tmp_path / "test_session_bench.jsonl"
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_LOG_FILE", fake_log)
    monkeypatch.setattr("ramllm.telemetry.calibration.BENCH_DIR", tmp_path)
    yield
