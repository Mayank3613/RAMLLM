"""System RAM and Swap capacity and bandwidth profiling."""

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor

import psutil

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


def get_memory_info() -> tuple[int, int, int]:
    """Retrieve system physical RAM and swap statistics.

    Returns:
        tuple[ram_total_bytes, ram_available_bytes, swap_total_bytes]
    """
    vmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return vmem.total, vmem.available, swap.total


def _worker_bandwidth_stream(size_bytes: int, iterations: int) -> float:
    """Benchmark read-only RAM transfer using STREAM-style read traversal in a single thread."""
    if np is not None:
        elem_count = max(size_bytes // 8, 1)
        arr = np.ones(elem_count, dtype=np.uint64)
        # Warmup
        _ = int(np.sum(arr))

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = int(np.sum(arr))
        t1 = time.perf_counter()
    else:
        # Fallback without numpy: memoryview read traversal
        buf = ctypes.create_string_buffer(size_bytes)
        mv = memoryview(buf)
        # Warmup
        _ = sum(mv[::4096])

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = sum(mv[::4096])
        t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-6)
    transferred_bytes = size_bytes * iterations  # Read-only memory traffic
    return transferred_bytes / elapsed


def _worker_bandwidth_copy(size_bytes: int, iterations: int) -> float:
    """Benchmark raw memory transfer using ctypes.memmove in a single thread."""
    src = ctypes.create_string_buffer(size_bytes)
    dst = ctypes.create_string_buffer(size_bytes)

    # Warmup
    ctypes.memmove(dst, src, size_bytes)

    t0 = time.perf_counter()
    for _ in range(iterations):
        ctypes.memmove(dst, src, size_bytes)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-6)
    transferred_bytes = size_bytes * iterations * 2  # read + write
    return transferred_bytes / elapsed


def measure_ram_bandwidth(num_threads: int = 4, chunk_mb: int = 64, iterations: int = 5) -> float:
    """Measure system RAM bandwidth in bytes per second using a read-only STREAM kernel.

    Runs quickly (typically < 0.5s) to avoid delaying application startup.
    LLM decode traffic is read-dominated; this kernel measures pure read throughput.
    """
    chunk_bytes = chunk_mb * 1024 * 1024
    num_threads = min(num_threads, os.cpu_count() or 4)

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(_worker_bandwidth_stream, chunk_bytes, iterations)
                for _ in range(num_threads)
            ]
            results = [f.result() for f in futures]
            return float(sum(results))
    except Exception:
        # Fallback conservative estimate: 30 GB/s for DDR4/DDR5
        return 30.0 * 1024**3


def measure_ram_bandwidth_memcpy(
    num_threads: int = 4, chunk_mb: int = 64, iterations: int = 5
) -> float:
    """Measure system RAM bandwidth using multithreaded memcpy (read + write).

    Preserved as a secondary diagnostic metric.
    """
    chunk_bytes = chunk_mb * 1024 * 1024
    num_threads = min(num_threads, os.cpu_count() or 4)

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(_worker_bandwidth_copy, chunk_bytes, iterations)
                for _ in range(num_threads)
            ]
            results = [f.result() for f in futures]
            return float(sum(results))
    except Exception:
        return 30.0 * 1024**3
