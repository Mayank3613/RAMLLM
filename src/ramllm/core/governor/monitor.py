"""Stage A Memory Governor: Continuous per-tier memory sampling and budget verification."""

import threading
import time
from collections.abc import Callable
from datetime import datetime

import psutil
from pydantic import BaseModel, ConfigDict, Field

from ramllm.backends.base import BackendProcess
from ramllm.core.governor.enforcement import EnforcementManager, get_enforcement_manager
from ramllm.core.schemas import ExecutionPlan, HardwareProfile


class TierMetrics(BaseModel):
    """Snapshot of real-time memory and bandwidth consumption across tiers."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.now)
    ram_rss_bytes: int
    ram_available_bytes: int
    swap_used_bytes: int
    swap_in_rate_bytes_s: float
    vram_used_bytes: dict[int, int]
    disk_read_rate_bytes_s: float
    predicted_ram_bytes: int
    predicted_vram_bytes: dict[int, int]
    budget_exceeded: bool
    enforcement_applied: bool = False
    pressure_detected: bool = False
    process_alive: bool = True
    warnings: list[str] = Field(default_factory=list)


class MemoryGovernor:
    """Monitors running backend process, samples tier allocations, and detects budget violations."""

    def __init__(
        self,
        process_handle: BackendProcess,
        hw: HardwareProfile,
        sample_interval_s: float = 0.25,
        tolerance: float = 0.15,
        enforce: bool = True,
        enforcement_manager: EnforcementManager | None = None,
        pressure_threshold_ram_bytes: int = 500 * 1024 * 1024,
        on_pressure: Callable[[TierMetrics], None] | None = None,
    ):
        self.process_handle = process_handle
        self.plan: ExecutionPlan = process_handle.plan
        self.hw = hw
        self.sample_interval_s = sample_interval_s
        self.tolerance = tolerance
        self.enforce = enforce
        self.enforcement_manager = enforcement_manager or get_enforcement_manager()
        self.enforcement_applied = False
        self.pressure_threshold_ram_bytes = pressure_threshold_ram_bytes
        self.on_pressure = on_pressure
        self.pressure_detected = False
        self._pressure_signaled = False

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._history: list[TierMetrics] = []
        self._max_history = 120  # Keep last 30-60s

        self._last_swap_sin = 0
        self._last_disk_read_bytes = 0
        self._last_sample_time = time.time()

        # Initialize baseline swap and disk counters
        try:
            sin = getattr(psutil.swap_memory(), "sin", 0) or 0
            self._last_swap_sin = sin
            io_counters = psutil.disk_io_counters()
            self._last_disk_read_bytes = io_counters.read_bytes if io_counters else 0
        except Exception:
            pass

    def start(self) -> None:
        """Start the background monitoring thread and apply OS enforcement if enabled."""
        if self._thread and self._thread.is_alive():
            return

        if self.enforce and self.process_handle.pid > 0:
            limit = self.plan.memory.ram_peak_bytes
            self.enforcement_applied = self.enforcement_manager.apply_limit(
                self.process_handle.pid, limit
            )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background monitoring thread and cleanup OS limits."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.enforce and self.process_handle.pid > 0:
            self.enforcement_manager.cleanup(self.process_handle.pid)

    def _sample_metrics(self) -> TierMetrics:
        now = time.time()
        dt = max(now - self._last_sample_time, 1e-4)
        self._last_sample_time = now

        # 1. Host Process RSS
        rss = 0
        try:
            if self.process_handle.pid > 0:
                p = psutil.Process(self.process_handle.pid)
                rss = p.memory_info().rss
                for child in p.children(recursive=True):
                    rss += child.memory_info().rss
        except Exception:
            rss = self.process_handle.initial_memory_report.ram_rss_bytes

        # 2. System RAM & Swap
        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        swap_sin = getattr(swap, "sin", 0) or 0
        swap_in_rate = max(0.0, float(swap_sin - self._last_swap_sin) / dt)
        self._last_swap_sin = swap_sin

        # 3. Disk read rate
        disk_rate = 0.0
        try:
            io = psutil.disk_io_counters()
            if io:
                disk_rate = max(0.0, float(io.read_bytes - self._last_disk_read_bytes) / dt)
                self._last_disk_read_bytes = io.read_bytes
        except Exception:
            pass

        # 4. VRAM allocations
        # On discrete GPUs or Metal, read from plan estimate / initial report
        vram_used = dict(self.process_handle.initial_memory_report.vram_used_bytes)

        # 5. Check against budget
        warnings: list[str] = []
        budget_exceeded = False

        # Check RAM RSS against predicted + tolerance
        predicted_ram = self.plan.memory.ram_peak_bytes
        ram_ceiling = int(predicted_ram * (1.0 + self.tolerance))
        if rss > ram_ceiling and predicted_ram > 0:
            budget_exceeded = True
            warnings.append(
                f"RAM RSS ({rss / (1024**2):.1f} MB) exceeded predicted peak ({predicted_ram / (1024**2):.1f} MB) by >{int(self.tolerance * 100)}%."
            )

        # Check for swap thrashing (> 15 MB/s swap-in activity)
        if swap_in_rate > 15.0 * 1024**2:
            warnings.append(
                f"Swap thrashing detected ({swap_in_rate / (1024**2):.1f} MB/s swap-in rate) - system memory under heavy pressure."
            )

        # Check backend process health
        process_alive = True
        if self.process_handle.process and self.process_handle.process.poll() is not None:
            process_alive = False
            code = self.process_handle.process.poll()
            warnings.append(f"Backend process terminated unexpectedly (exit code {code}).")

        # Check available RAM headroom and memory pressure
        is_pressure = vmem.available < self.pressure_threshold_ram_bytes
        if is_pressure:
            self.pressure_detected = True
            avail_mb = vmem.available / (1024**2)
            thresh_mb = self.pressure_threshold_ram_bytes / (1024**2)
            warnings.append(
                f"Critical system memory pressure: only {avail_mb:.1f} MB RAM available (threshold: {thresh_mb:.1f} MB)."
            )

        return TierMetrics(
            timestamp=datetime.now(),
            ram_rss_bytes=rss,
            ram_available_bytes=vmem.available,
            swap_used_bytes=swap.used,
            swap_in_rate_bytes_s=swap_in_rate,
            vram_used_bytes=vram_used,
            disk_read_rate_bytes_s=disk_rate,
            predicted_ram_bytes=predicted_ram,
            predicted_vram_bytes=self.plan.memory.vram_peak_bytes,
            budget_exceeded=budget_exceeded,
            enforcement_applied=self.enforcement_applied,
            pressure_detected=self.pressure_detected,
            process_alive=process_alive,
            warnings=warnings,
        )

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            metrics = self._sample_metrics()
            with self._lock:
                self._history.append(metrics)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

            # Invoke pressure callback if first detected
            if self.pressure_detected and not self._pressure_signaled and self.on_pressure:
                self._pressure_signaled = True
                try:
                    self.on_pressure(metrics)
                except Exception:
                    pass

            self._stop_event.wait(self.sample_interval_s)

    def get_latest_metrics(self) -> TierMetrics:
        """Get the most recent sampled tier metrics."""
        with self._lock:
            if self._history:
                return self._history[-1]
        return self._sample_metrics()

    def get_history(self) -> list[TierMetrics]:
        """Return shallow copy of recorded metrics history."""
        with self._lock:
            return list(self._history)
