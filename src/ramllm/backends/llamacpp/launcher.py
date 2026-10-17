"""Process launcher, buffer size log parser, and health monitoring for llama-server."""

import atexit
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import httpx
import psutil

from ramllm.backends.base import BackendProcess, MemoryReport
from ramllm.core.schemas import ExecutionPlan

# Regex patterns for llama.cpp memory buffer logging
BUFFER_REGEX = re.compile(
    r"(model|kv cache|kv|compute)\s+buffer\s+size\s*=\s*([\d\.]+)\s*(MiB|GiB|MB|GB|B|KiB)",
    re.IGNORECASE,
)


def _parse_size_bytes(val: float, unit: str) -> int:
    unit_upper = unit.upper()
    if unit_upper in ("GIB", "GB"):
        return int(val * 1024**3)
    if unit_upper in ("MIB", "MB"):
        return int(val * 1024**2)
    if unit_upper in ("KIB", "KB"):
        return int(val * 1024)
    return int(val)


class ProcessLogMonitor:
    """Monitors stdout/stderr stream from llama-server in background thread to parse memory sizes."""

    def __init__(self, process: subprocess.Popen[str]):
        self.process = process
        self.lines: list[str] = []
        self.model_buffer_bytes = 0
        self.kv_buffer_bytes = 0
        self.compute_buffer_bytes = 0
        self.thread = threading.Thread(target=self._read_stream, daemon=True)
        self.thread.start()

    def _read_stream(self) -> None:
        if not self.process.stdout:
            return
        for line in iter(self.process.stdout.readline, ""):
            self.lines.append(line)
            match = BUFFER_REGEX.search(line)
            if match:
                buf_type = match.group(1).lower()
                val = float(match.group(2))
                unit = match.group(3)
                bytes_val = _parse_size_bytes(val, unit)

                if "model" in buf_type:
                    self.model_buffer_bytes = bytes_val
                elif "kv" in buf_type:
                    self.kv_buffer_bytes = bytes_val
                elif "compute" in buf_type:
                    self.compute_buffer_bytes = bytes_val


_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()


def _cleanup_orphaned_processes() -> None:
    """atexit hook to ensure all managed llama-server child processes are terminated."""
    for p in list(_ACTIVE_PROCESSES):
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=1.5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


atexit.register(_cleanup_orphaned_processes)


def find_free_loopback_port() -> int:
    """Find an available ephemeral port bound strictly to IPv4 loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if the given port is already in use (bound or actively listening)."""
    # First, test if the port can be bound
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return True

    # Second, test if an active service is reachable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def launch_llamacpp_server(
    cmd: list[str],
    plan: ExecutionPlan,
    port: int = 8080,
    timeout_s: float = 30.0,
) -> BackendProcess:
    """Launch llama-server, wait for health endpoint, and return BackendProcess handle."""
    if is_port_in_use(port, "127.0.0.1"):
        raise RuntimeError(
            f"Cannot launch llama-server: Port {port} on 127.0.0.1 is already in use by another process."
        )

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    _ACTIVE_PROCESSES.add(proc)

    monitor = ProcessLogMonitor(proc)
    health_url = f"http://127.0.0.1:{port}/health"

    t0 = time.time()
    healthy = False

    while time.time() - t0 < timeout_s:
        # Check if process died
        ret = proc.poll()
        if ret is not None:
            err_output = "".join(monitor.lines[-25:])
            raise RuntimeError(
                f"llama-server exited unexpectedly with code {ret}.\nLast logs:\n{err_output}"
            )

        try:
            resp = httpx.get(health_url, timeout=0.5)
            if resp.status_code == 200:
                healthy = True
                break
        except Exception:
            pass

        time.sleep(0.3)

    if not healthy:
        _ACTIVE_PROCESSES.discard(proc)
        proc.terminate()
        raise TimeoutError(f"llama-server failed to become healthy within {timeout_s}s.")

    # Collect initial RSS
    rss = 0
    try:
        p = psutil.Process(proc.pid)
        rss = p.memory_info().rss
    except Exception:
        pass

    initial_report = MemoryReport(
        vram_used_bytes=plan.memory.vram_peak_bytes,
        ram_rss_bytes=rss,
        model_buffer_bytes=monitor.model_buffer_bytes,
        kv_buffer_bytes=monitor.kv_buffer_bytes,
        compute_buffer_bytes=monitor.compute_buffer_bytes,
    )

    return BackendProcess(
        pid=proc.pid,
        port=port,
        process=proc,
        plan=plan,
        initial_memory_report=initial_report,
    )
