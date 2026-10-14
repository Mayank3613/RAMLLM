"""Runtime state registry and process lifecycle management for background engines."""

import json
import re
import time
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

from ramllm.services.paths import get_logs_dir, get_run_dir


class RunningEngine(BaseModel):
    """Metadata tracking an active background or foreground RAMLLM engine."""

    id: str
    pid: int
    port: int
    host: str = "127.0.0.1"
    model_name: str
    model_path: str
    plan_id: str
    plan_summary: str = ""
    started_at: float = Field(default_factory=time.time)
    log_file: str
    mode: str = "serve"  # 'serve' or 'chat'
    api_key: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, time.time() - self.started_at)

    def is_alive(self) -> bool:
        """Check whether the engine process is still running and alive."""
        if not psutil.pid_exists(self.pid):
            return False
        try:
            p = psutil.Process(self.pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


def generate_engine_id(model_name: str) -> str:
    """Generate a clean, unique human-readable ID like 'qwen-2.5-7b-1'."""
    slug = re.sub(r"[^\w\-]", "-", model_name.lower()).strip("-")
    existing_ids = {e.id for e in list_running_engines(clean_stale=False)}

    candidate = f"{slug}-1"
    idx = 1
    while candidate in existing_ids:
        idx += 1
        candidate = f"{slug}-{idx}"
    return candidate


def register_engine(engine: RunningEngine) -> None:
    """Save running engine record to <data_dir>/run/<id>.json."""
    run_dir = get_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / f"{engine.id}.json"
    target.write_text(engine.model_dump_json(indent=2), encoding="utf-8")


def unregister_engine(engine_id: str) -> None:
    """Remove engine record from <data_dir>/run/<id>.json."""
    target = get_run_dir() / f"{engine_id}.json"
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass


def get_running_engine(engine_id: str) -> RunningEngine | None:
    """Retrieve engine record by ID, checking liveness."""
    target = get_run_dir() / f"{engine_id}.json"
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        engine = RunningEngine.model_validate(data)
        if not engine.is_alive():
            unregister_engine(engine_id)
            return None
        return engine
    except Exception:
        unregister_engine(engine_id)
        return None


def list_running_engines(clean_stale: bool = True) -> list[RunningEngine]:
    """List all running engine instances, optionally removing dead/stale records."""
    run_dir = get_run_dir()
    if not run_dir.exists():
        return []

    engines: list[RunningEngine] = []
    for f in run_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            engine = RunningEngine.model_validate(data)
            if engine.is_alive():
                engines.append(engine)
            elif clean_stale:
                f.unlink(missing_ok=True)
        except Exception:
            if clean_stale:
                f.unlink(missing_ok=True)

    return sorted(engines, key=lambda e: e.started_at)


def stop_engine(engine: RunningEngine, timeout_s: float = 5.0) -> bool:
    """Gracefully terminate engine and its child processes, forcing kill if unresponsive."""
    try:
        if psutil.pid_exists(engine.pid):
            parent = psutil.Process(engine.pid)
            children = parent.children(recursive=True)

            # Send SIGTERM to children then parent
            for c in children:
                try:
                    c.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            try:
                parent.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # Wait for termination
            _, alive = psutil.wait_procs([parent, *children], timeout=timeout_s)

            # Force kill any lingering survivors
            for p in alive:
                try:
                    p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    finally:
        unregister_engine(engine.id)

    return True


def stop_all_engines() -> list[str]:
    """Stop all active engine instances and return their IDs."""
    stopped_ids: list[str] = []
    for engine in list_running_engines(clean_stale=True):
        stop_engine(engine)
        stopped_ids.append(engine.id)
    return stopped_ids


def get_default_engine_log_file(engine_id: str) -> Path:
    """Generate path to engine log file under logs directory."""
    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{engine_id}.log"
