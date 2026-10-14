"""Linux cgroup v2 memory enforcement implementation (Stage B)."""

import os
from pathlib import Path

from ramllm.core.governor.enforcement.base import EnforcementManager, EnforcementStatus


class CgroupV2EnforcementManager(EnforcementManager):
    """Enforces memory budgets on Linux using cgroup v2 memory.high and memory.max controllers."""

    def __init__(self, cgroup_root: Path | None = None):
        self.cgroup_root = cgroup_root or Path("/sys/fs/cgroup")
        self._created_groups: dict[int, Path] = {}

    @property
    def name(self) -> str:
        return "cgroups_v2"

    def get_status(self) -> EnforcementStatus:
        if not self.cgroup_root.exists():
            return EnforcementStatus(
                platform="linux",
                mechanism="cgroup v2",
                supported=False,
                can_enforce=False,
                detail="cgroup filesystem not mounted at /sys/fs/cgroup.",
            )

        controllers_file = self.cgroup_root / "cgroup.controllers"
        if not controllers_file.exists():
            return EnforcementStatus(
                platform="linux",
                mechanism="cgroup v2",
                supported=False,
                can_enforce=False,
                detail="Legacy cgroup v1 detected or cgroup.controllers missing (soft monitoring only).",
            )

        try:
            controllers = controllers_file.read_text(encoding="utf-8")
            if "memory" not in controllers:
                return EnforcementStatus(
                    platform="linux",
                    mechanism="cgroup v2",
                    supported=True,
                    can_enforce=False,
                    detail="cgroup v2 present, but 'memory' controller is not enabled in root controllers.",
                )
        except Exception as e:
            return EnforcementStatus(
                platform="linux",
                mechanism="cgroup v2",
                supported=True,
                can_enforce=False,
                detail=f"Cannot read cgroup.controllers: {e}",
            )

        # Check write permissions
        test_dir = self.cgroup_root / "ramllm_test_probe"
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            test_dir.rmdir()
            can_write = True
            detail = (
                "cgroup v2 memory controller active with write permissions (hard limits enabled)."
            )
        except PermissionError:
            can_write = False
            detail = "cgroup v2 memory controller active but lacking write permissions (run with systemd-run or delegate cgroup; soft monitoring only)."
        except Exception as e:
            can_write = False
            detail = f"cgroup probe failed: {e}"

        return EnforcementStatus(
            platform="linux",
            mechanism="cgroup v2",
            supported=True,
            can_enforce=can_write,
            detail=detail,
        )

    def apply_limit(self, pid: int, limit_bytes: int) -> bool:
        """Create a dedicated cgroup slice for the process, set limits, and attach pid."""
        if not self.cgroup_root.exists():
            return False

        group_dir = self.cgroup_root / f"ramllm_{pid}"
        try:
            group_dir.mkdir(parents=True, exist_ok=True)
            self._created_groups[pid] = group_dir

            # memory.high: throttle threshold (exact limit)
            high_file = group_dir / "memory.high"
            if high_file.exists():
                high_file.write_text(str(limit_bytes), encoding="utf-8")

            # memory.max: hard OOM cap (10% overhead above target budget)
            max_file = group_dir / "memory.max"
            if max_file.exists():
                hard_cap = int(limit_bytes * 1.10)
                max_file.write_text(str(hard_cap), encoding="utf-8")

            # Attach PID to cgroup.procs
            procs_file = group_dir / "cgroup.procs"
            if procs_file.exists():
                procs_file.write_text(str(pid), encoding="utf-8")

            return True
        except (PermissionError, OSError):
            # Graceful fallback without crashing
            return False

    def cleanup(self, pid: int | None = None) -> None:
        """Remove process cgroup directory if empty."""
        targets = [pid] if pid is not None else list(self._created_groups.keys())
        for p in targets:
            g = self._created_groups.pop(p, None)
            if g and g.exists():
                try:
                    os.rmdir(g)
                except OSError:
                    pass
