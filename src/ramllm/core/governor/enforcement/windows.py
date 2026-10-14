"""Windows Job Objects memory enforcement implementation (Stage B)."""

import sys
from typing import Any

from ramllm.core.governor.enforcement.base import EnforcementManager, EnforcementStatus


class WindowsJobObjectEnforcementManager(EnforcementManager):
    """Enforces process memory limits on Windows via Job Objects."""

    def __init__(self) -> None:
        self._jobs: dict[int, Any] = {}

    @property
    def name(self) -> str:
        return "job_objects"

    def get_status(self) -> EnforcementStatus:
        if sys.platform != "win32":
            return EnforcementStatus(
                platform="windows",
                mechanism="Job Objects",
                supported=False,
                can_enforce=False,
                detail="Not running on Windows operating system.",
            )

        return EnforcementStatus(
            platform="windows",
            mechanism="Job Objects",
            supported=True,
            can_enforce=True,
            detail="Windows Job Objects supported with hard per-process memory limits.",
        )

    def apply_limit(self, pid: int, limit_bytes: int) -> bool:
        """Assign PID to a Win32 Job Object with JobObjectExtendedLimitInformation."""
        if sys.platform != "win32":
            return False

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

            # Constants
            JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
            JobObjectExtendedLimitInformation = 9
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_TERMINATE = 0x0001

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_uint64),
                    ("WriteOperationCount", ctypes.c_uint64),
                    ("OtherOperationCount", ctypes.c_uint64),
                    ("ReadTransferCount", ctypes.c_uint64),
                    ("WriteTransferCount", ctypes.c_uint64),
                    ("OtherTransferCount", ctypes.c_uint64),
                ]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            job_handle = kernel32.CreateJobObjectW(None, f"ramllm_job_{pid}")
            if not job_handle:
                return False

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = int(limit_bytes * 1.10)  # 10% tolerance overhead

            success = kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not success:
                kernel32.CloseHandle(job_handle)
                return False

            process_handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
            if not process_handle:
                kernel32.CloseHandle(job_handle)
                return False

            assign_success = kernel32.AssignProcessToJobObject(job_handle, process_handle)
            kernel32.CloseHandle(process_handle)

            if assign_success:
                self._jobs[pid] = job_handle
                return True
            else:
                kernel32.CloseHandle(job_handle)
                return False

        except Exception:
            return False

    def cleanup(self, pid: int | None = None) -> None:
        """Close job object handles upon termination."""
        if sys.platform != "win32":
            return
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        targets = [pid] if pid is not None else list(self._jobs.keys())
        for p in targets:
            handle = self._jobs.pop(p, None)
            if handle:
                try:
                    kernel32.CloseHandle(handle)
                except Exception:
                    pass
