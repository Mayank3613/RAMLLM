"""Base interfaces for operating system level memory enforcement (Stage B)."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict


class EnforcementStatus(BaseModel):
    """Diagnostic status of operating system memory enforcement mechanisms."""

    model_config = ConfigDict(frozen=True)

    platform: str
    mechanism: str
    supported: bool
    can_enforce: bool
    detail: str


class EnforcementManager(ABC):
    """Abstract manager for hard and soft OS-level process memory budget limits."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the enforcement mechanism (e.g., 'cgroups_v2', 'job_objects')."""
        ...

    @abstractmethod
    def get_status(self) -> EnforcementStatus:
        """Query system capabilities and write permissions for enforcement."""
        ...

    @abstractmethod
    def apply_limit(self, pid: int, limit_bytes: int) -> bool:
        """Apply memory limit to a target process PID. Returns True if successfully applied."""
        ...

    @abstractmethod
    def cleanup(self, pid: int | None = None) -> None:
        """Release OS resources (cgroup directories, Job handles) upon process termination."""
        ...
