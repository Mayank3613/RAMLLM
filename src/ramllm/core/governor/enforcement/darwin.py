"""macOS and general soft-monitoring memory enforcement fallback."""

import platform

from ramllm.core.governor.enforcement.base import EnforcementManager, EnforcementStatus


class NullEnforcementManager(EnforcementManager):
    """Fallback enforcement manager for platforms without hard per-process memory limits."""

    def __init__(self, platform_name: str | None = None):
        self._platform_name = platform_name or platform.system().lower()

    @property
    def name(self) -> str:
        return "soft_monitoring"

    def get_status(self) -> EnforcementStatus:
        if self._platform_name == "darwin":
            detail = "macOS memory pressure dispatch supported (soft limits & active monitoring)."
        else:
            detail = f"{self._platform_name.capitalize()} hard limits unavailable (soft monitoring only)."

        return EnforcementStatus(
            platform=self._platform_name,
            mechanism="soft monitoring",
            supported=False,
            can_enforce=False,
            detail=detail,
        )

    def apply_limit(self, pid: int, limit_bytes: int) -> bool:
        # Cannot apply hard limit, returns False gracefully
        return False

    def cleanup(self, pid: int | None = None) -> None:
        pass
