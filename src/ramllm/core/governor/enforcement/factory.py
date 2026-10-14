"""Factory for OS-specific memory enforcement managers."""

import platform
import sys

from ramllm.core.governor.enforcement.base import EnforcementManager, EnforcementStatus
from ramllm.core.governor.enforcement.cgroups import CgroupV2EnforcementManager
from ramllm.core.governor.enforcement.darwin import NullEnforcementManager
from ramllm.core.governor.enforcement.windows import WindowsJobObjectEnforcementManager


def get_enforcement_manager() -> EnforcementManager:
    """Return the native enforcement manager for the host operating system."""
    system = sys.platform
    if system.startswith("linux"):
        return CgroupV2EnforcementManager()
    elif system == "win32":
        return WindowsJobObjectEnforcementManager()
    elif system == "darwin":
        return NullEnforcementManager("darwin")
    else:
        return NullEnforcementManager(platform.system().lower())


def get_current_enforcement_status() -> EnforcementStatus:
    """Query current OS memory enforcement status and capabilities."""
    return get_enforcement_manager().get_status()
