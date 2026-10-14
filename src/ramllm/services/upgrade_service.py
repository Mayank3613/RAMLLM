"""Self-update detection and package manager dispatch (Milestone 8)."""

import os
import shutil
import sys
from typing import Literal

import httpx
from packaging.version import Version

from ramllm import __version__

InstallMethod = Literal["brew", "pipx", "uv", "pip", "standalone", "unknown"]


def detect_install_method() -> InstallMethod:
    """Detect how RAMLLM was installed on the host system."""
    # 1. PyInstaller standalone frozen executable
    if getattr(sys, "frozen", False):
        return "standalone"

    exec_path = sys.executable.lower()

    # 2. Homebrew
    if "cellar" in exec_path or "/opt/homebrew" in exec_path or "/usr/local/cellar" in exec_path:
        return "brew"
    if shutil.which("brew"):
        # Check if python prefix points to brew
        prefix = getattr(sys, "prefix", "").lower()
        if "homebrew" in prefix or "cellar" in prefix:
            return "brew"

    # 3. Pipx
    if "pipx/venvs" in exec_path or "pipx" in exec_path:
        return "pipx"

    # 4. uv tool
    if "uv/tools" in exec_path or "uv" in exec_path:
        return "uv"

    # 5. Standard pip / venv
    return "pip"


def check_for_updates(timeout_s: float = 3.0) -> tuple[str, str | None, bool]:
    """Check PyPI for newer versions of RAMLLM.

    Returns:
        (current_version, latest_version, is_update_available)
    """
    current_ver = __version__
    url = "https://pypi.org/pypi/ramllm/json"

    try:
        res = httpx.get(url, timeout=timeout_s)
        if res.status_code == 200:
            data = res.json()
            latest_ver = data.get("info", {}).get("version")
            if latest_ver:
                has_update = Version(latest_ver) > Version(current_ver)
                return current_ver, latest_ver, has_update
    except Exception:
        pass

    return current_ver, None, False


def get_upgrade_command(method: InstallMethod) -> str:
    """Return the exact command needed to upgrade RAMLLM according to its install method."""
    if method == "brew":
        return "brew upgrade ramllm"
    if method == "pipx":
        return "pipx upgrade ramllm"
    if method == "uv":
        return "uv tool upgrade ramllm"
    if method == "standalone":
        if os.name == "nt":
            return 'powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.ps1 | iex"'
        return "curl -fsSL https://raw.githubusercontent.com/Mayank3613/RAMLLM/main/scripts/install.sh | bash"
    # Default pip
    return "pip install --upgrade ramllm"
