"""Backend (llama.cpp) management, discovery, and automated installation."""

import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from ramllm.backends.base import BackendInstall
from ramllm.backends.llamacpp.install import detect_llamacpp
from ramllm.services.paths import get_backends_dir

LLAMACPP_REPOS_API = "https://api.github.com/repos/ggerganov/llama.cpp/releases"
REGISTRY_FILENAME = "registry.json"


@dataclass
class BackendDiscoveryResult:
    source: str  # "env", "config", "managed", "path"
    install: BackendInstall
    detail: str


def get_backend_registry_file() -> Path:
    return get_backends_dir() / REGISTRY_FILENAME


def load_backend_registry() -> dict[str, Any]:
    reg_file = get_backend_registry_file()
    if not reg_file.exists():
        return {"active": None, "installs": {}}
    try:
        data = json.loads(reg_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return cast(dict[str, Any], data)
        return {"active": None, "installs": {}}
    except Exception:
        return {"active": None, "installs": {}}


def save_backend_registry(reg: dict[str, Any]) -> None:
    get_backends_dir().mkdir(parents=True, exist_ok=True)
    reg_file = get_backend_registry_file()
    reg_file.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def discover_backend(custom_config_path: str | None = None) -> BackendDiscoveryResult | None:
    """Discover llama.cpp according to the strict priority order:

    1. RAMLLM_BACKEND_PATH environment variable
    2. Config file backend_path
    3. RAMLLM-managed install in <data_dir>/backends/
    4. System PATH
    """
    # 1. Environment variable
    env_path = os.environ.get("RAMLLM_BACKEND_PATH")
    if env_path:
        install = detect_llamacpp(env_path)
        if install:
            return BackendDiscoveryResult(
                source="env", install=install, detail=f"From RAMLLM_BACKEND_PATH: {env_path}"
            )

    # 2. Config file
    if custom_config_path:
        install = detect_llamacpp(custom_config_path)
        if install:
            return BackendDiscoveryResult(
                source="config",
                install=install,
                detail=f"From config backend_path: {custom_config_path}",
            )

    # 3. RAMLLM-managed install
    reg = load_backend_registry()
    active_tag = reg.get("active")
    if active_tag and active_tag in reg.get("installs", {}):
        managed_bin = reg["installs"][active_tag].get("binary_path")
        if managed_bin and Path(managed_bin).exists():
            install = detect_llamacpp(managed_bin)
            if install:
                return BackendDiscoveryResult(
                    source="managed",
                    install=install,
                    detail=f"Managed build ({active_tag}): {managed_bin}",
                )

    # Also check if any installed tag exists in managed directory
    managed_dir = get_backends_dir() / "llama.cpp"
    if managed_dir.exists():
        for tag_dir in sorted(managed_dir.iterdir(), reverse=True):
            if tag_dir.is_dir():
                for candidate in ["llama-server", "llama-server.exe"]:
                    candidate_path = tag_dir / candidate
                    if candidate_path.exists():
                        install = detect_llamacpp(str(candidate_path))
                        if install:
                            return BackendDiscoveryResult(
                                source="managed",
                                install=install,
                                detail=f"Managed build ({tag_dir.name}): {candidate_path}",
                            )

    # 4. System PATH
    path_install = detect_llamacpp()
    if path_install:
        return BackendDiscoveryResult(
            source="path",
            install=path_install,
            detail=f"From system PATH: {path_install.binary_path}",
        )

    return None


def normalize_os(raw_os: str | None = None) -> str:
    s = (raw_os or platform.system()).lower()
    if "darwin" in s or "mac" in s:
        return "darwin"
    if "win" in s:
        return "windows"
    return "linux"


def normalize_arch(raw_arch: str | None = None) -> str:
    m = (raw_arch or platform.machine()).lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("x86_64", "amd64", "x64"):
        return "x64"
    return m


def score_llama_asset(
    asset_name: str,
    target_os: str,
    arch: str,
    gpu_apis: list[str],
    cuda_version: str | None = None,
) -> int:
    """Score a llama.cpp release asset based on OS, architecture, and accelerator capabilities."""
    name = asset_name.lower()

    # Must be a binary distribution
    if not (name.startswith("llama-") and "-bin-" in name):
        return 0
    if not (name.endswith(".zip") or name.endswith(".tar.gz")):
        return 0
    if (
        "android" in name
        or "snapdragon" in name
        or "ui" in name
        or "xcframework" in name
        or "s390x" in name
    ):
        return 0

    score = 10

    # 1. OS Matching
    if target_os == "darwin":
        if "macos" not in name and "darwin" not in name:
            return 0
        if "win" in name or "ubuntu" in name or "linux" in name:
            return 0
    elif target_os == "windows":
        if "win" not in name:
            return 0
        if "macos" in name or "ubuntu" in name or "linux" in name:
            return 0
    elif target_os == "linux":
        if "ubuntu" not in name and "linux" not in name:
            return 0
        if "macos" in name or "win" in name:
            return 0

    # 2. Architecture Matching
    if arch == "arm64":
        if "arm64" not in name and "aarch64" not in name:
            return 0
    elif arch == "x64":
        if "arm64" in name or "aarch64" in name:
            return 0
        if "x64" not in name and "x86_64" not in name and "amd64" not in name:
            return 0

    # 3. Accelerator Matching
    normalized_apis = [api.lower() for api in gpu_apis]

    if "openvino" in name and "openvino" not in normalized_apis:
        return 0
    if "sycl" in name and "sycl" not in normalized_apis:
        return 0

    if target_os == "darwin":
        # macOS binaries natively bundle Metal
        score += 80
        return score

    # CUDA
    if "cuda" in normalized_apis:
        if "cuda" in name:
            score += 100
            if cuda_version and cuda_version in name:
                score += 20
        elif "vulkan" in name:
            score += 40
        elif "cpu" in name or "-x64." in name or "-arm64." in name:
            score += 10

    # Vulkan
    elif "vulkan" in normalized_apis:
        if "vulkan" in name:
            score += 90
        elif "cpu" in name or "-x64." in name or "-arm64." in name:
            score += 20

    # ROCm
    elif "rocm" in normalized_apis:
        if "rocm" in name:
            score += 90
        elif "vulkan" in name:
            score += 50
        elif "cpu" in name or "-x64." in name:
            score += 20

    else:
        # CPU only
        if "cpu" in name:
            score += 70
        elif (
            name.endswith("ubuntu-x64.tar.gz")
            or name.endswith("ubuntu-arm64.tar.gz")
            or "win-cpu" in name
        ):
            score += 65
        elif "vulkan" not in name and "cuda" not in name and "rocm" not in name:
            score += 50

    return score


def select_llama_assets(
    assets: list[dict[str, Any]],
    target_os: str | None = None,
    arch: str | None = None,
    gpu_apis: list[str] | None = None,
    cuda_version: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select the best main binary asset and optional companion CUDA runtime asset."""
    t_os = normalize_os(target_os)
    t_arch = normalize_arch(arch)
    apis = gpu_apis or []

    best_asset: dict[str, Any] | None = None
    best_score = 0

    for asset in assets:
        name = asset.get("name", "")
        sc = score_llama_asset(
            name, target_os=t_os, arch=t_arch, gpu_apis=apis, cuda_version=cuda_version
        )
        if sc > best_score:
            best_score = sc
            best_asset = asset

    # Windows CUDA matching cudart
    cudart_asset: dict[str, Any] | None = None
    if best_asset and t_os == "windows" and "cuda" in best_asset.get("name", "").lower():
        # E.g. llama-b11609-bin-win-cuda-12.4-x64.zip -> cudart-llama-bin-win-cuda-12.4-x64.zip
        main_name = best_asset["name"]
        for asset in assets:
            aname = asset.get("name", "")
            if aname.startswith("cudart-") and "win-cuda" in aname:
                # Match version if present
                for ver in ["13.4", "12.8", "12.4", "12.2", "11.8"]:
                    if ver in main_name and ver in aname:
                        cudart_asset = asset
                        break
                if cudart_asset:
                    break
        if not cudart_asset:
            # Pick any matching win cudart asset
            for asset in assets:
                aname = asset.get("name", "")
                if aname.startswith("cudart-") and "win" in aname:
                    cudart_asset = asset
                    break

    return best_asset, cudart_asset


def fetch_latest_release(timeout_s: float = 15.0) -> dict[str, Any]:
    """Fetch the latest release information from GitHub API that contains binary assets."""
    headers = {"User-Agent": "RAMLLM-BackendManager"}
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=timeout_s) as client:
            res = client.get(f"{LLAMACPP_REPOS_API}?per_page=10")
            if res.status_code != 200:
                raise RuntimeError(f"GitHub Releases API returned HTTP {res.status_code}")
            releases = res.json()
            for rel in releases:
                assets = rel.get("assets", [])
                has_bins = any(
                    a.get("name", "").startswith("llama-") and "-bin-" in a.get("name", "")
                    for a in assets
                )
                if has_bins:
                    return cast(dict[str, Any], rel)
    except Exception as e:
        suggestions = "\n".join(f"  • {s}" for s in get_fallback_suggestions())
        raise RuntimeError(
            f"Unable to reach GitHub to download llama.cpp releases ({e}).\n"
            f"You can install it manually or use an existing install:\n{suggestions}"
        ) from None

    raise RuntimeError("No suitable llama.cpp release containing prebuilt binaries found.")


def download_and_extract(
    asset_url: str,
    target_dir: Path,
    expected_size: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Download archive with progress and extract to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "download.tmp"

    headers = {"User-Agent": "RAMLLM-BackendManager"}
    downloaded = 0

    with (
        httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client,
        client.stream("GET", asset_url) as response,
    ):
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download asset from {asset_url}: HTTP {response.status_code}"
            )
        total = int(response.headers.get("content-length", expected_size))
        with open(archive_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress:
                    on_progress(downloaded, total)

    # Extract based on file type
    if str(asset_url).endswith(".zip") or archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
    else:
        # Assume tar.gz
        try:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(target_dir)
        except Exception:
            # Fallback try zipfile
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(target_dir)

    # Clean up temporary archive file
    if archive_path.exists():
        archive_path.unlink()


def install_backend(
    tag: str | None = None,
    target_os: str | None = None,
    arch: str | None = None,
    gpu_apis: list[str] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    consent_callback: Callable[[str, int], bool] | None = None,
) -> BackendInstall:
    """Download, extract, verify, and register a managed llama.cpp installation."""
    release = fetch_latest_release()
    rel_tag = tag or release.get("tag_name", "unknown")
    assets = release.get("assets", [])

    main_asset, cudart_asset = select_llama_assets(
        assets, target_os=target_os, arch=arch, gpu_apis=gpu_apis
    )
    if not main_asset:
        raise RuntimeError(
            f"No compatible prebuilt llama.cpp binary asset found for OS='{target_os}', arch='{arch}'"
        )

    total_bytes = main_asset.get("size", 0) + (cudart_asset.get("size", 0) if cudart_asset else 0)

    # Ask consent if callback given
    if consent_callback:
        approved = consent_callback(main_asset["name"], total_bytes)
        if not approved:
            raise RuntimeError("Backend installation aborted by user.")

    install_dir = get_backends_dir() / "llama.cpp" / rel_tag
    install_dir.mkdir(parents=True, exist_ok=True)

    # Download main asset
    download_and_extract(
        main_asset["browser_download_url"],
        install_dir,
        expected_size=main_asset.get("size", 0),
        on_progress=on_progress,
    )

    # Download cudart if present
    if cudart_asset:
        download_and_extract(
            cudart_asset["browser_download_url"],
            install_dir,
            expected_size=cudart_asset.get("size", 0),
        )

    # Find executable
    bin_name = "llama-server.exe" if normalize_os(target_os) == "windows" else "llama-server"
    binary_path: Path | None = None

    for root, _, files in os.walk(install_dir):
        if bin_name in files:
            binary_path = Path(root) / bin_name
            break

    if not binary_path or not binary_path.exists():
        raise RuntimeError(f"Extraction completed but '{bin_name}' was not found in {install_dir}")

    # Set executable permissions on POSIX
    if normalize_os(target_os) != "windows":
        for root, _, files in os.walk(install_dir):
            for f in files:
                fp = Path(root) / f
                if "llama" in f:
                    fp.chmod(fp.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    # macOS quarantine removal
    if normalize_os(target_os) == "darwin" and shutil.which("xattr"):
        try:
            subprocess.run(["xattr", "-d", "com.apple.quarantine", str(binary_path)], check=False)
        except Exception:
            pass

    # Verify and probe capabilities
    install = detect_llamacpp(str(binary_path.resolve()))
    if not install:
        raise RuntimeError(f"Engine verification failed: could not execute {binary_path}")

    # Register in registry.json
    reg = load_backend_registry()
    if "installs" not in reg:
        reg["installs"] = {}
    reg["installs"][rel_tag] = {
        "binary_path": str(binary_path.resolve()),
        "version": install.version,
        "capabilities": install.capabilities.model_dump(),
        "installed_at": str(Path(binary_path).stat().st_mtime),
    }
    reg["active"] = rel_tag
    save_backend_registry(reg)

    return install


def list_managed_backends() -> list[dict[str, Any]]:
    """List all managed engine installations."""
    reg = load_backend_registry()
    active_tag = reg.get("active")
    installs = reg.get("installs", {})
    results: list[dict[str, Any]] = []

    for tag, meta in installs.items():
        results.append(
            {
                "tag": tag,
                "binary_path": meta.get("binary_path"),
                "version": meta.get("version"),
                "is_active": (tag == active_tag),
            }
        )
    return results


def use_managed_backend(path_or_tag: str) -> bool:
    """Set the active managed backend by tag or file path."""
    reg = load_backend_registry()
    installs = reg.get("installs", {})

    if path_or_tag in installs:
        reg["active"] = path_or_tag
        save_backend_registry(reg)
        return True

    # If it is a file path
    p = Path(path_or_tag).resolve()
    if p.exists():
        install = detect_llamacpp(str(p))
        if install:
            tag = f"custom-{p.name}"
            if "installs" not in reg:
                reg["installs"] = {}
            reg["installs"][tag] = {
                "binary_path": str(p),
                "version": install.version,
                "capabilities": install.capabilities.model_dump(),
            }
            reg["active"] = tag
            save_backend_registry(reg)
            return True

    return False


def remove_managed_backend(tag: str) -> bool:
    """Remove a managed backend installation by tag."""
    reg = load_backend_registry()
    installs = reg.get("installs", {})
    if tag not in installs:
        return False

    meta = installs.pop(tag)
    bin_path = meta.get("binary_path")
    if bin_path:
        tag_dir = Path(bin_path).parent
        if tag_dir.name == tag and tag_dir.exists():
            shutil.rmtree(tag_dir, ignore_errors=True)

    if reg.get("active") == tag:
        reg["active"] = next(iter(installs.keys())) if installs else None

    save_backend_registry(reg)
    return True


def get_fallback_suggestions(raw_os: str | None = None) -> list[str]:
    """Get system-specific installation recommendations if automatic installation is unavailable."""
    target = normalize_os(raw_os)
    suggestions: list[str] = []
    if target == "darwin":
        suggestions.append("Install via Homebrew: brew install llama.cpp")
    elif target == "windows":
        suggestions.append("Install via winget: winget install llama.cpp")
    else:
        suggestions.append(
            "Download prebuilt binaries from https://github.com/ggerganov/llama.cpp/releases"
        )
    suggestions.append(
        "Or register any pre-installed engine with: ramllm backend use /path/to/llama-server"
    )
    return suggestions
