"""Unit tests for Backend Manager asset scoring, discovery, and registry management."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from ramllm.services.backend_manager import (
    discover_backend,
    get_fallback_suggestions,
    score_llama_asset,
    select_llama_assets,
)

FIXTURE_PATH = Path("tests/fixtures/releases/llama_cpp_b11609.json")


def load_release_assets() -> list[dict[str, Any]]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return cast(list[dict[str, Any]], data.get("assets", []))


def test_asset_selection_macos_arm64() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(assets, target_os="darwin", arch="arm64", gpu_apis=["metal"])
    assert main is not None
    assert main["name"] == "llama-b11609-bin-macos-arm64.tar.gz"
    assert cudart is None


def test_asset_selection_macos_x64() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(assets, target_os="darwin", arch="x64", gpu_apis=["metal"])
    assert main is not None
    assert main["name"] == "llama-b11609-bin-macos-x64.tar.gz"
    assert cudart is None


def test_asset_selection_linux_cuda() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(
        assets, target_os="linux", arch="x64", gpu_apis=["cuda"], cuda_version="12.8"
    )
    assert main is not None
    assert "cuda" in main["name"]
    assert "x64" in main["name"]
    assert main["name"] == "llama-b11609-bin-ubuntu-cuda-12.8-x64.tar.gz"


def test_asset_selection_linux_vulkan() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(assets, target_os="linux", arch="x64", gpu_apis=["vulkan"])
    assert main is not None
    assert main["name"] == "llama-b11609-bin-ubuntu-vulkan-x64.tar.gz"


def test_asset_selection_linux_cpu() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(assets, target_os="linux", arch="x64", gpu_apis=[])
    assert main is not None
    assert main["name"] == "llama-b11609-bin-ubuntu-x64.tar.gz"


def test_asset_selection_windows_cuda() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(
        assets, target_os="windows", arch="x64", gpu_apis=["cuda"], cuda_version="12.4"
    )
    assert main is not None
    assert main["name"] == "llama-b11609-bin-win-cuda-12.4-x64.zip"
    assert cudart is not None
    assert cudart["name"] == "cudart-llama-bin-win-cuda-12.4-x64.zip"


def test_asset_selection_windows_vulkan() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(assets, target_os="windows", arch="x64", gpu_apis=["vulkan"])
    assert main is not None
    assert main["name"] == "llama-b11609-bin-win-vulkan-x64.zip"
    assert cudart is None


def test_asset_selection_windows_cpu() -> None:
    assets = load_release_assets()
    main, cudart = select_llama_assets(assets, target_os="windows", arch="x64", gpu_apis=[])
    assert main is not None
    assert main["name"] == "llama-b11609-bin-win-cpu-x64.zip"
    assert cudart is None


def test_score_llama_asset_irrelevant() -> None:
    assert score_llama_asset("llama-b11609-ui.tar.gz", "linux", "x64", []) == 0
    assert score_llama_asset("llama-b11609-bin-android-arm64.tar.gz", "linux", "arm64", []) == 0


def test_discovery_priority_env(monkeypatch) -> None:
    monkeypatch.setenv("RAMLLM_BACKEND_PATH", "/fake/custom/llama-server")
    with patch("ramllm.services.backend_manager.detect_llamacpp") as mock_detect:
        from ramllm.backends.base import BackendInstall, Capabilities

        mock_detect.return_value = BackendInstall(
            name="llamacpp",
            binary_path="/fake/custom/llama-server",
            capabilities=Capabilities(),
        )
        res = discover_backend()
        assert res is not None
        assert res.source == "env"
        assert res.install.binary_path == "/fake/custom/llama-server"


def test_fallback_suggestions() -> None:
    s_mac = get_fallback_suggestions("darwin")
    assert any("brew install" in s for s in s_mac)

    s_win = get_fallback_suggestions("windows")
    assert any("winget install" in s for s in s_win)

    s_linux = get_fallback_suggestions("linux")
    assert any("releases" in s for s in s_linux)
