"""Discovery and capability introspection for llama.cpp installations."""

import shutil
import subprocess
from pathlib import Path

from ramllm.backends.base import BackendInstall, Capabilities


def detect_llamacpp(custom_path: str | None = None) -> BackendInstall | None:
    """Discover llama-server binary and introspect its capabilities.

    Returns None if no working llama-server executable is found.
    """
    binary = custom_path or shutil.which("llama-server") or shutil.which("llama-cli")
    if not binary:
        return None

    # Check version
    version_str: str | None = None
    build_info: str | None = None
    try:
        ver_res = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=5)
        out = ver_res.stdout + ver_res.stderr
        for line in out.splitlines():
            if "version:" in line.lower() or "build" in line.lower():
                version_str = line.strip()
            if "built with" in line.lower():
                build_info = line.strip()
    except Exception:
        pass

    # Check help output for supported flags
    help_text = ""
    try:
        help_res = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=5)
        help_text = help_res.stdout + help_res.stderr
    except Exception:
        pass

    # Derive capabilities from help text
    has_ngl = "-ngl" in help_text or "--gpu-layers" in help_text
    has_cpu_moe = "-cmoe" in help_text or "--cpu-moe" in help_text
    has_kv_quant = "-ctk" in help_text or "--cache-type-k" in help_text
    has_override_tensor = "-ot" in help_text or "--override-tensor" in help_text
    has_split_mode = "-sm" in help_text or "--split-mode" in help_text
    has_flash_attn = "-fa" in help_text or "--flash-attn" in help_text
    has_fit = "-fit" in help_text or "--fit" in help_text

    # Search for llama-fit-params
    fit_params_path: str | None = None
    if custom_path:
        p_cust = Path(custom_path).parent
        for candidate_name in ("llama-fit-params", "llama-fit-params.exe"):
            p_cand = p_cust / candidate_name
            if p_cand.is_file():
                fit_params_path = str(p_cand.resolve())
                break

    if not fit_params_path and binary:
        p_bin = Path(binary).parent
        for candidate_name in ("llama-fit-params", "llama-fit-params.exe"):
            p_cand = p_bin / candidate_name
            if p_cand.is_file():
                fit_params_path = str(p_cand.resolve())
                break

    if not fit_params_path:
        found = shutil.which("llama-fit-params")
        if found:
            fit_params_path = str(Path(found).resolve())

    has_fit_params = fit_params_path is not None

    apis: list[str] = []
    if "metal" in help_text.lower() or "metal" in (build_info or "").lower():
        apis.append("metal")
    if "cuda" in help_text.lower() or "cuda" in (build_info or "").lower():
        apis.append("cuda")
    if "vulkan" in help_text.lower():
        apis.append("vulkan")
    if "rocm" in help_text.lower():
        apis.append("rocm")

    caps = Capabilities(
        partial_gpu_offload=has_ngl,
        per_tensor_placement=has_override_tensor or has_cpu_moe,
        kv_on_gpu_or_cpu_choice=True,
        kv_quantization=has_kv_quant,
        multi_gpu_layer_split=has_split_mode,
        flash_attention=has_flash_attn,
        mmap_control=True,
        apis=apis,
        supports_fit=has_fit,
        has_fit_params=has_fit_params,
        fit_params_binary_path=fit_params_path,
    )

    return BackendInstall(
        name="llamacpp",
        binary_path=binary,
        version=version_str,
        build_info=build_info,
        capabilities=caps,
    )
