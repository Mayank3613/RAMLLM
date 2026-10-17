"""CPU inspection and capability profiling."""

import os
import platform

import psutil

try:
    import cpuinfo  # type: ignore[import-untyped]
except ImportError:
    cpuinfo = None


def get_cpu_info() -> tuple[str, int, int, list[str]]:
    """Detect CPU name, physical cores, logical cores, and instruction set features.

    Returns:
        tuple[cpu_name, physical_cores, logical_cores, features]
    """
    logical_cores = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    physical_cores = psutil.cpu_count(logical=False) or logical_cores

    cpu_name = platform.processor() or "Unknown CPU"
    features: list[str] = []

    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            if "brand_raw" in info and info["brand_raw"]:
                cpu_name = str(info["brand_raw"])
            if "flags" in info and info["flags"]:
                features = [str(f).lower() for f in info["flags"]]
        except Exception:
            pass

    # Normalize features for common LLM SIMD extensions
    normalized_features: list[str] = []
    feature_checks = ["avx", "avx2", "avx512", "fma", "neon", "fp16", "dotprod", "sve", "amx"]
    for feat in feature_checks:
        if any(feat in f for f in features):
            normalized_features.append(feat)

    # Fallback on Apple Silicon / ARM macOS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if "neon" not in normalized_features:
            normalized_features.extend(["neon", "fp16", "dotprod"])
        if cpu_name in ("Unknown CPU", "arm", "arm64", ""):
            import subprocess

            try:
                brand = (
                    subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
                    .decode()
                    .strip()
                )
                if brand:
                    cpu_name = brand
            except Exception:
                cpu_name = "Apple Silicon"

    return cpu_name, physical_cores, logical_cores, normalized_features


def estimate_cpu_gflops(
    physical_cores: int,
    logical_cores: int,
    features: list[str],
    calibration_factor: float = 1.0,
    clock_ghz: float = 3.0,
) -> float:
    """Estimate effective CPU decode compute throughput in GFLOPs.

    Derives theoretical peak throughput from physical cores, clock frequency, and SIMD instructions,
    then applies an empirical GEMV/GEMM CPU efficiency factor (0.65) and calibration multiplier.
    """
    cores = max(1, physical_cores)
    flops_per_cycle = 4.0  # scalar fallback

    feat_set = {f.lower() for f in features}
    if any("amx" in f for f in feat_set) or any("avx512" in f for f in feat_set):
        flops_per_cycle = 32.0
    elif any("avx2" in f for f in feat_set) or any("fma" in f for f in feat_set):
        flops_per_cycle = 16.0
    elif any("dotprod" in f for f in feat_set) or any("neon" in f for f in feat_set):
        flops_per_cycle = 16.0
    elif any("avx" in f for f in feat_set):
        flops_per_cycle = 8.0

    theoretical_gflops = cores * clock_ghz * flops_per_cycle
    # Typical CPU utilization on quantized matrix-vector decode kernels is ~65%
    effective = theoretical_gflops * 0.65 * max(0.2, min(calibration_factor, 5.0))
    return round(max(5.0, effective), 1)
