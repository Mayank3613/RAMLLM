"""Structured error handling and friendly user-facing formatting."""

from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console

from ramllm.cli.ui.theme import icons, is_plain_mode

# Exit codes
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_USAGE = 2
EXIT_INFEASIBLE = 3
EXIT_BACKEND = 4
EXIT_NETWORK = 5
EXIT_INTERRUPTED = 6


@dataclass
class RamllmError(Exception):
    """Base class for all domain errors presented gracefully to the user."""

    code: str
    title: str
    cause: str
    remedies: Sequence[str]
    exit_code: int = EXIT_GENERIC
    doc_anchor: str | None = None

    def __str__(self) -> str:
        return f"[{self.code}] {self.title}: {self.cause}"


class BackendNotFoundError(RamllmError):
    def __init__(self, cause: str = "llama.cpp engine executable not found on system."):
        super().__init__(
            code="RAM-E001",
            title="Backend Engine Missing",
            cause=cause,
            remedies=[
                "Install managed engine automatically: ramllm backend install",
                "Or register existing install: ramllm backend use /path/to/llama-server",
                "Or diagnose backend status: ramllm doctor --fix",
            ],
            exit_code=EXIT_BACKEND,
            doc_anchor="ram-e001",
        )


class InsufficientMemoryError(RamllmError):
    def __init__(self, needed_bytes: int, available_bytes: int):
        needed_gb = needed_bytes / (1024**3)
        avail_gb = available_bytes / (1024**3)
        super().__init__(
            code="RAM-E012",
            title="Not Enough Memory For This Model",
            cause=f"Model needs approximately {needed_gb:.1f} GB memory, but only {avail_gb:.1f} GB is available.",
            remedies=[
                "Find models that fit your computer: ramllm recommend",
                "Or run with a more aggressive memory mode: ramllm plan <model> --mode max",
                "Or use a smaller context length: --context 2048",
            ],
            exit_code=EXIT_INFEASIBLE,
            doc_anchor="ram-e012",
        )


class DownloadFailedError(RamllmError):
    def __init__(self, target: str, reason: str):
        super().__init__(
            code="RAM-E005",
            title="Download Failed",
            cause=f"Could not download '{target}': {reason}",
            remedies=[
                "Check your internet connection and proxy settings",
                "Ensure sufficient disk space: ramllm models du",
                "If model requires Hugging Face authentication, set HF_TOKEN env var",
            ],
            exit_code=EXIT_NETWORK,
            doc_anchor="ram-e005",
        )


class ChecksumMismatchError(RamllmError):
    def __init__(self, file_path: str, expected: str, actual: str):
        super().__init__(
            code="RAM-E006",
            title="File Integrity Verification Failed",
            cause=f"SHA-256 mismatch for {file_path}. Expected {expected[:12]}..., got {actual[:12]}...",
            remedies=[
                "The file may be corrupted or partially downloaded.",
                "Re-download the model: ramllm models pull <target>",
                "Or remove the invalid file: ramllm models remove <target>",
            ],
            exit_code=EXIT_GENERIC,
            doc_anchor="ram-e006",
        )


class PortInUseError(RamllmError):
    def __init__(self, port: int):
        super().__init__(
            code="RAM-E007",
            title="Port Already In Use",
            cause=f"Port {port} is occupied by another process.",
            remedies=[
                "Check running RAMLLM engines: ramllm ps",
                f"Specify a different port: --port {port + 1}",
                "Stop existing server: ramllm stop",
            ],
            exit_code=EXIT_USAGE,
            doc_anchor="ram-e007",
        )


class BackendOutdatedError(RamllmError):
    def __init__(self, installed_version: str, required_version: str):
        super().__init__(
            code="RAM-E002",
            title="Backend Engine Outdated",
            cause=f"Installed engine build ({installed_version}) does not meet requirement ({required_version}).",
            remedies=[
                "Update managed backend: ramllm backend update",
                "Or repair using doctor: ramllm doctor --fix",
            ],
            exit_code=EXIT_BACKEND,
            doc_anchor="ram-e002",
        )


class DiskSpaceExhaustedError(RamllmError):
    def __init__(self, required_bytes: int, available_bytes: int, path: str):
        req_gb = required_bytes / (1024**3)
        avail_gb = available_bytes / (1024**3)
        super().__init__(
            code="RAM-E008",
            title="Insufficient Disk Space",
            cause=f"Need {req_gb:.1f} GB disk space at '{path}', but only {avail_gb:.1f} GB is available.",
            remedies=[
                "Clean up unused models: ramllm models list and ramllm models remove <model>",
                "Check disk usage: ramllm models du",
                "Change models directory: ramllm config set models_dir /path/to/larger/disk",
            ],
            exit_code=EXIT_GENERIC,
            doc_anchor="ram-e008",
        )


class GatedModelAuthError(RamllmError):
    def __init__(self, model_repo: str):
        super().__init__(
            code="RAM-E009",
            title="Hugging Face Authentication Required",
            cause=f"Model repository '{model_repo}' is gated and requires authentication.",
            remedies=[
                "Accept terms on Hugging Face: https://huggingface.co/" + model_repo,
                "Export your token: export HF_TOKEN=hf_...",
                "Or configure token env in RAMLLM: ramllm config set hf_token_env HF_TOKEN",
            ],
            exit_code=EXIT_NETWORK,
            doc_anchor="ram-e009",
        )


class CorruptedModelError(RamllmError):
    def __init__(self, model_path: str, details: str):
        super().__init__(
            code="RAM-E010",
            title="Model File Corrupted",
            cause=f"Failed to read GGUF file '{model_path}': {details}",
            remedies=[
                "Verify file integrity: ramllm models verify <model>",
                "Re-download the model: ramllm models pull <model>",
            ],
            exit_code=EXIT_GENERIC,
            doc_anchor="ram-e010",
        )


class UnsupportedArchitectureError(RamllmError):
    def __init__(self, arch: str):
        super().__init__(
            code="RAM-E011",
            title="Unsupported Model Architecture",
            cause=f"The model architecture '{arch}' is not currently supported by the llama.cpp engine.",
            remedies=[
                "Check for supported model architectures with: ramllm models search",
                "Ensure your backend engine is up to date: ramllm backend update",
            ],
            exit_code=EXIT_USAGE,
            doc_anchor="ram-e011",
        )


class GpuDriverError(RamllmError):
    def __init__(self, details: str):
        super().__init__(
            code="RAM-E013",
            title="GPU Driver Communication Error",
            cause=f"Could not initialize GPU accelerator: {details}",
            remedies=[
                "Run diagnostics: ramllm doctor",
                "Run in CPU-only mode: ramllm run <model> --cpu-only",
                "Update GPU drivers (CUDA/Metal/Vulkan) or reinstall backend",
            ],
            exit_code=EXIT_BACKEND,
            doc_anchor="ram-e013",
        )


class PermissionDeniedError(RamllmError):
    def __init__(self, path: str, action: str = "access"):
        super().__init__(
            code="RAM-E014",
            title="Permission Denied",
            cause=f"Cannot {action} '{path}': insufficient operating system permissions.",
            remedies=[
                f"Verify ownership and read/write permissions for '{path}'",
                "Run doctor to inspect file permissions: ramllm doctor",
            ],
            exit_code=EXIT_GENERIC,
            doc_anchor="ram-e014",
        )


class ConfigInvalidError(RamllmError):
    def __init__(self, details: str):
        super().__init__(
            code="RAM-E015",
            title="Invalid Configuration",
            cause=f"Configuration file contains invalid settings: {details}",
            remedies=[
                "Inspect config with: ramllm config list",
                "Reset corrupted configuration: ramllm config reset",
            ],
            exit_code=EXIT_USAGE,
            doc_anchor="ram-e015",
        )


def render_error(err: RamllmError, console: Console | None = None) -> None:
    """Format and print a user-friendly error block."""
    from ramllm.cli.ui.console import get_console

    c = console or get_console()

    lines = [
        f"{icons.fail} {err.title} ({err.code})",
        f"  Why:  {err.cause}",
    ]
    if err.remedies:
        lines.append("  Try:")
        for r in err.remedies:
            lines.append(f"    {icons.bullet} {r}")
    if err.doc_anchor:
        lines.append(f"  More: ramllm explain {err.code}")

    text = "\n".join(lines)
    if is_plain_mode():
        c.print(text)
    else:
        c.print(f"[bold red]{text}[/bold red]")


ERROR_CLASSES: list[type[RamllmError]] = [
    BackendNotFoundError,
    BackendOutdatedError,
    DownloadFailedError,
    ChecksumMismatchError,
    PortInUseError,
    DiskSpaceExhaustedError,
    GatedModelAuthError,
    CorruptedModelError,
    UnsupportedArchitectureError,
    InsufficientMemoryError,
    GpuDriverError,
    PermissionDeniedError,
    ConfigInvalidError,
]
