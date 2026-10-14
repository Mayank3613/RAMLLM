"""Unit tests for RAMLLM error catalog, exit codes, and rendering."""

from io import StringIO

from rich.console import Console
from typer.testing import CliRunner

from ramllm.cli.app import app
from ramllm.cli.ui.errors import (
    ERROR_CLASSES,
    BackendNotFoundError,
    BackendOutdatedError,
    ChecksumMismatchError,
    ConfigInvalidError,
    CorruptedModelError,
    DiskSpaceExhaustedError,
    DownloadFailedError,
    GatedModelAuthError,
    GpuDriverError,
    InsufficientMemoryError,
    PermissionDeniedError,
    PortInUseError,
    RamllmError,
    UnsupportedArchitectureError,
    render_error,
)
from ramllm.cli.ui.theme import set_plain_mode

runner = CliRunner()


def test_error_classes_instantiation() -> None:
    sample_instances: list[RamllmError] = [
        BackendNotFoundError(),
        BackendOutdatedError("b1000", "b1100"),
        DownloadFailedError("repo/model", "timeout"),
        ChecksumMismatchError("/path/to/file", "abc123456789", "def987654321"),
        PortInUseError(8080),
        DiskSpaceExhaustedError(10 * 1024**3, 2 * 1024**3, "/tmp"),
        GatedModelAuthError("meta-llama/Llama-3-8B"),
        CorruptedModelError("test.gguf", "bad magic number"),
        UnsupportedArchitectureError("exotic_arch"),
        InsufficientMemoryError(16 * 1024**3, 8 * 1024**3),
        GpuDriverError("CUDA driver version mismatch"),
        PermissionDeniedError("/etc/shadow"),
        ConfigInvalidError("Unknown mode 'extreme'"),
    ]

    assert len(sample_instances) == len(ERROR_CLASSES)

    for err in sample_instances:
        assert err.code.startswith("RAM-E")
        assert len(err.title) > 0
        assert len(err.cause) > 0
        assert len(err.remedies) > 0
        assert 0 <= err.exit_code <= 6
        assert err.doc_anchor is not None
        assert err.doc_anchor.lower() == err.code.lower()


def test_render_error_output() -> None:
    err = InsufficientMemoryError(16 * 1024**3, 8 * 1024**3)
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=False, width=120)

    render_error(err, console=test_console)
    output = buf.getvalue()

    assert "RAM-E012" in output
    assert "Not Enough Memory" in output
    assert "Why:" in output
    assert "Try:" in output
    assert "More: ramllm explain RAM-E012" in output


def test_render_error_plain_mode() -> None:
    set_plain_mode(True)
    try:
        err = BackendNotFoundError()
        buf = StringIO()
        test_console = Console(file=buf, force_terminal=False, width=120)

        render_error(err, console=test_console)
        output = buf.getvalue()
        assert "[x]" in output or "✘" in output or "Backend Engine Missing" in output
        assert "RAM-E001" in output
    finally:
        set_plain_mode(False)


def test_explain_command_error_codes() -> None:
    res = runner.invoke(app, ["explain", "RAM-E001"])
    assert res.exit_code == 0
    assert "RAM-E001" in res.output
    assert "llama-server" in res.output

    res_mem = runner.invoke(app, ["explain", "ram-e012"])
    assert res_mem.exit_code == 0
    assert "RAM-E012" in res_mem.output
