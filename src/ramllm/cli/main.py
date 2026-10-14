"""RAMLLM Command Line Interface entry point."""

from ramllm.cli.app import _parse_gpu_args as _parse_gpu_args
from ramllm.cli.app import app as app
from ramllm.cli.app import backend_app as backend_app
from ramllm.cli.app import models_app as models_app

__all__ = ["app", "models_app", "backend_app", "_parse_gpu_args"]

if __name__ == "__main__":
    app()
