#!/usr/bin/env python3
"""CLI script to generate lightweight (<5 MB) synthetic GGUF model fixtures for headless CI and tests."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.make_gguf import (  # noqa: E402
    create_tiny_dense_gguf,
    create_tiny_moe_gguf,
)

from ramllm.core.analyzer.gguf import analyze_gguf  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate valid synthetic GGUF model fixtures for headless testing without large downloads."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "models" / "synthetic_tiny.gguf",
        help="Destination path for the generated GGUF fixture file.",
    )
    parser.add_argument(
        "--arch",
        choices=["llama", "moe"],
        default="llama",
        help="Model architecture to generate (llama for dense, moe for mixture-of-experts).",
    )
    args = parser.parse_args()

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic GGUF fixture ({args.arch}) at: {out_path}")
    if args.arch == "moe":
        create_tiny_moe_gguf(out_path)
    else:
        create_tiny_dense_gguf(out_path)

    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Created {out_path.name}: {size_mb:.2f} MB ({size_bytes} bytes)")

    # Verify that RAMLLM analyzer parses it cleanly
    info = analyze_gguf(out_path)
    print(
        f"Validation successful: arch={info.arch}, layers={info.n_layers}, "
        f"params={info.n_params:,}, is_moe={info.is_moe}"
    )
    assert size_mb < 5.0, f"Fixture size ({size_mb:.2f} MB) exceeds 5 MB limit."
    return 0


if __name__ == "__main__":
    sys.exit(main())
