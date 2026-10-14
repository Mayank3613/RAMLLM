"""Utility parsing functions for human-readable sizes and token context lengths."""

import re


def parse_bytes(val: str | int | None) -> int | None:
    """Parse human strings like '6GB', '6GiB', '512MB', '16G' into exact bytes."""
    if val is None:
        return None
    if isinstance(val, int):
        return val

    s = str(val).strip().upper()
    if not s:
        return None

    match = re.match(r"^([\d\.]+)\s*([A-Z]*)$", s)
    if not match:
        raise ValueError(f"Invalid byte size specification: '{val}'")

    num = float(match.group(1))
    unit = match.group(2)

    multipliers = {
        "": 1,
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "KIB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "MIB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "GIB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
        "TIB": 1024**4,
    }

    if unit not in multipliers:
        raise ValueError(f"Unrecognized byte size unit '{unit}' in '{val}'")

    return int(num * multipliers[unit])


def parse_context(val: str | int | None, default: int = 4096) -> int:
    """Parse context length like '8k', '8192', '32k', '16384' into an integer."""
    if val is None:
        return default
    if isinstance(val, int):
        return val

    s = str(val).strip().lower()
    if not s:
        return default

    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)

    return int(s)
