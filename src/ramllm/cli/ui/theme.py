"""Theme, colors, and iconography for RAMLLM CLI with ASCII fallbacks."""

import os

_NO_COLOR = bool(os.environ.get("NO_COLOR"))
_PLAIN_MODE = False


def set_plain_mode(plain: bool) -> None:
    global _PLAIN_MODE
    _PLAIN_MODE = plain


def is_plain_mode() -> bool:
    return _PLAIN_MODE or _NO_COLOR


class Icons:
    @property
    def ok(self) -> str:
        return "[ok]" if is_plain_mode() else "✔"

    @property
    def fail(self) -> str:
        return "[x]" if is_plain_mode() else "✘"

    @property
    def warn(self) -> str:
        return "[!]" if is_plain_mode() else "!"

    @property
    def arrow(self) -> str:
        return "->" if is_plain_mode() else "→"

    @property
    def bullet(self) -> str:
        return "*" if is_plain_mode() else "•"

    @property
    def pointer(self) -> str:
        return ">" if is_plain_mode() else "▸"


icons = Icons()
