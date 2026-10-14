"""Live streaming token renderer with Rich Markdown formatting and plain fallback."""

import sys
from collections.abc import Generator

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from ramllm.cli.ui.console import get_console


class ChatRenderer:
    """Renders streamed tokens to the terminal, formatting Markdown syntax on the fly."""

    def __init__(self, console: Console | None = None, plain: bool = False):
        self.console = console or get_console()
        self.plain = plain

    def render_stream(self, stream: Generator[str, None, None]) -> tuple[str, bool]:
        """Stream chunks to the screen, returning (full_text, was_interrupted).

        If the user presses Ctrl-C during streaming, stops streaming, marks
        the response as interrupted, and preserves the partial output.
        """
        chunks: list[str] = []
        interrupted = False

        if self.plain or not sys.stdout.isatty():
            # Plain streaming mode: write chunks directly without formatting
            try:
                for chunk in stream:
                    chunks.append(chunk)
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                interrupted = True
                sys.stdout.write(" [interrupted]\n")
                sys.stdout.flush()
        else:
            # Rich Live streaming mode with live Markdown formatting
            current_text = ""
            try:
                with Live(
                    Markdown(""),
                    console=self.console,
                    refresh_per_second=15,
                    vertical_overflow="visible",
                ) as live:
                    for chunk in stream:
                        chunks.append(chunk)
                        current_text = "".join(chunks)
                        live.update(Markdown(current_text))
            except KeyboardInterrupt:
                interrupted = True
                self.console.print(
                    "\n[yellow dim][!] Generation stopped by user (Ctrl-C)[/yellow dim]"
                )
            except Exception:
                # If Markdown parsing fails on mid-stream tokens, fall back to plain chunk writing
                for chunk in chunks:
                    sys.stdout.write(chunk)
                sys.stdout.write("\n")
                sys.stdout.flush()

        full_text = "".join(chunks)
        return full_text, interrupted
