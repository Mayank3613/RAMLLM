"""Save and resume chat transcripts in Markdown and JSON formats."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ramllm.cli.chat.client import ChatMessage
from ramllm.services.paths import get_user_data_dir


def get_default_transcript_path(model_name: str) -> Path:
    """Generate default timestamped file path for automatic transcript archiving."""
    chats_dir = get_user_data_dir() / "chats"
    chats_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", model_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return chats_dir / f"{timestamp}_{safe_name}.md"


def save_transcript(
    messages: list[ChatMessage],
    model_name: str,
    plan_id: str | None = None,
    dest_path: Path | str | None = None,
) -> Path:
    """Save conversation history to Markdown or JSON file."""
    if dest_path:
        out_path = Path(dest_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = get_default_transcript_path(model_name)

    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if out_path.suffix.lower() == ".json":
        data = {
            "metadata": {
                "date": now_iso,
                "model": model_name,
                "plan": plan_id or "unknown",
            },
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "interrupted": m.interrupted,
                }
                for m in messages
            ],
        }
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        # Markdown format
        lines = [
            "# RAMLLM Chat Transcript",
            f"- **Date:** {now_iso}",
            f"- **Model:** {model_name}",
        ]
        if plan_id:
            lines.append(f"- **Plan:** {plan_id}")
        lines.append("\n---\n")

        for m in messages:
            role_header = m.role.capitalize()
            interrupted_tag = " _[interrupted]_" if m.interrupted else ""
            lines.append(f"### {role_header}{interrupted_tag}\n\n{m.content}\n")

        out_path.write_text("\n".join(lines), encoding="utf-8")

    return out_path


def load_transcript(path: Path | str) -> list[ChatMessage]:
    """Load conversation messages from an existing Markdown or JSON transcript."""
    in_path = Path(path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {in_path}")

    text = in_path.read_text(encoding="utf-8")

    if in_path.suffix.lower() == ".json":
        data: dict[str, Any] = json.loads(text)
        msgs_raw = data.get("messages", [])
        return [
            ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", 0.0),
                interrupted=m.get("interrupted", False),
            )
            for m in msgs_raw
        ]

    # Parse Markdown transcript
    # Header sections look like: ### User\n... or ### Assistant _[interrupted]_\n...
    messages: list[ChatMessage] = []
    pattern = re.compile(
        r"^###\s+(User|Assistant|System)(?:\s+_?[\[\(]interrupted[\]\)]_?)?\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    for idx, match in enumerate(matches):
        role = match.group(1).lower()
        full_header = match.group(0)
        is_interrupted = "interrupted" in full_header.lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content.endswith("---"):
            content = content[:-3].strip()
        if content:
            messages.append(ChatMessage(role=role, content=content, interrupted=is_interrupted))

    return messages
