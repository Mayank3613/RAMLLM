"""CLI Documentation Generator for RAMLLM.

Extracts command specifications from the central Typer application
and generates an up-to-date, markdown-formatted reference in docs/cli.md.
"""

from pathlib import Path
from typing import Any

from typer.main import get_command

from ramllm.cli.app import app


def generate_cli_markdown() -> str:
    """Generate Markdown text for all registered RAMLLM commands."""
    click_app = get_command(app)

    lines: list[str] = [
        "# RAMLLM CLI Reference",
        "",
        "> Auto-generated from Typer CLI specifications. Do not edit manually.",
        "> Run `uv run python scripts/generate_cli_docs.py` to regenerate.",
        "",
        "## Overview",
        "",
        "```bash",
        "ramllm [OPTIONS] COMMAND [ARGS]...",
        "```",
        "",
        "### Global Options",
        "",
        "| Flag | Type | Default | Description |",
        "|---|---|---|---|",
        "| `--version`, `-v` | Flag | `False` | Show RAMLLM and backend version |",
        "| `--plain` | Flag | `False` | Disable rich formatting, emojis, and colors |",
        "| `--debug` | Flag | `False` | Enable verbose debug logging and stack traces |",
        "| `--help` | Flag | | Show global help and exit |",
        "",
        "---",
        "",
    ]

    # Group commands
    groups: dict[str, list[dict[str, Any]]] = {}

    for name, cmd in sorted(click_app.commands.items()):  # type: ignore[attr-defined]
        panel = getattr(cmd, "rich_help_panel", None) or "COMMANDS"

        subcommands: list[dict[str, Any]] = []
        if hasattr(cmd, "commands"):
            for sub_name, sub_cmd in sorted(cmd.commands.items()):
                subcommands.append(
                    {
                        "name": f"{name} {sub_name}",
                        "help": sub_cmd.help or sub_cmd.short_help or "",
                        "params": sub_cmd.params,
                    }
                )

        cmd_info = {
            "name": name,
            "help": cmd.help or cmd.short_help or "",
            "params": cmd.params,
            "subcommands": subcommands,
        }
        groups.setdefault(panel, []).append(cmd_info)

    # Order panels
    panel_order = ["GET STARTED", "CHAT & RUN", "MODELS", "SYSTEM", "COMMANDS"]
    ordered_panels = [p for p in panel_order if p in groups] + [
        p for p in groups if p not in panel_order
    ]

    for panel in ordered_panels:
        lines.append(f"## {panel}")
        lines.append("")

        for item in groups[panel]:
            cmd_name = item["name"]
            help_text = item["help"].strip()
            lines.append(f"### `ramllm {cmd_name}`")
            lines.append("")
            lines.append(f"{help_text}")
            lines.append("")

            # Parameters table
            params = item["params"]
            if params:
                lines.append("| Option / Argument | Type | Default | Description |")
                lines.append("|---|---|---|---|")
                for p in params:
                    opt_names = ", ".join(f"`{o}`" for o in p.opts) if p.opts else f"`{p.name}`"
                    is_required = getattr(p, "required", False)
                    default_val = getattr(p, "default", None)
                    if default_val is None or default_val == "":
                        default_str = "*required*" if is_required else "none"
                    else:
                        default_str = f"`{default_val}`"

                    param_type = getattr(p.type, "name", str(p.type))
                    param_help = getattr(p, "help", "") or ""
                    lines.append(f"| {opt_names} | {param_type} | {default_str} | {param_help} |")
                lines.append("")

            # Subcommands if any
            if item["subcommands"]:
                lines.append("#### Subcommands")
                lines.append("")
                for sub in item["subcommands"]:
                    sub_name = sub["name"]
                    sub_help = sub["help"].strip()
                    lines.append(f"- **`ramllm {sub_name}`**: {sub_help}")
                    sub_params = sub["params"]
                    if sub_params:
                        lines.append("")
                        lines.append("  | Option / Argument | Type | Default | Description |")
                        lines.append("  |---|---|---|---|")
                        for sp in sub_params:
                            sp_names = (
                                ", ".join(f"`{o}`" for o in sp.opts) if sp.opts else f"`{sp.name}`"
                            )
                            sp_req = getattr(sp, "required", False)
                            sp_def = getattr(sp, "default", None)
                            sp_def_str = (
                                "*required*"
                                if sp_req
                                else (f"`{sp_def}`" if sp_def is not None else "none")
                            )
                            sp_type = getattr(sp.type, "name", str(sp.type))
                            sp_help = getattr(sp, "help", "") or ""
                            lines.append(f"  | {sp_names} | {sp_type} | {sp_def_str} | {sp_help} |")
                        lines.append("")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Generate docs/cli.md file."""
    output_path = Path(__file__).resolve().parent.parent / "docs" / "cli.md"
    content = generate_cli_markdown()
    output_path.write_text(content, encoding="utf-8")
    print(f"Generated CLI documentation: {output_path} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
