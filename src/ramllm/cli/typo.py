"""Click/Typer Group subclass providing fuzzy typo suggestions for mistyped commands."""

import difflib
from typing import Any

import click
from typer.core import TyperGroup


class SuggestGroup(TyperGroup):
    """TyperGroup that suggests close command matches when an invalid command is entered."""

    def resolve_command(self, ctx: Any, args: list[str]) -> Any:
        if not args:
            return super().resolve_command(ctx, args)

        cmd_name = click.utils.make_str(args[0])
        cmd = self.get_command(ctx, cmd_name)
        if cmd is None and cmd_name and not cmd_name.startswith("-"):
            available = self.list_commands(ctx)
            matches = difflib.get_close_matches(cmd_name, available, n=2, cutoff=0.5)
            if matches:
                ctx.fail(f"No such command '{cmd_name}'. Did you mean '{matches[0]}'?")

        return super().resolve_command(ctx, args)
