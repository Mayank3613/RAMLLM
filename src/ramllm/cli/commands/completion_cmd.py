"""Shell completion installer command (ramllm completion install)."""

import os
from pathlib import Path
from typing import Annotated, Literal

import typer

from ramllm.cli.ui.console import get_console, print_error, print_success
from ramllm.cli.ui.theme import icons

ShellKind = Literal["bash", "zsh", "fish", "powershell"]


def _detect_shell() -> ShellKind:
    """Detect current active user shell from environment."""
    shell_env = os.environ.get("SHELL", "").lower()
    if "zsh" in shell_env:
        return "zsh"
    if "fish" in shell_env:
        return "fish"
    if "bash" in shell_env:
        return "bash"
    if os.name == "nt" or "powershell" in shell_env or "pwsh" in shell_env:
        return "powershell"
    return "bash"


completion_app = typer.Typer(
    name="completion",
    help="Install and manage shell auto-completion for tab completion.",
    no_args_is_help=True,
)


@completion_app.command("install")
def install_completion(
    shell: Annotated[
        str | None,
        typer.Argument(help="Target shell ('bash', 'zsh', 'fish', 'powershell')"),
    ] = None,
) -> None:
    """Install tab auto-completion for your shell."""
    console = get_console()
    target_shell: ShellKind = shell.lower() if shell else _detect_shell()  # type: ignore[assignment]

    home = Path.home()

    if target_shell == "zsh":
        zsh_dir = home / ".zfunc"
        zsh_dir.mkdir(parents=True, exist_ok=True)
        comp_file = zsh_dir / "_ramllm"
        # Generate completion script
        script = "#compdef ramllm\n_ramllm() {\n  eval $(env _RAMLLM_COMPLETE=zsh_complete ramllm)\n}\ncompdef _ramllm ramllm\n"
        comp_file.write_text(script, encoding="utf-8")
        print_success(f"Installed Zsh completion script to {comp_file}")
        console.print("\n[dim]Add the following to your ~/.zshrc if not already present:[/dim]")
        console.print("  [cyan]fpath=(~/.zfunc $fpath)[/cyan]")
        console.print("  [cyan]autoload -Uz compinit && compinit[/cyan]\n")
        console.print(
            f"[green]{icons.ok} Reload your shell or run:[/green] [cyan]source ~/.zshrc[/cyan]"
        )

    elif target_shell == "bash":
        bash_dir = home / ".local" / "share" / "bash-completion" / "completions"
        bash_dir.mkdir(parents=True, exist_ok=True)
        comp_file = bash_dir / "ramllm"
        script = 'eval "$(_RAMLLM_COMPLETE=bash_complete ramllm)"\n'
        comp_file.write_text(script, encoding="utf-8")
        print_success(f"Installed Bash completion script to {comp_file}")
        console.print(
            f"\n[green]{icons.ok} Reload your shell or run:[/green] [cyan]source ~/.bashrc[/cyan]"
        )

    elif target_shell == "fish":
        fish_dir = home / ".config" / "fish" / "completions"
        fish_dir.mkdir(parents=True, exist_ok=True)
        comp_file = fish_dir / "ramllm.fish"
        script = "eval (env _RAMLLM_COMPLETE=fish_complete ramllm)\n"
        comp_file.write_text(script, encoding="utf-8")
        print_success(f"Installed Fish completion script to {comp_file}")

    elif target_shell == "powershell":
        console.print("[dim]For PowerShell, add the following line to your $PROFILE:[/dim]")
        console.print(
            "  [cyan]Register-ArgumentCompleter -Native -CommandName ramllm -ScriptBlock {[/cyan]"
        )
        console.print("  [cyan]    param($wordToComplete, $commandAst, $cursorPosition)[/cyan]")
        console.print(
            '  [cyan]    [System.Management.Automation.CompletionResult]::new($wordToComplete, $wordToComplete, "ParameterValue", $wordToComplete)[/cyan]'
        )
        console.print("  [cyan]}[/cyan]")
    else:
        print_error(
            f"Unsupported shell '{target_shell}'. Supported shells: bash, zsh, fish, powershell."
        )
        raise typer.Exit(code=1)


@completion_app.command("show")
def show_completion(
    shell: Annotated[str | None, typer.Argument(help="Shell type ('bash', 'zsh', 'fish')")] = None,
) -> None:
    """Print the raw shell completion script to stdout."""
    target_shell = shell.lower() if shell else _detect_shell()
    env_var = f"_RAMLLM_COMPLETE={target_shell}_source"
    os.system(f"{env_var} ramllm")
