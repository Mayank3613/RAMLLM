"""Tests for the UX test harness itself."""

from rich.table import Table
from tests.ux.test_harness import assert_clean_process_state, render_to_plain_text, run_cli_command

from ramllm.cli.main import app


def test_ux_harness_render_plain() -> None:
    table = Table(title="Test Table", min_width=30)
    table.add_column("Col A")
    table.add_row("Val 1")
    text = render_to_plain_text(table)
    assert "Test Table" in text
    assert "Col A" in text
    assert "Val 1" in text


def test_ux_harness_run_cli() -> None:
    res = run_cli_command(app, ["--help"])
    assert res.exit_code == 0
    assert "Run bigger models on smaller machines" in res.stdout


def test_ux_harness_process_hygiene() -> None:
    # Assert no stray processes matching an absurd name
    assert_clean_process_state("nonexistent_dummy_process_12345")
