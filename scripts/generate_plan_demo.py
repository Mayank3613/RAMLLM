"""Generate high-quality SVG and PNG screenshots of `ramllm plan` for documentation."""

import io
import os
import re
import subprocess
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def generate_plan_demo():
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "ramllm_plan_demo.svg"
    html_path = output_dir / "ramllm_plan_demo.html"
    png_path = output_dir / "ramllm_plan_demo.png"

    # Use io.StringIO to ensure Console strictly uses width=122 without terminal clipping
    console = Console(file=io.StringIO(), record=True, width=122)

    # 1. Terminal Prompt
    console.print(
        "[bold green]cassain@laptop[/bold green]:[bold blue]~[/bold blue]$ [bold white]ramllm plan mistral-7b-instruct-v0.3.Q4_K_M.gguf --context 8192 --details --explain[/bold white]\n"
    )

    # 2. Plain-English Plan Summary
    summary_text = (
        "[bold green]* mistral-7b-instruct-v0.3.Q4_K_M.gguf will run well on your computer[/bold green]\n"
        "  • [bold]Speed:[/bold] fast — about [bold green]35–52 words per second[/bold green]   [bold green][COMFORTABLE][/bold green]\n"
        "  • [bold]Memory:[/bold] 4.8 GB allocated of your 6.0 GB graphics memory, 1.2 GB of RAM\n"
        "  • [bold]Where it runs:[/bold] 31 of 32 layers fit on GPU (RTX 3060), 1 layer in host RAM\n"
        "  • [bold]Optimization:[/bold] KV-cache quantized to [cyan]q4_0[/cyan] (saves 1.5 GB VRAM with negligible quality loss)"
    )
    console.print(
        Panel(
            summary_text,
            title="[bold cyan]Plan Summary[/bold cyan]",
            box=box.ROUNDED,
            border_style="cyan",
        )
    )
    console.print(
        "[dim]Using up to 6.0 GB of your 16.0 GB memory and 5.0 GB of your 6.0 GB graphics memory.[/dim]\n"
    )

    # 3. Execution Plans Table
    table = Table(
        title="[bold]Execution Plans for mistral-7b-instruct-v0.3.Q4_K_M.gguf (4.4 GB)[/bold]",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold cyan",
    )
    table.add_column("Rank", style="bold cyan", width=6, justify="center")
    table.add_column("Strategy", style="white", min_width=30)
    table.add_column("VRAM Peak", style="magenta", justify="right", width=11)
    table.add_column("RAM Peak", style="blue", justify="right", width=10)
    table.add_column("Decode Speed", style="green", justify="right", min_width=18)
    table.add_column("Bottleneck", style="yellow", justify="center", width=12)
    table.add_column("Verdict", style="bold", justify="center", width=13)

    plans_data = [
        (
            "#1",
            "partial_offload_31L_q4_0_gpu",
            "4.8 GB",
            "1.2 GB",
            "~35.3–51.7 tok/s",
            "vram_bw",
            "[bold green]COMFORTABLE[/bold green]",
        ),
        (
            "#2",
            "partial_offload_29L_q8_0_gpu",
            "4.9 GB",
            "1.5 GB",
            "~24.1–35.9 tok/s",
            "vram_bw",
            "[bold green]COMFORTABLE[/bold green]",
        ),
        (
            "#3",
            "partial_offload_25L_f16_gpu",
            "4.7 GB",
            "2.1 GB",
            "~14.7–22.4 tok/s",
            "ram_bw",
            "[bold green]COMFORTABLE[/bold green]",
        ),
        (
            "#4",
            "cpu_only_f16",
            "0.0 GB",
            "5.8 GB",
            "~3.5–5.4 tok/s",
            "ram_bw",
            "[bold yellow]CONSTRAINED[/bold yellow]",
        ),
    ]

    for rank, strat, vram, ram, speed, bneck, verdict in plans_data:
        table.add_row(rank, strat, vram, ram, speed, bneck, verdict)

    console.print(table)

    # 4. Detailed Explanation & Reasoning
    console.print("\n[bold]Plan #1 Reasoning & Breakdown:[/bold]")
    console.print(
        "  • [cyan]Split Offload Distribution:[/cyan] 31 of 32 transformer blocks placed in NVIDIA GeForce RTX 3060 VRAM."
    )
    console.print(
        "  • [cyan]KV Cache Budget:[/cyan] 8192 tokens with q4_0/q4_0 quant consumes 0.6 GB (vs 2.1 GB in standard F16)."
    )
    console.print(
        "  • [cyan]Primary Bottleneck:[/cyan] GPU VRAM memory bus bandwidth (336 GB/s). Host PCIe transfer adds minimal latency."
    )
    console.print(
        "  • [cyan]Reserve Headroom:[/cyan] 1.2 GB VRAM and 10.0 GB system RAM remain available for display & OS."
    )

    # Save SVG
    console.save_svg(str(svg_path), title="ramllm plan mistral-7b-instruct-v0.3.Q4_K_M.gguf")
    print(f"Generated SVG: {svg_path}")

    # Parse SVG dimensions
    svg_text = svg_path.read_text()
    m = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg_text)
    svg_w = int(float(m.group(1))) if m else 1200
    svg_h = int(float(m.group(2))) if m else 900

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 24px;
    background: #0d1117;
    display: inline-block;
  }}
  svg {{
    width: {svg_w}px;
    height: {svg_h}px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.6);
    border-radius: 10px;
    display: block;
  }}
</style>
</head>
<body>
{svg_text}
</body>
</html>"""
    html_path.write_text(html)
    print(f"Generated HTML: {html_path}")

    # Generate PNG using headless Chrome
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if os.path.exists(chrome_bin):
        abs_html = html_path.resolve()
        win_w = svg_w + 48
        win_h = svg_h + 48
        cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            f"--window-size={win_w},{win_h}",
            "--default-background-color=0d1117",
            f"--screenshot={png_path.resolve()}",
            f"file://{abs_html}",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=15)
            print(f"Generated PNG: {png_path} ({win_w}x{win_h})")
        except Exception as e:
            print(f"Failed to generate PNG screenshot via Chrome: {e}")


if __name__ == "__main__":
    generate_plan_demo()
