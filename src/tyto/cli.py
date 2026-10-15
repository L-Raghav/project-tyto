"""Typer CLI interface for Project Tyto."""

from __future__ import annotations

import logging
import platform
import shutil
import sys
import time
from typing import Annotated

import psutil
import torch
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tyto.modeling.registry import default_registry

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="tyto",
    help="Project Tyto: Translation-quality mining and prediction CLI.",
    no_args_is_help=True,
    add_completion=False,
)

# Sub-command groups
data_app = typer.Typer(
    help="Data ingestion, preprocessing, and splitting.",
    invoke_without_command=True,
)
features_app = typer.Typer(
    help="Feature extraction and embeddings.",
    invoke_without_command=True,
)
mine_app = typer.Typer(
    help="Pattern mining and association rules.",
    invoke_without_command=True,
)
train_app = typer.Typer(
    help="Model training and evaluation.",
    invoke_without_command=True,
)
review_app = typer.Typer(
    help="Streamlit review application.",
    invoke_without_command=True,
)
export_app = typer.Typer(
    help="Power BI and artifact export.",
    invoke_without_command=True,
)


@data_app.callback(invoke_without_command=True)
def data_group(ctx: typer.Context) -> None:
    """Data processing command group."""
    if ctx.invoked_subcommand is None:
        typer.echo("not implemented yet")


@features_app.callback(invoke_without_command=True)
def features_group(ctx: typer.Context) -> None:
    """Feature extraction command group."""
    if ctx.invoked_subcommand is None:
        typer.echo("not implemented yet")


@mine_app.callback(invoke_without_command=True)
def mine_group(ctx: typer.Context) -> None:
    """Pattern mining command group."""
    if ctx.invoked_subcommand is None:
        typer.echo("not implemented yet")


@train_app.callback(invoke_without_command=True)
def train_group(ctx: typer.Context) -> None:
    """Model training command group."""
    if ctx.invoked_subcommand is None:
        typer.echo("not implemented yet")


@review_app.callback(invoke_without_command=True)
def review_group(ctx: typer.Context) -> None:
    """Streamlit review app command group."""
    if ctx.invoked_subcommand is None:
        typer.echo("not implemented yet")


@export_app.callback(invoke_without_command=True)
def export_group(ctx: typer.Context) -> None:
    """Export command group."""
    if ctx.invoked_subcommand is None:
        typer.echo("not implemented yet")


app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(mine_app, name="mine")
app.add_typer(train_app, name="train")
app.add_typer(review_app, name="app")
app.add_typer(export_app, name="export")


@app.command(name="doctor")
def doctor(
    cpu_ok: Annotated[
        bool,
        typer.Option(
            "--cpu-ok",
            help="Allow doctor to pass even if CUDA is unavailable.",
        ),
    ] = False,
    smoke_model: Annotated[
        str,
        typer.Option(
            "--smoke-model",
            help="Model identifier to use for smoke test (default: labse).",
        ),
    ] = "labse",
) -> None:
    """System health check and GPU LaBSE smoke test."""
    console.print(Panel.fit("[bold cyan]Project Tyto: Environment & Hardware Doctor[/bold cyan]"))

    table = Table(title="System Environment", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    # OS Information
    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    table.add_row("Operating System", os_name)

    # Python Information
    python_ver = f"{platform.python_version()} ({sys.executable})"
    table.add_row("Python", python_ver)

    # PyTorch Information
    torch_ver = torch.__version__
    table.add_row("PyTorch Version", torch_ver)

    # CUDA Availability
    cuda_avail = torch.cuda.is_available()
    cuda_str = "[green]Available[/green]" if cuda_avail else "[yellow]Unavailable[/yellow]"
    table.add_row("CUDA Available", cuda_str)

    # GPU & VRAM Information
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info()
        vram_str = (
            f"Total: {total_b / (1024**3):.2f} GB ({total_b / (1024**2):.0f} MB), "
            f"Free: {free_b / (1024**3):.2f} GB ({free_b / (1024**2):.0f} MB)"
        )
    else:
        gpu_name = "N/A"
        vram_str = "N/A"

    table.add_row("GPU Name", gpu_name)
    table.add_row("GPU VRAM", vram_str)

    # Host RAM
    mem = psutil.virtual_memory()
    ram_str = (
        f"Total: {mem.total / (1024**3):.2f} GB, "
        f"Available: {mem.available / (1024**3):.2f} GB"
    )
    table.add_row("Host RAM", ram_str)

    # Free Disk
    disk = shutil.disk_usage(".")
    disk_str = (
        f"Free: {disk.free / (1024**3):.2f} GB / "
        f"Total: {disk.total / (1024**3):.2f} GB"
    )
    table.add_row("Free Disk", disk_str)

    console.print(table)

    # Verify CUDA status
    if not cuda_avail:
        if not cpu_ok:
            console.print(
                "[bold red]ERROR: CUDA is unavailable and --cpu-ok was not provided. "
                "Exiting.[/bold red]"
            )
            raise typer.Exit(code=1)
        console.print(
            "[yellow]WARNING: CUDA unavailable, but --cpu-ok was passed. "
            "Continuing on CPU.[/yellow]"
        )

    # Run LaBSE embedding smoke test
    device = "cuda" if cuda_avail else "cpu"
    console.print(
        f"\n[bold]Running embedding smoke test with '{smoke_model}' on {device}...[/bold]"
    )

    test_sentences = [
        "Hello world, this is a subtitle translation quality test.",
        "வணக்கம் உலகம், இது வசன மொழிபெயர்ப்பு தர சோதனை.",
    ]

    try:
        if cuda_avail:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        model = default_registry.get(smoke_model, device=device, fp16=cuda_avail)

        _ = model.encode(test_sentences, device=device)

        if cuda_avail:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        smoke_table = Table(
            title="Embedding Smoke Test Results", show_header=True, header_style="bold green"
        )
        smoke_table.add_column("Metric", style="bold")
        smoke_table.add_column("Value")

        smoke_table.add_row("Model", smoke_model)
        smoke_table.add_row("Device", device)
        smoke_table.add_row("Batch Size", str(len(test_sentences)))
        smoke_table.add_row("Latency", f"{latency_ms:.2f} ms")

        if cuda_avail:
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            smoke_table.add_row("Peak VRAM", f"{peak_vram_mb:.2f} MB")
        else:
            smoke_table.add_row("Peak VRAM", "N/A (CPU)")

        console.print(smoke_table)
    except Exception as e:
        console.print(f"[bold red]Smoke test failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e
    finally:
        default_registry.unload(smoke_model)

    console.print("[bold green]Doctor check completed successfully.[/bold green]")


def main() -> None:
    """Entrypoint function for CLI."""
    app()


if __name__ == "__main__":
    main()
