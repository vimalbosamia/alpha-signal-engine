"""
CLI: Train agent on historical Binance data.

Downloads 1 year of OHLCV data from data.binance.vision and runs it through
the full pipeline + backtest engine. Every trade outcome feeds the self-training
coordinator to build pattern scores, tune strategy params, and calibrate RL weights.

Usage:
    # Download + train on all default symbols (20 symbols × 3 timeframes)
    uv run python -m apps.cli.train_historical

    # Download only
    uv run python -m apps.cli.train_historical download --months 12

    # Train only (assumes data already downloaded)
    uv run python -m apps.cli.train_historical train

    # Custom symbols
    uv run python -m apps.cli.train_historical --symbols BTCUSDT,ETHUSDT,SOLUSDT

    # Custom timeframes
    uv run python -m apps.cli.train_historical --timeframes 5m,15m,1h,4h
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

app = typer.Typer(help="Train agent on historical Binance data")
console = Console()


def _parse_list(val: str) -> list[str]:
    return [s.strip() for s in val.split(",") if s.strip()]


@app.command("download")
def download(
    symbols: str = typer.Option(
        "",
        "--symbols", "-s",
        help="Comma-separated symbols (default: full watchlist)",
    ),
    timeframes: str = typer.Option(
        "15m,1h,4h",
        "--timeframes", "-t",
        help="Comma-separated timeframes to download",
    ),
    months: int = typer.Option(12, "--months", "-m", help="Months of history"),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached"),
    futures: bool = typer.Option(False, "--futures", help="Also download futures data"),
):
    """Download historical data from Binance (spot + optional futures)."""
    from libs.data.providers.historical.downloader import BinanceHistoricalDownloader
    from libs.training.historical_trainer import DEFAULT_TRAINING_SYMBOLS, DEFAULT_FUTURES_SYMBOLS

    sym_list = _parse_list(symbols) if symbols else DEFAULT_TRAINING_SYMBOLS
    tf_list = _parse_list(timeframes)

    # Spot download
    console.print(f"\n[bold cyan]SPOT: Downloading {len(sym_list)} symbols × {len(tf_list)} timeframes × {months} months[/]")
    console.print(f"Symbols: {', '.join(sym_list[:10])}{'...' if len(sym_list) > 10 else ''}")
    console.print(f"Timeframes: {', '.join(tf_list)}\n")

    dl_spot = BinanceHistoricalDownloader(market_type="spot")

    async def _run():
        total = 0
        for tf in tf_list:
            console.print(f"[yellow]  Spot {tf}...[/]")
            results = await dl_spot.download_all(sym_list, tf, months, force, max_concurrent=3)
            total += len(results)
            console.print(f"    ✓ {len(results)}/{len(sym_list)} symbols")

        if futures:
            fut_syms = _parse_list(symbols) if symbols else DEFAULT_FUTURES_SYMBOLS
            console.print(f"\n[bold cyan]FUTURES: Downloading {len(fut_syms)} symbols × {len(tf_list)} timeframes[/]")
            dl_fut = BinanceHistoricalDownloader(market_type="futures")
            for tf in tf_list:
                console.print(f"[yellow]  Futures {tf}...[/]")
                results = await dl_fut.download_all(fut_syms, tf, months, force, max_concurrent=3)
                total += len(results)
                console.print(f"    ✓ {len(results)}/{len(fut_syms)} symbols")
        return total

    total = asyncio.run(_run())
    console.print(f"\n[bold green]Download complete: {total} parquet files saved to data/historical/[/]\n")


@app.command("train")
def train(
    symbols: str = typer.Option(
        "",
        "--symbols", "-s",
        help="Comma-separated symbols",
    ),
    timeframes: str = typer.Option(
        "15m,1h,4h",
        "--timeframes", "-t",
        help="Comma-separated timeframes",
    ),
    months: int = typer.Option(12, "--months", "-m", help="Months of history to train on"),
    fast: bool = typer.Option(False, "--fast", help="Use 6 core strategies only (3-4x faster)"),
):
    """Train agent on downloaded historical data."""
    from libs.training.historical_trainer import (
        HistoricalTrainer, DEFAULT_TRAINING_SYMBOLS,
    )

    sym_list = _parse_list(symbols) if symbols else DEFAULT_TRAINING_SYMBOLS
    tf_list = _parse_list(timeframes)

    mode = "[bold yellow]FAST MODE[/] " if fast else ""
    console.print(f"\n{mode}[bold cyan]Training on {len(sym_list)} symbols × {len(tf_list)} timeframes[/]")
    console.print(f"Symbols: {', '.join(sym_list[:10])}{'...' if len(sym_list) > 10 else ''}")
    console.print(f"Timeframes: {', '.join(tf_list)}\n")

    trainer = HistoricalTrainer()

    async def _run():
        return await trainer.train(symbols=sym_list, timeframes=tf_list, months=months, fast_mode=fast)

    summary = asyncio.run(_run())

    # Display results
    table = Table(title="Training Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Trades", str(summary.total_trades))
    table.add_row("Wins", str(summary.total_wins))
    table.add_row("Losses", str(summary.total_losses))
    table.add_row("Win Rate", f"{summary.win_rate:.1%}")
    table.add_row("Phase Before", summary.phase_before)
    table.add_row("Phase After", summary.phase_after)
    table.add_row("Duration", f"{summary.duration_seconds:.1f}s")

    console.print(table)

    # Per-symbol breakdown
    if summary.results:
        detail = Table(title="Per Symbol Detail", show_header=True)
        detail.add_column("Symbol")
        detail.add_column("TF")
        detail.add_column("Trades", justify="right")
        detail.add_column("Wins", justify="right")
        detail.add_column("Losses", justify="right")
        detail.add_column("Win%", justify="right")
        detail.add_column("PnL R", justify="right")
        detail.add_column("Errors", justify="right")

        for r in sorted(summary.results, key=lambda x: x.total_trades, reverse=True):
            if r.total_trades == 0:
                continue
            wr = r.wins / r.total_trades if r.total_trades > 0 else 0
            style = "green" if wr > 0.5 else "red" if wr < 0.4 else "yellow"
            detail.add_row(
                r.symbol, r.timeframe,
                str(r.total_trades), str(r.wins), str(r.losses),
                f"[{style}]{wr:.0%}[/]",
                f"{r.total_pnl_r:+.1f}R",
                str(r.errors) if r.errors else "",
            )
        console.print(detail)

    # Show training progress
    console.print("\n[bold]Training Progress:[/]")
    try:
        from libs.learning.coordinator import get_coordinator
        progress = get_coordinator().get_training_progress()
        console.print(f"  Phase: [cyan]{progress['phase']}[/] ({progress['phase_number']}/8)")
        console.print(f"  Progress to next: [yellow]{progress['phase_progress_pct']}%[/]")
        console.print(f"  Next phase at: {progress['next_phase_at_trades']} trades")
        console.print(f"  Active subsystems: {', '.join(progress['active_subsystems'])}")
    except Exception:
        pass

    console.print("\n[bold green]Training state saved. Agent is now pre-trained.[/]\n")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    symbols: str = typer.Option("", "--symbols", "-s"),
    timeframes: str = typer.Option("15m,1h,4h", "--timeframes", "-t"),
    months: int = typer.Option(12, "--months", "-m"),
    force: bool = typer.Option(False, "--force"),
    fast: bool = typer.Option(False, "--fast", help="Use 6 core strategies only"),
):
    """Download + train in one step (default action)."""
    if ctx.invoked_subcommand is not None:
        return

    console.print("[bold]Step 1/2: Download historical data[/]")
    ctx.invoke(download, symbols=symbols, timeframes=timeframes, months=months, force=force)

    console.print("[bold]Step 2/2: Train agent[/]")
    ctx.invoke(train, symbols=symbols, timeframes=timeframes, months=months, fast=fast)


if __name__ == "__main__":
    app()
