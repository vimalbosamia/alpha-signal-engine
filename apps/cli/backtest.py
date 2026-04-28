"""
CLI command: backtest

Runs a strategy against historical OHLCV data and prints a performance report.

Usage:
    python -m apps.cli.backtest --help
    python -m apps.cli.backtest --symbol BTC/USDT --asset-class crypto --timeframe 15m --strategy macd_crossover
    python -m apps.cli.backtest --symbol AAPL --asset-class stock --timeframe 1h --strategy ema_crossover --days 180
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from libs.backtesting.engine import BacktestConfig, BacktestEngine
from libs.core.models.domain import AssetClass, Timeframe

app = typer.Typer(help="AI Trading Signal Agent — Backtesting CLI")
console = Console()

# ── Strategy registry ──────────────────────────────────────────────────────────

def _load_strategy(name: str):  # type: ignore[return]
    """Resolve strategy name to an instance. Raises typer.Exit on unknown."""
    from libs.strategies.momentum.macd_crossover import MACDCrossoverStrategy
    from libs.strategies.momentum.rsi_strategy import RSIStrategy as RSIMeanReversionStrategy
    from libs.strategies.trend.ema_crossover import EMACrossoverStrategy
    from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy
    from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
    from libs.strategies.continuation.pullback import PullbackContinuationStrategy

    registry = {
        "macd_crossover": MACDCrossoverStrategy,
        "rsi_mean_reversion": RSIMeanReversionStrategy,
        "ema_crossover": EMACrossoverStrategy,
        "hammer_reversal": HammerReversalStrategy,
        "resistance_breakout": ResistanceBreakoutStrategy,
        "pullback_continuation": PullbackContinuationStrategy,
    }

    cls = registry.get(name)
    if cls is None:
        console.print(
            f"[red]Unknown strategy '{name}'. Available:[/red] {', '.join(registry)}"
        )
        raise typer.Exit(1)
    return cls()


def _load_df(
    symbol: str,
    asset_class: AssetClass,
    timeframe: Timeframe,
    days: int,
) -> "import pandas; pandas.DataFrame":  # type: ignore[return, name-defined]
    """
    Load OHLCV data from the configured provider.
    Falls back to a simple synthetic DataFrame when provider is unavailable
    (so the CLI can be exercised without live credentials).
    """
    import asyncio
    import pandas as pd

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    try:
        from libs.core.config.settings import get_settings
        from libs.data.providers.base import BaseDataProvider

        settings = get_settings()
        provider_name = settings.data_provider if hasattr(settings, "data_provider") else "alpaca"

        # Lazy import to avoid hard dependency when credentials are absent
        if asset_class == AssetClass.CRYPTO:
            from libs.data.providers.binance_provider import BinanceDataProvider
            provider: BaseDataProvider = BinanceDataProvider()
        else:
            from libs.data.providers.alpaca_provider import AlpacaDataProvider
            provider = AlpacaDataProvider()

        df = asyncio.run(provider.get_candles(symbol, timeframe, start, now))
        df.attrs["symbol"] = symbol
        return df

    except Exception as exc:
        console.print(f"[yellow]Provider unavailable ({exc}); generating synthetic data.[/yellow]")
        return _synthetic_df(symbol, timeframe, days)


def _synthetic_df(symbol: str, timeframe: Timeframe, days: int) -> "pandas.DataFrame":  # type: ignore[name-defined]
    """Minimal synthetic OHLCV for offline testing."""
    import numpy as np
    import pandas as pd

    # Derive bar count from timeframe
    tf_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
    }.get(timeframe.value, 60)
    bars = (days * 24 * 60) // tf_minutes
    bars = max(bars, 300)

    rng = np.random.default_rng(42)
    price = 100.0
    prices = [price]
    for _ in range(bars - 1):
        price *= 1.0 + rng.normal(0.0002, 0.005)
        prices.append(price)

    closes = np.array(prices)
    highs = closes * (1.0 + rng.uniform(0.001, 0.010, bars))
    lows = closes * (1.0 - rng.uniform(0.001, 0.010, bars))
    opens = closes * (1.0 + rng.normal(0.0, 0.002, bars))
    volumes = rng.uniform(500_000, 5_000_000, bars)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.date_range(
            end=datetime.now(timezone.utc),
            periods=bars,
            freq=f"{tf_minutes}min",
            tz="UTC",
        ),
    )
    df.attrs["symbol"] = symbol
    return df


# ── CLI command ────────────────────────────────────────────────────────────────

@app.command()
def run(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Symbol (e.g. BTC/USDT, AAPL)")] = "BTC/USDT",
    asset_class: Annotated[str, typer.Option("--asset-class", "-a", help="crypto or stock")] = "crypto",
    timeframe: Annotated[str, typer.Option("--timeframe", "-t", help="1m 5m 15m 1h 4h 1d")] = "15m",
    strategy: Annotated[str, typer.Option("--strategy", help="Strategy name")] = "macd_crossover",
    days: Annotated[int, typer.Option("--days", help="Lookback period in days")] = 90,
    capital: Annotated[float, typer.Option("--capital", help="Initial capital")] = 10_000.0,
    risk_pct: Annotated[float, typer.Option("--risk-pct", help="Risk per trade (%)")] = 1.0,
    min_score: Annotated[float, typer.Option("--min-score", help="Min confluence score (0-1)")] = 0.50,
    show_trades: Annotated[bool, typer.Option("--trades/--no-trades", help="Show trade log")] = False,
) -> None:
    """Run a backtest for a strategy on historical data."""

    # Validate enums
    try:
        ac = AssetClass(asset_class)
    except ValueError:
        console.print(f"[red]Invalid asset-class '{asset_class}'. Use: crypto, stock[/red]")
        raise typer.Exit(1)

    try:
        tf = Timeframe(timeframe)
    except ValueError:
        valid = [t.value for t in Timeframe]
        console.print(f"[red]Invalid timeframe '{timeframe}'. Use: {', '.join(valid)}[/red]")
        raise typer.Exit(1)

    strat = _load_strategy(strategy)
    config = BacktestConfig(
        initial_capital=capital,
        risk_per_trade_pct=risk_pct,
        min_confluence_score=min_score,
    )

    console.print(f"\n[bold]Backtesting[/bold] {symbol} | {strategy} | {timeframe} | {days}d")
    console.print("Loading data...", end=" ")

    df = _load_df(symbol, ac, tf, days)
    console.print(f"[green]{len(df)} bars loaded.[/green]")

    engine = BacktestEngine()
    console.print("Running backtest...", end=" ")
    result = engine.run(df, strat, ac, tf, config)
    console.print("[green]done.[/green]\n")

    # ── Summary table ──────────────────────────────────────────────────────────
    summary = Table(
        title=f"Backtest Results — {symbol} [{strategy}]",
        box=box.ROUNDED,
        show_header=False,
    )
    summary.add_column("Metric", style="bold cyan", no_wrap=True)
    summary.add_column("Value", justify="right")

    def _pct(v: float) -> str:
        return f"{v:+.2f}%"

    def _r(v: float) -> str:
        return f"{v:+.3f}R"

    summary.add_row("Symbol", symbol)
    summary.add_row("Strategy", strategy)
    summary.add_row("Timeframe", timeframe)
    summary.add_row("Period", f"{days} days")
    summary.add_row("Bars Tested", str(result.bars_tested))
    summary.add_row("", "")
    summary.add_row("Total Signals", str(result.total_signals))
    summary.add_row("Total Trades", str(result.total_trades))
    summary.add_row("Wins", f"[green]{result.wins}[/green]")
    summary.add_row("Losses", f"[red]{result.losses}[/red]")
    summary.add_row("Timeouts", str(result.timeouts))
    summary.add_row("Win Rate", f"[{'green' if result.win_rate >= 0.5 else 'red'}]{result.win_rate:.1%}[/]")
    summary.add_row("", "")
    summary.add_row("Avg Win", _r(result.avg_win_r))
    summary.add_row("Avg Loss", _r(result.avg_loss_r))
    summary.add_row(
        "Expectancy",
        f"[{'green' if result.expectancy_r >= 0 else 'red'}]{_r(result.expectancy_r)}[/]",
    )
    summary.add_row("Profit Factor", f"{result.profit_factor:.2f}x")
    summary.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    summary.add_row("", "")
    summary.add_row("Initial Capital", f"${config.initial_capital:,.2f}")
    summary.add_row("Final Capital", f"${result.final_capital:,.2f}")
    summary.add_row(
        "Total Return",
        f"[{'green' if result.total_return_pct >= 0 else 'red'}]{_pct(result.total_return_pct)}[/]",
    )
    summary.add_row(
        "Max Drawdown",
        f"[{'red' if result.max_drawdown_pct > 10 else 'yellow'}]-{result.max_drawdown_pct:.1f}%[/]",
    )

    console.print(summary)

    # ── Trade log ──────────────────────────────────────────────────────────────
    if show_trades and result.trades:
        trade_table = Table(
            title="Trade Log",
            box=box.SIMPLE,
            show_lines=False,
        )
        trade_table.add_column("#", style="dim", width=4)
        trade_table.add_column("Action", width=6)
        trade_table.add_column("Entry", justify="right", width=10)
        trade_table.add_column("Exit", justify="right", width=10)
        trade_table.add_column("Outcome", width=12)
        trade_table.add_column("PnL R", justify="right", width=8)
        trade_table.add_column("PnL %", justify="right", width=8)
        trade_table.add_column("Bars", justify="right", width=6)

        for t in result.trades:
            color = "green" if t.is_win else ("red" if t.is_loss else "yellow")
            trade_table.add_row(
                str(t.trade_idx),
                t.action,
                f"{t.entry_price:.4f}",
                f"{t.exit_price:.4f}",
                f"[{color}]{t.outcome}[/]",
                f"[{color}]{t.pnl_r:+.2f}R[/]",
                f"[{color}]{t.pnl_pct:+.2f}%[/]",
                str(t.bars_held),
            )

        console.print(trade_table)

    console.print(f"\n[dim]{result.summary()}[/dim]\n")


if __name__ == "__main__":
    app()
