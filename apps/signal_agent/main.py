"""
CLI entrypoint for the AI Trading Signal Agent.

Usage:
    uv run python -m apps.signal_agent.main run
    uv run python -m apps.signal_agent.main run --timeframe 15m
    uv run python -m apps.signal_agent.main once --symbol AAPL --asset-class stock
"""
from __future__ import annotations
import asyncio
from enum import Enum
import typer
from libs.core.config.settings import get_settings
from libs.core.logging.logger import configure_logging, get_logger

app = typer.Typer(help="AI Trading Signal Agent — signal-only, never trades")
log = get_logger(__name__)


class TimeframeChoice(str, Enum):
    m1 = "1m"; m3 = "3m"; m5 = "5m"; m15 = "15m"; m30 = "30m"
    h1 = "1h"; h4 = "4h"; d1 = "1d"


@app.command()
def run(
    timeframe: TimeframeChoice = typer.Option(TimeframeChoice.m5, help="Bar timeframe"),
    interval: int = typer.Option(300, help="Seconds between scan passes"),
) -> None:
    """Run the agent in a continuous loop."""
    from libs.core.models.domain import Timeframe
    from apps.signal_agent.runner import SignalRunner

    settings = get_settings()
    configure_logging(settings.observability.log_level)

    tf = Timeframe(timeframe.value)
    log.info("agent_starting", mode=settings.agent_mode.value, timeframe=tf.value)
    typer.echo(f"Starting signal agent | mode={settings.agent_mode.value} | tf={tf.value}")
    typer.echo("This agent NEVER places trades. Signals only.")

    runner = SignalRunner(timeframe=tf)
    asyncio.run(runner.run_loop(interval_seconds=interval))


@app.command()
def once(
    symbol: str = typer.Argument(..., help="Symbol to scan, e.g. AAPL or BTCUSDT"),
    asset_class: str = typer.Option("stock", help="stock or crypto"),
    timeframe: TimeframeChoice = typer.Option(TimeframeChoice.m5),
) -> None:
    """Run a single scan pass for one symbol."""
    from libs.core.models.domain import AssetClass, Timeframe
    from libs.data.providers.alpaca.provider import AlpacaDataProvider
    from libs.data.providers.binance.provider import BinanceDataProvider
    from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy
    from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
    from libs.strategies.continuation.pullback import PullbackContinuationStrategy
    from apps.signal_agent.pipeline import SignalPipeline

    settings = get_settings()
    configure_logging(settings.observability.log_level)

    ac = AssetClass(asset_class)
    tf = Timeframe(timeframe.value)
    provider = BinanceDataProvider() if ac == AssetClass.CRYPTO else AlpacaDataProvider()

    pipeline = SignalPipeline(
        provider=provider,
        strategies=[
            HammerReversalStrategy(),
            ResistanceBreakoutStrategy(),
            PullbackContinuationStrategy(),
        ],
    )

    async def _run():
        outputs = await pipeline.run_once(symbol, ac, tf)
        if not outputs:
            typer.echo(f"No signals found for {symbol}")
            return
        for out in outputs:
            typer.echo(out.to_display())

    asyncio.run(_run())


@app.command()
def serve(
    timeframe: TimeframeChoice = typer.Option(TimeframeChoice.m15, help="Scan timeframe"),
    interval: int = typer.Option(300, help="Seconds between scan passes"),
    port: int = typer.Option(8000, help="Dashboard port"),
) -> None:
    """Run scanner + localhost dashboard together (http://localhost:<port>)."""
    import threading
    import uvicorn
    from libs.core.models.domain import Timeframe
    from apps.signal_agent.runner import SignalRunner
    from apps.dashboard.server import app as dashboard_app, set_shared_watcher
    from libs.data.storage.db import init_db

    settings = get_settings()
    configure_logging(settings.observability.log_level)
    tf = Timeframe(timeframe.value)

    typer.echo(f"Starting AI Trading Signal Agent")
    typer.echo(f"  Mode: {settings.agent_mode.value} | Timeframe: {tf.value} | Interval: {interval}s")
    typer.echo(f"  Dashboard: http://localhost:{port}")
    typer.echo(f"  Position watcher: alerts when TP/SL is hit (polls every 30s)")
    typer.echo(f"  This agent NEVER places trades. Signals only.")
    typer.echo(f"  Press Ctrl+C to stop.")

    async def _scanner():
        await init_db()
        runner = SignalRunner(timeframe=tf)
        set_shared_watcher(runner._watcher)
        await runner.run_loop(interval_seconds=interval)

    def _run_scanner():
        asyncio.run(_scanner())

    # Start scanner in background thread
    scanner_thread = threading.Thread(target=_run_scanner, daemon=True)
    scanner_thread.start()

    # Run dashboard in main thread (blocking)
    uvicorn.run(dashboard_app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    app()
