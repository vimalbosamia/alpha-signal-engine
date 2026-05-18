"""
Multi-symbol pipeline runner.

Schedules periodic pipeline runs for each watchlist symbol
using APScheduler. One run per symbol per bar interval.

No trades are executed. Signals are only produced and logged.
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from libs.core.config.settings import get_settings
from libs.core.logging.logger import get_logger, configure_logging
from libs.core.models.domain import AssetClass, Timeframe
from libs.data.providers.alpaca.provider import AlpacaDataProvider
from libs.data.providers.binance.provider import BinanceDataProvider
from libs.data.providers.binance_futures.provider import BinanceFuturesProvider
from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy
from libs.strategies.reversal.shooting_star_reversal import ShootingStarReversalStrategy
from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
from libs.strategies.breakout.support_breakdown import SupportBreakdownStrategy
from libs.strategies.continuation.pullback import PullbackContinuationStrategy
from libs.strategies.continuation.pullback_bear import PullbackBearContinuationStrategy
from libs.strategies.momentum.rsi_strategy import RSIStrategy
from libs.strategies.momentum.macd_crossover import MACDCrossoverStrategy
from libs.strategies.trend.ema_crossover import EMACrossoverStrategy
from libs.audit.audit_log import AuditLog
from libs.monitoring.metrics import MetricsCollector
from libs.core.events.bus import EventBus
from libs.monitoring.position_watcher import PositionWatcher
from libs.monitoring.outcome_tracker import SignalOutcomeTracker, load_muted_strategies_from_db
from apps.signal_agent.pipeline import SignalPipeline

log = get_logger(__name__)

DEFAULT_STRATEGIES = [
    HammerReversalStrategy(),
    ShootingStarReversalStrategy(),
    ResistanceBreakoutStrategy(),
    SupportBreakdownStrategy(),
    PullbackContinuationStrategy(),
    PullbackBearContinuationStrategy(),
    RSIStrategy(),
    MACDCrossoverStrategy(),
    EMACrossoverStrategy(),
]


class SignalRunner:
    """
    Runs signal pipelines for all configured symbols.
    Call run_all() once per bar interval.
    """

    def __init__(
        self,
        timeframe: Timeframe = Timeframe.FIVE_MIN,
        audit_log: AuditLog | None = None,
        metrics: MetricsCollector | None = None,
        event_bus: EventBus | None = None,
        position_poll_interval: int = 30,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self._timeframe = timeframe
        self._audit = audit_log or AuditLog(
            log_dir=settings.storage.audit_log_dir,
            enabled=settings.storage.enable_audit_log,
        )
        self._metrics = metrics or MetricsCollector()
        self._bus = event_bus or EventBus()

        # Build provider-specific pipelines
        self._stock_pipeline: SignalPipeline | None = None
        self._crypto_pipeline: SignalPipeline | None = None
        self._futures_pipeline: SignalPipeline | None = None
        self._alpaca: AlpacaDataProvider | None = None
        self._binance: BinanceDataProvider | None = None
        self._binance_futures: BinanceFuturesProvider | None = None

        if settings.enable_stocks:
            self._alpaca = AlpacaDataProvider()
            self._stock_pipeline = SignalPipeline(
                provider=self._alpaca,
                strategies=DEFAULT_STRATEGIES,
                audit_log=self._audit,
                metrics=self._metrics,
                event_bus=self._bus,
            )

        if settings.enable_crypto:
            self._binance = BinanceDataProvider()
            self._crypto_pipeline = SignalPipeline(
                provider=self._binance,
                strategies=DEFAULT_STRATEGIES,
                audit_log=self._audit,
                metrics=self._metrics,
                event_bus=self._bus,
            )

        if settings.signal.enable_futures:
            self._binance_futures = BinanceFuturesProvider()
            self._futures_pipeline = SignalPipeline(
                provider=self._binance_futures,
                strategies=DEFAULT_STRATEGIES,
                audit_log=self._audit,
                metrics=self._metrics,
                event_bus=self._bus,
            )

        # Position watcher: alerts when open signals hit TP or SL
        self._watcher = PositionWatcher(
            crypto_provider=self._binance,
            stock_provider=self._alpaca,
            futures_provider=self._binance_futures,
            event_bus=self._bus,
            poll_interval=position_poll_interval,
        )

        # Outcome tracker: checks if signals were directionally correct after N mins
        self._outcome_tracker = SignalOutcomeTracker(
            crypto_provider=self._binance,
            stock_provider=self._alpaca,
            check_after_minutes=30,
            poll_interval=30,
        )

    async def run_all(self) -> None:
        """Run one pass over all configured symbols."""
        tasks = []

        if self._stock_pipeline:
            for symbol in self._settings.signal.stock_symbols:
                tasks.append(
                    self._stock_pipeline.run_once(symbol, AssetClass.STOCK, self._timeframe)
                )

        if self._crypto_pipeline:
            for symbol in self._settings.signal.crypto_symbols:
                tasks.append(
                    self._crypto_pipeline.run_once(symbol, AssetClass.CRYPTO, self._timeframe)
                )

        if self._futures_pipeline:
            for symbol in self._settings.signal.futures_symbols:
                tasks.append(
                    self._futures_pipeline.run_once(symbol, AssetClass.CRYPTO, self._timeframe)
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error("runner_task_error", error=str(r))
                elif isinstance(r, list):
                    for signal in r:
                        # Position watcher: alert when TP/SL hit
                        self._watcher.track(signal)
                        # Outcome tracker: record if signal direction was correct
                        self._outcome_tracker.enqueue(signal)
                        # Paper trading: dispatch to all bots
                        if hasattr(self, '_paper_engine'):
                            self._paper_engine.dispatch_signal(signal)

    async def run_loop(self, interval_seconds: int = 300) -> None:
        """Run continuously, sleeping between passes. Starts position watcher."""
        log.info("runner_started", interval_seconds=interval_seconds)
        # Restore muted strategies from DB before the pipeline starts
        await load_muted_strategies_from_db()
        # Launch position watcher as a concurrent background task
        watcher_task = asyncio.create_task(self._watcher.run_loop())
        outcome_task = asyncio.create_task(self._outcome_tracker.run_loop())

        # Start paper trading engine
        from libs.paper_trading.engine import PaperTradingEngine
        from apps.dashboard.paper_page import set_paper_engine
        self._paper_engine = PaperTradingEngine()
        await self._paper_engine.start(self._bus)
        set_paper_engine(self._paper_engine)
        log.info("paper_trading_engine_wired")

        # Paper trading exit checker — polls prices every 30s
        async def _paper_exit_loop():
            while True:
                try:
                    # Collect symbols from open positions
                    positions = self._paper_engine.get_all_open_positions()
                    if positions:
                        symbols = {p["symbol"] for p in positions}
                        prices: dict[str, float] = {}
                        for sym in symbols:
                            try:
                                # Use Binance for crypto (USDT pairs), Alpaca for stocks
                                if sym.endswith("USDT") and self._binance:
                                    price = await self._binance.get_latest_price(sym)
                                elif self._alpaca:
                                    price = await self._alpaca.get_latest_price(sym)
                                else:
                                    price = None
                                if price:
                                    prices[sym] = price
                            except Exception:
                                pass
                        if prices:
                            closed = self._paper_engine.check_all_exits(prices)
                            if closed:
                                log.info("paper_exits_resolved", count=len(closed))
                            self._paper_engine.snapshot_equity(prices)
                except Exception as exc:
                    log.debug("paper_exit_loop_error", error=str(exc))
                await asyncio.sleep(30)

        paper_exit_task = asyncio.create_task(_paper_exit_loop())

        try:
            while True:
                try:
                    await self.run_all()
                except Exception as exc:
                    log.error("runner_loop_error", error=str(exc))
                await asyncio.sleep(interval_seconds)
        finally:
            watcher_task.cancel()
            outcome_task.cancel()
            paper_exit_task.cancel()
