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
# Reversal (5)
from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy
from libs.strategies.reversal.shooting_star_reversal import ShootingStarReversalStrategy
from libs.strategies.reversal.candle_reversal import CandlestickReversalStrategy
from libs.strategies.reversal.failed_breakout import FailedBreakoutReversalStrategy
from libs.strategies.reversal.liquidity_sweep import LiquiditySweepReversalStrategy
# Breakout (7)
from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
from libs.strategies.breakout.support_breakdown import SupportBreakdownStrategy
from libs.strategies.breakout.volume_breakout import VolumeBreakoutStrategy
from libs.strategies.breakout.atr_breakout import ATRBreakoutStrategy
from libs.strategies.breakout.range_breakout import RangeBreakoutStrategy
from libs.strategies.breakout.break_retest import BreakRetestStrategy
from libs.strategies.breakout.volatility_squeeze import VolatilitySqueezeStrategy
from libs.strategies.breakout.opening_range import OpeningRangeBreakoutStrategy
# Continuation (3)
from libs.strategies.continuation.pullback import PullbackContinuationStrategy
from libs.strategies.continuation.pullback_bear import PullbackBearContinuationStrategy
from libs.strategies.continuation.momentum_continuation import MomentumContinuationStrategy
# Momentum (2)
from libs.strategies.momentum.rsi_strategy import RSIStrategy
from libs.strategies.momentum.macd_crossover import MACDCrossoverStrategy
# Trend (3)
from libs.strategies.trend.ema_crossover import EMACrossoverStrategy
from libs.strategies.trend.sma_crossover import SMACrossoverStrategy
from libs.strategies.trend.trend_following import TrendFollowingStrategy
# Mean Reversion (3)
from libs.strategies.mean_reversion.bollinger_reversion import BollingerReversionStrategy
from libs.strategies.mean_reversion.gap_fill import GapFillStrategy
from libs.strategies.mean_reversion.range_fade import RangeFadeStrategy
# Level (2)
from libs.strategies.level.vwap_reclaim import VWAPReclaimStrategy
from libs.strategies.level.fibonacci_bounce import FibonacciBounceStrategy
# Confirmation (1)
from libs.strategies.confirmation.mtf_alignment import MTFAlignmentStrategy
from libs.audit.audit_log import AuditLog
from libs.monitoring.metrics import MetricsCollector
from libs.core.events.bus import EventBus
from libs.monitoring.position_watcher import PositionWatcher
from libs.monitoring.outcome_tracker import SignalOutcomeTracker, load_muted_strategies_from_db
from apps.signal_agent.pipeline import SignalPipeline

log = get_logger(__name__)

DEFAULT_STRATEGIES = [
    # Reversal (5)
    HammerReversalStrategy(),
    ShootingStarReversalStrategy(),
    CandlestickReversalStrategy(),
    FailedBreakoutReversalStrategy(),
    LiquiditySweepReversalStrategy(),
    # Breakout (7)
    ResistanceBreakoutStrategy(),
    SupportBreakdownStrategy(),
    VolumeBreakoutStrategy(),
    ATRBreakoutStrategy(),
    RangeBreakoutStrategy(),
    BreakRetestStrategy(),
    VolatilitySqueezeStrategy(),
    OpeningRangeBreakoutStrategy(),
    # Continuation (3)
    PullbackContinuationStrategy(),
    PullbackBearContinuationStrategy(),
    MomentumContinuationStrategy(),
    # Momentum (2)
    RSIStrategy(),
    MACDCrossoverStrategy(),
    # Trend (3)
    EMACrossoverStrategy(),
    SMACrossoverStrategy(),
    TrendFollowingStrategy(),
    # Mean Reversion (3)
    BollingerReversionStrategy(),
    GapFillStrategy(),
    RangeFadeStrategy(),
    # Level (2)
    VWAPReclaimStrategy(),
    FibonacciBounceStrategy(),
    # Confirmation (1)
    MTFAlignmentStrategy(),
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
                trading_mode="futures",
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
                    positions = self._paper_engine.get_all_open_positions()
                    if positions:
                        # Separate spot and futures symbols
                        spot_symbols = set()
                        futures_symbols = set()
                        for p in positions:
                            mode = p.get("trading_mode", "spot")
                            if "futures" in str(mode):
                                futures_symbols.add(p["symbol"])
                            else:
                                spot_symbols.add(p["symbol"])

                        prices: dict[str, float] = {}
                        # Fetch spot prices
                        for sym in spot_symbols:
                            try:
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
                        # Fetch futures prices (different endpoint)
                        for sym in futures_symbols:
                            try:
                                if self._binance_futures:
                                    price = await self._binance_futures.get_latest_price(sym)
                                    if price:
                                        prices[sym] = price
                            except Exception:
                                pass
                        if prices:
                            closed = self._paper_engine.check_all_exits(prices)
                            if closed:
                                log.info("paper_exits_resolved", count=len(closed),
                                         trades=[c.get("symbol", "?") for c in closed])
                            self._paper_engine.snapshot_equity(prices)
                        log.debug("paper_exit_tick", positions=len(positions),
                                  prices_fetched=len(prices))
                    else:
                        # No positions — still snapshot balance for equity curve
                        self._paper_engine.snapshot_equity({})
                except Exception as exc:
                    log.warning("paper_exit_loop_error", error=str(exc))
                await asyncio.sleep(10)  # Check every 10s for fast exits

        paper_exit_task = asyncio.create_task(_paper_exit_loop())

        # Trade re-analysis loop — every 5 min, re-run bias on open positions
        # If bias flipped against the trade → close it
        async def _trade_reanalysis_loop():
            while True:
                await asyncio.sleep(300)  # Every 5 minutes
                try:
                    positions = self._paper_engine.get_all_open_positions()
                    if not positions:
                        continue

                    symbols_to_check = {p["symbol"] for p in positions}
                    log.info("trade_reanalysis_start", symbols=len(symbols_to_check))

                    for sym in symbols_to_check:
                        try:
                            # Fetch fresh candles
                            provider = self._binance if sym.endswith("USDT") and self._binance else self._alpaca
                            if not provider:
                                continue

                            from datetime import timedelta
                            from libs.core.models.domain import Timeframe
                            from libs.data.candles.builder import CandleBuilder
                            from libs.analysis.indicators.engine import IndicatorsEngine
                            from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
                            from libs.analysis.structure.market_structure import MarketStructureAnalyzer
                            from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
                            from libs.analysis.regime.engine import RegimeEngine

                            now = datetime.now()
                            df = await provider.get_candles(sym, Timeframe.FIFTEEN_MIN, now - timedelta(days=3), now)
                            df = CandleBuilder().enrich(df)

                            if len(df) < 20:
                                continue

                            # Re-compute bias
                            indicators = IndicatorsEngine().compute(df)
                            regime = RegimeEngine().analyze(df)
                            structure = MarketStructureAnalyzer().analyze(df)

                            last_close = float(df["close"].iloc[-1])
                            rel_vol = float(df["relative_volume"].iloc[-1]) if "relative_volume" in df else 1.0
                            is_bull = bool(df["is_bullish"].iloc[-1]) if "is_bullish" in df else True

                            indicator_bias = IndicatorBiasAnalyzer().analyze_all(
                                rsi=indicators.rsi or 50, rsi_prev=indicators.rsi_prev or 50,
                                macd_line=indicators.macd_line or 0, macd_signal=indicators.macd_signal or 0,
                                histogram=indicators.macd_histogram or 0, histogram_prev=indicators.macd_histogram_prev or 0,
                                close=last_close, bb_upper=indicators.bb_upper or last_close + 1,
                                bb_lower=indicators.bb_lower or last_close - 1, bb_pct_b=indicators.bb_pct_b or 0.5,
                                ema_9=indicators.ema_9 or last_close, ema_20=indicators.ema_20 or last_close,
                                ema_50=indicators.ema_50 or last_close,
                                adx=indicators.adx or 0,
                                relative_volume=rel_vol, is_bullish_candle=is_bull,
                            )

                            new_bias = BullBearBiasEngine().score(BiasInput(
                                indicator_bullish=indicator_bias.bullish_score,
                                indicator_bearish=indicator_bias.bearish_score,
                                structure_bias=structure.trend_bias,
                                structure_strength=structure.strength,
                                candle_bullish_count=0, candle_bearish_count=0, candle_total=1,
                                regime_supports_direction=True,
                                volume_confirms=rel_vol > 1.2,
                                htf_bias="neutral",
                            ))

                            # Update bias status + check for 2-check exit on all positions
                            from libs.paper_trading.bias_exit import (
                                compute_bias_status,
                                should_exit_on_bias_flip,
                            )

                            for bot in self._paper_engine.bots:
                                for trade in list(bot.portfolio.open_trades):
                                    if trade.symbol != sym:
                                        continue

                                    # Update live bias fields on trade
                                    trade.current_bias = new_bias.net_bias
                                    trade.bias_status = compute_bias_status(
                                        trade.direction, new_bias.net_bias,
                                    )

                                    # 2-check exit rule (document 4, section 9)
                                    should_close, new_count = should_exit_on_bias_flip(
                                        direction=trade.direction,
                                        market_mode=trade.market_mode,
                                        current_bias=new_bias.net_bias,
                                        bias_adverse_count=trade.bias_adverse_count,
                                    )
                                    trade.bias_adverse_count = new_count

                                    if should_close:
                                        price = await provider.get_latest_price(sym)
                                        if price:
                                            reason = (
                                                f"Bias flipped to {new_bias.net_bias.upper()} "
                                                f"for {new_count} checks — closing {trade.direction}"
                                            )
                                            trade.exit_reason = reason
                                            result = bot.portfolio.close_trade(
                                                trade.id, price, f"BIAS_FLIP: {reason}",
                                            )
                                            log.info("trade_bias_flip_closed",
                                                     bot=bot.name, symbol=sym,
                                                     direction=trade.direction,
                                                     market_mode=trade.market_mode,
                                                     adverse_checks=new_count,
                                                     new_bias=new_bias.net_bias,
                                                     pnl=result.get("realized_pnl", 0))
                                    elif trade.bias_status == "CONFLICT":
                                        log.info("trade_bias_conflict_warning",
                                                 bot=bot.name, symbol=sym,
                                                 direction=trade.direction,
                                                 adverse_count=new_count,
                                                 bias=new_bias.net_bias)

                        except Exception as sym_exc:
                            log.debug("reanalysis_symbol_error", symbol=sym, error=str(sym_exc))

                    log.info("trade_reanalysis_complete", symbols=len(symbols_to_check))
                except Exception as exc:
                    log.warning("trade_reanalysis_error", error=str(exc))

        reanalysis_task = asyncio.create_task(_trade_reanalysis_loop())

        # Periodic validation loop — runs Monte Carlo, VaR, walk-forward every 30 min
        async def _validation_loop():
            while True:
                await asyncio.sleep(1800)  # Every 30 minutes
                try:
                    from libs.validation.monte_carlo import MonteCarloEngine
                    from libs.risk.var import VaREngine
                    from libs.validation.walk_forward import WalkForwardValidator
                    from libs.learning.tracker import PerformanceTracker
                    from libs.learning.model_registry import ModelVersionRegistry

                    # Collect trade returns from paper engine
                    closed = self._paper_engine.get_closed_trades(limit=500)
                    if len(closed) < 10:
                        continue

                    returns = [t.get("pnl_pct", 0) for t in closed]

                    # Monte Carlo robustness check
                    mc = MonteCarloEngine().simulate(returns)
                    log.info("validation_monte_carlo",
                             robust=mc.is_robust, prob_profit=round(mc.probability_of_profit, 2),
                             median_return=round(mc.median_return_pct, 2))

                    # VaR calculation
                    daily_returns = returns[-30:] if len(returns) >= 30 else returns
                    var = VaREngine().calculate(daily_returns)
                    log.info("validation_var",
                             var_95=round(var.var_95, 2), cvar_95=round(var.cvar_95, 2),
                             acceptable=var.is_acceptable)

                    # Walk-forward per strategy
                    strategy_outcomes: dict[str, list[bool]] = {}
                    for t in closed:
                        strat = t.get("strategy_name", "unknown")
                        won = t.get("realized_pnl", 0) > 0
                        strategy_outcomes.setdefault(strat, []).append(won)

                    validator = WalkForwardValidator()
                    for strat, outcomes in strategy_outcomes.items():
                        if len(outcomes) >= 20:
                            result = validator.validate(strat, outcomes)
                            log.info("validation_walkforward",
                                     strategy=strat, status=result.validation_status,
                                     cap=result.confidence_cap)

                    # Track performance per strategy
                    tracker = PerformanceTracker()
                    for t in closed:
                        tracker.record_outcome(
                            t.get("strategy_name", "unknown"), "strategy",
                            won=t.get("realized_pnl", 0) > 0,
                            pnl=t.get("realized_pnl", 0),
                        )

                    log.info("validation_loop_complete", trades=len(closed))
                except Exception as exc:
                    log.debug("validation_loop_error", error=str(exc))

        validation_task = asyncio.create_task(_validation_loop())

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
            reanalysis_task.cancel()
            validation_task.cancel()
