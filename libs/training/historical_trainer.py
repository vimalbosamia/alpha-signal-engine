"""
HistoricalTrainer — trains the self-training agent on 1 year of Binance data.

Runs the full signal pipeline on historical candle windows, simulates trades
via the backtesting engine, and feeds every outcome to the SelfTrainingCoordinator.

This is NOT live trading. It replays historical data to build pattern scores,
tune strategy parameters, calibrate RL reward weights, and advance training phases
— so the agent is pre-trained before it starts paper trading.

Usage:
    trainer = HistoricalTrainer()
    await trainer.download_data(symbols, timeframe="15m", months=12)
    results = await trainer.train(symbols, timeframe="15m")
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, Timeframe

log = get_logger(__name__)

# Full crypto watchlist for training — matches settings.py crypto_watchlist
DEFAULT_TRAINING_SYMBOLS = [
    "BTCUSDT", "ETHUSDT",
    "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "NEARUSDT", "SUIUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT", "INJUSDT",
    "ARBUSDT", "OPUSDT", "IMXUSDT",
    "PEPEUSDT", "SHIBUSDT",
    "LTCUSDT", "APTUSDT", "STXUSDT", "FETUSDT", "RENDERUSDT", "TONUSDT", "BCHUSDT",
    "POLUSDT",
]

# Futures watchlist — subset that trades on Binance futures
DEFAULT_FUTURES_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT",
    "POLUSDT", "LTCUSDT", "NEARUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT",
    "INJUSDT", "FETUSDT", "RENDERUSDT", "TONUSDT",
]

# Multiple timeframes for richer training — match available data
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]

# Fast mode: top 6 strategies that generate most signals
FAST_STRATEGIES_NAMES = {
    "ema_crossover", "macd_crossover",
    "resistance_breakout", "support_breakdown",
    "hammer_reversal", "pullback_continuation",
}


@dataclass
class TrainingResult:
    """Results from a historical training run."""
    symbol: str
    timeframe: str
    total_signals: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_r: float = 0.0
    errors: int = 0
    duration_seconds: float = 0.0


@dataclass
class TrainingSummary:
    """Aggregate results across all symbols and timeframes."""
    results: list[TrainingResult] = field(default_factory=list)
    total_signals: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    phase_before: str = ""
    phase_after: str = ""
    duration_seconds: float = 0.0


class HistoricalTrainer:
    """Trains the self-training agent on historical Binance data."""

    def __init__(self, data_dir: str = "data/historical") -> None:
        self._data_dir = data_dir

    async def download_data(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "15m",
        months: int = 12,
        force: bool = False,
        market_type: str = "spot",
    ) -> dict[str, Any]:
        """Download historical data from Binance.

        Args:
            market_type: "spot" or "futures" — determines data source.

        Returns dict of symbol → parquet path.
        """
        from libs.data.providers.historical.downloader import BinanceHistoricalDownloader

        if market_type == "futures":
            symbols = symbols or DEFAULT_FUTURES_SYMBOLS
        else:
            symbols = symbols or DEFAULT_TRAINING_SYMBOLS

        dl = BinanceHistoricalDownloader(data_dir=self._data_dir, market_type=market_type)

        log.info("download_start", symbols=len(symbols), timeframe=timeframe,
                 months=months, market_type=market_type)

        results = await dl.download_all(
            symbols=symbols,
            timeframe=timeframe,
            months=months,
            force=force,
            max_concurrent=3,
        )

        log.info("download_complete", success=len(results),
                 failed=len(symbols) - len(results), market_type=market_type)
        return results

    async def train(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        months: int = 12,
        fast_mode: bool = False,
    ) -> TrainingSummary:
        """Run full training loop on historical data.

        For each symbol+timeframe:
          1. Load historical candles from parquet
          2. Walk through data in windows (simulating real-time)
          3. Run full pipeline analysis on each window
          4. For signals generated, simulate trades using backtest engine
          5. Feed every trade outcome to SelfTrainingCoordinator

        Args:
            fast_mode: Use only 6 core strategies for faster training.

        Returns TrainingSummary with aggregate results.
        """
        from libs.backtesting.engine import BacktestEngine, BacktestConfig
        from libs.data.providers.historical.provider import HistoricalDataProvider
        from libs.learning.coordinator import get_coordinator
        from apps.signal_agent.runner import DEFAULT_STRATEGIES

        symbols = symbols or DEFAULT_TRAINING_SYMBOLS
        timeframes = timeframes or DEFAULT_TIMEFRAMES

        strategies = DEFAULT_STRATEGIES
        if fast_mode:
            strategies = [s for s in DEFAULT_STRATEGIES if s.name in FAST_STRATEGIES_NAMES]
            log.info("training.fast_mode", strategies=len(strategies))

        coordinator = get_coordinator()

        # Restore existing learning state before training so we build on prior progress
        try:
            from libs.paper_trading.state_persistence import _restore_learning_systems
            _restore_learning_systems()
            log.info("training.restored_prior_state")
        except Exception as exc:
            log.debug("training.no_prior_state", error=str(exc))

        phase_before = coordinator.current_phase.name

        provider = HistoricalDataProvider(data_dir=self._data_dir)
        backtest_engine = BacktestEngine()
        config = BacktestConfig(
            initial_capital=10_000.0,
            risk_per_trade_pct=1.0,
            commission_pct=0.05,
            slippage_pct=0.02,
            max_hold_bars=48,
            min_confluence_score=0.45,  # lower bar for more training data
        )

        summary = TrainingSummary(phase_before=phase_before)
        start_time = datetime.now(timezone.utc)

        total_combos = len(timeframes) * len(symbols)
        combo_idx = 0

        for tf_str in timeframes:
            tf = Timeframe(tf_str)

            for symbol in symbols:
                combo_idx += 1
                result = await self._train_symbol(
                    symbol=symbol,
                    timeframe=tf,
                    tf_str=tf_str,
                    provider=provider,
                    backtest_engine=backtest_engine,
                    config=config,
                    strategies=strategies,
                    coordinator=coordinator,
                )
                summary.results.append(result)
                summary.total_signals += result.total_signals
                summary.total_trades += result.total_trades
                summary.total_wins += result.wins
                summary.total_losses += result.losses

                log.info("symbol_trained",
                         symbol=symbol, timeframe=tf_str,
                         progress=f"{combo_idx}/{total_combos}",
                         trades=result.total_trades,
                         wins=result.wins, losses=result.losses,
                         cumulative_trades=summary.total_trades)

        summary.win_rate = (
            summary.total_wins / summary.total_trades
            if summary.total_trades > 0 else 0
        )
        summary.phase_after = coordinator.current_phase.name
        summary.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        # Save training state
        self._save_training_state()

        log.info("training_complete",
                 total_trades=summary.total_trades,
                 win_rate=round(summary.win_rate, 4),
                 phase_before=summary.phase_before,
                 phase_after=summary.phase_after,
                 duration=round(summary.duration_seconds, 1))

        return summary

    async def _train_symbol(
        self,
        symbol: str,
        timeframe: Timeframe,
        tf_str: str,
        provider: Any,
        backtest_engine: Any,
        config: Any,
        strategies: list,
        coordinator: Any,
    ) -> TrainingResult:
        """Train on one symbol+timeframe using backtest engine."""
        result = TrainingResult(symbol=symbol, timeframe=tf_str)
        start = datetime.now(timezone.utc)

        # Preload data
        if not provider.preload(symbol, tf_str):
            log.warning("no_data_skip", symbol=symbol, timeframe=tf_str)
            return result

        df = provider._downloader.load_symbol(symbol, tf_str)
        if df is None or len(df) < 200:
            log.warning("insufficient_data", symbol=symbol, rows=len(df) if df is not None else 0)
            return result

        df.attrs["symbol"] = symbol

        # Run backtest for each strategy
        for strategy in strategies:
            try:
                bt_result = backtest_engine.run(
                    df=df,
                    strategy=strategy,
                    asset_class=AssetClass.CRYPTO,
                    timeframe=timeframe,
                    config=config,
                )

                result.total_trades += len(bt_result.trades)
                result.total_signals += len(bt_result.trades)

                # Feed each trade to coordinator
                for trade in bt_result.trades:
                    won = trade.pnl_r > 0
                    if won:
                        result.wins += 1
                    else:
                        result.losses += 1
                    result.total_pnl_r += trade.pnl_r

                    # Extract pattern info from trade
                    patterns = []
                    if hasattr(trade, "patterns") and trade.patterns:
                        patterns = trade.patterns if isinstance(trade.patterns, list) else [trade.patterns]

                    regime = getattr(trade, "regime", "unknown") or "unknown"
                    strategy_name = getattr(strategy, "name", str(strategy.__class__.__name__))

                    # Build market context from trade data for vector memory
                    rr_val = abs(trade.pnl_r) if trade.pnl_r > 0 else abs(trade.pnl_r) * -1
                    market_ctx = {
                        "rsi": getattr(trade, "rsi", 50.0) or 50.0,
                        "macd_histogram": getattr(trade, "macd_histogram", 0.0) or 0.0,
                        "atr_pct": getattr(trade, "atr_pct", 1.0) or 1.0,
                        "volume_ratio": getattr(trade, "volume_ratio", 1.0) or 1.0,
                        "regime": regime,
                        "market_regime": regime,
                        "trend_strength": getattr(trade, "adx", 25.0) or 25.0,
                        "bb_position": getattr(trade, "bb_pct_b", 0.5) or 0.5,
                        "ema_alignment": getattr(trade, "ema_ratio", 1.0) or 1.0,
                    }

                    # Feed to coordinator (force_all=True bypasses phase gating
                    # so ALL subsystems learn from every historical trade)
                    coordinator.on_trade_close(
                        strategy=strategy_name,
                        won=won,
                        pnl=trade.pnl_r,
                        rr=rr_val,
                        confidence=trade.confidence,
                        patterns=patterns,
                        regime=regime,
                        disciplined_exit=trade.outcome in ("WIN_TP1", "WIN_TP2", "LOSS_SL"),
                        regime_aligned=won,
                        force_all=True,
                        # Extra kwargs for vector memory + RL engine
                        symbol=symbol,
                        market_context=market_ctx,
                        trade_id=f"hist_{symbol}_{tf_str}_{trade.trade_idx}",
                    )

            except Exception as exc:
                result.errors += 1
                log.warning("backtest_error", symbol=symbol, strategy=str(strategy),
                            error=str(exc))

        result.duration_seconds = (datetime.now(timezone.utc) - start).total_seconds()
        return result

    def _save_training_state(self) -> None:
        """Persist all learning state after training."""
        try:
            from libs.paper_trading.state_persistence import (
                _save_learning_systems, _ensure_dir,
            )
            _ensure_dir()
            _save_learning_systems()
            log.info("training_state_saved")
        except Exception as exc:
            log.warning("training_state_save_failed", error=str(exc))
