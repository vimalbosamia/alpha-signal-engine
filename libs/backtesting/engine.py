"""
Backtesting Engine.

Replays a single strategy over a historical OHLCV DataFrame, bar-by-bar,
and measures signal quality using simulated trade outcomes.

Design rules:
  - No live data or async I/O — accepts a pre-loaded DataFrame.
  - Outcome simulation is conservative: when both SL and TP1 could be touched
    on the same bar, assume SL was hit first (worst-case fill).
  - Entry price = midpoint of entry_zone ± slippage.
  - Position sizing: risk a fixed percentage of capital per trade.
  - Never raises — errors are logged and the bar is skipped.
  - Returns an immutable BacktestResult.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from libs.analysis.indicators.engine import IndicatorsEngine
from libs.analysis.levels.engine import KeyLevelsEngine
from libs.analysis.regime.engine import RegimeEngine
from libs.analysis.structure.engine import MarketStructureEngine
from libs.analysis.volume.engine import VolumeEngine
from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, SignalAction, Timeframe
from libs.signals.confluence.engine import ConfluenceEngine

if TYPE_CHECKING:
    from libs.strategies.base.strategy import BaseStrategy

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_HOLD_BARS = 48          # close trade after this many bars if not resolved
TRADE_OPEN = "OPEN"
OUTCOME_WIN_TP1 = "WIN_TP1"
OUTCOME_WIN_TP2 = "WIN_TP2"
OUTCOME_LOSS_SL = "LOSS_SL"
OUTCOME_TIMEOUT = "TIMEOUT"


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestConfig:
    """Simulation parameters."""
    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 1.0     # % of capital risked per trade
    commission_pct: float = 0.05        # round-trip commission (% of notional)
    slippage_pct: float = 0.02          # % slippage on entry fill
    min_warmup_bars: int = 50           # bars to skip before generating signals
    max_hold_bars: int = MAX_HOLD_BARS  # force-close after this many bars
    min_confluence_score: float = 0.50  # minimum score to count as a signal


# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Single backtested trade."""
    trade_idx: int
    symbol: str
    strategy_name: str
    action: str                   # BUY or SELL
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    entry_bar_idx: int

    exit_price: float = 0.0
    exit_bar_idx: int = -1
    outcome: str = TRADE_OPEN
    pnl_pct: float = 0.0          # % return on entry price
    pnl_r: float = 0.0            # profit in units of R (1R = stop distance)
    bars_held: int = 0

    @property
    def risk_per_unit(self) -> float:
        """Distance from entry to stop (in price units)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def is_win(self) -> bool:
        return self.outcome in (OUTCOME_WIN_TP1, OUTCOME_WIN_TP2)

    @property
    def is_loss(self) -> bool:
        return self.outcome == OUTCOME_LOSS_SL

    @property
    def is_closed(self) -> bool:
        return self.outcome != TRADE_OPEN


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BacktestResult:
    """Aggregate metrics for a completed backtest."""
    symbol: str
    strategy_name: str
    timeframe: str
    asset_class: str
    config: BacktestConfig

    # Bar counts
    total_bars: int
    bars_tested: int          # after warmup

    # Signal & trade counts
    total_signals: int        # candidates that passed confluence threshold
    total_trades: int         # closed trades (signals that became entries)
    wins: int
    losses: int
    timeouts: int             # trades closed at max_hold_bars

    # Rate metrics
    win_rate: float           # wins / total_trades
    loss_rate: float

    # R-based metrics
    avg_win_r: float          # avg profit in R on winning trades
    avg_loss_r: float         # avg loss in R on losing trades (negative)
    expectancy_r: float       # avg PnL in R per trade (includes timeouts)
    profit_factor: float      # gross_wins_r / abs(gross_losses_r)

    # Capital metrics
    total_return_pct: float
    max_drawdown_pct: float
    final_capital: float
    sharpe_ratio: float       # simplified: expectancy / std(pnl_r)

    # Per-trade records
    trades: list[Trade] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"{self.symbol} [{self.strategy_name}] "
            f"| Trades: {self.total_trades} "
            f"| Win%: {self.win_rate:.1%} "
            f"| Expectancy: {self.expectancy_r:+.2f}R "
            f"| Return: {self.total_return_pct:+.1f}% "
            f"| MaxDD: {self.max_drawdown_pct:.1f}%"
        )


# ── Engine ─────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Replay a strategy over historical data and return BacktestResult.

    Usage:
        engine = BacktestEngine()
        result = engine.run(df, strategy, AssetClass.CRYPTO, Timeframe.FIFTEEN_MIN)
    """

    def __init__(self) -> None:
        self._structure = MarketStructureEngine()
        self._levels = KeyLevelsEngine()
        self._volume = VolumeEngine()
        self._regime = RegimeEngine()
        self._indicators = IndicatorsEngine()
        self._confluence = ConfluenceEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy: "BaseStrategy",
        asset_class: AssetClass,
        timeframe: Timeframe,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        """
        Run the full backtest. Returns BacktestResult. Never raises.
        """
        if config is None:
            config = BacktestConfig()

        symbol = str(df.index.name or "UNKNOWN")
        # Allow caller to embed symbol via a column or attribute
        if hasattr(df, "attrs") and "symbol" in df.attrs:
            symbol = df.attrs["symbol"]

        try:
            return self._run(df, strategy, asset_class, timeframe, config, symbol)
        except Exception as exc:
            log.error("backtest_engine_error", error=str(exc))
            return self._empty_result(symbol, strategy.name, timeframe, asset_class, config, df)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(
        self,
        df: pd.DataFrame,
        strategy: "BaseStrategy",
        asset_class: AssetClass,
        timeframe: Timeframe,
        config: BacktestConfig,
        symbol: str,
    ) -> BacktestResult:
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame missing columns: {required_cols - set(df.columns)}")

        n = len(df)
        warmup = max(config.min_warmup_bars, strategy.min_bars_required)
        bars_tested = n - warmup

        if bars_tested <= 0:
            return self._empty_result(symbol, strategy.name, timeframe, asset_class, config, df)

        capital = config.initial_capital
        peak_capital = capital
        max_drawdown_pct = 0.0
        capital_curve: list[float] = [capital]

        trades: list[Trade] = []
        open_trades: list[Trade] = []
        total_signals = 0
        trade_counter = 0

        for bar_idx in range(warmup, n):
            df_slice = df.iloc[: bar_idx + 1]
            current_bar = df.iloc[bar_idx]

            # ── Update open trades for this bar ───────────────────────────────
            still_open: list[Trade] = []
            for trade in open_trades:
                self._update_trade(trade, current_bar, bar_idx, config)
                if trade.is_closed:
                    # Apply PnL to capital
                    risk_amount = capital * (config.risk_per_trade_pct / 100.0)
                    if trade.risk_per_unit > 0:
                        pnl_money = trade.pnl_r * risk_amount
                    else:
                        pnl_money = 0.0
                    # Deduct commission
                    commission = capital * (config.commission_pct / 100.0) * 2
                    capital = max(0.0, capital + pnl_money - commission)
                    capital_curve.append(capital)
                    peak_capital = max(peak_capital, capital)
                    dd = (peak_capital - capital) / peak_capital * 100.0 if peak_capital > 0 else 0.0
                    max_drawdown_pct = max(max_drawdown_pct, dd)
                    trades.append(trade)
                else:
                    still_open.append(trade)
            open_trades = still_open

            # ── Generate new signal ───────────────────────────────────────────
            # Only generate one active signal per bar
            if len(open_trades) > 0:
                continue  # already in a trade; no pyramiding

            try:
                structure = self._structure.analyze(df_slice)
                levels = self._levels.analyze(df_slice, asset_class)
                regime = self._regime.analyze(df_slice)
                indicators = self._indicators.compute(df_slice)

                candidate = strategy.generate_candidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    df=df_slice,
                    df_htf=None,
                    session=None,
                    quality=None,
                    structure=structure,
                    levels=levels,
                    volume=None,
                    regime=regime,
                    indicators=indicators,
                )
            except Exception as exc:
                log.debug("backtest_strategy_error", bar=bar_idx, error=str(exc))
                continue

            if candidate is None:
                continue

            volume = self._volume.analyze(df_slice, candidate.proposed_action)
            breakdown = self._confluence.score(candidate, structure, levels, volume, regime)

            if breakdown.weighted_total < config.min_confluence_score:
                continue

            total_signals += 1

            # Build trade
            entry_mid = (candidate.entry_zone_low + candidate.entry_zone_high) / 2.0
            slip_factor = config.slippage_pct / 100.0
            if candidate.proposed_action == SignalAction.BUY:
                entry_price = entry_mid * (1.0 + slip_factor)
            else:
                entry_price = entry_mid * (1.0 - slip_factor)

            trade_counter += 1
            trade = Trade(
                trade_idx=trade_counter,
                symbol=symbol,
                strategy_name=strategy.name,
                action=candidate.proposed_action.value,
                confidence=breakdown.weighted_total,
                entry_price=entry_price,
                stop_loss=candidate.stop_loss,
                take_profit_1=candidate.take_profit_1,
                take_profit_2=candidate.take_profit_2,
                entry_bar_idx=bar_idx,
            )
            open_trades.append(trade)

        # Force-close any remaining open trades at last bar close
        if n > 0:
            last_bar = df.iloc[-1]
            for trade in open_trades:
                trade.exit_price = float(last_bar["close"])
                trade.exit_bar_idx = n - 1
                trade.outcome = OUTCOME_TIMEOUT
                trade.bars_held = (n - 1) - trade.entry_bar_idx
                trade.pnl_r = self._calc_pnl_r(trade, trade.exit_price)
                trade.pnl_pct = self._calc_pnl_pct(trade, trade.exit_price)
                trades.append(trade)

        return self._build_result(
            symbol=symbol,
            strategy_name=strategy.name,
            timeframe=timeframe,
            asset_class=asset_class,
            config=config,
            total_bars=n,
            bars_tested=bars_tested,
            total_signals=total_signals,
            trades=trades,
            initial_capital=config.initial_capital,
            final_capital=capital,
            max_drawdown_pct=max_drawdown_pct,
        )

    def _update_trade(
        self,
        trade: Trade,
        bar: pd.Series,
        bar_idx: int,
        config: BacktestConfig,
    ) -> None:
        """Check if bar hits SL, TP1, TP2, or timeout. Updates trade in-place."""
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        bars_held = bar_idx - trade.entry_bar_idx

        # Timeout
        if bars_held >= config.max_hold_bars:
            exit_price = float(bar["close"])
            trade.exit_price = exit_price
            trade.exit_bar_idx = bar_idx
            trade.outcome = OUTCOME_TIMEOUT
            trade.bars_held = bars_held
            trade.pnl_r = self._calc_pnl_r(trade, exit_price)
            trade.pnl_pct = self._calc_pnl_pct(trade, exit_price)
            return

        if trade.action == SignalAction.BUY.value:
            sl_hit = bar_low <= trade.stop_loss
            tp1_hit = bar_high >= trade.take_profit_1
            tp2_hit = (
                trade.take_profit_2 is not None
                and bar_high >= trade.take_profit_2
            )

            if sl_hit and tp1_hit:
                # Ambiguous bar — conservative: assume SL hit first
                # unless TP1 is much closer to open
                bar_open = float(bar["open"])
                sl_dist = abs(bar_open - trade.stop_loss)
                tp1_dist = abs(trade.take_profit_1 - bar_open)
                if tp1_dist < sl_dist:
                    tp1_hit, sl_hit = True, False
                else:
                    sl_hit, tp1_hit = True, False

            if sl_hit:
                self._close_trade(trade, trade.stop_loss, bar_idx, bars_held, OUTCOME_LOSS_SL)
            elif tp2_hit:
                self._close_trade(trade, trade.take_profit_2, bar_idx, bars_held, OUTCOME_WIN_TP2)  # type: ignore[arg-type]
            elif tp1_hit:
                self._close_trade(trade, trade.take_profit_1, bar_idx, bars_held, OUTCOME_WIN_TP1)

        else:  # SELL
            sl_hit = bar_high >= trade.stop_loss
            tp1_hit = bar_low <= trade.take_profit_1
            tp2_hit = (
                trade.take_profit_2 is not None
                and bar_low <= trade.take_profit_2
            )

            if sl_hit and tp1_hit:
                bar_open = float(bar["open"])
                sl_dist = abs(bar_open - trade.stop_loss)
                tp1_dist = abs(trade.take_profit_1 - bar_open)
                if tp1_dist < sl_dist:
                    tp1_hit, sl_hit = True, False
                else:
                    sl_hit, tp1_hit = True, False

            if sl_hit:
                self._close_trade(trade, trade.stop_loss, bar_idx, bars_held, OUTCOME_LOSS_SL)
            elif tp2_hit:
                self._close_trade(trade, trade.take_profit_2, bar_idx, bars_held, OUTCOME_WIN_TP2)  # type: ignore[arg-type]
            elif tp1_hit:
                self._close_trade(trade, trade.take_profit_1, bar_idx, bars_held, OUTCOME_WIN_TP1)

    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        bar_idx: int,
        bars_held: int,
        outcome: str,
    ) -> None:
        trade.exit_price = exit_price
        trade.exit_bar_idx = bar_idx
        trade.outcome = outcome
        trade.bars_held = bars_held
        trade.pnl_r = self._calc_pnl_r(trade, exit_price)
        trade.pnl_pct = self._calc_pnl_pct(trade, exit_price)

    @staticmethod
    def _calc_pnl_r(trade: Trade, exit_price: float) -> float:
        """Return PnL in units of R (positive = profit)."""
        risk = trade.risk_per_unit
        if risk <= 0:
            return 0.0
        if trade.action == SignalAction.BUY.value:
            return (exit_price - trade.entry_price) / risk
        else:
            return (trade.entry_price - exit_price) / risk

    @staticmethod
    def _calc_pnl_pct(trade: Trade, exit_price: float) -> float:
        """Return % PnL relative to entry price."""
        if trade.entry_price <= 0:
            return 0.0
        if trade.action == SignalAction.BUY.value:
            return (exit_price - trade.entry_price) / trade.entry_price * 100.0
        else:
            return (trade.entry_price - exit_price) / trade.entry_price * 100.0

    # ── Result builder ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        *,
        symbol: str,
        strategy_name: str,
        timeframe: Timeframe,
        asset_class: AssetClass,
        config: BacktestConfig,
        total_bars: int,
        bars_tested: int,
        total_signals: int,
        trades: list[Trade],
        initial_capital: float,
        final_capital: float,
        max_drawdown_pct: float,
    ) -> BacktestResult:
        closed = [t for t in trades if t.is_closed]
        wins = [t for t in closed if t.is_win]
        losses = [t for t in closed if t.is_loss]
        timeouts = [t for t in closed if t.outcome == OUTCOME_TIMEOUT]

        total_trades = len(closed)
        win_count = len(wins)
        loss_count = len(losses)
        timeout_count = len(timeouts)

        win_rate = win_count / total_trades if total_trades > 0 else 0.0
        loss_rate = loss_count / total_trades if total_trades > 0 else 0.0

        win_rs = [t.pnl_r for t in wins]
        loss_rs = [t.pnl_r for t in losses]
        all_rs = [t.pnl_r for t in closed]

        avg_win_r = statistics.mean(win_rs) if win_rs else 0.0
        avg_loss_r = statistics.mean(loss_rs) if loss_rs else 0.0
        expectancy_r = statistics.mean(all_rs) if all_rs else 0.0

        gross_win = sum(r for r in all_rs if r > 0)
        gross_loss = abs(sum(r for r in all_rs if r < 0))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

        total_return_pct = (final_capital - initial_capital) / initial_capital * 100.0

        # Simplified Sharpe: expectancy / std(pnl_r), annualised not applied
        if len(all_rs) >= 2:
            std_r = statistics.stdev(all_rs)
            sharpe = expectancy_r / std_r if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            timeframe=timeframe.value,
            asset_class=asset_class.value,
            config=config,
            total_bars=total_bars,
            bars_tested=bars_tested,
            total_signals=total_signals,
            total_trades=total_trades,
            wins=win_count,
            losses=loss_count,
            timeouts=timeout_count,
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            expectancy_r=expectancy_r,
            profit_factor=profit_factor,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            final_capital=final_capital,
            sharpe_ratio=sharpe,
            trades=closed,
        )

    @staticmethod
    def _empty_result(
        symbol: str,
        strategy_name: str,
        timeframe: Timeframe,
        asset_class: AssetClass,
        config: BacktestConfig,
        df: pd.DataFrame,
    ) -> BacktestResult:
        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name,
            timeframe=timeframe.value,
            asset_class=asset_class.value,
            config=config,
            total_bars=len(df),
            bars_tested=0,
            total_signals=0,
            total_trades=0,
            wins=0,
            losses=0,
            timeouts=0,
            win_rate=0.0,
            loss_rate=0.0,
            avg_win_r=0.0,
            avg_loss_r=0.0,
            expectancy_r=0.0,
            profit_factor=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            final_capital=config.initial_capital,
            sharpe_ratio=0.0,
            trades=[],
        )
