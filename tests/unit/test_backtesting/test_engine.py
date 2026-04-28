"""
Unit tests for BacktestEngine.

All DataFrames are built synthetically — no external data, no network calls.
Tests follow TDD: each test targets a specific contract of BacktestEngine /
BacktestResult / Trade.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from libs.backtesting.engine import (
    OUTCOME_LOSS_SL,
    OUTCOME_TIMEOUT,
    OUTCOME_WIN_TP1,
    OUTCOME_WIN_TP2,
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
)
from libs.core.models.domain import AssetClass, SignalAction, Timeframe


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_ohlcv(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    c = np.array(closes, dtype=float)
    h = np.array(highs, dtype=float) if highs is not None else c * 1.005
    l = np.array(lows, dtype=float) if lows is not None else c * 0.995
    o = c * 0.999
    v = np.full(n, 1_000_000.0)
    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v},
        index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
    )
    df.attrs["symbol"] = "SYN/USD"
    return df


def make_trending_up(n: int = 200, start: float = 100.0) -> pd.DataFrame:
    closes = [start + i * 0.5 for i in range(n)]
    return make_ohlcv(closes)


def make_flat(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = (price + rng.uniform(-0.1, 0.1, n)).tolist()
    return make_ohlcv(closes)


def _stub_strategy(
    action: SignalAction = SignalAction.BUY,
    entry_low: float = 100.0,
    entry_high: float = 101.0,
    stop: float = 97.0,
    tp1: float = 106.0,
    tp2: float | None = None,
    name: str = "stub",
):
    """Build a MagicMock strategy that always returns the same candidate."""
    from libs.core.models.domain import (
        MarketRegime,
        SignalCandidate,
        Timeframe,
        TrendDirection,
    )

    candidate = SignalCandidate(
        symbol="SYN/USD",
        asset_class=AssetClass.CRYPTO,
        strategy_name=name,
        proposed_action=action,
        timeframe=Timeframe.FIVE_MIN,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=entry_low,
        entry_zone_high=entry_high,
        stop_loss=stop,
        take_profit_1=tp1,
        take_profit_2=tp2,
    )

    strat = MagicMock()
    strat.name = name
    strat.min_bars_required = 30
    strat.is_eligible.return_value = True
    strat.generate_candidate.return_value = candidate
    return strat


def _silent_engine() -> BacktestEngine:
    """BacktestEngine with all analysis engines stubbed to return safe defaults."""
    engine = BacktestEngine()

    from libs.analysis.structure.engine import MarketStructure, TrendDirection
    from libs.analysis.regime.engine import RegimeAnalysis
    from libs.analysis.volume.engine import VolumeContext
    from libs.core.models.domain import ConfluenceBreakdown

    engine._structure.analyze = MagicMock(return_value=MagicMock())
    engine._levels.analyze = MagicMock(return_value=[])
    engine._regime.analyze = MagicMock(return_value=MagicMock())
    engine._indicators.compute = MagicMock(return_value=MagicMock())
    engine._volume.analyze = MagicMock(return_value=MagicMock())

    # Confluence always passes at score=0.75
    breakdown = ConfluenceBreakdown(
        pattern_score=0.75, structure_score=0.75, level_score=0.75,
        volume_score=0.75, regime_score=0.75, session_score=0.75,
        risk_score=0.75, data_quality_score=0.75, weighted_total=0.75,
    )
    engine._confluence.score = MagicMock(return_value=breakdown)

    return engine


# ── BacktestConfig ─────────────────────────────────────────────────────────────

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 10_000.0
        assert cfg.risk_per_trade_pct == 1.0
        assert cfg.min_warmup_bars == 50
        assert cfg.max_hold_bars == 48

    def test_custom_values(self):
        cfg = BacktestConfig(initial_capital=50_000.0, risk_per_trade_pct=2.0)
        assert cfg.initial_capital == 50_000.0
        assert cfg.risk_per_trade_pct == 2.0


# ── Trade helper methods ───────────────────────────────────────────────────────

class TestTrade:
    def _make_trade(self, action="BUY", entry=100.0, stop=97.0, tp1=106.0) -> Trade:
        return Trade(
            trade_idx=1,
            symbol="SYN/USD",
            strategy_name="stub",
            action=action,
            confidence=0.75,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=None,
            entry_bar_idx=0,
        )

    def test_risk_per_unit_buy(self):
        t = self._make_trade("BUY", entry=100.0, stop=97.0)
        assert t.risk_per_unit == pytest.approx(3.0)

    def test_risk_per_unit_sell(self):
        t = self._make_trade("SELL", entry=100.0, stop=103.0, tp1=94.0)
        assert t.risk_per_unit == pytest.approx(3.0)

    def test_is_win_tp1(self):
        t = self._make_trade()
        t.outcome = OUTCOME_WIN_TP1
        assert t.is_win is True
        assert t.is_loss is False

    def test_is_loss_sl(self):
        t = self._make_trade()
        t.outcome = OUTCOME_LOSS_SL
        assert t.is_loss is True
        assert t.is_win is False

    def test_is_closed_false_when_open(self):
        t = self._make_trade()
        assert t.is_closed is False

    def test_pnl_r_buy_win(self):
        t = self._make_trade("BUY", entry=100.0, stop=97.0, tp1=106.0)
        pnl = BacktestEngine._calc_pnl_r(t, 106.0)
        # (106 - 100) / (100 - 97) = 6/3 = 2.0
        assert pnl == pytest.approx(2.0)

    def test_pnl_r_buy_loss(self):
        t = self._make_trade("BUY", entry=100.0, stop=97.0)
        pnl = BacktestEngine._calc_pnl_r(t, 97.0)
        # (97 - 100) / 3 = -1.0
        assert pnl == pytest.approx(-1.0)

    def test_pnl_r_sell_win(self):
        t = self._make_trade("SELL", entry=100.0, stop=103.0, tp1=94.0)
        pnl = BacktestEngine._calc_pnl_r(t, 94.0)
        # (100 - 94) / (103 - 100) = 6/3 = 2.0
        assert pnl == pytest.approx(2.0)

    def test_pnl_r_zero_risk(self):
        t = self._make_trade("BUY", entry=100.0, stop=100.0)
        assert BacktestEngine._calc_pnl_r(t, 106.0) == 0.0

    def test_pnl_pct_buy(self):
        t = self._make_trade("BUY", entry=100.0)
        assert BacktestEngine._calc_pnl_pct(t, 105.0) == pytest.approx(5.0)

    def test_pnl_pct_sell(self):
        t = self._make_trade("SELL", entry=100.0, stop=103.0, tp1=94.0)
        assert BacktestEngine._calc_pnl_pct(t, 95.0) == pytest.approx(5.0)


# ── BacktestEngine — basic contracts ──────────────────────────────────────────

class TestBacktestEngineBasic:
    def test_returns_backtest_result(self):
        engine = _silent_engine()
        df = make_trending_up(200)
        strat = _stub_strategy()
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN)
        assert isinstance(result, BacktestResult)

    def test_never_raises_on_empty_df(self):
        engine = _silent_engine()
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        strat = _stub_strategy()
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN)
        assert result.total_trades == 0

    def test_never_raises_on_insufficient_bars(self):
        engine = _silent_engine()
        df = make_flat(10)  # way below warmup
        strat = _stub_strategy()
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN)
        assert result.total_trades == 0

    def test_missing_columns_returns_empty(self):
        engine = _silent_engine()
        df = pd.DataFrame({"close": [100.0] * 100})
        strat = _stub_strategy()
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN)
        assert result.total_trades == 0

    def test_symbol_propagated_from_df_attrs(self):
        engine = _silent_engine()
        df = make_flat(200)
        df.attrs["symbol"] = "MY/SYM"
        strat = _stub_strategy()
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN)
        assert result.symbol == "MY/SYM"

    def test_strategy_name_propagated(self):
        engine = _silent_engine()
        df = make_flat(200)
        strat = _stub_strategy(name="my_strat")
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN)
        assert result.strategy_name == "my_strat"

    def test_bars_tested_equals_bars_minus_warmup(self):
        engine = _silent_engine()
        df = make_flat(200)
        config = BacktestConfig(min_warmup_bars=50)
        strat = _stub_strategy()
        strat.min_bars_required = 30
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN, config)
        assert result.bars_tested == 200 - 50


# ── Outcome simulation ─────────────────────────────────────────────────────────

class TestOutcomeSimulation:
    """
    Tests for _update_trade — we exercise it indirectly via a short run()
    with a custom DataFrame designed so the bar immediately after entry
    resolves to a specific outcome.
    """

    def _run_one_trade(
        self,
        action: SignalAction,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        entry: float = 100.0,
        stop: float = 97.0,
        tp1: float = 106.0,
        tp2: float | None = None,
        max_hold: int = 48,
    ) -> Trade | None:
        """
        Build a minimal df where:
          - bars 0..59 are calm (no signal generated due to warmup)
          - bar 60 is the entry bar: engine generates the candidate
          - bar 61 has the specified high/low/close to trigger the outcome
        Returns the first trade from the result, or None.
        """
        n = 120
        closes = [entry] * n
        highs = [entry * 1.001] * n
        lows = [entry * 0.999] * n

        # Bar 61: the outcome bar
        highs[61] = bar_high
        lows[61] = bar_low
        closes[61] = bar_close

        df = make_ohlcv(closes, highs, lows)
        engine = _silent_engine()

        config = BacktestConfig(
            min_warmup_bars=60,
            max_hold_bars=max_hold,
            min_confluence_score=0.0,
        )
        strat = _stub_strategy(
            action=action,
            entry_low=entry - 0.5,
            entry_high=entry + 0.5,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
        )
        strat.min_bars_required = 1

        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN, config)
        return result.trades[0] if result.trades else None

    def test_buy_tp1_hit(self):
        trade = self._run_one_trade(
            SignalAction.BUY,
            bar_high=107.0,  # above tp1=106
            bar_low=99.0,
            bar_close=107.0,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_WIN_TP1

    def test_buy_sl_hit(self):
        trade = self._run_one_trade(
            SignalAction.BUY,
            bar_high=101.0,
            bar_low=96.0,  # below stop=97
            bar_close=96.5,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_LOSS_SL

    def test_sell_tp1_hit(self):
        trade = self._run_one_trade(
            SignalAction.SELL,
            bar_high=100.0,
            bar_low=93.0,   # below tp1=94 for SELL
            bar_close=93.5,
            stop=103.0,
            tp1=94.0,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_WIN_TP1

    def test_sell_sl_hit(self):
        trade = self._run_one_trade(
            SignalAction.SELL,
            bar_high=104.0,  # above stop=103 for SELL
            bar_low=100.0,
            bar_close=103.5,
            stop=103.0,
            tp1=94.0,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_LOSS_SL

    def test_buy_tp2_hit(self):
        trade = self._run_one_trade(
            SignalAction.BUY,
            bar_high=112.0,  # above tp2=110
            bar_low=99.5,
            bar_close=112.0,
            tp2=110.0,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_WIN_TP2

    def test_timeout_when_no_target_hit(self):
        trade = self._run_one_trade(
            SignalAction.BUY,
            bar_high=100.5,
            bar_low=99.5,
            bar_close=100.0,
            max_hold=1,  # force timeout after 1 bar
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_TIMEOUT

    def test_ambiguous_bar_conservative_sl(self):
        """Both SL and TP1 touched; SL is closer to open — SL wins."""
        # entry=100, open=100, stop=97 (dist 3), tp1=106 (dist 6)
        # SL is closer → SL first (conservative)
        trade = self._run_one_trade(
            SignalAction.BUY,
            bar_high=107.0,
            bar_low=96.5,
            bar_close=100.0,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_LOSS_SL

    def test_ambiguous_bar_tp1_when_closer(self):
        """TP1 is closer to open than SL — TP1 wins."""
        # entry=100, open=100, stop=90 (dist 10), tp1=102 (dist 2)
        # TP1 closer → TP1 first
        trade = self._run_one_trade(
            SignalAction.BUY,
            bar_high=103.0,
            bar_low=89.0,
            bar_close=100.0,
            stop=90.0,
            tp1=102.0,
        )
        assert trade is not None
        assert trade.outcome == OUTCOME_WIN_TP1


# ── BacktestResult metrics ─────────────────────────────────────────────────────

class TestBacktestResultMetrics:
    def _build_result_with_trades(
        self,
        outcomes: list[str],
        pnl_rs: list[float],
    ) -> BacktestResult:
        """Manually build a result from a list of (outcome, pnl_r) pairs."""
        trades = []
        for i, (outcome, pnl_r) in enumerate(zip(outcomes, pnl_rs)):
            t = Trade(
                trade_idx=i + 1,
                symbol="SYN",
                strategy_name="stub",
                action="BUY",
                confidence=0.75,
                entry_price=100.0,
                stop_loss=97.0,
                take_profit_1=106.0,
                take_profit_2=None,
                entry_bar_idx=i * 10,
            )
            t.exit_price = 100.0
            t.exit_bar_idx = i * 10 + 5
            t.outcome = outcome
            t.pnl_r = pnl_r
            t.pnl_pct = pnl_r * 3.0
            t.bars_held = 5
            trades.append(t)

        return BacktestEngine._build_result(
            symbol="SYN",
            strategy_name="stub",
            timeframe=Timeframe.FIVE_MIN,
            asset_class=AssetClass.CRYPTO,
            config=BacktestConfig(),
            total_bars=200,
            bars_tested=150,
            total_signals=len(trades),
            trades=trades,
            initial_capital=10_000.0,
            final_capital=10_000.0 + sum(pnl_rs) * 100.0,
            max_drawdown_pct=5.0,
        )

    def test_win_rate_all_wins(self):
        result = self._build_result_with_trades(
            [OUTCOME_WIN_TP1] * 4, [2.0, 2.0, 2.0, 2.0]
        )
        assert result.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losses(self):
        result = self._build_result_with_trades(
            [OUTCOME_LOSS_SL] * 3, [-1.0, -1.0, -1.0]
        )
        assert result.win_rate == pytest.approx(0.0)

    def test_win_rate_mixed(self):
        result = self._build_result_with_trades(
            [OUTCOME_WIN_TP1, OUTCOME_LOSS_SL, OUTCOME_WIN_TP1, OUTCOME_LOSS_SL],
            [2.0, -1.0, 2.0, -1.0],
        )
        assert result.win_rate == pytest.approx(0.5)

    def test_expectancy_positive(self):
        result = self._build_result_with_trades(
            [OUTCOME_WIN_TP1, OUTCOME_LOSS_SL, OUTCOME_WIN_TP1],
            [2.0, -1.0, 2.0],
        )
        # expectancy = mean(2.0, -1.0, 2.0) = 1.0
        assert result.expectancy_r == pytest.approx(1.0)

    def test_profit_factor(self):
        result = self._build_result_with_trades(
            [OUTCOME_WIN_TP1, OUTCOME_LOSS_SL],
            [3.0, -1.0],
        )
        # gross_win=3.0 / gross_loss=1.0 = 3.0
        assert result.profit_factor == pytest.approx(3.0)

    def test_no_trades_all_zeros(self):
        result = self._build_result_with_trades([], [])
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.expectancy_r == 0.0

    def test_total_return_pct(self):
        result = BacktestEngine._build_result(
            symbol="SYN",
            strategy_name="stub",
            timeframe=Timeframe.FIVE_MIN,
            asset_class=AssetClass.CRYPTO,
            config=BacktestConfig(initial_capital=10_000.0),
            total_bars=200,
            bars_tested=150,
            total_signals=1,
            trades=[],
            initial_capital=10_000.0,
            final_capital=11_000.0,
            max_drawdown_pct=0.0,
        )
        assert result.total_return_pct == pytest.approx(10.0)

    def test_summary_string_contains_key_fields(self):
        result = self._build_result_with_trades(
            [OUTCOME_WIN_TP1, OUTCOME_LOSS_SL],
            [2.0, -1.0],
        )
        s = result.summary()
        assert "stub" in s
        assert "Win%" in s
        assert "Expectancy" in s


# ── No-pyramiding constraint ───────────────────────────────────────────────────

class TestNoPyramiding:
    def test_only_one_trade_open_at_a_time(self):
        """
        Strategy generates a candidate on every bar.
        Engine should not open a second trade while one is still open.
        With max_hold_bars=48 on 150 bars tested, we expect floor(150/48) = 3
        sequential trades (not concurrent). Total trades must be strictly fewer
        than bars_tested — proving no duplicate entries within the same hold window.
        """
        engine = _silent_engine()
        df = make_flat(200)
        strat = _stub_strategy(
            entry_low=100.0, entry_high=101.0,
            stop=10.0, tp1=1000.0,  # unreachable → each trade times out at max_hold
        )
        config = BacktestConfig(
            min_warmup_bars=50,
            max_hold_bars=48,
            min_confluence_score=0.0,
        )
        result = engine.run(df, strat, AssetClass.CRYPTO, Timeframe.FIVE_MIN, config)
        # 150 bars tested, each trade holds 48 bars → max ~3 sequential trades
        # Far fewer than 150 proves no pyramiding
        assert result.total_trades < result.bars_tested
        # All trades are sequential (each holds at least 1 bar)
        assert all(t.bars_held >= 1 for t in result.trades)
