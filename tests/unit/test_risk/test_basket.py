"""
Unit tests for libs/risk/basket.py — BasketExpectancyEngine.

Tests follow the AAA (Arrange-Act-Assert) pattern.
All tests use only frozen dataclasses so there are no hidden side-effects.
"""
from __future__ import annotations

import pytest

from libs.risk.basket import BasketAnalysis, BasketExpectancyEngine, BasketSignal


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_signal(
    symbol: str = "AAPL",
    asset_class: str = "stock",
    action: str = "BUY",
    tp1_pct: float = 6.0,
    stop_pct: float = 2.0,
    risk_reward: float = 3.0,
    confidence: float = 0.75,
) -> BasketSignal:
    return BasketSignal(
        symbol=symbol,
        asset_class=asset_class,
        action=action,
        tp1_pct=tp1_pct,
        stop_pct=stop_pct,
        risk_reward=risk_reward,
        confidence=confidence,
    )


def make_engine() -> BasketExpectancyEngine:
    return BasketExpectancyEngine()


# ── Test 1: Single signal → basic stats ──────────────────────────────────────

class TestSingleSignal:
    """A basket with exactly one signal should produce correct basic metrics."""

    def test_counts_correct(self) -> None:
        # Arrange
        signals = [make_signal(action="BUY")]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.signal_count == 1
        assert result.buy_count == 1
        assert result.sell_count == 0

    def test_avg_metrics_match_single_signal(self) -> None:
        # Arrange
        sig = make_signal(tp1_pct=5.0, stop_pct=2.5, risk_reward=2.0)
        engine = make_engine()

        # Act
        result = engine.analyze([sig])

        # Assert
        assert result.avg_risk_reward == pytest.approx(2.0)
        assert result.avg_tp1_pct == pytest.approx(5.0)
        assert result.avg_stop_pct == pytest.approx(2.5)

    def test_worst_best_case_single_signal(self) -> None:
        # Arrange
        sig = make_signal(tp1_pct=6.0, stop_pct=2.0)
        engine = make_engine()

        # Act
        result = engine.analyze([sig])

        # Assert — worst = -stop_pct, best = +tp1_pct
        assert result.worst_case_loss_pct == pytest.approx(-2.0)
        assert result.best_case_gain_pct == pytest.approx(6.0)

    def test_returns_basket_analysis_type(self) -> None:
        result = make_engine().analyze([make_signal()])
        assert isinstance(result, BasketAnalysis)


# ── Test 2: Breakeven win rate — R:R=2 → breakeven ~33% ─────────────────────

class TestBreakevenWinRate:
    """breakeven_win_rate = 1 / (1 + avg_risk_reward)."""

    def test_rr2_gives_33_percent(self) -> None:
        # Arrange
        signals = [make_signal(risk_reward=2.0)]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert  1/(1+2) = 0.333...
        assert result.breakeven_win_rate == pytest.approx(1 / 3, rel=1e-4)

    def test_rr3_gives_25_percent(self) -> None:
        # Arrange
        signals = [make_signal(risk_reward=3.0)]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert  1/(1+3) = 0.25
        assert result.breakeven_win_rate == pytest.approx(0.25, rel=1e-4)

    def test_multiple_signals_uses_avg_rr(self) -> None:
        # Arrange  avg R:R = (2 + 4) / 2 = 3.0 → breakeven = 0.25
        signals = [
            make_signal(risk_reward=2.0),
            make_signal(risk_reward=4.0),
        ]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.avg_risk_reward == pytest.approx(3.0)
        assert result.breakeven_win_rate == pytest.approx(0.25, rel=1e-4)


# ── Test 3: Expected returns at 50%, 60% ─────────────────────────────────────

class TestExpectedReturns:
    """expected_return(wr) = (wr * avg_tp1) - ((1 - wr) * avg_stop)."""

    def test_keys_present(self) -> None:
        result = make_engine().analyze([make_signal()])
        assert "40%" in result.expected_returns
        assert "50%" in result.expected_returns
        assert "55%" in result.expected_returns
        assert "60%" in result.expected_returns
        assert "65%" in result.expected_returns
        assert "70%" in result.expected_returns

    def test_50_percent_win_rate(self) -> None:
        # Arrange  avg_tp1=6, avg_stop=2
        # expected at 50% = 0.5*6 - 0.5*2 = 3 - 1 = 2.0
        signals = [make_signal(tp1_pct=6.0, stop_pct=2.0)]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.expected_returns["50%"] == pytest.approx(2.0, rel=1e-4)

    def test_60_percent_win_rate(self) -> None:
        # Arrange  avg_tp1=6, avg_stop=2
        # expected at 60% = 0.6*6 - 0.4*2 = 3.6 - 0.8 = 2.8
        signals = [make_signal(tp1_pct=6.0, stop_pct=2.0)]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.expected_returns["60%"] == pytest.approx(2.8, rel=1e-4)

    def test_higher_win_rate_gives_higher_return(self) -> None:
        result = make_engine().analyze([make_signal(tp1_pct=6.0, stop_pct=2.0)])
        returns = result.expected_returns
        # Monotonically increasing
        assert returns["40%"] < returns["50%"] < returns["55%"] < returns["60%"] < returns["65%"] < returns["70%"]


# ── Test 4: Worst/best case math ─────────────────────────────────────────────

class TestWorstBestCase:
    """worst_case = sum of all stop_pct (negative); best_case = sum of all tp1_pct."""

    def test_two_signals_worst_case(self) -> None:
        # Arrange  stop = 2 + 3 = 5 → worst = -5
        signals = [
            make_signal(stop_pct=2.0, tp1_pct=6.0),
            make_signal(stop_pct=3.0, tp1_pct=8.0),
        ]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.worst_case_loss_pct == pytest.approx(-5.0)

    def test_two_signals_best_case(self) -> None:
        # Arrange  tp1 = 6 + 8 = 14
        signals = [
            make_signal(stop_pct=2.0, tp1_pct=6.0),
            make_signal(stop_pct=3.0, tp1_pct=8.0),
        ]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.best_case_gain_pct == pytest.approx(14.0)

    def test_worst_case_is_always_negative(self) -> None:
        signals = [make_signal(stop_pct=1.5), make_signal(stop_pct=2.5)]
        result = make_engine().analyze(signals)
        assert result.worst_case_loss_pct < 0


# ── Test 5: Concentration risk — all crypto → "high" ─────────────────────────

class TestConcentrationHigh:
    """All crypto signals → concentration_risk = 'high'."""

    def test_all_crypto_is_high(self) -> None:
        # Arrange — 5 crypto signals
        signals = [make_signal(asset_class="crypto") for _ in range(5)]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.concentration_risk == "high"
        assert result.crypto_exposure_pct == pytest.approx(100.0)

    def test_all_stock_is_high(self) -> None:
        # 100% stock → also high concentration
        signals = [make_signal(asset_class="stock") for _ in range(4)]
        result = make_engine().analyze(signals)
        assert result.concentration_risk == "high"

    def test_mixed_exposure_is_lower_risk(self) -> None:
        # 50% crypto / 50% stock with unique symbols → not high
        signals = [
            make_signal(symbol="BTCUSDT", asset_class="crypto"),
            make_signal(symbol="ETHUSDT", asset_class="crypto"),
            make_signal(symbol="AAPL", asset_class="stock"),
            make_signal(symbol="MSFT", asset_class="stock"),
        ]
        result = make_engine().analyze(signals)
        assert result.concentration_risk in ("low", "moderate")

    def test_crypto_exposure_pct_correct(self) -> None:
        # 2 crypto out of 4 signals → 50%
        signals = [
            make_signal(asset_class="crypto"),
            make_signal(asset_class="crypto"),
            make_signal(asset_class="stock"),
            make_signal(asset_class="stock"),
        ]
        result = make_engine().analyze(signals)
        assert result.crypto_exposure_pct == pytest.approx(50.0)
        assert result.stock_exposure_pct == pytest.approx(50.0)

    def test_70_percent_crypto_is_high(self) -> None:
        # 7 crypto, 3 stock → 70% crypto → high
        signals = (
            [make_signal(asset_class="crypto")] * 7
            + [make_signal(asset_class="stock")] * 3
        )
        result = make_engine().analyze(signals)
        assert result.concentration_risk == "high"


# ── Test 6: Duplicate direction warning ───────────────────────────────────────

class TestDuplicateDirectionWarning:
    """Same symbol+action appearing 3+ times should raise a warning and count."""

    def test_three_same_symbol_action_triggers_warning(self) -> None:
        # Arrange — BTCUSDT BUY × 3
        signals = [
            make_signal(symbol="BTCUSDT", action="BUY"),
            make_signal(symbol="BTCUSDT", action="BUY"),
            make_signal(symbol="BTCUSDT", action="BUY"),
        ]
        engine = make_engine()

        # Act
        result = engine.analyze(signals)

        # Assert
        assert result.duplicate_direction_count > 0
        assert any("symbol" in w.lower() or "duplicate" in w.lower() or "multiple" in w.lower()
                   for w in result.warnings)

    def test_two_same_symbol_action_not_a_dup(self) -> None:
        # Two occurrences → duplicate_direction_count should reflect this
        # (spec says "count symbols that appear more than once with same action")
        signals = [
            make_signal(symbol="BTCUSDT", action="BUY"),
            make_signal(symbol="BTCUSDT", action="BUY"),
        ]
        result = make_engine().analyze(signals)
        # duplicate_direction_count counts pairs where the same symbol+action > 1
        assert result.duplicate_direction_count >= 1

    def test_different_directions_same_symbol_not_dup(self) -> None:
        # BUY + SELL on same symbol → not a duplicate direction
        signals = [
            make_signal(symbol="AAPL", action="BUY"),
            make_signal(symbol="AAPL", action="SELL"),
        ]
        result = make_engine().analyze(signals)
        assert result.duplicate_direction_count == 0

    def test_low_avg_rr_triggers_warning(self) -> None:
        # avg R:R = 1.0 → should warn about low R:R
        signals = [make_signal(risk_reward=1.0), make_signal(risk_reward=1.0)]
        result = make_engine().analyze(signals)
        assert any("r:r" in w.lower() or "risk" in w.lower() for w in result.warnings)


# ── Test 7: Empty basket → safe defaults ─────────────────────────────────────

class TestEmptyBasket:
    """analyze([]) must not raise and must return safe neutral values."""

    def test_no_exception_on_empty(self) -> None:
        result = make_engine().analyze([])
        assert isinstance(result, BasketAnalysis)

    def test_zero_counts(self) -> None:
        result = make_engine().analyze([])
        assert result.signal_count == 0
        assert result.buy_count == 0
        assert result.sell_count == 0

    def test_zero_avg_metrics(self) -> None:
        result = make_engine().analyze([])
        assert result.avg_risk_reward == pytest.approx(0.0)
        assert result.avg_tp1_pct == pytest.approx(0.0)
        assert result.avg_stop_pct == pytest.approx(0.0)

    def test_zero_worst_best_case(self) -> None:
        result = make_engine().analyze([])
        assert result.worst_case_loss_pct == pytest.approx(0.0)
        assert result.best_case_gain_pct == pytest.approx(0.0)

    def test_zero_breakeven_win_rate(self) -> None:
        result = make_engine().analyze([])
        assert result.breakeven_win_rate == pytest.approx(0.0)

    def test_zero_exposure(self) -> None:
        result = make_engine().analyze([])
        assert result.crypto_exposure_pct == pytest.approx(0.0)
        assert result.stock_exposure_pct == pytest.approx(0.0)

    def test_expected_returns_all_zero(self) -> None:
        result = make_engine().analyze([])
        for v in result.expected_returns.values():
            assert v == pytest.approx(0.0)

    def test_concentration_risk_low_on_empty(self) -> None:
        result = make_engine().analyze([])
        assert result.concentration_risk == "low"
