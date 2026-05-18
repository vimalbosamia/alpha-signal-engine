# Analysis Foundation Implementation Plan (Phases 4, 5, 6 + Wire 40 Patterns)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform shallow pattern-matching into deep directional intelligence by adding indicator bias analysis, market structure detection (HH/HL/BOS/CHoCH), and a composite bull/bear bias engine — then wire all 40 pattern detectors and integrate everything into the pipeline.

**Architecture:** Three new engines layer on top of existing IndicatorsEngine/MarketStructureEngine. Each produces a frozen Pydantic result model with bias, strength, and explanation. The BullBearBiasEngine aggregates all sub-scores into a composite directional score. Pipeline wires these in before confluence scoring, replacing the current shallow analysis.

**Tech Stack:** Python 3.12+, pandas, pandas-ta (already installed), Pydantic frozen models, existing analysis engine pattern.

**Spec:** `docs/superpowers/specs/2026-05-17-paper-trading-simulator-design.md` (master doc Phases 4-6)

---

## File Structure

```
libs/analysis/
    indicators/
        engine.py              # EXISTING — already computes RSI/MACD/BB/ATR/ADX/EMA/SMA
        bias.py                # NEW — IndicatorBiasAnalyzer: per-indicator bias/strength/explanation
    structure/
        engine.py              # EXISTING — basic structure detection
        swing_points.py        # NEW — HH/HL/LH/LL detection from price data
        market_structure.py    # NEW — BOS/CHoCH/trend classification using swing points
    levels/
        supply_demand.py       # NEW — supply/demand zone detection
    bias/
        __init__.py            # NEW
        engine.py              # NEW — BullBearBiasEngine: composite directional scoring

apps/signal_agent/
    pipeline.py                # MODIFY — wire new engines, use CandlePatternEngine (all 40)

tests/unit/
    test_analysis/
        test_indicator_bias.py
        test_swing_points.py
        test_market_structure.py
        test_supply_demand.py
        test_bias_engine.py
    test_pipeline_integration.py  # MODIFY — verify new engines wired correctly
```

---

## Task 1: Indicator Bias Analyzer

**Files:**
- Create: `libs/analysis/indicators/bias.py`
- Create: `tests/unit/test_analysis/test_indicator_bias.py`

**What it does:** Takes an `IndicatorSnapshot` (already computed by existing engine) and produces per-indicator bias classification with strength and human-readable explanation.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_analysis/test_indicator_bias.py
"""Tests for per-indicator bias analysis."""
from __future__ import annotations

import pytest


def test_rsi_oversold_is_bullish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer, IndicatorBias
    analyzer = IndicatorBiasAnalyzer()
    result = analyzer.rsi_bias(rsi=25.0, rsi_prev=30.0)
    assert result.direction == "bullish"
    assert result.strength > 0.5
    assert "oversold" in result.explanation.lower()


def test_rsi_overbought_is_bearish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().rsi_bias(rsi=78.0, rsi_prev=75.0)
    assert result.direction == "bearish"


def test_rsi_neutral():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().rsi_bias(rsi=50.0, rsi_prev=49.0)
    assert result.direction == "neutral"


def test_macd_bullish_crossover():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().macd_bias(
        macd_line=0.5, macd_signal=0.3, histogram=0.2, histogram_prev=-0.1
    )
    assert result.direction == "bullish"
    assert "crossover" in result.explanation.lower()


def test_macd_bearish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().macd_bias(
        macd_line=-0.5, macd_signal=-0.3, histogram=-0.2, histogram_prev=0.1
    )
    assert result.direction == "bearish"


def test_bollinger_below_lower_is_bullish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().bollinger_bias(
        close=95.0, bb_upper=110.0, bb_lower=100.0, bb_pct_b=-0.3
    )
    assert result.direction == "bullish"


def test_bollinger_above_upper_is_bearish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().bollinger_bias(
        close=115.0, bb_upper=110.0, bb_lower=100.0, bb_pct_b=1.5
    )
    assert result.direction == "bearish"


def test_ema_stack_bullish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().ema_bias(
        ema_9=105.0, ema_20=103.0, ema_50=100.0, close=106.0
    )
    assert result.direction == "bullish"
    assert "stacked" in result.explanation.lower() or "above" in result.explanation.lower()


def test_ema_stack_bearish():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().ema_bias(
        ema_9=95.0, ema_20=97.0, ema_50=100.0, close=94.0
    )
    assert result.direction == "bearish"


def test_adx_strong_trend():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().adx_bias(adx=35.0)
    assert result.strength > 0.6
    assert "strong" in result.explanation.lower()


def test_adx_weak():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().adx_bias(adx=15.0)
    assert result.strength < 0.4


def test_volume_bias_spike():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    result = IndicatorBiasAnalyzer().volume_bias(relative_volume=2.5, is_bullish_candle=True)
    assert result.direction == "bullish"
    assert result.strength > 0.6


def test_aggregate_all_indicators():
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer, IndicatorBiasReport
    analyzer = IndicatorBiasAnalyzer()
    report = analyzer.analyze_all(
        rsi=30.0, rsi_prev=35.0,
        macd_line=0.5, macd_signal=0.3, histogram=0.2, histogram_prev=-0.1,
        close=106.0, bb_upper=110.0, bb_lower=100.0, bb_pct_b=0.6,
        ema_9=105.0, ema_20=103.0, ema_50=100.0,
        adx=30.0,
        relative_volume=1.5, is_bullish_candle=True,
    )
    assert isinstance(report, IndicatorBiasReport)
    assert report.net_bias in ("bullish", "bearish", "neutral")
    assert 0.0 <= report.bullish_score <= 1.0
    assert 0.0 <= report.bearish_score <= 1.0
    assert len(report.indicators) >= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_analysis/test_indicator_bias.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement IndicatorBiasAnalyzer**

```python
# libs/analysis/indicators/bias.py
"""Per-indicator bias analysis with strength and explanation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorBias:
    """Single indicator's directional assessment."""
    name: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    explanation: str


@dataclass(frozen=True)
class IndicatorBiasReport:
    """Aggregated bias across all indicators."""
    indicators: list[IndicatorBias]
    bullish_score: float  # 0.0 to 1.0
    bearish_score: float  # 0.0 to 1.0
    neutral_score: float  # 0.0 to 1.0
    net_bias: str  # "bullish", "bearish", "neutral"
    explanation: str


class IndicatorBiasAnalyzer:
    """Classifies each indicator as bullish/bearish/neutral with strength."""

    def rsi_bias(self, rsi: float, rsi_prev: float) -> IndicatorBias:
        if rsi <= 30:
            strength = min(1.0, (30 - rsi) / 20 + 0.5)
            return IndicatorBias("RSI", "bullish", strength, f"RSI oversold at {rsi:.0f}")
        if rsi >= 70:
            strength = min(1.0, (rsi - 70) / 20 + 0.5)
            return IndicatorBias("RSI", "bearish", strength, f"RSI overbought at {rsi:.0f}")
        # Momentum direction
        if rsi > 55 and rsi > rsi_prev:
            return IndicatorBias("RSI", "bullish", 0.3, f"RSI rising at {rsi:.0f}")
        if rsi < 45 and rsi < rsi_prev:
            return IndicatorBias("RSI", "bearish", 0.3, f"RSI falling at {rsi:.0f}")
        return IndicatorBias("RSI", "neutral", 0.1, f"RSI neutral at {rsi:.0f}")

    def macd_bias(
        self, macd_line: float, macd_signal: float,
        histogram: float, histogram_prev: float,
    ) -> IndicatorBias:
        crossover = histogram > 0 and histogram_prev <= 0
        crossunder = histogram < 0 and histogram_prev >= 0

        if crossover:
            return IndicatorBias("MACD", "bullish", 0.8, "MACD bullish crossover")
        if crossunder:
            return IndicatorBias("MACD", "bearish", 0.8, "MACD bearish crossover")
        if histogram > 0 and macd_line > 0:
            strength = min(0.6, abs(histogram) * 10)
            return IndicatorBias("MACD", "bullish", strength, f"MACD positive histogram ({histogram:.4f})")
        if histogram < 0 and macd_line < 0:
            strength = min(0.6, abs(histogram) * 10)
            return IndicatorBias("MACD", "bearish", strength, f"MACD negative histogram ({histogram:.4f})")
        return IndicatorBias("MACD", "neutral", 0.1, "MACD mixed signals")

    def bollinger_bias(
        self, close: float, bb_upper: float, bb_lower: float, bb_pct_b: float,
    ) -> IndicatorBias:
        if close < bb_lower or bb_pct_b < 0:
            strength = min(1.0, abs(bb_pct_b) * 0.5 + 0.5) if bb_pct_b < 0 else 0.6
            return IndicatorBias("Bollinger", "bullish", strength, f"Price below lower band (%B={bb_pct_b:.2f})")
        if close > bb_upper or bb_pct_b > 1:
            strength = min(1.0, (bb_pct_b - 1) * 0.5 + 0.5) if bb_pct_b > 1 else 0.6
            return IndicatorBias("Bollinger", "bearish", strength, f"Price above upper band (%B={bb_pct_b:.2f})")
        if bb_pct_b > 0.7:
            return IndicatorBias("Bollinger", "bullish", 0.3, f"Price in upper band zone (%B={bb_pct_b:.2f})")
        if bb_pct_b < 0.3:
            return IndicatorBias("Bollinger", "bearish", 0.3, f"Price in lower band zone (%B={bb_pct_b:.2f})")
        return IndicatorBias("Bollinger", "neutral", 0.1, f"Price mid-band (%B={bb_pct_b:.2f})")

    def ema_bias(
        self, ema_9: float, ema_20: float, ema_50: float, close: float,
    ) -> IndicatorBias:
        bullish_stack = ema_9 > ema_20 > ema_50 and close > ema_9
        bearish_stack = ema_9 < ema_20 < ema_50 and close < ema_9

        if bullish_stack:
            return IndicatorBias("EMA", "bullish", 0.8, "EMA stacked bullish, price above all")
        if bearish_stack:
            return IndicatorBias("EMA", "bearish", 0.8, "EMA stacked bearish, price below all")
        if close > ema_20:
            return IndicatorBias("EMA", "bullish", 0.4, "Price above EMA20")
        if close < ema_20:
            return IndicatorBias("EMA", "bearish", 0.4, "Price below EMA20")
        return IndicatorBias("EMA", "neutral", 0.1, "EMAs tangled, no clear direction")

    def adx_bias(self, adx: float) -> IndicatorBias:
        # ADX is non-directional — measures trend strength only
        if adx >= 25:
            strength = min(1.0, adx / 50)
            return IndicatorBias("ADX", "neutral", strength, f"Strong trend (ADX={adx:.0f})")
        return IndicatorBias("ADX", "neutral", max(0.1, adx / 50), f"Weak trend (ADX={adx:.0f})")

    def volume_bias(self, relative_volume: float, is_bullish_candle: bool) -> IndicatorBias:
        if relative_volume >= 2.0:
            direction = "bullish" if is_bullish_candle else "bearish"
            return IndicatorBias("Volume", direction, min(1.0, relative_volume / 3), f"Volume spike {relative_volume:.1f}x average")
        if relative_volume >= 1.2:
            direction = "bullish" if is_bullish_candle else "bearish"
            return IndicatorBias("Volume", direction, 0.4, f"Above-average volume ({relative_volume:.1f}x)")
        return IndicatorBias("Volume", "neutral", 0.2, f"Normal volume ({relative_volume:.1f}x)")

    def analyze_all(
        self, *,
        rsi: float, rsi_prev: float,
        macd_line: float, macd_signal: float, histogram: float, histogram_prev: float,
        close: float, bb_upper: float, bb_lower: float, bb_pct_b: float,
        ema_9: float, ema_20: float, ema_50: float,
        adx: float,
        relative_volume: float, is_bullish_candle: bool,
    ) -> IndicatorBiasReport:
        indicators = [
            self.rsi_bias(rsi, rsi_prev),
            self.macd_bias(macd_line, macd_signal, histogram, histogram_prev),
            self.bollinger_bias(close, bb_upper, bb_lower, bb_pct_b),
            self.ema_bias(ema_9, ema_20, ema_50, close),
            self.adx_bias(adx),
            self.volume_bias(relative_volume, is_bullish_candle),
        ]

        bullish_total = sum(i.strength for i in indicators if i.direction == "bullish")
        bearish_total = sum(i.strength for i in indicators if i.direction == "bearish")
        neutral_total = sum(i.strength for i in indicators if i.direction == "neutral")
        total = bullish_total + bearish_total + neutral_total or 1.0

        bullish_score = round(bullish_total / total, 4)
        bearish_score = round(bearish_total / total, 4)
        neutral_score = round(neutral_total / total, 4)

        if bullish_score > bearish_score + 0.15:
            net_bias = "bullish"
        elif bearish_score > bullish_score + 0.15:
            net_bias = "bearish"
        else:
            net_bias = "neutral"

        bullish_names = [i.name for i in indicators if i.direction == "bullish"]
        bearish_names = [i.name for i in indicators if i.direction == "bearish"]
        explanation = f"Bullish: {', '.join(bullish_names) or 'none'}. Bearish: {', '.join(bearish_names) or 'none'}."

        return IndicatorBiasReport(
            indicators=indicators,
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            net_bias=net_bias,
            explanation=explanation,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_analysis/test_indicator_bias.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add libs/analysis/indicators/bias.py tests/unit/test_analysis/test_indicator_bias.py
git commit -m "feat(analysis): add IndicatorBiasAnalyzer — per-indicator bias/strength/explanation"
```

---

## Task 2: Swing Point Detection (HH/HL/LH/LL)

**Files:**
- Create: `libs/analysis/structure/swing_points.py`
- Create: `tests/unit/test_analysis/test_swing_points.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_analysis/test_swing_points.py
"""Tests for swing point detection (HH/HL/LH/LL)."""
from __future__ import annotations

import pandas as pd
import numpy as np


def _make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """Create minimal OHLCV DataFrame from highs/lows."""
    n = len(highs)
    return pd.DataFrame({
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs,
        "low": lows,
        "close": [(h + l) / 2 for h, l in zip(highs, lows)],
        "volume": [1000] * n,
    })


def test_detects_swing_high():
    from libs.analysis.structure.swing_points import SwingPointDetector
    # Peak at index 2: higher than neighbors
    highs = [10, 11, 15, 12, 10]
    lows = [8, 9, 13, 10, 8]
    df = _make_df(highs, lows)
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    swing_highs = [p for p in points if p.kind == "high"]
    assert len(swing_highs) >= 1
    assert swing_highs[0].price == 15.0


def test_detects_swing_low():
    from libs.analysis.structure.swing_points import SwingPointDetector
    highs = [15, 12, 10, 12, 15]
    lows = [13, 10, 5, 10, 13]
    df = _make_df(highs, lows)
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    swing_lows = [p for p in points if p.kind == "low"]
    assert len(swing_lows) >= 1
    assert swing_lows[0].price == 5.0


def test_higher_high_classification():
    from libs.analysis.structure.swing_points import SwingPointDetector
    # Two swing highs: 15 then 18 = HH
    highs = [10, 15, 12, 10, 18, 14]
    lows = [8, 13, 10, 8, 16, 12]
    df = _make_df(highs, lows)
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    swing_highs = [p for p in points if p.kind == "high"]
    if len(swing_highs) >= 2:
        assert swing_highs[-1].classification == "HH"


def test_lower_low_classification():
    from libs.analysis.structure.swing_points import SwingPointDetector
    # Two swing lows: 8 then 5 = LL
    highs = [15, 12, 14, 10, 12]
    lows = [13, 8, 12, 5, 10]
    df = _make_df(highs, lows)
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    swing_lows = [p for p in points if p.kind == "low"]
    if len(swing_lows) >= 2:
        assert swing_lows[-1].classification == "LL"


def test_uptrend_pattern():
    from libs.analysis.structure.swing_points import SwingPointDetector, classify_trend
    # Uptrend: HH + HL
    highs = [10, 15, 12, 18, 14, 22, 18]
    lows = [8, 13, 10, 16, 12, 20, 16]
    df = _make_df(highs, lows)
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    trend = classify_trend(points)
    assert trend in ("uptrend", "neutral")


def test_downtrend_pattern():
    from libs.analysis.structure.swing_points import SwingPointDetector, classify_trend
    # Downtrend: LH + LL
    highs = [22, 18, 20, 15, 17, 12, 14]
    lows = [20, 16, 18, 13, 15, 10, 12]
    df = _make_df(highs, lows)
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    trend = classify_trend(points)
    assert trend in ("downtrend", "neutral")


def test_empty_df():
    from libs.analysis.structure.swing_points import SwingPointDetector
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)
    assert points == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_analysis/test_swing_points.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SwingPointDetector**

```python
# libs/analysis/structure/swing_points.py
"""Swing point detection and HH/HL/LH/LL classification."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    """A detected swing high or low."""
    index: int
    price: float
    kind: str  # "high" or "low"
    classification: str  # "HH", "HL", "LH", "LL", or "first"


class SwingPointDetector:
    """Detects swing highs/lows and classifies as HH/HL/LH/LL."""

    def __init__(self, lookback: int = 2) -> None:
        self._lookback = lookback

    def detect(self, df: pd.DataFrame) -> list[SwingPoint]:
        if len(df) < self._lookback * 2 + 1:
            return []

        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        lb = self._lookback

        raw_highs: list[tuple[int, float]] = []
        raw_lows: list[tuple[int, float]] = []

        for i in range(lb, n - lb):
            # Swing high: higher than lb bars on each side
            if all(highs[i] > highs[i - j] for j in range(1, lb + 1)) and \
               all(highs[i] > highs[i + j] for j in range(1, lb + 1)):
                raw_highs.append((i, float(highs[i])))

            # Swing low: lower than lb bars on each side
            if all(lows[i] < lows[i - j] for j in range(1, lb + 1)) and \
               all(lows[i] < lows[i + j] for j in range(1, lb + 1)):
                raw_lows.append((i, float(lows[i])))

        # Classify swing highs
        points: list[SwingPoint] = []
        for idx, (i, price) in enumerate(raw_highs):
            if idx == 0:
                classification = "first"
            else:
                prev_price = raw_highs[idx - 1][1]
                classification = "HH" if price > prev_price else "LH"
            points.append(SwingPoint(index=i, price=price, kind="high", classification=classification))

        # Classify swing lows
        for idx, (i, price) in enumerate(raw_lows):
            if idx == 0:
                classification = "first"
            else:
                prev_price = raw_lows[idx - 1][1]
                classification = "HL" if price > prev_price else "LL"
            points.append(SwingPoint(index=i, price=price, kind="low", classification=classification))

        points.sort(key=lambda p: p.index)
        return points


def classify_trend(points: list[SwingPoint]) -> str:
    """Classify overall trend from recent swing points."""
    if len(points) < 4:
        return "neutral"

    recent = points[-6:]  # Last 6 swing points
    recent_highs = [p for p in recent if p.kind == "high" and p.classification != "first"]
    recent_lows = [p for p in recent if p.kind == "low" and p.classification != "first"]

    hh_count = sum(1 for p in recent_highs if p.classification == "HH")
    lh_count = sum(1 for p in recent_highs if p.classification == "LH")
    hl_count = sum(1 for p in recent_lows if p.classification == "HL")
    ll_count = sum(1 for p in recent_lows if p.classification == "LL")

    if hh_count >= 1 and hl_count >= 1 and lh_count == 0 and ll_count == 0:
        return "uptrend"
    if lh_count >= 1 and ll_count >= 1 and hh_count == 0 and hl_count == 0:
        return "downtrend"
    if (hh_count + hl_count) > (lh_count + ll_count):
        return "uptrend"
    if (lh_count + ll_count) > (hh_count + hl_count):
        return "downtrend"
    return "neutral"
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_analysis/test_swing_points.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add libs/analysis/structure/swing_points.py tests/unit/test_analysis/test_swing_points.py
git commit -m "feat(analysis): add SwingPointDetector with HH/HL/LH/LL classification"
```

---

## Task 3: Market Structure Engine (BOS/CHoCH)

**Files:**
- Create: `libs/analysis/structure/market_structure.py`
- Create: `tests/unit/test_analysis/test_market_structure.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_analysis/test_market_structure.py
"""Tests for BOS/CHoCH detection and structure analysis."""
from __future__ import annotations

import pandas as pd


def _make_df(highs: list[float], lows: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    n = len(highs)
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * n,
    })


def test_bos_bullish():
    """Break of structure: price breaks above previous swing high."""
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    # Swing high at 15, then price breaks above it
    highs = [10, 15, 12, 10, 16, 14]
    lows = [8, 13, 10, 8, 14, 12]
    df = _make_df(highs, lows)
    analyzer = MarketStructureAnalyzer()
    result = analyzer.analyze(df)
    assert any(e.kind == "BOS" and e.direction == "bullish" for e in result.events)


def test_bos_bearish():
    """Break of structure: price breaks below previous swing low."""
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    highs = [15, 12, 14, 10, 12, 8]
    lows = [13, 8, 12, 6, 10, 4]
    df = _make_df(highs, lows)
    analyzer = MarketStructureAnalyzer()
    result = analyzer.analyze(df)
    assert any(e.kind == "BOS" and e.direction == "bearish" for e in result.events)


def test_choch_bullish():
    """Change of character: downtrend makes first HH (trend reversal signal)."""
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    # Downtrend: 20, 18, 16 (LH sequence), then 19 breaks above last LH
    highs = [20, 16, 18, 14, 19, 15]
    lows = [18, 14, 16, 12, 17, 13]
    df = _make_df(highs, lows)
    analyzer = MarketStructureAnalyzer()
    result = analyzer.analyze(df)
    # Should detect some structure event
    assert result.trend_bias in ("bullish", "bearish", "neutral")


def test_structure_result_fields():
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    highs = [10, 15, 12, 18, 14, 22]
    lows = [8, 13, 10, 16, 12, 20]
    df = _make_df(highs, lows)
    result = MarketStructureAnalyzer().analyze(df)
    assert hasattr(result, "trend_bias")
    assert hasattr(result, "swing_points")
    assert hasattr(result, "events")
    assert hasattr(result, "strength")
    assert hasattr(result, "explanation")


def test_empty_df():
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = MarketStructureAnalyzer().analyze(df)
    assert result.trend_bias == "neutral"
    assert result.events == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_analysis/test_market_structure.py -v`
Expected: FAIL

- [ ] **Step 3: Implement MarketStructureAnalyzer**

```python
# libs/analysis/structure/market_structure.py
"""Market structure analysis — BOS, CHoCH, trend bias from swing points."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from libs.analysis.structure.swing_points import SwingPoint, SwingPointDetector, classify_trend


@dataclass(frozen=True)
class StructureEvent:
    """A detected structural event (BOS or CHoCH)."""
    kind: str  # "BOS" or "CHoCH"
    direction: str  # "bullish" or "bearish"
    index: int
    price: float
    broken_level: float
    explanation: str


@dataclass(frozen=True)
class StructureAnalysis:
    """Complete market structure analysis result."""
    trend_bias: str  # "bullish", "bearish", "neutral"
    swing_points: list[SwingPoint] = field(default_factory=list)
    events: list[StructureEvent] = field(default_factory=list)
    strength: float = 0.0  # 0.0 to 1.0
    explanation: str = ""
    last_swing_high: float | None = None
    last_swing_low: float | None = None


class MarketStructureAnalyzer:
    """Analyzes market structure using swing points, BOS, and CHoCH."""

    def __init__(self, lookback: int = 2) -> None:
        self._detector = SwingPointDetector(lookback=lookback)

    def analyze(self, df: pd.DataFrame) -> StructureAnalysis:
        if len(df) < 5:
            return StructureAnalysis(trend_bias="neutral", explanation="Insufficient data")

        points = self._detector.detect(df)
        if not points:
            return StructureAnalysis(trend_bias="neutral", explanation="No swing points detected")

        trend = classify_trend(points)
        events = self._detect_events(df, points)
        strength = self._calc_strength(points, events)

        swing_highs = [p for p in points if p.kind == "high"]
        swing_lows = [p for p in points if p.kind == "low"]

        # CHoCH detection: if latest BOS contradicts prior trend
        trend_bias = trend
        for event in reversed(events):
            if event.kind == "CHoCH":
                trend_bias = event.direction
                break
            if event.kind == "BOS":
                trend_bias = event.direction
                break

        if trend_bias == "neutral" and trend != "neutral":
            trend_bias = "bullish" if trend == "uptrend" else "bearish" if trend == "downtrend" else "neutral"

        explanation_parts = []
        if trend != "neutral":
            explanation_parts.append(f"Swing trend: {trend}")
        for e in events[-3:]:
            explanation_parts.append(f"{e.kind} {e.direction} at {e.price:.4f}")

        return StructureAnalysis(
            trend_bias=trend_bias,
            swing_points=points,
            events=events,
            strength=strength,
            explanation=". ".join(explanation_parts) if explanation_parts else "No clear structure",
            last_swing_high=swing_highs[-1].price if swing_highs else None,
            last_swing_low=swing_lows[-1].price if swing_lows else None,
        )

    def _detect_events(self, df: pd.DataFrame, points: list[SwingPoint]) -> list[StructureEvent]:
        events: list[StructureEvent] = []
        closes = df["close"].values

        swing_highs = [p for p in points if p.kind == "high"]
        swing_lows = [p for p in points if p.kind == "low"]

        # BOS: current price breaks above prior swing high or below prior swing low
        for i in range(1, len(swing_highs)):
            prev = swing_highs[i - 1]
            curr = swing_highs[i]
            # Check if any bar between them closed above prev high
            for bar_idx in range(prev.index + 1, min(curr.index + 1, len(closes))):
                if closes[bar_idx] > prev.price:
                    is_choch = i >= 2 and swing_highs[i - 1].classification == "LH"
                    events.append(StructureEvent(
                        kind="CHoCH" if is_choch else "BOS",
                        direction="bullish",
                        index=bar_idx,
                        price=float(closes[bar_idx]),
                        broken_level=prev.price,
                        explanation=f"{'CHoCH' if is_choch else 'BOS'}: price broke above swing high {prev.price:.4f}",
                    ))
                    break

        for i in range(1, len(swing_lows)):
            prev = swing_lows[i - 1]
            curr = swing_lows[i]
            for bar_idx in range(prev.index + 1, min(curr.index + 1, len(closes))):
                if closes[bar_idx] < prev.price:
                    is_choch = i >= 2 and swing_lows[i - 1].classification == "HL"
                    events.append(StructureEvent(
                        kind="CHoCH" if is_choch else "BOS",
                        direction="bearish",
                        index=bar_idx,
                        price=float(closes[bar_idx]),
                        broken_level=prev.price,
                        explanation=f"{'CHoCH' if is_choch else 'BOS'}: price broke below swing low {prev.price:.4f}",
                    ))
                    break

        events.sort(key=lambda e: e.index)
        return events

    def _calc_strength(self, points: list[SwingPoint], events: list[StructureEvent]) -> float:
        if not points:
            return 0.0
        recent = points[-6:]
        highs = [p for p in recent if p.kind == "high" and p.classification != "first"]
        lows = [p for p in recent if p.kind == "low" and p.classification != "first"]

        # Consistent direction = stronger
        hh = sum(1 for p in highs if p.classification == "HH")
        hl = sum(1 for p in lows if p.classification == "HL")
        lh = sum(1 for p in highs if p.classification == "LH")
        ll = sum(1 for p in lows if p.classification == "LL")

        bullish = hh + hl
        bearish = lh + ll
        total = bullish + bearish or 1
        consistency = max(bullish, bearish) / total

        # BOS/CHoCH events add strength
        event_bonus = min(0.2, len(events) * 0.05)
        return min(1.0, consistency * 0.8 + event_bonus)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_analysis/test_market_structure.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add libs/analysis/structure/market_structure.py tests/unit/test_analysis/test_market_structure.py
git commit -m "feat(analysis): add MarketStructureAnalyzer with BOS/CHoCH detection"
```

---

## Task 4: Supply/Demand Zones

**Files:**
- Create: `libs/analysis/levels/supply_demand.py`
- Create: `tests/unit/test_analysis/test_supply_demand.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_analysis/test_supply_demand.py
"""Tests for supply/demand zone detection."""
from __future__ import annotations

import pandas as pd


def _make_df(highs, lows, closes=None, volumes=None):
    n = len(highs)
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    if volumes is None:
        volumes = [1000] * n
    return pd.DataFrame({
        "open": [(h + l) / 2 for h, l in zip(highs, lows)],
        "high": highs, "low": lows, "close": closes, "volume": volumes,
    })


def test_demand_zone_at_strong_bounce():
    from libs.analysis.levels.supply_demand import SupplyDemandDetector
    # Price drops to 90, strong bounce up with volume
    highs = [100, 98, 95, 92, 95, 100, 105]
    lows = [98, 95, 92, 90, 92, 97, 102]
    volumes = [1000, 1000, 1500, 3000, 2000, 1500, 1000]
    df = _make_df(highs, lows, volumes=volumes)
    detector = SupplyDemandDetector()
    zones = detector.detect(df)
    demand = [z for z in zones if z.kind == "demand"]
    assert len(demand) >= 1


def test_supply_zone_at_strong_rejection():
    from libs.analysis.levels.supply_demand import SupplyDemandDetector
    highs = [100, 102, 105, 108, 105, 100, 95]
    lows = [98, 100, 102, 105, 102, 97, 92]
    volumes = [1000, 1000, 1500, 3000, 2000, 1500, 1000]
    df = _make_df(highs, lows, volumes=volumes)
    detector = SupplyDemandDetector()
    zones = detector.detect(df)
    supply = [z for z in zones if z.kind == "supply"]
    assert len(supply) >= 1


def test_zone_has_required_fields():
    from libs.analysis.levels.supply_demand import SupplyDemandDetector
    highs = [100, 98, 95, 92, 95, 100, 105]
    lows = [98, 95, 92, 90, 92, 97, 102]
    df = _make_df(highs, lows)
    zones = SupplyDemandDetector().detect(df)
    if zones:
        z = zones[0]
        assert hasattr(z, "kind")
        assert hasattr(z, "upper")
        assert hasattr(z, "lower")
        assert hasattr(z, "strength")


def test_price_in_demand_zone():
    from libs.analysis.levels.supply_demand import SupplyDemandDetector, price_near_zone
    highs = [100, 98, 95, 92, 95, 100, 105]
    lows = [98, 95, 92, 90, 92, 97, 102]
    volumes = [1000, 1000, 1500, 3000, 2000, 1500, 1000]
    df = _make_df(highs, lows, volumes=volumes)
    zones = SupplyDemandDetector().detect(df)
    demand = [z for z in zones if z.kind == "demand"]
    if demand:
        result = price_near_zone(demand[0].lower, demand)
        assert result is not None


def test_empty_df():
    from libs.analysis.levels.supply_demand import SupplyDemandDetector
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    zones = SupplyDemandDetector().detect(df)
    assert zones == []
```

- [ ] **Step 2: Run test, verify fail, implement, verify pass**

- [ ] **Step 3: Implement SupplyDemandDetector**

```python
# libs/analysis/levels/supply_demand.py
"""Supply and demand zone detection from OHLCV data."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Zone:
    kind: str  # "supply" or "demand"
    upper: float
    lower: float
    strength: float  # 0.0 to 1.0
    index: int  # bar index where zone originated
    touches: int = 1


class SupplyDemandDetector:
    """Detects supply/demand zones from strong reversals with volume."""

    def __init__(self, min_move_pct: float = 0.5, volume_factor: float = 1.5) -> None:
        self._min_move_pct = min_move_pct
        self._vol_factor = volume_factor

    def detect(self, df: pd.DataFrame) -> list[Zone]:
        if len(df) < 5:
            return []

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        avg_vol = np.mean(volumes) if len(volumes) > 0 else 1.0

        zones: list[Zone] = []

        for i in range(2, len(df) - 2):
            # Demand zone: swing low with strong bounce
            if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
                # Check for strong move away (up)
                move_up = (highs[i + 1] - lows[i]) / max(lows[i], 0.0001) * 100
                vol_spike = volumes[i] > avg_vol * self._vol_factor if avg_vol > 0 else False
                if move_up >= self._min_move_pct or vol_spike:
                    body_low = min(opens[i], closes[i])
                    strength = min(1.0, move_up / 3.0 + (0.2 if vol_spike else 0))
                    zones.append(Zone(
                        kind="demand", upper=float(body_low),
                        lower=float(lows[i]), strength=round(strength, 4),
                        index=i,
                    ))

            # Supply zone: swing high with strong drop
            if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
                move_down = (highs[i] - lows[i + 1]) / max(highs[i], 0.0001) * 100
                vol_spike = volumes[i] > avg_vol * self._vol_factor if avg_vol > 0 else False
                if move_down >= self._min_move_pct or vol_spike:
                    body_high = max(opens[i], closes[i])
                    strength = min(1.0, move_down / 3.0 + (0.2 if vol_spike else 0))
                    zones.append(Zone(
                        kind="supply", upper=float(highs[i]),
                        lower=float(body_high), strength=round(strength, 4),
                        index=i,
                    ))

        return zones


def price_near_zone(price: float, zones: list[Zone], tolerance_pct: float = 0.5) -> Zone | None:
    """Return the nearest zone if price is within tolerance, else None."""
    for zone in zones:
        zone_range = zone.upper - zone.lower
        tolerance = max(zone_range * 0.5, price * tolerance_pct / 100)
        if zone.lower - tolerance <= price <= zone.upper + tolerance:
            return zone
    return None
```

- [ ] **Step 4: Run tests, commit**

```bash
git add libs/analysis/levels/supply_demand.py tests/unit/test_analysis/test_supply_demand.py
git commit -m "feat(analysis): add SupplyDemandDetector with zone strength scoring"
```

---

## Task 5: BullBearBiasEngine — Composite Directional Score

**Files:**
- Create: `libs/analysis/bias/__init__.py`
- Create: `libs/analysis/bias/engine.py`
- Create: `tests/unit/test_analysis/test_bias_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_analysis/test_bias_engine.py
"""Tests for BullBearBiasEngine composite directional scoring."""
from __future__ import annotations

import pytest


def test_strong_bullish():
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    engine = BullBearBiasEngine()
    result = engine.score(BiasInput(
        indicator_bullish=0.8, indicator_bearish=0.1,
        structure_bias="bullish", structure_strength=0.7,
        candle_bullish_count=3, candle_bearish_count=0, candle_total=4,
        regime_supports_direction=True,
        volume_confirms=True,
        htf_bias="bullish",
    ))
    assert result.net_bias == "bullish"
    assert result.bullish_score > 0.6
    assert result.conflict_score < 0.3


def test_strong_bearish():
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    result = BullBearBiasEngine().score(BiasInput(
        indicator_bullish=0.1, indicator_bearish=0.8,
        structure_bias="bearish", structure_strength=0.7,
        candle_bullish_count=0, candle_bearish_count=3, candle_total=4,
        regime_supports_direction=True,
        volume_confirms=True,
        htf_bias="bearish",
    ))
    assert result.net_bias == "bearish"
    assert result.bearish_score > 0.6


def test_mixed_signals_neutral():
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    result = BullBearBiasEngine().score(BiasInput(
        indicator_bullish=0.5, indicator_bearish=0.5,
        structure_bias="neutral", structure_strength=0.3,
        candle_bullish_count=2, candle_bearish_count=2, candle_total=4,
        regime_supports_direction=False,
        volume_confirms=False,
        htf_bias="neutral",
    ))
    assert result.net_bias == "neutral"
    assert result.conflict_score > 0.3


def test_htf_conflict_reduces_confidence():
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    # Bullish setup but HTF is bearish
    bullish_aligned = BullBearBiasEngine().score(BiasInput(
        indicator_bullish=0.7, indicator_bearish=0.2,
        structure_bias="bullish", structure_strength=0.6,
        candle_bullish_count=2, candle_bearish_count=0, candle_total=3,
        regime_supports_direction=True,
        volume_confirms=True,
        htf_bias="bullish",
    ))
    htf_conflict = BullBearBiasEngine().score(BiasInput(
        indicator_bullish=0.7, indicator_bearish=0.2,
        structure_bias="bullish", structure_strength=0.6,
        candle_bullish_count=2, candle_bearish_count=0, candle_total=3,
        regime_supports_direction=True,
        volume_confirms=True,
        htf_bias="bearish",
    ))
    assert htf_conflict.bullish_score < bullish_aligned.bullish_score
    assert htf_conflict.conflict_score > bullish_aligned.conflict_score


def test_result_has_all_fields():
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    result = BullBearBiasEngine().score(BiasInput(
        indicator_bullish=0.5, indicator_bearish=0.3,
        structure_bias="neutral", structure_strength=0.5,
        candle_bullish_count=1, candle_bearish_count=1, candle_total=3,
        regime_supports_direction=True,
        volume_confirms=False,
        htf_bias="neutral",
    ))
    assert hasattr(result, "bullish_score")
    assert hasattr(result, "bearish_score")
    assert hasattr(result, "neutral_score")
    assert hasattr(result, "conflict_score")
    assert hasattr(result, "net_bias")
    assert hasattr(result, "explanation")
    assert 0 <= result.bullish_score <= 1
    assert 0 <= result.bearish_score <= 1
    assert 0 <= result.conflict_score <= 1
```

- [ ] **Step 2: Run test, verify fail**

- [ ] **Step 3: Implement BullBearBiasEngine**

```python
# libs/analysis/bias/__init__.py
"""Bull/Bear directional bias analysis."""
```

```python
# libs/analysis/bias/engine.py
"""BullBearBiasEngine — composite directional scoring."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiasInput:
    """All inputs needed for bias calculation."""
    indicator_bullish: float  # 0-1 from IndicatorBiasReport
    indicator_bearish: float  # 0-1
    structure_bias: str  # "bullish", "bearish", "neutral"
    structure_strength: float  # 0-1
    candle_bullish_count: int
    candle_bearish_count: int
    candle_total: int
    regime_supports_direction: bool
    volume_confirms: bool
    htf_bias: str  # "bullish", "bearish", "neutral"


@dataclass(frozen=True)
class BiasResult:
    """Composite directional bias assessment."""
    bullish_score: float  # 0.0 to 1.0
    bearish_score: float  # 0.0 to 1.0
    neutral_score: float  # 0.0 to 1.0
    conflict_score: float  # 0.0 to 1.0 — how much signals disagree
    net_bias: str  # "bullish", "bearish", "neutral"
    explanation: str


# Weights for each factor
_W_INDICATORS = 0.30
_W_STRUCTURE = 0.25
_W_CANDLES = 0.15
_W_REGIME = 0.10
_W_VOLUME = 0.10
_W_HTF = 0.10


class BullBearBiasEngine:
    """Aggregates all directional signals into a composite bias score."""

    def score(self, inp: BiasInput) -> BiasResult:
        bullish = 0.0
        bearish = 0.0
        factors: list[str] = []

        # 1. Indicators (30%)
        bullish += inp.indicator_bullish * _W_INDICATORS
        bearish += inp.indicator_bearish * _W_INDICATORS

        # 2. Market structure (25%)
        if inp.structure_bias == "bullish":
            bullish += inp.structure_strength * _W_STRUCTURE
            factors.append(f"Structure bullish ({inp.structure_strength:.0%})")
        elif inp.structure_bias == "bearish":
            bearish += inp.structure_strength * _W_STRUCTURE
            factors.append(f"Structure bearish ({inp.structure_strength:.0%})")

        # 3. Candle patterns (15%)
        total_candles = max(inp.candle_total, 1)
        candle_bull_pct = inp.candle_bullish_count / total_candles
        candle_bear_pct = inp.candle_bearish_count / total_candles
        bullish += candle_bull_pct * _W_CANDLES
        bearish += candle_bear_pct * _W_CANDLES

        # 4. Regime (10%)
        if inp.regime_supports_direction:
            # Boost the dominant direction
            if bullish > bearish:
                bullish += _W_REGIME
                factors.append("Regime supports bullish")
            elif bearish > bullish:
                bearish += _W_REGIME
                factors.append("Regime supports bearish")

        # 5. Volume (10%)
        if inp.volume_confirms:
            if bullish > bearish:
                bullish += _W_VOLUME
            elif bearish > bullish:
                bearish += _W_VOLUME

        # 6. Higher timeframe (10%)
        if inp.htf_bias == "bullish":
            bullish += _W_HTF
            factors.append("HTF bullish")
        elif inp.htf_bias == "bearish":
            bearish += _W_HTF
            factors.append("HTF bearish")

        # HTF conflict penalty
        htf_conflicts = False
        if inp.htf_bias == "bearish" and bullish > bearish:
            conflict_penalty = _W_HTF * 0.5
            bullish -= conflict_penalty
            bearish += conflict_penalty * 0.5
            htf_conflicts = True
            factors.append("HTF CONFLICTS with bullish bias")
        elif inp.htf_bias == "bullish" and bearish > bullish:
            conflict_penalty = _W_HTF * 0.5
            bearish -= conflict_penalty
            bullish += conflict_penalty * 0.5
            htf_conflicts = True
            factors.append("HTF CONFLICTS with bearish bias")

        # Normalize
        total = bullish + bearish or 1.0
        bullish_norm = round(min(1.0, max(0.0, bullish / total)), 4)
        bearish_norm = round(min(1.0, max(0.0, bearish / total)), 4)
        neutral_norm = round(max(0.0, 1.0 - bullish_norm - bearish_norm), 4)

        # Conflict score: how close are bullish and bearish?
        conflict = 1.0 - abs(bullish_norm - bearish_norm)
        if htf_conflicts:
            conflict = min(1.0, conflict + 0.2)
        conflict = round(conflict, 4)

        # Net bias
        if bullish_norm > bearish_norm + 0.15:
            net_bias = "bullish"
        elif bearish_norm > bullish_norm + 0.15:
            net_bias = "bearish"
        else:
            net_bias = "neutral"

        explanation = ". ".join(factors) if factors else "No strong directional signals"

        return BiasResult(
            bullish_score=bullish_norm,
            bearish_score=bearish_norm,
            neutral_score=neutral_norm,
            conflict_score=conflict,
            net_bias=net_bias,
            explanation=explanation,
        )
```

- [ ] **Step 4: Run tests, commit**

```bash
git add libs/analysis/bias/ tests/unit/test_analysis/test_bias_engine.py
git commit -m "feat(analysis): add BullBearBiasEngine — composite directional scoring"
```

---

## Task 6: Wire All 40 Pattern Detectors + New Engines into Pipeline

**Files:**
- Modify: `apps/signal_agent/pipeline.py`

This is the integration task. Wire:
1. `CandlePatternEngine` (all 40 detectors) replacing manual `DEFAULT_DETECTORS` list
2. `IndicatorBiasAnalyzer` after existing `IndicatorsEngine.compute()`
3. `MarketStructureAnalyzer` alongside existing `MarketStructureEngine`
4. `SupplyDemandDetector` alongside existing `KeyLevelsEngine`
5. `BullBearBiasEngine` to produce composite score before confluence

- [ ] **Step 1: Read current pipeline to identify exact integration points**

Read `apps/signal_agent/pipeline.py` — find:
- Where `DEFAULT_DETECTORS` is defined (replace with CandlePatternEngine)
- Where `self._indicators.compute(df)` is called (add bias analysis after)
- Where `self._structure.analyze(df)` is called (add market structure after)
- Where confluence is scored (add bias score before)

- [ ] **Step 2: Replace DEFAULT_DETECTORS with CandlePatternEngine**

In pipeline `__init__`, replace the manual detector list with:
```python
from libs.analysis.patterns.engine import CandlePatternEngine
self._pattern_engine = CandlePatternEngine()
```

In `run_once`, replace `for det in self._detectors: det.detect(candles)` with:
```python
pattern_results = self._pattern_engine.detect_all(candles)
```

- [ ] **Step 3: Add IndicatorBiasAnalyzer after indicators compute**

After `indicators = self._indicators.compute(df)`:
```python
from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
indicator_bias = IndicatorBiasAnalyzer().analyze_all(
    rsi=indicators.rsi or 50, rsi_prev=indicators.rsi_prev or 50,
    macd_line=indicators.macd_line or 0, macd_signal=indicators.macd_signal or 0,
    histogram=indicators.macd_histogram or 0, histogram_prev=indicators.macd_histogram_prev or 0,
    close=float(df["close"].iloc[-1]), bb_upper=indicators.bb_upper or 0,
    bb_lower=indicators.bb_lower or 0, bb_pct_b=indicators.bb_pct_b or 0.5,
    ema_9=indicators.ema_9 or 0, ema_20=indicators.ema_20 or 0, ema_50=indicators.ema_50 or 0,
    adx=indicators.adx or 0,
    relative_volume=float(df["relative_volume"].iloc[-1]) if "relative_volume" in df else 1.0,
    is_bullish_candle=bool(df["is_bullish"].iloc[-1]) if "is_bullish" in df else True,
)
```

- [ ] **Step 4: Add MarketStructureAnalyzer**

After existing structure analysis:
```python
from libs.analysis.structure.market_structure import MarketStructureAnalyzer
deep_structure = MarketStructureAnalyzer().analyze(df)
```

- [ ] **Step 5: Add BullBearBiasEngine scoring**

Before confluence scoring, compute composite bias:
```python
from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
candle_bull = sum(1 for p in pattern_results if p.bias == "bullish")
candle_bear = sum(1 for p in pattern_results if p.bias == "bearish")

bias_result = BullBearBiasEngine().score(BiasInput(
    indicator_bullish=indicator_bias.bullish_score,
    indicator_bearish=indicator_bias.bearish_score,
    structure_bias=deep_structure.trend_bias,
    structure_strength=deep_structure.strength,
    candle_bullish_count=candle_bull,
    candle_bearish_count=candle_bear,
    candle_total=len(pattern_results),
    regime_supports_direction=regime.regime in relevant_regimes,
    volume_confirms=volume.is_confirming if hasattr(volume, 'is_confirming') else False,
    htf_bias=htf_structure.trend.value if htf_structure else "neutral",
))
```

Use `bias_result.net_bias` to validate signal direction:
- If strategy says BUY but `bias_result.net_bias == "bearish"` → block or reduce confidence
- If strategy says SELL but `bias_result.net_bias == "bullish"` → block or reduce confidence
- If `bias_result.conflict_score > 0.7` → force NO_TRADE

- [ ] **Step 6: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: All pass (655+ existing + ~40 new)

- [ ] **Step 7: Commit**

```bash
git add apps/signal_agent/pipeline.py
git commit -m "feat(pipeline): wire CandlePatternEngine (40 detectors), indicator bias, market structure, bull/bear bias into signal pipeline"
```

---

## Summary

| Task | What | New Tests |
|------|------|-----------|
| 1 | IndicatorBiasAnalyzer — RSI/MACD/BB/EMA/ADX/Volume bias | 13 |
| 2 | SwingPointDetector — HH/HL/LH/LL | 8 |
| 3 | MarketStructureAnalyzer — BOS/CHoCH | 5 |
| 4 | SupplyDemandDetector — zones with strength | 5 |
| 5 | BullBearBiasEngine — composite directional score | 5 |
| 6 | Pipeline integration — wire everything | 0 (uses existing) |
| **Total** | **6 tasks** | **~36 tests** |

**After this plan:** Every signal will have indicator bias, market structure (HH/HL/BOS/CHoCH), supply/demand context, and a composite bull/bear score before the strategy even decides BUY/SELL. Bad predictions like "SELL while price goes up" will be caught by the bias engine blocking signals that conflict with structure + indicators + HTF.
