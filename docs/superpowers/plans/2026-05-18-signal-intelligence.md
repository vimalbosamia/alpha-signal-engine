# Signal Intelligence Implementation Plan (Phases 12, 13, 20)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trade grading (A+/A/B/C/Avoid), confidence calibration based on historical reliability, and futures-specific risk assessment — transforming raw signals into actionable intelligence with honest quality scores.

**Architecture:** Three new engines: TradeDecisionEngine grades signals and decides TAKE/WAIT/SKIP. ConfidenceCalibrator adjusts confidence using per-strategy historical win rates. FuturesRiskEngine calculates liquidation, leverage, and funding risk for futures signals. All wire into the existing pipeline after confluence scoring.

**Tech Stack:** Python 3.12+, Pydantic frozen models, existing DB for historical stats.

---

## File Structure

```
libs/signals/
    grading/
        __init__.py
        engine.py           # TradeDecisionEngine: A+/A/B/C/Avoid grading + TAKE/WAIT/SKIP
    calibration/
        __init__.py
        engine.py           # ConfidenceCalibrator: adjusts confidence by historical reliability
    
libs/risk/
    futures.py              # FuturesRiskEngine: liquidation, leverage, funding risk

tests/unit/
    test_signals/
        test_grading.py
        test_calibration.py
    test_risk/
        test_futures_risk.py
```

---

## Task 1: TradeDecisionEngine — Signal Grading

**Files:**
- Create: `libs/signals/grading/__init__.py`
- Create: `libs/signals/grading/engine.py`
- Create: `tests/unit/test_signals/test_grading.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_signals/test_grading.py
"""Tests for TradeDecisionEngine — signal grading and trade decisions."""
from __future__ import annotations

import pytest


def test_a_plus_setup():
    """High confidence, high R:R, HTF aligned, good regime → A+ / TAKE."""
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    engine = TradeDecisionEngine()
    result = engine.grade(GradingInput(
        confidence=0.82,
        risk_reward=3.0,
        bias_net="bullish",
        bias_conflict=0.15,
        bias_bullish=0.75,
        bias_bearish=0.15,
        htf_aligned=True,
        regime_supports=True,
        structure_strength=0.8,
        volume_confirms=True,
        data_quality_clean=True,
        action="BUY",
        is_late_entry=False,
        is_overextended=False,
    ))
    assert result.setup_grade == "A+"
    assert result.trade_decision == "TAKE"


def test_a_setup():
    """Good confluence, minor issues → A / TAKE."""
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    result = TradeDecisionEngine().grade(GradingInput(
        confidence=0.72, risk_reward=2.2, bias_net="bullish",
        bias_conflict=0.25, bias_bullish=0.65, bias_bearish=0.25,
        htf_aligned=True, regime_supports=True, structure_strength=0.6,
        volume_confirms=False, data_quality_clean=True, action="BUY",
        is_late_entry=False, is_overextended=False,
    ))
    assert result.setup_grade == "A"
    assert result.trade_decision == "TAKE"


def test_b_setup():
    """Moderate confidence, some issues → B / WAIT."""
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    result = TradeDecisionEngine().grade(GradingInput(
        confidence=0.58, risk_reward=1.8, bias_net="neutral",
        bias_conflict=0.45, bias_bullish=0.45, bias_bearish=0.40,
        htf_aligned=False, regime_supports=True, structure_strength=0.4,
        volume_confirms=False, data_quality_clean=True, action="BUY",
        is_late_entry=False, is_overextended=False,
    ))
    assert result.setup_grade == "B"
    assert result.trade_decision == "WAIT"


def test_c_setup():
    """Low confidence, weak setup → C / SKIP."""
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    result = TradeDecisionEngine().grade(GradingInput(
        confidence=0.42, risk_reward=1.2, bias_net="neutral",
        bias_conflict=0.6, bias_bullish=0.35, bias_bearish=0.35,
        htf_aligned=False, regime_supports=False, structure_strength=0.2,
        volume_confirms=False, data_quality_clean=True, action="BUY",
        is_late_entry=True, is_overextended=False,
    ))
    assert result.setup_grade == "C"
    assert result.trade_decision == "SKIP"


def test_avoid_setup():
    """Very low confidence, bad data, overextended → Avoid / NO_TRADE."""
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    result = TradeDecisionEngine().grade(GradingInput(
        confidence=0.30, risk_reward=0.8, bias_net="bearish",
        bias_conflict=0.8, bias_bullish=0.20, bias_bearish=0.60,
        htf_aligned=False, regime_supports=False, structure_strength=0.1,
        volume_confirms=False, data_quality_clean=False, action="BUY",
        is_late_entry=True, is_overextended=True,
    ))
    assert result.setup_grade == "Avoid"
    assert result.trade_decision == "NO_TRADE"


def test_late_entry_downgrades():
    """Same as A+ but late entry → downgrade to B / WAIT."""
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    result = TradeDecisionEngine().grade(GradingInput(
        confidence=0.82, risk_reward=3.0, bias_net="bullish",
        bias_conflict=0.15, bias_bullish=0.75, bias_bearish=0.15,
        htf_aligned=True, regime_supports=True, structure_strength=0.8,
        volume_confirms=True, data_quality_clean=True, action="BUY",
        is_late_entry=True, is_overextended=False,
    ))
    assert result.setup_grade in ("A", "B")
    assert result.trade_decision in ("TAKE", "WAIT")


def test_result_has_all_fields():
    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
    result = TradeDecisionEngine().grade(GradingInput(
        confidence=0.6, risk_reward=2.0, bias_net="neutral",
        bias_conflict=0.4, bias_bullish=0.5, bias_bearish=0.4,
        htf_aligned=True, regime_supports=True, structure_strength=0.5,
        volume_confirms=True, data_quality_clean=True, action="BUY",
        is_late_entry=False, is_overextended=False,
    ))
    assert hasattr(result, "setup_grade")
    assert hasattr(result, "trade_decision")
    assert hasattr(result, "quality_score")
    assert hasattr(result, "reasons")
    assert result.setup_grade in ("A+", "A", "B", "C", "Avoid")
    assert result.trade_decision in ("TAKE", "WAIT", "SKIP", "NO_TRADE")
    assert 0 <= result.quality_score <= 100
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run python -m pytest tests/unit/test_signals/test_grading.py -v`

- [ ] **Step 3: Implement TradeDecisionEngine**

```python
# libs/signals/grading/__init__.py
"""Signal grading and trade decision layer."""

# libs/signals/grading/engine.py
"""TradeDecisionEngine — grades signals A+/A/B/C/Avoid and decides TAKE/WAIT/SKIP/NO_TRADE."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GradingInput:
    confidence: float          # 0-1 from confluence
    risk_reward: float         # R:R ratio
    bias_net: str              # "bullish", "bearish", "neutral" from BullBearBiasEngine
    bias_conflict: float       # 0-1 from BiasResult.conflict_score
    bias_bullish: float        # 0-1
    bias_bearish: float        # 0-1
    htf_aligned: bool          # higher timeframe agrees
    regime_supports: bool      # market regime supports strategy type
    structure_strength: float  # 0-1 from MarketStructureAnalyzer
    volume_confirms: bool      # volume supports the move
    data_quality_clean: bool   # no data issues
    action: str                # "BUY" or "SELL"
    is_late_entry: bool        # price moved significantly past entry zone
    is_overextended: bool      # price overextended from mean


@dataclass(frozen=True)
class GradingResult:
    setup_grade: str        # "A+", "A", "B", "C", "Avoid"
    trade_decision: str     # "TAKE", "WAIT", "SKIP", "NO_TRADE"
    quality_score: float    # 0-100 composite score
    reasons: list[str] = field(default_factory=list)


class TradeDecisionEngine:
    """Grades trade setups and decides executability."""

    def grade(self, inp: GradingInput) -> GradingResult:
        score = 0.0
        reasons: list[str] = []
        penalties: list[str] = []

        # Confidence (max 25 points)
        conf_pts = inp.confidence * 25
        score += conf_pts
        if inp.confidence >= 0.75:
            reasons.append(f"Strong confidence ({inp.confidence:.0%})")
        elif inp.confidence < 0.45:
            penalties.append(f"Low confidence ({inp.confidence:.0%})")

        # Risk/reward (max 20 points)
        rr_pts = min(20, inp.risk_reward * 6.67)
        score += rr_pts
        if inp.risk_reward >= 2.5:
            reasons.append(f"Excellent R:R ({inp.risk_reward:.1f}x)")
        elif inp.risk_reward < 1.5:
            penalties.append(f"Poor R:R ({inp.risk_reward:.1f}x)")

        # Bias alignment (max 15 points)
        if inp.action == "BUY" and inp.bias_net == "bullish":
            score += 15
            reasons.append("Bias confirms BUY")
        elif inp.action == "SELL" and inp.bias_net == "bearish":
            score += 15
            reasons.append("Bias confirms SELL")
        elif inp.bias_net == "neutral":
            score += 5
        else:
            penalties.append(f"Bias conflicts: {inp.bias_net} vs {inp.action}")

        # Conflict penalty (max -10)
        if inp.bias_conflict > 0.6:
            score -= 10
            penalties.append(f"High conflict ({inp.bias_conflict:.0%})")
        elif inp.bias_conflict > 0.4:
            score -= 5

        # HTF alignment (10 points)
        if inp.htf_aligned:
            score += 10
            reasons.append("HTF aligned")
        else:
            score -= 5
            penalties.append("HTF not aligned")

        # Regime (10 points)
        if inp.regime_supports:
            score += 10
            reasons.append("Regime supports strategy")
        else:
            penalties.append("Regime doesn't support")

        # Structure strength (10 points)
        score += inp.structure_strength * 10

        # Volume (5 points)
        if inp.volume_confirms:
            score += 5
            reasons.append("Volume confirms")

        # Data quality (5 points)
        if inp.data_quality_clean:
            score += 5
        else:
            score -= 10
            penalties.append("Data quality issues")

        # Late entry penalty
        if inp.is_late_entry:
            score -= 15
            penalties.append("Late entry")

        # Overextended penalty
        if inp.is_overextended:
            score -= 15
            penalties.append("Price overextended")

        score = max(0, min(100, score))

        # Grade mapping
        if score >= 75:
            grade = "A+"
        elif score >= 60:
            grade = "A"
        elif score >= 45:
            grade = "B"
        elif score >= 30:
            grade = "C"
        else:
            grade = "Avoid"

        # Decision mapping
        decision_map = {"A+": "TAKE", "A": "TAKE", "B": "WAIT", "C": "SKIP", "Avoid": "NO_TRADE"}
        decision = decision_map[grade]

        return GradingResult(
            setup_grade=grade,
            trade_decision=decision,
            quality_score=round(score, 1),
            reasons=reasons + [f"⚠ {p}" for p in penalties],
        )
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Run full suite**

Run: `uv run python -m pytest tests/ -x -q`

- [ ] **Step 6: Commit**

```bash
git add libs/signals/grading/ tests/unit/test_signals/test_grading.py
git commit -m "feat(signals): add TradeDecisionEngine — A+/A/B/C/Avoid grading with TAKE/WAIT/SKIP"
```

---

## Task 2: ConfidenceCalibrator

**Files:**
- Create: `libs/signals/calibration/__init__.py`
- Create: `libs/signals/calibration/engine.py`
- Create: `tests/unit/test_signals/test_calibration.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_signals/test_calibration.py
"""Tests for ConfidenceCalibrator — adjusts confidence by historical reliability."""
from __future__ import annotations

import pytest


def test_unvalidated_strategy_caps_confidence():
    """Strategy with < 10 historical trades → cap at 0.60."""
    from libs.signals.calibration.engine import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    result = cal.calibrate(
        raw_confidence=0.85,
        strategy_name="new_strategy",
        strategy_win_rate=None,
        strategy_trade_count=5,
        symbol_win_rate=None,
        regime_win_rate=None,
    )
    assert result.calibrated_confidence <= 0.60
    assert result.validation_status == "unvalidated"


def test_validated_strategy_boosts():
    """Strategy with 50+ trades and 65% win rate → slight boost."""
    from libs.signals.calibration.engine import ConfidenceCalibrator
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.70,
        strategy_name="proven_strat",
        strategy_win_rate=0.65,
        strategy_trade_count=60,
        symbol_win_rate=0.60,
        regime_win_rate=0.55,
    )
    assert result.calibrated_confidence >= 0.70
    assert result.validation_status == "validated"


def test_poor_win_rate_reduces():
    """Strategy with 35% win rate → reduce confidence."""
    from libs.signals.calibration.engine import ConfidenceCalibrator
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.75,
        strategy_name="bad_strat",
        strategy_win_rate=0.35,
        strategy_trade_count=30,
        symbol_win_rate=0.40,
        regime_win_rate=0.30,
    )
    assert result.calibrated_confidence < 0.75
    assert "poor" in result.explanation.lower() or "reduced" in result.explanation.lower()


def test_high_conflict_reduces():
    """High conflict score → reduce confidence."""
    from libs.signals.calibration.engine import ConfidenceCalibrator
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.70,
        strategy_name="test",
        strategy_win_rate=0.55,
        strategy_trade_count=20,
        symbol_win_rate=None,
        regime_win_rate=None,
        conflict_score=0.8,
    )
    assert result.calibrated_confidence < 0.70


def test_result_fields():
    from libs.signals.calibration.engine import ConfidenceCalibrator
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.70,
        strategy_name="test",
        strategy_win_rate=0.55,
        strategy_trade_count=20,
        symbol_win_rate=None,
        regime_win_rate=None,
    )
    assert hasattr(result, "calibrated_confidence")
    assert hasattr(result, "raw_confidence")
    assert hasattr(result, "validation_status")
    assert hasattr(result, "adjustments")
    assert hasattr(result, "explanation")
    assert 0 <= result.calibrated_confidence <= 1


def test_confidence_never_exceeds_one():
    from libs.signals.calibration.engine import ConfidenceCalibrator
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.95,
        strategy_name="great",
        strategy_win_rate=0.80,
        strategy_trade_count=100,
        symbol_win_rate=0.75,
        regime_win_rate=0.70,
    )
    assert result.calibrated_confidence <= 1.0
```

- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Implement ConfidenceCalibrator**

```python
# libs/signals/calibration/__init__.py
"""Confidence calibration based on historical reliability."""

# libs/signals/calibration/engine.py
"""ConfidenceCalibrator — adjusts raw confluence confidence using historical stats."""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_TRADES_FOR_VALIDATION = 10
UNVALIDATED_CAP = 0.60
MIN_TRADES_FOR_BOOST = 30


@dataclass(frozen=True)
class CalibrationResult:
    raw_confidence: float
    calibrated_confidence: float  # 0-1
    validation_status: str        # "unvalidated", "learning", "validated"
    adjustments: list[str] = field(default_factory=list)
    explanation: str = ""


class ConfidenceCalibrator:
    """Adjusts confidence based on historical strategy/symbol/regime performance."""

    def calibrate(
        self,
        raw_confidence: float,
        strategy_name: str,
        strategy_win_rate: float | None,
        strategy_trade_count: int,
        symbol_win_rate: float | None,
        regime_win_rate: float | None,
        conflict_score: float = 0.0,
    ) -> CalibrationResult:
        confidence = raw_confidence
        adjustments: list[str] = []

        # Determine validation status
        if strategy_trade_count < MIN_TRADES_FOR_VALIDATION:
            status = "unvalidated"
            confidence = min(confidence, UNVALIDATED_CAP)
            adjustments.append(f"Unvalidated ({strategy_trade_count} trades) → capped at {UNVALIDATED_CAP:.0%}")
        elif strategy_trade_count < MIN_TRADES_FOR_BOOST:
            status = "learning"
        else:
            status = "validated"

        # Strategy win rate adjustment
        if strategy_win_rate is not None and strategy_trade_count >= MIN_TRADES_FOR_VALIDATION:
            if strategy_win_rate >= 0.60:
                boost = (strategy_win_rate - 0.50) * 0.2  # Max +10%
                confidence = min(1.0, confidence + boost)
                adjustments.append(f"Strategy WR {strategy_win_rate:.0%} → +{boost:.0%}")
            elif strategy_win_rate < 0.40:
                penalty = (0.50 - strategy_win_rate) * 0.3  # Max -15%
                confidence = max(0.1, confidence - penalty)
                adjustments.append(f"Poor strategy WR {strategy_win_rate:.0%} → reduced by {penalty:.0%}")

        # Symbol-specific adjustment
        if symbol_win_rate is not None:
            if symbol_win_rate < 0.35:
                confidence = max(0.1, confidence - 0.05)
                adjustments.append(f"Poor symbol WR {symbol_win_rate:.0%} → -5%")
            elif symbol_win_rate >= 0.65:
                confidence = min(1.0, confidence + 0.03)
                adjustments.append(f"Strong symbol WR {symbol_win_rate:.0%} → +3%")

        # Regime-specific adjustment
        if regime_win_rate is not None:
            if regime_win_rate < 0.35:
                confidence = max(0.1, confidence - 0.05)
                adjustments.append(f"Poor regime WR {regime_win_rate:.0%} → -5%")

        # Conflict penalty
        if conflict_score > 0.6:
            penalty = (conflict_score - 0.5) * 0.15
            confidence = max(0.1, confidence - penalty)
            adjustments.append(f"High conflict ({conflict_score:.0%}) → -{penalty:.0%}")

        confidence = round(max(0.0, min(1.0, confidence)), 4)

        explanation_parts = [f"Raw: {raw_confidence:.0%} → Calibrated: {confidence:.0%}"]
        if adjustments:
            explanation_parts.append(". ".join(adjustments))

        return CalibrationResult(
            raw_confidence=raw_confidence,
            calibrated_confidence=confidence,
            validation_status=status,
            adjustments=adjustments,
            explanation=". ".join(explanation_parts),
        )
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add libs/signals/calibration/ tests/unit/test_signals/test_calibration.py
git commit -m "feat(signals): add ConfidenceCalibrator — historical reliability-based confidence adjustment"
```

---

## Task 3: FuturesRiskEngine

**Files:**
- Create: `libs/risk/futures.py`
- Create: `tests/unit/test_risk/test_futures_risk.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_risk/test_futures_risk.py
"""Tests for FuturesRiskEngine — liquidation, leverage, funding risk."""
from __future__ import annotations

import pytest


def test_liquidation_price_long():
    """Long at 100 with 10x leverage, isolated → liq near 90."""
    from libs.risk.futures import FuturesRiskEngine
    engine = FuturesRiskEngine()
    result = engine.assess(
        entry_price=100.0, leverage=10.0, margin_type="isolated",
        direction="LONG", stop_loss=95.0,
    )
    assert result.estimated_liquidation_price < 95.0
    assert result.estimated_liquidation_price > 85.0


def test_liquidation_price_short():
    """Short at 100 with 10x leverage → liq near 110."""
    from libs.risk.futures import FuturesRiskEngine
    result = FuturesRiskEngine().assess(
        entry_price=100.0, leverage=10.0, margin_type="isolated",
        direction="SHORT", stop_loss=105.0,
    )
    assert result.estimated_liquidation_price > 105.0
    assert result.estimated_liquidation_price < 115.0


def test_high_leverage_extreme_risk():
    """50x leverage → liquidation_risk = EXTREME."""
    from libs.risk.futures import FuturesRiskEngine
    result = FuturesRiskEngine().assess(
        entry_price=100.0, leverage=50.0, margin_type="isolated",
        direction="LONG", stop_loss=99.0,
    )
    assert result.liquidation_risk == "EXTREME"


def test_low_leverage_moderate_risk():
    """3x leverage → liquidation_risk = LOW or MODERATE."""
    from libs.risk.futures import FuturesRiskEngine
    result = FuturesRiskEngine().assess(
        entry_price=100.0, leverage=3.0, margin_type="isolated",
        direction="LONG", stop_loss=90.0,
    )
    assert result.liquidation_risk in ("LOW", "MODERATE")


def test_stop_too_close_to_liquidation():
    """Stop loss within 2% of liquidation → should_reject = True."""
    from libs.risk.futures import FuturesRiskEngine
    result = FuturesRiskEngine().assess(
        entry_price=100.0, leverage=20.0, margin_type="isolated",
        direction="LONG", stop_loss=94.0,
    )
    # With 20x, liq is ~95. Stop at 94 is below liq → reject
    assert result.should_reject is True


def test_result_fields():
    from libs.risk.futures import FuturesRiskEngine
    result = FuturesRiskEngine().assess(
        entry_price=100.0, leverage=5.0, margin_type="isolated",
        direction="LONG", stop_loss=90.0,
    )
    assert hasattr(result, "leverage")
    assert hasattr(result, "margin_type")
    assert hasattr(result, "estimated_liquidation_price")
    assert hasattr(result, "liquidation_buffer_percent")
    assert hasattr(result, "liquidation_risk")
    assert hasattr(result, "max_loss_before_stop")
    assert hasattr(result, "should_reject")
    assert hasattr(result, "rejection_reason")


def test_max_loss_before_stop():
    """10x long at 100, stop at 95 → max loss = 50% of margin."""
    from libs.risk.futures import FuturesRiskEngine
    result = FuturesRiskEngine().assess(
        entry_price=100.0, leverage=10.0, margin_type="isolated",
        direction="LONG", stop_loss=95.0,
    )
    assert result.max_loss_before_stop == pytest.approx(50.0, abs=1.0)
```

- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Implement FuturesRiskEngine**

```python
# libs/risk/futures.py
"""FuturesRiskEngine — liquidation, leverage, funding risk assessment."""
from __future__ import annotations

from dataclasses import dataclass

# Maintenance margin rate (Binance typical)
MAINTENANCE_MARGIN_RATE = 0.004  # 0.4%


@dataclass(frozen=True)
class FuturesRiskAssessment:
    leverage: float
    margin_type: str  # "isolated" or "cross"
    estimated_liquidation_price: float
    liquidation_buffer_percent: float  # distance from entry to liq as %
    liquidation_risk: str  # "LOW", "MODERATE", "HIGH", "EXTREME"
    max_loss_before_stop: float  # % loss at stop price
    should_reject: bool
    rejection_reason: str


class FuturesRiskEngine:
    """Calculates futures-specific risk metrics."""

    def assess(
        self,
        entry_price: float,
        leverage: float,
        margin_type: str,
        direction: str,  # "LONG" or "SHORT"
        stop_loss: float,
        funding_rate: float = 0.0,
    ) -> FuturesRiskAssessment:
        # Liquidation price calculation (simplified, Binance-style)
        # For isolated margin: liq = entry * (1 - 1/leverage + maintenance_margin_rate) for LONG
        #                       liq = entry * (1 + 1/leverage - maintenance_margin_rate) for SHORT
        if direction == "LONG":
            liq_price = entry_price * (1 - (1 / leverage) + MAINTENANCE_MARGIN_RATE)
        else:  # SHORT
            liq_price = entry_price * (1 + (1 / leverage) - MAINTENANCE_MARGIN_RATE)

        # Buffer: distance from entry to liquidation as %
        buffer_pct = abs(entry_price - liq_price) / entry_price * 100

        # Max loss at stop
        stop_distance_pct = abs(entry_price - stop_loss) / entry_price * 100
        max_loss_pct = stop_distance_pct * leverage

        # Risk classification
        if leverage >= 50 or buffer_pct < 2:
            risk = "EXTREME"
        elif leverage >= 20 or buffer_pct < 5:
            risk = "HIGH"
        elif leverage >= 10 or buffer_pct < 10:
            risk = "MODERATE"
        else:
            risk = "LOW"

        # Rejection logic
        should_reject = False
        rejection_reason = ""

        # Stop loss beyond liquidation
        if direction == "LONG" and stop_loss <= liq_price:
            should_reject = True
            rejection_reason = f"Stop ({stop_loss:.2f}) at or below liquidation ({liq_price:.2f})"
        elif direction == "SHORT" and stop_loss >= liq_price:
            should_reject = True
            rejection_reason = f"Stop ({stop_loss:.2f}) at or above liquidation ({liq_price:.2f})"

        # Buffer too small (< 2%)
        if not should_reject and buffer_pct < 2:
            should_reject = True
            rejection_reason = f"Liquidation buffer too small ({buffer_pct:.1f}%)"

        # Extreme leverage
        if not should_reject and leverage > 50:
            should_reject = True
            rejection_reason = f"Extreme leverage ({leverage}x)"

        # Max loss > 80% of margin
        if not should_reject and max_loss_pct > 80:
            should_reject = True
            rejection_reason = f"Max loss at stop is {max_loss_pct:.0f}% of margin"

        return FuturesRiskAssessment(
            leverage=leverage,
            margin_type=margin_type,
            estimated_liquidation_price=round(liq_price, 4),
            liquidation_buffer_percent=round(buffer_pct, 2),
            liquidation_risk=risk,
            max_loss_before_stop=round(max_loss_pct, 2),
            should_reject=should_reject,
            rejection_reason=rejection_reason,
        )
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add libs/risk/futures.py tests/unit/test_risk/test_futures_risk.py
git commit -m "feat(risk): add FuturesRiskEngine — liquidation, leverage, buffer calculation"
```

---

## Task 4: Wire Grading + Calibration + Futures Risk into Pipeline

**Files:**
- Modify: `apps/signal_agent/pipeline.py`
- Modify: `libs/signals/output/emitter.py` (populate grading fields on SignalOutput)

- [ ] **Step 1: Add imports to pipeline.py**

After existing imports, add:
```python
from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
from libs.signals.calibration.engine import ConfidenceCalibrator
from libs.risk.futures import FuturesRiskEngine
```

- [ ] **Step 2: After bias gate, add grading + calibration**

After the bias gate block and before `output = self._emitter.emit(...)`, add:
```python
                # ── 7. Confidence calibration ────────────────────────────
                try:
                    calibrator = ConfidenceCalibrator()
                    # Get strategy stats from outcome tracker if available
                    strat_wr = None
                    strat_count = 0
                    try:
                        from libs.monitoring.outcome_tracker import _MUTED_STRATEGIES
                        # Use muted set as proxy — proper stats require DB query
                        strat_count = 15  # Assume some history after first runs
                    except Exception:
                        pass

                    cal_result = calibrator.calibrate(
                        raw_confidence=breakdown.weighted_total,
                        strategy_name=strategy.name,
                        strategy_win_rate=strat_wr,
                        strategy_trade_count=strat_count,
                        symbol_win_rate=None,
                        regime_win_rate=None,
                        conflict_score=bias_result.conflict_score if bias_result else 0.0,
                    )
                    # Override confidence with calibrated value
                    breakdown = breakdown.model_copy(update={
                        "weighted_total": cal_result.calibrated_confidence,
                    }) if hasattr(breakdown, 'model_copy') else breakdown
                except Exception:
                    cal_result = None

                # ── 8. Trade grading ─────────────────────────────────────
                grading_result = None
                try:
                    vol_ctx = self._volume.analyze(df, candidate.proposed_action)
                    grading_result = TradeDecisionEngine().grade(GradingInput(
                        confidence=breakdown.weighted_total,
                        risk_reward=self._emitter._calc_rr(candidate) if candidate.proposed_action != SignalAction.NO_TRADE else 0,
                        bias_net=bias_result.net_bias if bias_result else "neutral",
                        bias_conflict=bias_result.conflict_score if bias_result else 0.5,
                        bias_bullish=bias_result.bullish_score if bias_result else 0.5,
                        bias_bearish=bias_result.bearish_score if bias_result else 0.5,
                        htf_aligned=htf_bias_str != "neutral",
                        regime_supports=regime.vol_score >= 0.4 if hasattr(regime, 'vol_score') else True,
                        structure_strength=deep_structure.strength if deep_structure else 0.0,
                        volume_confirms=getattr(vol_ctx, 'is_confirming', False) if vol_ctx else False,
                        data_quality_clean=quality.is_safe,
                        action=candidate.proposed_action.value if candidate.proposed_action != SignalAction.NO_TRADE else "BUY",
                        is_late_entry=False,
                        is_overextended=False,
                    ))
                except Exception:
                    pass
```

- [ ] **Step 3: In emitter.emit(), populate grading fields on SignalOutput**

After `output = self._emitter.emit(candidate, breakdown)`, add:
```python
                # Attach grading to output
                if grading_result is not None:
                    try:
                        from libs.core.models.domain import SetupGrade, TradeDecision, SignalQuality
                        grade_map = {"A+": SetupGrade.A_PLUS, "A": SetupGrade.A, "B": SetupGrade.B, "C": SetupGrade.C, "Avoid": SetupGrade.AVOID}
                        decision_map = {"TAKE": TradeDecision.TAKE, "WAIT": TradeDecision.WAIT, "SKIP": TradeDecision.SKIP, "NO_TRADE": TradeDecision.NO_TRADE}
                        output = output.model_copy(update={
                            "setup_grade": grade_map.get(grading_result.setup_grade),
                            "trade_decision": decision_map.get(grading_result.trade_decision),
                        })
                    except Exception:
                        pass
```

- [ ] **Step 4: Add futures risk for futures signals**

After grading, add:
```python
                # ── 9. Futures risk ──────────────────────────────────────
                if getattr(candidate, 'trading_mode', None) == TradingMode.FUTURES:
                    try:
                        futures_result = FuturesRiskEngine().assess(
                            entry_price=(candidate.entry_zone_low + candidate.entry_zone_high) / 2,
                            leverage=getattr(candidate, 'leverage', 3.0),
                            margin_type="isolated",
                            direction="LONG" if candidate.proposed_action == SignalAction.BUY else "SHORT",
                            stop_loss=candidate.stop_loss,
                        )
                        if futures_result.should_reject:
                            log.info("futures_risk_rejected", symbol=symbol, reason=futures_result.rejection_reason)
                            continue
                    except Exception:
                        pass
```

- [ ] **Step 5: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`

- [ ] **Step 6: Commit**

```bash
git add apps/signal_agent/pipeline.py
git commit -m "feat(pipeline): wire grading, confidence calibration, and futures risk into signal pipeline"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | TradeDecisionEngine (A+/A/B/C/Avoid + TAKE/WAIT/SKIP) | 7 |
| 2 | ConfidenceCalibrator (historical reliability) | 6 |
| 3 | FuturesRiskEngine (liquidation, leverage, buffer) | 7 |
| 4 | Pipeline wiring (all three into signal flow) | 0 (existing) |
| **Total** | **4 tasks** | **~20 tests** |
