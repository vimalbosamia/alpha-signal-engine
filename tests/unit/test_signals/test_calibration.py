"""Tests for ConfidenceCalibrator."""
from __future__ import annotations

from libs.signals.calibration.engine import ConfidenceCalibrator


def test_unvalidated_strategy_caps_confidence():
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.85, strategy_name="new", strategy_win_rate=None,
        strategy_trade_count=5, symbol_win_rate=None, regime_win_rate=None,
    )
    assert result.calibrated_confidence <= 0.60
    assert result.validation_status == "unvalidated"


def test_validated_strategy_boosts():
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.70, strategy_name="proven", strategy_win_rate=0.65,
        strategy_trade_count=60, symbol_win_rate=0.60, regime_win_rate=0.55,
    )
    assert result.calibrated_confidence >= 0.70
    assert result.validation_status == "validated"


def test_poor_win_rate_reduces():
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.75, strategy_name="bad", strategy_win_rate=0.35,
        strategy_trade_count=30, symbol_win_rate=0.40, regime_win_rate=0.30,
    )
    assert result.calibrated_confidence < 0.75


def test_high_conflict_reduces():
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.70, strategy_name="test", strategy_win_rate=0.55,
        strategy_trade_count=20, symbol_win_rate=None, regime_win_rate=None,
        conflict_score=0.8,
    )
    assert result.calibrated_confidence < 0.70


def test_result_fields():
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.70, strategy_name="test", strategy_win_rate=0.55,
        strategy_trade_count=20, symbol_win_rate=None, regime_win_rate=None,
    )
    assert hasattr(result, "calibrated_confidence")
    assert hasattr(result, "raw_confidence")
    assert hasattr(result, "validation_status")
    assert hasattr(result, "adjustments")
    assert hasattr(result, "explanation")
    assert 0 <= result.calibrated_confidence <= 1


def test_confidence_never_exceeds_one():
    result = ConfidenceCalibrator().calibrate(
        raw_confidence=0.95, strategy_name="great", strategy_win_rate=0.80,
        strategy_trade_count=100, symbol_win_rate=0.75, regime_win_rate=0.70,
    )
    assert result.calibrated_confidence <= 1.0
