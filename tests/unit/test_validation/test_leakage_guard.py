"""Tests for LeakageGuard."""
from __future__ import annotations

import pytest

from libs.validation.leakage_guard import LeakageAudit, LeakageCheck, LeakageGuard


# ---------------------------------------------------------------------------
# Test 1 — sufficient bars: all indicators safe
# ---------------------------------------------------------------------------

def test_sufficient_bars_all_safe():
    guard = LeakageGuard()
    # max warmup is 200 (ema_200 / sma_200), so 200 bars covers everything
    audit = guard.audit(available_bars=200)

    assert audit.safe_count == audit.total_indicators
    assert audit.unsafe_count == 0
    assert all(c.is_safe for c in audit.checks)
    assert "200" in audit.explanation  # references the bar count or warmup


# ---------------------------------------------------------------------------
# Test 2 — insufficient bars: most indicators unsafe
# ---------------------------------------------------------------------------

def test_insufficient_bars():
    guard = LeakageGuard()
    # With only 5 bars, only vwap (warmup=1) should be safe
    audit = guard.audit(available_bars=5)

    assert audit.unsafe_count > 0
    # vwap only needs 1 bar — confirm at least that one is safe
    vwap_check = next(c for c in audit.checks if c.indicator_name == "vwap")
    assert vwap_check.is_safe is True

    # ema_50 needs 50 bars — must be unsafe
    ema50_check = next(c for c in audit.checks if c.indicator_name == "ema_50")
    assert ema50_check.is_safe is False
    assert "Need 50 bars" in ema50_check.explanation


# ---------------------------------------------------------------------------
# Test 3 — partial safety: exact boundary cases
# ---------------------------------------------------------------------------

def test_partial_safe():
    guard = LeakageGuard()
    # 20 bars: covers warmup <= 20
    audit = guard.audit(available_bars=20)

    safe_names = {c.indicator_name for c in audit.checks if c.is_safe}
    unsafe_names = {c.indicator_name for c in audit.checks if not c.is_safe}

    # These should be safe (warmup <= 20)
    assert "vwap" in safe_names       # warmup 1
    assert "rsi" in safe_names        # warmup 14
    assert "bb_upper" in safe_names   # warmup 20

    # These should be unsafe (warmup > 20)
    assert "ema_50" in unsafe_names   # warmup 50
    assert "ema_200" in unsafe_names  # warmup 200
    assert "adx" in unsafe_names      # warmup 28

    assert audit.safe_count + audit.unsafe_count == audit.total_indicators


# ---------------------------------------------------------------------------
# Test 4 — is_safe_for_trading: custom indicator subsets
# ---------------------------------------------------------------------------

def test_is_safe_for_trading():
    guard = LeakageGuard()

    # Only need vwap and rsi (warmup 1 and 14) — 14 bars is enough
    assert guard.is_safe_for_trading(14, required_indicators=["vwap", "rsi"]) is True

    # Need ema_200 (warmup 200) — 199 bars is not enough
    assert guard.is_safe_for_trading(199, required_indicators=["ema_200"]) is False
    assert guard.is_safe_for_trading(200, required_indicators=["ema_200"]) is True

    # Default (all indicators) — need 200 bars
    assert guard.is_safe_for_trading(199) is False
    assert guard.is_safe_for_trading(200) is True


# ---------------------------------------------------------------------------
# Test 5 — result fields typed correctly
# ---------------------------------------------------------------------------

def test_result_fields():
    guard = LeakageGuard()
    audit = guard.audit(available_bars=50)

    assert isinstance(audit, LeakageAudit)
    assert isinstance(audit.checks, list)
    assert isinstance(audit.total_indicators, int)
    assert isinstance(audit.safe_count, int)
    assert isinstance(audit.unsafe_count, int)
    assert isinstance(audit.min_warmup_bars, int)
    assert isinstance(audit.explanation, str)

    assert audit.total_indicators == len(guard.WARMUP)
    assert audit.safe_count + audit.unsafe_count == audit.total_indicators
    assert audit.min_warmup_bars == max(guard.WARMUP.values())

    for check in audit.checks:
        assert isinstance(check, LeakageCheck)
        assert isinstance(check.indicator_name, str)
        assert isinstance(check.warmup_bars, int)
        assert isinstance(check.is_safe, bool)
        assert isinstance(check.explanation, str)
        assert check.indicator_name in guard.WARMUP
