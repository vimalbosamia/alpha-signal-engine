"""
Unit tests for libs/risk/trailing_stop.py.
"""
from __future__ import annotations

import pytest

from libs.risk.trailing_stop import TrailingStopManager, TrailingStopState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(trail_pct: float = 1.0, activation_pct: float = 0.5) -> TrailingStopManager:
    return TrailingStopManager(trail_pct=trail_pct, activation_pct=activation_pct)


# ---------------------------------------------------------------------------
# Test 1 — BUY trailing activates once price moves enough
# ---------------------------------------------------------------------------

def test_buy_trailing_activates():
    """
    For a BUY trade, the trailing stop should activate once price has moved
    far enough above the original stop, and the stop should trail the peak.
    """
    manager = _make_manager(trail_pct=1.0, activation_pct=0.5)
    state = manager.create(entry_price=100.0, original_stop=98.0, action="BUY")

    assert state.activated is False
    assert state.current_stop == 98.0

    # Move price to 102 — profit from original stop = (102-98)/98 * 100 ≈ 4.1% > 0.5%
    state = manager.update(state, current_price=102.0, action="BUY")

    assert state.activated is True
    # Stop should be at 102 * (1 - 0.01) = 100.98
    assert state.current_stop == pytest.approx(100.98, abs=0.01)
    assert state.highest_price == pytest.approx(102.0)


# ---------------------------------------------------------------------------
# Test 2 — BUY stop only moves up, never down
# ---------------------------------------------------------------------------

def test_buy_stop_only_moves_up():
    """
    After the trail activates, if price falls back down the stop should
    remain at its highest achieved level — it never widens.
    """
    manager = _make_manager(trail_pct=1.0, activation_pct=0.5)
    state = manager.create(entry_price=100.0, original_stop=97.0, action="BUY")

    # Push price up to activate and set a high stop
    state = manager.update(state, current_price=110.0, action="BUY")
    stop_after_high = state.current_stop  # 110 * 0.99 = 108.9

    # Price pulls back
    state = manager.update(state, current_price=105.0, action="BUY")

    assert state.current_stop == pytest.approx(stop_after_high, abs=0.01)
    assert state.highest_price == pytest.approx(110.0)  # peak is preserved


# ---------------------------------------------------------------------------
# Test 3 — SELL trailing tracks lowest price and stop moves down only
# ---------------------------------------------------------------------------

def test_sell_trailing():
    """
    For a SELL trade, the trailing stop should track the lowest price and
    only move in the downward direction (i.e. stop decreases as price falls).
    """
    manager = _make_manager(trail_pct=1.0, activation_pct=0.5)
    state = manager.create(entry_price=100.0, original_stop=103.0, action="SELL")

    assert state.activated is False

    # Drop price — profit from stop = (103-98)/103 * 100 ≈ 4.9% > 0.5%
    state = manager.update(state, current_price=98.0, action="SELL")

    assert state.activated is True
    # Stop should be at 98 * (1 + 0.01) = 98.98
    assert state.current_stop == pytest.approx(98.98, abs=0.01)
    assert state.lowest_price == pytest.approx(98.0)

    # Price falls further — stop should move down
    state = manager.update(state, current_price=95.0, action="SELL")
    assert state.current_stop == pytest.approx(95.95, abs=0.01)

    # Price bounces up — stop should NOT move up
    stop_before_bounce = state.current_stop
    state = manager.update(state, current_price=97.0, action="SELL")
    assert state.current_stop == pytest.approx(stop_before_bounce, abs=0.01)


# ---------------------------------------------------------------------------
# Test 4 — stop does NOT activate before the threshold is reached
# ---------------------------------------------------------------------------

def test_not_activated_before_threshold():
    """
    If price barely moves from entry, the trailing stop should remain
    inactive and the stop should stay at its original level.
    """
    manager = _make_manager(trail_pct=1.0, activation_pct=2.0)
    state = manager.create(entry_price=100.0, original_stop=98.0, action="BUY")

    # Move price only slightly — not enough to exceed activation_pct=2%
    # profit from stop = (98.5 - 98) / 98 * 100 ≈ 0.51% < 2%
    state = manager.update(state, current_price=98.5, action="BUY")

    assert state.activated is False
    assert state.current_stop == pytest.approx(98.0)


# ---------------------------------------------------------------------------
# Test 5 — create() sets correct initial state
# ---------------------------------------------------------------------------

def test_create_initial():
    """create() should initialise state correctly for both BUY and SELL."""
    manager = _make_manager(trail_pct=1.5, activation_pct=1.0)

    buy_state = manager.create(entry_price=200.0, original_stop=195.0, action="BUY")
    assert buy_state.original_stop == 195.0
    assert buy_state.current_stop == 195.0
    assert buy_state.highest_price == 200.0
    assert buy_state.lowest_price == 200.0
    assert buy_state.trail_distance_pct == 1.5
    assert buy_state.activated is False

    sell_state = manager.create(entry_price=200.0, original_stop=205.0, action="SELL")
    assert sell_state.original_stop == 205.0
    assert sell_state.current_stop == 205.0
    assert sell_state.highest_price == 200.0
    assert sell_state.lowest_price == 200.0
    assert sell_state.activated is False
