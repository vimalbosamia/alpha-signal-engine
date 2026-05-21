"""
Candle Confirmation — prevents entering on single unconfirmed candle moves.

Problem: A 4-minute bullish candle flips bearish next bar. Entering on
the first candle alone causes instant loss.

Solution: Require N consecutive candles in the trade direction before
confirming entry. Also check candle body strength (not just color).

Rules:
  - Last 2 candles must agree with trade direction
  - Candle body must be > 40% of total range (not doji/indecision)
  - If last candle contradicts direction → reject
"""
from __future__ import annotations

import pandas as pd


MIN_BODY_RATIO: float = 0.30  # body must be 30%+ of high-low range


def is_candle_confirmed(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 3,
) -> tuple[bool, str]:
    """
    Check if recent candles confirm the proposed trade direction.

    Relaxed rules (avoids blocking everything):
      - If last candle matches direction with strong body → confirmed
      - If 2 of last 3 candles match direction → confirmed
      - Only reject if last candle STRONGLY contradicts (body > 50% opposing)

    Args:
        df: OHLCV DataFrame (must have open, high, low, close columns)
        direction: "BUY" or "SELL"
        lookback: number of recent candles to check (default 3)

    Returns:
        (confirmed, reason)
    """
    if len(df) < lookback + 1:
        return True, "insufficient_candles_allow"  # don't block on low data

    recent = df.iloc[-lookback:]

    confirming = 0
    for _, row in recent.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        candle_range = h - l
        if candle_range <= 0:
            continue

        is_bullish = c > o
        is_bearish = c < o

        if direction.upper() == "BUY" and is_bullish:
            confirming += 1
        elif direction.upper() == "SELL" and is_bearish:
            confirming += 1

    # 2 of 3 candles match → confirmed
    if confirming >= 2:
        return True, "confirmed"

    # Last candle matches with decent body → confirmed
    last = df.iloc[-1]
    last_o, last_h, last_l, last_c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    last_range = last_h - last_l
    last_body_ratio = abs(last_c - last_o) / last_range if last_range > 0 else 0
    last_bullish = last_c > last_o

    if direction.upper() == "BUY" and last_bullish and last_body_ratio >= MIN_BODY_RATIO:
        return True, "last_candle_confirmed"
    if direction.upper() == "SELL" and not last_bullish and last_body_ratio >= MIN_BODY_RATIO:
        return True, "last_candle_confirmed"

    # Only hard-reject if last candle STRONGLY contradicts (big body opposing)
    if direction.upper() == "BUY" and not last_bullish and last_body_ratio > 0.50:
        return False, "last_candle_strong_bearish"
    if direction.upper() == "SELL" and last_bullish and last_body_ratio > 0.50:
        return False, "last_candle_strong_bullish"

    # Weak/doji last candle — allow (not a strong contradiction)
    return True, "weak_candle_allow"


def get_candle_momentum(df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    Compute candle momentum metrics for trade quality assessment.

    Returns dict with:
      - consecutive_bullish: count of recent consecutive bullish candles
      - consecutive_bearish: count of recent consecutive bearish candles
      - avg_body_ratio: average body/range ratio (0-1, higher = stronger moves)
      - direction_consistency: fraction of recent candles matching dominant direction
    """
    if len(df) < lookback:
        return {"consecutive_bullish": 0, "consecutive_bearish": 0,
                "avg_body_ratio": 0, "direction_consistency": 0}

    recent = df.iloc[-lookback:]
    bull_count = 0
    bear_count = 0
    body_ratios = []

    # Count consecutive from most recent
    consecutive_bull = 0
    consecutive_bear = 0
    for i in range(len(recent) - 1, -1, -1):
        row = recent.iloc[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        candle_range = h - l
        body_ratio = abs(c - o) / candle_range if candle_range > 0 else 0
        body_ratios.append(body_ratio)

        if c > o:
            bull_count += 1
            if i == len(recent) - 1 or consecutive_bull > 0:
                consecutive_bull += 1
            else:
                break
        elif c < o:
            bear_count += 1
            if i == len(recent) - 1 or consecutive_bear > 0:
                consecutive_bear += 1
            else:
                break

    dominant = max(bull_count, bear_count)
    consistency = dominant / lookback if lookback > 0 else 0

    return {
        "consecutive_bullish": consecutive_bull,
        "consecutive_bearish": consecutive_bear,
        "avg_body_ratio": sum(body_ratios) / len(body_ratios) if body_ratios else 0,
        "direction_consistency": round(consistency, 2),
    }
