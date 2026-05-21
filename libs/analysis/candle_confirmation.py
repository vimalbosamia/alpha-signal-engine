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


MIN_CONFIRMING_CANDLES: int = 2
MIN_BODY_RATIO: float = 0.40  # body must be 40%+ of high-low range


def is_candle_confirmed(
    df: pd.DataFrame,
    direction: str,
    lookback: int = MIN_CONFIRMING_CANDLES,
) -> tuple[bool, str]:
    """
    Check if recent candles confirm the proposed trade direction.

    Args:
        df: OHLCV DataFrame (must have open, high, low, close columns)
        direction: "BUY" or "SELL"
        lookback: number of recent candles to check (default 2)

    Returns:
        (confirmed, reason)
    """
    if len(df) < lookback + 1:
        return False, "insufficient_candles"

    recent = df.iloc[-lookback:]

    confirming = 0
    for _, row in recent.iterrows():
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        candle_range = h - l
        if candle_range <= 0:
            continue

        body = abs(c - o)
        body_ratio = body / candle_range
        is_bullish = c > o
        is_bearish = c < o

        # Check direction match + body strength
        if direction.upper() == "BUY" and is_bullish and body_ratio >= MIN_BODY_RATIO:
            confirming += 1
        elif direction.upper() == "SELL" and is_bearish and body_ratio >= MIN_BODY_RATIO:
            confirming += 1

    if confirming >= lookback:
        return True, "confirmed"

    # Check if last candle contradicts
    last = df.iloc[-1]
    last_bullish = float(last["close"]) > float(last["open"])
    if direction.upper() == "BUY" and not last_bullish:
        return False, "last_candle_bearish"
    if direction.upper() == "SELL" and last_bullish:
        return False, "last_candle_bullish"

    return False, f"only_{confirming}_of_{lookback}_confirmed"


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
