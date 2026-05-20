"""
OrderBlockDetector — identifies institutional order blocks from OHLCV data.

Order blocks are the last opposing candle before a strong impulsive move.
Institutions leave limit orders at these levels, so price often returns to them.

Bullish OB: last bearish candle before a strong bullish impulse (up > ATR×2 in 3 bars).
            Zone = (low, open) of that bearish candle.
Bearish OB: last bullish candle before a strong bearish impulse (down > ATR×2 in 3 bars).
            Zone = (close, high) of that bullish candle.

Design rules:
  - Never raises on empty or short DataFrames — returns [] safely.
  - All output objects are frozen dataclasses (immutable).
  - No hardcoded magic numbers; all thresholds are constructor parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_ATR_WINDOW: int = 14          # bars for ATR calculation
_IMPULSE_BARS: int = 3         # bars after OB to confirm impulsive move
_MIN_BARS: int = _ATR_WINDOW + _IMPULSE_BARS + 2  # minimum required
_DEFAULT_ATR_FACTOR: float = 2.0  # impulse must exceed ATR × this factor
_STRENGTH_BASE: float = 0.5    # base strength for any confirmed OB
_STRENGTH_ATR_SCALE: float = 3.0  # impulse / (ATR × factor) normalised to add 0.5


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrderBlock:
    """A single institutional order block."""

    kind: str        # "bullish_ob" or "bearish_ob"
    upper: float     # top boundary of the OB zone
    lower: float     # bottom boundary of the OB zone
    strength: float  # 0.0 to 1.0
    index: int       # bar index of the order block candle
    explanation: str


# ── Detector ──────────────────────────────────────────────────────────────────

class OrderBlockDetector:
    """
    Detects institutional order blocks from an OHLCV DataFrame.

    Parameters
    ----------
    atr_factor : float
        The subsequent move (measured over ``impulse_bars``) must exceed
        ``atr_factor × ATR`` to confirm an order block.
    impulse_bars : int
        Number of bars after the OB candle used to measure the impulse move.
    """

    def __init__(
        self,
        atr_factor: float = _DEFAULT_ATR_FACTOR,
        impulse_bars: int = _IMPULSE_BARS,
    ) -> None:
        self._atr_factor = atr_factor
        self._impulse_bars = impulse_bars

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> list[OrderBlock]:
        """
        Scan *df* for order blocks and return a list of ``OrderBlock`` objects.

        Returns an empty list for empty or too-short DataFrames without raising.
        """
        if df is None or df.empty or len(df) < _MIN_BARS:
            return []

        required = {"high", "low", "open", "close"}
        if not required.issubset(df.columns):
            return []

        try:
            return self._detect_impl(df)
        except Exception:
            return []

    def _detect_impl(self, df: pd.DataFrame) -> list[OrderBlock]:
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)

        atr = _compute_atr(highs, lows, closes, window=_ATR_WINDOW)
        n = len(closes)
        threshold_bars = self._impulse_bars
        order_blocks: list[OrderBlock] = []

        # Scan each bar (leave room for impulse_bars look-ahead)
        for i in range(n - threshold_bars):
            bar_atr = float(atr[i]) if atr[i] > 0 else 1e-8
            threshold = bar_atr * self._atr_factor

            is_bearish = closes[i] < opens[i]
            is_bullish = closes[i] > opens[i]

            # ── Bullish OB: bearish candle followed by strong bullish move ──
            if is_bearish:
                end = min(i + threshold_bars + 1, n)
                future_high = float(highs[i + 1: end].max())
                impulse_up = future_high - float(highs[i])
                if impulse_up >= threshold:
                    strength = _compute_strength(impulse_up, bar_atr, self._atr_factor)
                    order_blocks.append(
                        OrderBlock(
                            kind="bullish_ob",
                            upper=float(opens[i]),   # open of bearish candle
                            lower=float(lows[i]),    # low of bearish candle
                            strength=round(strength, 4),
                            index=i,
                            explanation=(
                                f"Bearish candle at bar {i} (O={opens[i]:.4f}, "
                                f"C={closes[i]:.4f}) followed by bullish impulse "
                                f"{impulse_up:.4f} > ATR×{self._atr_factor} ({threshold:.4f})."
                            ),
                        )
                    )

            # ── Bearish OB: bullish candle followed by strong bearish move ──
            if is_bullish:
                end = min(i + threshold_bars + 1, n)
                future_low = float(lows[i + 1: end].min())
                impulse_down = float(lows[i]) - future_low
                if impulse_down >= threshold:
                    strength = _compute_strength(impulse_down, bar_atr, self._atr_factor)
                    order_blocks.append(
                        OrderBlock(
                            kind="bearish_ob",
                            upper=float(highs[i]),   # high of bullish candle
                            lower=float(closes[i]),  # close of bullish candle
                            strength=round(strength, 4),
                            index=i,
                            explanation=(
                                f"Bullish candle at bar {i} (O={opens[i]:.4f}, "
                                f"C={closes[i]:.4f}) followed by bearish impulse "
                                f"{impulse_down:.4f} > ATR×{self._atr_factor} ({threshold:.4f})."
                            ),
                        )
                    )

        return order_blocks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    window: int = _ATR_WINDOW,
) -> np.ndarray:
    """
    Compute a simple ATR (Average True Range) for each bar.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    For bar 0 we use high-low as the true range.
    """
    n = len(highs)
    tr = np.empty(n, dtype=float)
    tr[0] = highs[0] - lows[0]

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hpc, lpc)

    # Rolling mean of true range
    atr = np.empty(n, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        atr[i] = float(tr[start: i + 1].mean())

    return atr


def _compute_strength(impulse: float, bar_atr: float, atr_factor: float) -> float:
    """
    Compute OB strength in [0.0, 1.0].

    Scales the impulse size relative to the ATR threshold:
    at exactly threshold → 0.5; at 3× threshold → 1.0.
    """
    threshold = bar_atr * atr_factor
    if threshold <= 0:
        return _STRENGTH_BASE

    ratio = impulse / threshold  # ≥ 1.0 for any confirmed OB
    # Map: ratio=1 → 0.5, ratio=_STRENGTH_ATR_SCALE → 1.0
    scale = (ratio - 1.0) / (_STRENGTH_ATR_SCALE - 1.0)
    return min(1.0, _STRENGTH_BASE + scale * _STRENGTH_BASE)
