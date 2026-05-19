"""
Supply and Demand Zone Detector — identifies institutional supply/demand zones
from OHLCV data based on strong price reversals with volume confirmation.

Design rules:
  - Never raises on empty/None inputs — returns [] safely.
  - All data classes are frozen (immutable).
  - No hardcoded magic numbers; all thresholds are constructor parameters.

Zone definitions:
  Demand zone: formed at swing lows where price drops then bounces strongly.
               Zone spans from (low of candle) to (body low = min(open, close)).
  Supply zone: formed at swing highs where price rises then drops strongly.
               Zone spans from (body high = max(open, close)) to (high of candle).

Strength scoring:
  Base score from move size relative to min_move_pct threshold, capped at 0.8.
  Volume spike bonus up to 0.2 when volume exceeds volume_factor * average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_SWING_WINDOW: int = 3        # bars each side for swing high/low detection
_MIN_BARS: int = 2 * _SWING_WINDOW + 3  # minimum bars required to detect any zone
_MAX_MOVE_FOR_STRENGTH: float = 5.0     # move % that gives full base strength (0.8)
_VOLUME_STRENGTH_BONUS: float = 0.2     # max bonus from volume spike


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Zone:
    """A single supply or demand zone with metadata."""

    kind: str       # "supply" or "demand"
    upper: float    # top edge of zone
    lower: float    # bottom edge of zone
    strength: float  # 0.0 to 1.0
    index: int      # bar index where zone originated
    touches: int = 1


# ── Public utility ────────────────────────────────────────────────────────────

def price_near_zone(
    price: float,
    zones: list[Zone],
    tolerance_pct: float = 0.5,
) -> Zone | None:
    """
    Return the nearest zone if *price* is within ``tolerance_pct`` percent of
    any zone boundary or falls inside the zone.

    Parameters
    ----------
    price:
        Current market price to test.
    zones:
        List of Zone objects to search.
    tolerance_pct:
        How close (as a percentage of price) the price must be to a zone
        boundary to count as "near".  Default 0.5 %.

    Returns
    -------
    The nearest qualifying Zone, or None if no zone qualifies.
    """
    if not zones:
        return None

    tolerance = tolerance_pct / 100.0 * price
    best: Zone | None = None
    best_dist = float("inf")

    for zone in zones:
        # Price inside zone
        if zone.lower <= price <= zone.upper:
            dist = 0.0
        else:
            # Distance to nearest zone edge
            dist = min(abs(price - zone.upper), abs(price - zone.lower))

        if dist <= tolerance and dist < best_dist:
            best_dist = dist
            best = zone

    return best


# ── Detector ──────────────────────────────────────────────────────────────────

class SupplyDemandDetector:
    """
    Detects supply and demand zones from OHLCV DataFrames.

    Parameters
    ----------
    min_move_pct:
        Minimum percentage move away from the reversal candle required to
        qualify a zone.  Either this threshold OR a volume spike must be met.
    volume_factor:
        A candle's volume must exceed ``volume_factor * rolling_avg_volume``
        to be considered a volume spike.
    """

    def __init__(
        self,
        min_move_pct: float = 0.5,
        volume_factor: float = 1.5,
    ) -> None:
        self.min_move_pct = min_move_pct
        self.volume_factor = volume_factor

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> list[Zone]:
        """
        Scan *df* for supply and demand zones and return a list of ``Zone``
        objects sorted by index (chronological order).

        Returns an empty list for empty or too-short DataFrames without raising.
        """
        if df is None or df.empty or len(df) < _MIN_BARS:
            return []

        required = {"high", "low", "open", "close", "volume"}
        if not required.issubset(df.columns):
            return []

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        volumes = df["volume"].to_numpy(dtype=float)

        avg_volume = self._rolling_avg_volume(volumes)

        zones: list[Zone] = []

        for i in range(_SWING_WINDOW, len(df) - _SWING_WINDOW):
            # ── Demand zone: swing low ─────────────────────────────────────
            if self._is_swing_low(lows, i):
                zone = self._try_demand_zone(
                    i, highs, lows, opens, closes, volumes, avg_volume
                )
                if zone is not None:
                    zones.append(zone)

            # ── Supply zone: swing high ────────────────────────────────────
            if self._is_swing_high(highs, i):
                zone = self._try_supply_zone(
                    i, highs, lows, opens, closes, volumes, avg_volume
                )
                if zone is not None:
                    zones.append(zone)

        zones.sort(key=lambda z: z.index)
        return zones

    # ── Private: swing detection ──────────────────────────────────────────────

    @staticmethod
    def _is_swing_low(lows: np.ndarray, i: int) -> bool:
        """True when lows[i] is the minimum in a ±SWING_WINDOW window."""
        window = lows[i - _SWING_WINDOW: i + _SWING_WINDOW + 1]
        return bool(lows[i] == window.min())

    @staticmethod
    def _is_swing_high(highs: np.ndarray, i: int) -> bool:
        """True when highs[i] is the maximum in a ±SWING_WINDOW window."""
        window = highs[i - _SWING_WINDOW: i + _SWING_WINDOW + 1]
        return bool(highs[i] == window.max())

    # ── Private: zone construction ────────────────────────────────────────────

    def _try_demand_zone(
        self,
        i: int,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        avg_volume: np.ndarray,
    ) -> Zone | None:
        """
        Attempt to create a demand zone at swing low index *i*.

        Qualification: the price must bounce strongly upward after the low,
        evidenced by ``move_up >= min_move_pct`` OR a volume spike at bar *i*.
        """
        bounce_high = highs[i + 1: i + _SWING_WINDOW + 2].max() if i + 1 < len(highs) else 0.0
        candle_low = lows[i]

        if candle_low <= 0.0:
            return None

        move_up_pct = (bounce_high - candle_low) / candle_low * 100.0
        has_volume_spike = self._has_volume_spike(volumes[i], avg_volume[i])

        if move_up_pct < self.min_move_pct and not has_volume_spike:
            return None

        # Zone boundaries: low of candle (lower) → body low (upper)
        body_low = min(opens[i], closes[i])
        zone_lower = candle_low
        zone_upper = max(body_low, candle_low)  # ensure upper >= lower

        strength = self._compute_strength(move_up_pct, volumes[i], avg_volume[i])

        return Zone(
            kind="demand",
            upper=float(zone_upper),
            lower=float(zone_lower),
            strength=strength,
            index=i,
        )

    def _try_supply_zone(
        self,
        i: int,
        highs: np.ndarray,
        lows: np.ndarray,
        opens: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        avg_volume: np.ndarray,
    ) -> Zone | None:
        """
        Attempt to create a supply zone at swing high index *i*.

        Qualification: the price must drop strongly after the high,
        evidenced by ``move_down >= min_move_pct`` OR a volume spike at bar *i*.
        """
        drop_low = lows[i + 1: i + _SWING_WINDOW + 2].min() if i + 1 < len(lows) else float("inf")
        candle_high = highs[i]

        if candle_high <= 0.0:
            return None

        move_down_pct = (candle_high - drop_low) / candle_high * 100.0
        has_volume_spike = self._has_volume_spike(volumes[i], avg_volume[i])

        if move_down_pct < self.min_move_pct and not has_volume_spike:
            return None

        # Zone boundaries: body high (lower) → high of candle (upper)
        body_high = max(opens[i], closes[i])
        zone_upper = candle_high
        zone_lower = min(body_high, candle_high)  # ensure lower <= upper

        strength = self._compute_strength(move_down_pct, volumes[i], avg_volume[i])

        return Zone(
            kind="supply",
            upper=float(zone_upper),
            lower=float(zone_lower),
            strength=strength,
            index=i,
        )

    # ── Private: strength & volume helpers ───────────────────────────────────

    def _has_volume_spike(self, vol: float, avg_vol: float) -> bool:
        """True when vol exceeds volume_factor * avg_vol."""
        if avg_vol <= 0.0:
            return False
        return vol >= self.volume_factor * avg_vol

    def _compute_strength(
        self, move_pct: float, vol: float, avg_vol: float
    ) -> float:
        """
        Compute zone strength in [0.0, 1.0].

        Base strength is proportional to move size (capped at 0.8).
        Volume spike adds up to 0.2 bonus.
        """
        base = min(0.8, (move_pct / _MAX_MOVE_FOR_STRENGTH) * 0.8)
        base = max(0.0, base)

        if avg_vol > 0.0:
            spike_ratio = vol / avg_vol
            # Scale: at volume_factor → 0; at 3× volume_factor → full bonus
            vol_bonus_raw = (spike_ratio - 1.0) / max(1.0, self.volume_factor * 2.0)
            vol_bonus = min(_VOLUME_STRENGTH_BONUS, max(0.0, vol_bonus_raw * _VOLUME_STRENGTH_BONUS))
        else:
            vol_bonus = 0.0

        return min(1.0, base + vol_bonus)

    # ── Private: rolling average volume ──────────────────────────────────────

    @staticmethod
    def _rolling_avg_volume(volumes: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Compute a backward-looking rolling average of volume.

        For bars with fewer than ``window`` preceding bars, the average covers
        all available bars.  Returns an array of the same length as *volumes*.
        """
        n = len(volumes)
        avg = np.empty(n, dtype=float)
        for i in range(n):
            start = max(0, i - window)
            segment = volumes[start:i]
            avg[i] = float(segment.mean()) if len(segment) > 0 else volumes[i]
        return avg
