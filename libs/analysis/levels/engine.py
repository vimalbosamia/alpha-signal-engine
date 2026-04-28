"""
Key Levels Engine — identifies support/resistance, VWAP, prior-day levels,
weekly levels, opening-range, and gap zones from OHLCV DataFrames.

Design rules:
  - Never raises on empty/None inputs — returns [] or None safely.
  - All data classes are frozen (immutable).
  - No hardcoded magic numbers; all thresholds are parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from libs.core.models.domain import AssetClass

# ── Constants ─────────────────────────────────────────────────────────────────

_SR_WINDOW: int = 5          # half-window for pivot detection (bars each side)
_MAX_TOUCH_NORMALISER: int = 5  # touch_count / this → strength (capped at 1.0)

_PDH_PDL_STRENGTH: float = 0.85
_VWAP_STRENGTH: float = 0.80
_OR_STRENGTH: float = 0.75
_WEEKLY_STRENGTH: float = 0.75
_GAP_STRENGTH: float = 0.70


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeyLevel:
    """A single key price level with metadata."""

    price: float
    level_type: str          # see module docstring for valid values
    strength: float          # 0.0–1.0
    asset_class: AssetClass
    touch_count: int = 1
    notes: str = ""


# ── Engine ────────────────────────────────────────────────────────────────────

class KeyLevelsEngine:
    """
    Detects key price levels from intraday, daily, and weekly OHLCV data.

    Parameters
    ----------
    sr_lookback:
        Number of recent bars to consider when scanning for S/R pivots.
    cluster_pct:
        Two levels closer than this fraction of price are merged into one.
    """

    def __init__(
        self,
        sr_lookback: int = 50,
        cluster_pct: float = 0.003,
    ) -> None:
        self.sr_lookback = sr_lookback
        self.cluster_pct = cluster_pct

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        asset_class: AssetClass,
        df_daily: pd.DataFrame | None = None,
        df_weekly: pd.DataFrame | None = None,
        opening_range_bars: int = 30,
    ) -> list[KeyLevel]:
        """
        Run all level-detection helpers, cluster nearby levels, and return
        a price-ascending list of ``KeyLevel`` objects.
        """
        if df is None or df.empty:
            return []

        levels: list[KeyLevel] = []

        levels.extend(self._find_sr(df, asset_class))

        vwap_level = self._find_vwap(df, asset_class)
        if vwap_level is not None:
            levels.append(vwap_level)

        levels.extend(self._find_opening_range(df, asset_class, opening_range_bars))

        if df_daily is not None and not df_daily.empty:
            levels.extend(self._find_prior_day(df_daily, asset_class))
            levels.extend(self._find_gap_zones(df_daily, asset_class))

        if df_weekly is not None and not df_weekly.empty:
            levels.extend(self._find_weekly(df_weekly, asset_class))

        levels = self._cluster_levels(levels)
        levels.sort(key=lambda lvl: lvl.price)
        return levels

    def nearest_level(
        self,
        price: float,
        levels: list[KeyLevel],
        max_dist_pct: float = 0.005,
    ) -> KeyLevel | None:
        """
        Return the ``KeyLevel`` whose price is closest to *price* and within
        ``max_dist_pct * price`` distance.  Returns ``None`` if none qualify.
        """
        if not levels:
            return None

        threshold = max_dist_pct * price
        best: KeyLevel | None = None
        best_dist = float("inf")

        for lvl in levels:
            dist = abs(lvl.price - price)
            if dist <= threshold and dist < best_dist:
                best_dist = dist
                best = lvl

        return best

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_sr(self, df: pd.DataFrame, asset_class: AssetClass) -> list[KeyLevel]:
        """
        Detect local pivot highs (resistance) and lows (support) using a
        ±``_SR_WINDOW``-bar rolling window over the most recent
        ``sr_lookback`` bars.
        """
        if df is None or len(df) < 2 * _SR_WINDOW + 1:
            return []

        subset = df.tail(self.sr_lookback)
        highs = subset["high"].to_numpy(dtype=float)
        lows = subset["low"].to_numpy(dtype=float)
        n = len(highs)

        levels: list[KeyLevel] = []

        for i in range(_SR_WINDOW, n - _SR_WINDOW):
            window_high = highs[max(0, i - _SR_WINDOW): i + _SR_WINDOW + 1]
            window_low = lows[max(0, i - _SR_WINDOW): i + _SR_WINDOW + 1]

            # Pivot high → resistance
            if highs[i] == window_high.max():
                touch_count = self._count_touches(highs[i], subset, self.cluster_pct)
                strength = min(1.0, touch_count / _MAX_TOUCH_NORMALISER)
                levels.append(
                    KeyLevel(
                        price=float(highs[i]),
                        level_type="resistance",
                        strength=strength,
                        asset_class=asset_class,
                        touch_count=touch_count,
                    )
                )

            # Pivot low → support
            if lows[i] == window_low.min():
                touch_count = self._count_touches(lows[i], subset, self.cluster_pct)
                strength = min(1.0, touch_count / _MAX_TOUCH_NORMALISER)
                levels.append(
                    KeyLevel(
                        price=float(lows[i]),
                        level_type="support",
                        strength=strength,
                        asset_class=asset_class,
                        touch_count=touch_count,
                    )
                )

        return levels

    def _find_prior_day(
        self, df_daily: pd.DataFrame, asset_class: AssetClass
    ) -> list[KeyLevel]:
        """Emit PDH (resistance) and PDL (support) from the second-to-last daily bar."""
        if df_daily is None or len(df_daily) < 2:
            return []

        prior = df_daily.iloc[-2]
        return [
            KeyLevel(
                price=float(prior["high"]),
                level_type="pdh",
                strength=_PDH_PDL_STRENGTH,
                asset_class=asset_class,
                notes="Prior day high",
            ),
            KeyLevel(
                price=float(prior["low"]),
                level_type="pdl",
                strength=_PDH_PDL_STRENGTH,
                asset_class=asset_class,
                notes="Prior day low",
            ),
        ]

    def _find_weekly(
        self, df_weekly: pd.DataFrame, asset_class: AssetClass
    ) -> list[KeyLevel]:
        """Emit weekly high / low from the second-to-last weekly bar."""
        if df_weekly is None or len(df_weekly) < 2:
            return []

        prior_week = df_weekly.iloc[-2]
        return [
            KeyLevel(
                price=float(prior_week["high"]),
                level_type="weekly_high",
                strength=_WEEKLY_STRENGTH,
                asset_class=asset_class,
                notes="Prior week high",
            ),
            KeyLevel(
                price=float(prior_week["low"]),
                level_type="weekly_low",
                strength=_WEEKLY_STRENGTH,
                asset_class=asset_class,
                notes="Prior week low",
            ),
        ]

    def _find_gap_zones(
        self, df_daily: pd.DataFrame, asset_class: AssetClass
    ) -> list[KeyLevel]:
        """
        Detect overnight gap zones.  A gap up occurs when today's open is
        above the prior day's high; gap down when below the prior day's low.
        """
        if df_daily is None or len(df_daily) < 2:
            return []

        prior = df_daily.iloc[-2]
        today = df_daily.iloc[-1]

        today_open = float(today["open"])
        prior_high = float(prior["high"])
        prior_low = float(prior["low"])

        levels: list[KeyLevel] = []

        if today_open > prior_high:
            # Gap up: zone between prior high and today's open
            levels.append(
                KeyLevel(
                    price=today_open,
                    level_type="gap_zone_top",
                    strength=_GAP_STRENGTH,
                    asset_class=asset_class,
                    notes="Gap-up top (today open)",
                )
            )
            levels.append(
                KeyLevel(
                    price=prior_high,
                    level_type="gap_zone_bottom",
                    strength=_GAP_STRENGTH,
                    asset_class=asset_class,
                    notes="Gap-up bottom (prior high)",
                )
            )
        elif today_open < prior_low:
            # Gap down: zone between today's open and prior low
            levels.append(
                KeyLevel(
                    price=prior_low,
                    level_type="gap_zone_top",
                    strength=_GAP_STRENGTH,
                    asset_class=asset_class,
                    notes="Gap-down top (prior low)",
                )
            )
            levels.append(
                KeyLevel(
                    price=today_open,
                    level_type="gap_zone_bottom",
                    strength=_GAP_STRENGTH,
                    asset_class=asset_class,
                    notes="Gap-down bottom (today open)",
                )
            )

        return levels

    def _find_vwap(
        self, df: pd.DataFrame, asset_class: AssetClass
    ) -> KeyLevel | None:
        """
        Compute cumulative VWAP from df and return the final value as a level.
        Returns ``None`` if the ``volume`` column is absent or all-zero.
        """
        if df is None or df.empty or "volume" not in df.columns:
            return None

        volume = df["volume"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        if volume.sum() == 0:
            return None

        cum_pv = np.cumsum(close * volume)
        cum_vol = np.cumsum(volume)
        vwap_series = cum_pv / cum_vol
        vwap_price = float(vwap_series[-1])

        return KeyLevel(
            price=vwap_price,
            level_type="vwap",
            strength=_VWAP_STRENGTH,
            asset_class=asset_class,
            notes="Cumulative intraday VWAP",
        )

    def _find_opening_range(
        self,
        df: pd.DataFrame,
        asset_class: AssetClass,
        n_bars: int,
    ) -> list[KeyLevel]:
        """
        Derive opening-range high and low from the first ``n_bars`` rows.
        Returns [] if the DataFrame has fewer rows than requested.
        """
        if df is None or df.empty or n_bars > len(df):
            return []

        or_slice = df.iloc[:n_bars]
        or_high = float(or_slice["high"].max())
        or_low = float(or_slice["low"].min())

        return [
            KeyLevel(
                price=or_high,
                level_type="or_high",
                strength=_OR_STRENGTH,
                asset_class=asset_class,
                notes=f"Opening range high ({n_bars} bars)",
            ),
            KeyLevel(
                price=or_low,
                level_type="or_low",
                strength=_OR_STRENGTH,
                asset_class=asset_class,
                notes=f"Opening range low ({n_bars} bars)",
            ),
        ]

    def _cluster_levels(self, levels: list[KeyLevel]) -> list[KeyLevel]:
        """
        Merge levels whose prices are within ``cluster_pct`` of each other.

        Merging rules:
          - Price     → average of the group.
          - Strength  → maximum of the group.
          - touch_count → sum of the group.
          - level_type  → from the highest-strength member.
          - asset_class → carried from the highest-strength member.
          - notes     → carried from the highest-strength member.
        """
        if not levels:
            return []

        # Sort by price so nearby levels are adjacent.
        sorted_levels = sorted(levels, key=lambda lvl: lvl.price)

        clusters: list[list[KeyLevel]] = []
        current_cluster: list[KeyLevel] = [sorted_levels[0]]

        for lvl in sorted_levels[1:]:
            reference_price = current_cluster[0].price
            if abs(lvl.price - reference_price) <= self.cluster_pct * reference_price:
                current_cluster.append(lvl)
            else:
                clusters.append(current_cluster)
                current_cluster = [lvl]
        clusters.append(current_cluster)

        merged: list[KeyLevel] = []
        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
                continue

            avg_price = float(np.mean([lvl.price for lvl in cluster]))
            total_touches = sum(lvl.touch_count for lvl in cluster)
            strongest = max(cluster, key=lambda lvl: lvl.strength)

            merged.append(
                KeyLevel(
                    price=avg_price,
                    level_type=strongest.level_type,
                    strength=strongest.strength,
                    asset_class=strongest.asset_class,
                    touch_count=total_touches,
                    notes=strongest.notes,
                )
            )

        return merged

    # ── Internal utility ──────────────────────────────────────────────────────

    @staticmethod
    def _count_touches(
        level_price: float, df: pd.DataFrame, cluster_pct: float
    ) -> int:
        """
        Count bars whose high or low is within ``cluster_pct`` of *level_price*.
        """
        threshold = cluster_pct * level_price
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)

        touches = int(
            np.sum(
                (np.abs(highs - level_price) <= threshold)
                | (np.abs(lows - level_price) <= threshold)
            )
        )
        return max(1, touches)
