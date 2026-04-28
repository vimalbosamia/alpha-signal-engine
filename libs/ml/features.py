"""
Feature extraction for the ML signal classifier.

Converts a SignalOutput (+ optional extra context) into a fixed-length
numeric feature vector used for training and inference.

Features (13 total):
  0  confluence_score       float  0–1
  1  estimated_risk_reward  float
  2  action_enc             int    BUY=1, SELL=0
  3  regime_enc             int    trending_up=3, ranging=2, volatile=1, trending_down=0
  4  patterns_count         int
  5  hour_sin               float  cyclical hour-of-day
  6  hour_cos               float
  7  day_of_week            int    0=Mon … 6=Sun
  8  strategy_enc           int    see STRATEGY_MAP
  9  asset_class_enc        int    crypto=1, stock=0
 10  entry_zone_width_pct   float  (high-low)/mid * 100
 11  stop_dist_pct          float  |entry_mid - stop| / entry_mid * 100
 12  tp1_dist_pct           float  |tp1 - entry_mid| / entry_mid * 100
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from libs.core.models.domain import SignalOutput

STRATEGY_MAP: dict[str, int] = {
    "hammer_reversal":         0,
    "resistance_breakout":     1,
    "pullback_continuation":   2,
    "rsi_mean_reversion":      3,
    "macd_crossover":          4,
    "ema_crossover":           5,
}

REGIME_MAP: dict[str, int] = {
    "trending_up":    3,
    "ranging":        2,
    "volatile":       1,
    "trending_down":  0,
}

FEATURE_NAMES = [
    "confluence_score",
    "estimated_risk_reward",
    "action_enc",
    "regime_enc",
    "patterns_count",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "strategy_enc",
    "asset_class_enc",
    "entry_zone_width_pct",
    "stop_dist_pct",
    "tp1_dist_pct",
]


def extract(signal: "SignalOutput") -> list[float]:
    """Return a 13-element feature vector from a SignalOutput."""
    mid = (signal.entry_zone_low + signal.entry_zone_high) / 2 if signal.entry_zone_low else 1.0

    # Cyclical hour encoding
    hour = signal.generated_at.hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    entry_width_pct = (
        abs(signal.entry_zone_high - signal.entry_zone_low) / mid * 100
        if mid else 0.0
    )
    stop_dist_pct = abs(mid - signal.stop_loss) / mid * 100 if mid else 0.0
    tp1_dist_pct  = abs(signal.take_profit_1 - mid) / mid * 100 if mid and signal.take_profit_1 else 0.0

    return [
        float(signal.confidence),
        float(signal.estimated_risk_reward),
        1.0 if signal.action.value == "BUY" else 0.0,
        float(REGIME_MAP.get(signal.market_regime.value, 1)),
        float(len(signal.patterns_detected or [])),
        hour_sin,
        hour_cos,
        float(signal.generated_at.weekday()),
        float(STRATEGY_MAP.get(signal.strategy_name, -1)),
        1.0 if signal.asset_class.value == "crypto" else 0.0,
        entry_width_pct,
        stop_dist_pct,
        tp1_dist_pct,
    ]


def extract_from_dict(d: dict) -> list[float]:
    """Extract features from a stored signal dict (from DB payload)."""
    from datetime import datetime, timezone

    mid = (d.get("entry_zone_low", 0) + d.get("entry_zone_high", 0)) / 2 or 1.0
    generated_at_str = d.get("generated_at", "")
    try:
        generated_at = datetime.fromisoformat(generated_at_str.replace("Z", "+00:00"))
    except Exception:
        generated_at = datetime.now(timezone.utc)

    hour = generated_at.hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    entry_width_pct = (
        abs(d.get("entry_zone_high", mid) - d.get("entry_zone_low", mid)) / mid * 100
        if mid else 0.0
    )
    stop_dist_pct = abs(mid - d.get("stop_loss", mid)) / mid * 100 if mid else 0.0
    tp1_dist_pct  = abs(d.get("take_profit_1", mid) - mid) / mid * 100 if mid else 0.0

    action = d.get("action", "BUY")
    regime = d.get("market_regime", "ranging")
    strategy = d.get("strategy_name", "")
    asset_class = d.get("asset_class", "crypto")

    return [
        float(d.get("confidence", 0.5)),
        float(d.get("estimated_risk_reward", 2.0)),
        1.0 if action == "BUY" else 0.0,
        float(REGIME_MAP.get(regime, 1)),
        float(len(d.get("patterns", []))),
        hour_sin,
        hour_cos,
        float(generated_at.weekday()),
        float(STRATEGY_MAP.get(strategy, -1)),
        1.0 if asset_class == "crypto" else 0.0,
        entry_width_pct,
        stop_dist_pct,
        tp1_dist_pct,
    ]
