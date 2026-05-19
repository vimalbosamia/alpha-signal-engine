"""
Unit tests for SupplyDemandDetector.

All DataFrames are built synthetically — no external data or fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from libs.analysis.levels.supply_demand import SupplyDemandDetector, Zone, price_near_zone


# ── DataFrame helper ──────────────────────────────────────────────────────────

def _make_df(
    highs: list[float],
    lows: list[float],
    closes: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from explicit high/low arrays."""
    n = len(highs)
    assert len(lows) == n, "highs and lows must be the same length"

    highs_arr = np.array(highs, dtype=float)
    lows_arr = np.array(lows, dtype=float)

    if closes is None:
        closes_arr = (highs_arr + lows_arr) / 2.0
    else:
        closes_arr = np.array(closes, dtype=float)

    opens_arr = closes_arr * 0.999

    if volumes is None:
        volumes_arr = np.full(n, 1000.0)
    else:
        volumes_arr = np.array(volumes, dtype=float)

    return pd.DataFrame(
        {
            "open": opens_arr,
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": volumes_arr,
        },
        index=pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC"),
    )


# ── Test 1: demand zone detected at strong bounce ─────────────────────────────

def test_demand_zone_at_strong_bounce() -> None:
    """
    Price drops from 100 to 90, then bounces strongly to 105 with a volume spike.
    A demand zone should be detected near the low of the reversal candle.
    """
    detector = SupplyDemandDetector(min_move_pct=0.5, volume_factor=1.5)

    # Build a sequence: 10 bars declining to ~90, then 5 bars rising sharply to ~105
    avg_vol = 1000.0
    spike_vol = avg_vol * 2.0  # volume spike at reversal

    highs =  [100, 99, 98, 97, 96, 95, 94, 93, 92, 91,  92,  96, 100, 103, 106]
    lows  =  [ 99, 98, 97, 96, 95, 94, 93, 92, 91, 90,  91,  94,  98, 101, 104]
    closes = [ 99, 98, 97, 96, 95, 94, 93, 92, 91, 90,  92,  96, 100, 103, 105]
    vols   = [avg_vol] * 9 + [spike_vol] + [avg_vol] * 5

    df = _make_df(highs, lows, closes, vols)

    zones = detector.detect(df)

    demand_zones = [z for z in zones if z.kind == "demand"]
    assert len(demand_zones) >= 1, f"Expected at least one demand zone, got zones={zones}"

    # The demand zone should be in the area of the low (~90)
    zone = demand_zones[0]
    assert zone.lower <= 92.0, f"Demand zone lower {zone.lower} expected near 90"
    assert zone.upper >= zone.lower, "Zone upper must be >= lower"


# ── Test 2: supply zone detected at strong rejection ─────────────────────────

def test_supply_zone_at_strong_rejection() -> None:
    """
    Price rises from 95 to 108, then drops sharply to 95 with a volume spike.
    A supply zone should be detected near the high of the reversal candle.
    """
    detector = SupplyDemandDetector(min_move_pct=0.5, volume_factor=1.5)

    avg_vol = 1000.0
    spike_vol = avg_vol * 2.0

    highs =  [ 96,  97,  99, 101, 103, 105, 107, 109,  108, 106, 103, 100, 97]
    lows  =  [ 95,  96,  97,  99, 101, 103, 105, 107,  106, 103, 100,  97, 95]
    closes = [ 96,  97,  98, 100, 102, 104, 106, 108,  106, 103, 100,  97, 95]
    vols   = [avg_vol] * 7 + [spike_vol] + [avg_vol] * 5

    df = _make_df(highs, lows, closes, vols)

    zones = detector.detect(df)

    supply_zones = [z for z in zones if z.kind == "supply"]
    assert len(supply_zones) >= 1, f"Expected at least one supply zone, got zones={zones}"

    zone = supply_zones[0]
    assert zone.upper >= 107.0, f"Supply zone upper {zone.upper} expected near 108–109"
    assert zone.upper >= zone.lower, "Zone upper must be >= lower"


# ── Test 3: zone has required fields ─────────────────────────────────────────

def test_zone_has_required_fields() -> None:
    """Every Zone object must have kind, upper, lower, and strength attributes."""
    detector = SupplyDemandDetector()

    avg_vol = 1000.0
    spike_vol = avg_vol * 2.0

    highs =  [100, 99, 98, 97, 96, 95, 94, 93, 92, 91,  92,  96, 100, 103, 106]
    lows  =  [ 99, 98, 97, 96, 95, 94, 93, 92, 91, 90,  91,  94,  98, 101, 104]
    closes = [ 99, 98, 97, 96, 95, 94, 93, 92, 91, 90,  92,  96, 100, 103, 105]
    vols   = [avg_vol] * 9 + [spike_vol] + [avg_vol] * 5

    df = _make_df(highs, lows, closes, vols)
    zones = detector.detect(df)

    assert len(zones) >= 1, "Expected at least one zone to test fields on"

    for zone in zones:
        assert hasattr(zone, "kind"), "Zone missing 'kind' field"
        assert hasattr(zone, "upper"), "Zone missing 'upper' field"
        assert hasattr(zone, "lower"), "Zone missing 'lower' field"
        assert hasattr(zone, "strength"), "Zone missing 'strength' field"
        assert hasattr(zone, "index"), "Zone missing 'index' field"
        assert zone.kind in ("supply", "demand"), f"Unexpected kind: {zone.kind}"
        assert zone.upper >= zone.lower, "Zone upper must be >= lower"
        assert 0.0 <= zone.strength <= 1.0, f"Strength {zone.strength} out of [0, 1]"


# ── Test 4: price_near_zone returns zone when price is within tolerance ───────

def test_price_in_demand_zone() -> None:
    """price_near_zone must return the matching zone when price is near it."""
    zone = Zone(kind="demand", upper=92.0, lower=90.0, strength=0.7, index=9)
    other = Zone(kind="supply", upper=110.0, lower=108.0, strength=0.6, index=5)

    # Price at 91.0 — inside the demand zone range
    result = price_near_zone(91.0, [zone, other], tolerance_pct=0.5)
    assert result is not None, "Expected zone to be returned for price=91.0"
    assert result.kind == "demand"


def test_price_not_near_any_zone_returns_none() -> None:
    """price_near_zone must return None when price is not near any zone."""
    zone = Zone(kind="demand", upper=92.0, lower=90.0, strength=0.7, index=9)

    # Price at 100.0 — far from the demand zone
    result = price_near_zone(100.0, [zone], tolerance_pct=0.5)
    assert result is None


# ── Test 5: empty DataFrame → empty list ─────────────────────────────────────

def test_empty_df() -> None:
    """detect() on an empty DataFrame must return an empty list without raising."""
    detector = SupplyDemandDetector()
    df = pd.DataFrame()
    result = detector.detect(df)
    assert result == [], f"Expected [], got {result}"


# ── Test: Zone is immutable (frozen dataclass) ────────────────────────────────

def test_zone_is_immutable() -> None:
    """Zone must be frozen — mutation must raise FrozenInstanceError."""
    zone = Zone(kind="demand", upper=92.0, lower=90.0, strength=0.7, index=9)
    with pytest.raises(Exception):
        zone.kind = "supply"  # type: ignore[misc]


# ── Test: price_near_zone with empty list returns None ────────────────────────

def test_price_near_zone_empty_list() -> None:
    """price_near_zone with empty zones list must return None."""
    result = price_near_zone(100.0, [], tolerance_pct=0.5)
    assert result is None
