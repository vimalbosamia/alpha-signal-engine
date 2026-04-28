"""
Unit tests for KeyLevelsEngine.

All DataFrames are built synthetically — no external data or fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from libs.analysis.levels.engine import KeyLevel, KeyLevelsEngine
from libs.core.models.domain import AssetClass


# ── DataFrame helpers ─────────────────────────────────────────────────────────

def make_intraday(
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal intraday (1-min) OHLCV DataFrame from close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    vols = volumes if volumes is not None else [1000.0] * n
    return pd.DataFrame(
        {
            "open": closes_arr * 0.999,
            "high": closes_arr * 1.003,
            "low": closes_arr * 0.997,
            "close": closes_arr,
            "volume": vols,
        },
        index=pd.date_range(
            "2024-01-02 09:30", periods=n, freq="1min", tz="UTC"
        ),
    )


def make_daily(closes: list[float]) -> pd.DataFrame:
    """Build a minimal daily OHLCV DataFrame from close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes_arr * 0.998,
            "high": closes_arr * 1.01,
            "low": closes_arr * 0.99,
            "close": closes_arr,
            "volume": [1_000_000.0] * n,
        },
        index=pd.date_range("2023-12-20", periods=n, freq="B"),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> KeyLevelsEngine:
    return KeyLevelsEngine(sr_lookback=50, cluster_pct=0.003)


@pytest.fixture
def asset() -> AssetClass:
    return AssetClass.STOCK


# ── Test 1: analyze with only intraday data — no exception ────────────────────

def test_analyze_intraday_only_no_exception(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """
    ``analyze`` called with only intraday data must return a list
    (possibly empty) and must NOT raise any exception.
    """
    # 60 bars of slowly rising prices
    closes = [100.0 + i * 0.1 for i in range(60)]
    df = make_intraday(closes)

    result = engine.analyze(df, asset)

    assert isinstance(result, list)
    # All items must be KeyLevel instances
    for lvl in result:
        assert isinstance(lvl, KeyLevel)


# ── Test 2: PDH / PDL detected from daily data ────────────────────────────────

def test_pdh_pdl_detected(engine: KeyLevelsEngine, asset: AssetClass) -> None:
    """
    When ``df_daily`` has ≥ 2 rows, a PDH and PDL level must appear in the
    output (after clustering they may be merged with nearby levels, but their
    level_type remains if they are the strongest in the cluster).

    We verify using level_type strings rather than exact prices to be
    cluster-tolerant.
    """
    closes = [100.0 + i * 0.1 for i in range(60)]
    df = make_intraday(closes)

    # Use prices far from the intraday range so no clustering occurs
    daily_closes = [50.0, 200.0]   # prior-day close way below/above intraday
    df_daily = make_daily(daily_closes)

    result = engine.analyze(df, asset, df_daily=df_daily)

    types = {lvl.level_type for lvl in result}
    assert "pdh" in types, f"Expected 'pdh' in level types, got {types}"
    assert "pdl" in types, f"Expected 'pdl' in level types, got {types}"


# ── Test 3: VWAP level present when volume present ───────────────────────────

def test_vwap_level_present(engine: KeyLevelsEngine, asset: AssetClass) -> None:
    """A ``vwap`` level must appear when the intraday df has a volume column."""
    closes = [100.0 + i * 0.05 for i in range(60)]
    df = make_intraday(closes)

    result = engine.analyze(df, asset)

    types = {lvl.level_type for lvl in result}
    assert "vwap" in types, f"Expected 'vwap' in level types, got {types}"


def test_vwap_absent_without_volume(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """When volume column is absent, no ``vwap`` level should appear."""
    closes = [100.0 + i * 0.05 for i in range(60)]
    df = make_intraday(closes)
    df = df.drop(columns=["volume"])

    result = engine.analyze(df, asset)

    types = {lvl.level_type for lvl in result}
    assert "vwap" not in types


# ── Test 4: Opening range levels returned when n_bars <= len(df) ─────────────

def test_opening_range_levels_present(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """
    ``_find_opening_range`` must emit or_high and or_low when n_bars ≤ len(df).

    We verify the helper directly (pre-clustering) to avoid false negatives
    caused by or_high / or_low being legitimately merged with S/R pivots that
    share the same price (correct engine behavior).  The test also confirms
    that the prices match the expected OR range.
    """
    closes = [100.0 + i * 0.02 for i in range(60)]
    df = make_intraday(closes)

    or_levels = engine._find_opening_range(df, asset, n_bars=30)

    types = {lvl.level_type for lvl in or_levels}
    assert "or_high" in types, f"Expected 'or_high' in {types}"
    assert "or_low" in types, f"Expected 'or_low' in {types}"

    or_high = next(l for l in or_levels if l.level_type == "or_high")
    or_low = next(l for l in or_levels if l.level_type == "or_low")

    # OR high must equal the max of the first 30 bars' high column
    expected_high = float(df.iloc[:30]["high"].max())
    expected_low = float(df.iloc[:30]["low"].min())

    assert or_high.price == pytest.approx(expected_high)
    assert or_low.price == pytest.approx(expected_low)
    assert or_high.price >= or_low.price


def test_opening_range_absent_when_n_bars_exceeds_length(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """When ``opening_range_bars`` > len(df), no OR levels should appear."""
    closes = [100.0] * 10
    df = make_intraday(closes)

    result = engine.analyze(df, asset, opening_range_bars=30)

    types = {lvl.level_type for lvl in result}
    assert "or_high" not in types
    assert "or_low" not in types


# ── Test 5: nearest_level returns correct nearest level ──────────────────────

def test_nearest_level_returns_closest(asset: AssetClass) -> None:
    """``nearest_level`` should return the level closest to the query price."""
    engine = KeyLevelsEngine()
    levels = [
        KeyLevel(price=100.0, level_type="support", strength=0.5, asset_class=asset),
        KeyLevel(price=105.0, level_type="resistance", strength=0.6, asset_class=asset),
        KeyLevel(price=110.0, level_type="resistance", strength=0.7, asset_class=asset),
    ]

    # Query at 104.5 → closest is 105.0 (dist 0.5), within 0.5% of 104.5
    result = engine.nearest_level(104.5, levels, max_dist_pct=0.01)

    assert result is not None
    assert result.price == pytest.approx(105.0)


# ── Test 6: nearest_level returns None when all levels are too far ───────────

def test_nearest_level_returns_none_when_too_far(asset: AssetClass) -> None:
    """``nearest_level`` must return ``None`` when no level is within threshold."""
    engine = KeyLevelsEngine()
    levels = [
        KeyLevel(price=100.0, level_type="support", strength=0.5, asset_class=asset),
        KeyLevel(price=200.0, level_type="resistance", strength=0.6, asset_class=asset),
    ]

    # Query at 150 — both levels are >30% away; threshold is 0.5%
    result = engine.nearest_level(150.0, levels, max_dist_pct=0.005)

    assert result is None


def test_nearest_level_empty_list(asset: AssetClass) -> None:
    """``nearest_level`` with an empty list must return ``None``."""
    engine = KeyLevelsEngine()
    assert engine.nearest_level(100.0, [], max_dist_pct=0.01) is None


# ── Test 7: Clustering — two levels within 0.3% merge into one ───────────────

def test_clustering_merges_nearby_levels(asset: AssetClass) -> None:
    """
    Two levels whose prices differ by < cluster_pct should be merged into
    exactly one level.
    """
    engine = KeyLevelsEngine(cluster_pct=0.003)   # 0.3 %

    # 100.0 and 100.2 → differ by 0.2 % (< 0.3 %)
    levels = [
        KeyLevel(price=100.0, level_type="support", strength=0.4, asset_class=asset, touch_count=2),
        KeyLevel(price=100.2, level_type="resistance", strength=0.7, asset_class=asset, touch_count=3),
    ]

    merged = engine._cluster_levels(levels)

    assert len(merged) == 1, f"Expected 1 merged level, got {len(merged)}"
    merged_lvl = merged[0]

    # Price is the average of the two
    assert merged_lvl.price == pytest.approx(100.1)
    # touch_count is the sum
    assert merged_lvl.touch_count == 5
    # strength is the maximum
    assert merged_lvl.strength == pytest.approx(0.7)
    # level_type comes from the strongest member
    assert merged_lvl.level_type == "resistance"


def test_clustering_does_not_merge_distant_levels(asset: AssetClass) -> None:
    """
    Two levels that differ by more than cluster_pct must remain separate.
    """
    engine = KeyLevelsEngine(cluster_pct=0.003)   # 0.3 %

    # 100.0 and 101.0 → differ by 1.0 % (> 0.3 %)
    levels = [
        KeyLevel(price=100.0, level_type="support", strength=0.5, asset_class=asset),
        KeyLevel(price=101.0, level_type="resistance", strength=0.6, asset_class=asset),
    ]

    result = engine._cluster_levels(levels)

    assert len(result) == 2


# ── Test 8: Empty DataFrame → returns [] ─────────────────────────────────────

def test_empty_dataframe_returns_empty_list(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """``analyze`` on an empty DataFrame must return an empty list, not raise."""
    df = pd.DataFrame()
    result = engine.analyze(df, asset)
    assert result == []


def test_none_daily_does_not_raise(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """Passing ``df_daily=None`` must not raise."""
    closes = [100.0] * 20
    df = make_intraday(closes)
    result = engine.analyze(df, asset, df_daily=None)
    assert isinstance(result, list)


def test_none_weekly_does_not_raise(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """Passing ``df_weekly=None`` must not raise."""
    closes = [100.0] * 20
    df = make_intraday(closes)
    result = engine.analyze(df, asset, df_weekly=None)
    assert isinstance(result, list)


# ── Test: result is sorted by price ascending ─────────────────────────────────

def test_result_sorted_by_price(engine: KeyLevelsEngine, asset: AssetClass) -> None:
    """The returned list must always be sorted by price in ascending order."""
    closes = [100.0 + i * 0.1 for i in range(60)]
    df = make_intraday(closes)
    df_daily = make_daily([50.0, 200.0])   # wide range to get PDH/PDL far apart

    result = engine.analyze(df, asset, df_daily=df_daily)

    prices = [lvl.price for lvl in result]
    assert prices == sorted(prices), f"Levels not sorted: {prices}"


# ── Test: KeyLevel is immutable (frozen dataclass) ────────────────────────────

def test_key_level_is_immutable(asset: AssetClass) -> None:
    """KeyLevel must be frozen — mutation must raise."""
    lvl = KeyLevel(price=100.0, level_type="support", strength=0.5, asset_class=asset)
    with pytest.raises(Exception):
        lvl.price = 200.0  # type: ignore[misc]


# ── Test: weekly levels detected ─────────────────────────────────────────────

def test_weekly_levels_detected(engine: KeyLevelsEngine, asset: AssetClass) -> None:
    """``weekly_high`` and ``weekly_low`` must appear when df_weekly has ≥ 2 rows."""
    closes = [100.0 + i * 0.1 for i in range(60)]
    df = make_intraday(closes)

    # Weekly data far from intraday range to avoid clustering
    n = 4
    weekly_closes = np.array([50.0, 300.0, 310.0, 320.0], dtype=float)
    df_weekly = pd.DataFrame(
        {
            "open": weekly_closes * 0.998,
            "high": weekly_closes * 1.02,
            "low": weekly_closes * 0.98,
            "close": weekly_closes,
            "volume": [5_000_000.0] * n,
        },
        index=pd.date_range("2023-12-01", periods=n, freq="W"),
    )

    result = engine.analyze(df, asset, df_weekly=df_weekly)

    types = {lvl.level_type for lvl in result}
    assert "weekly_high" in types, f"Expected 'weekly_high' in {types}"
    assert "weekly_low" in types, f"Expected 'weekly_low' in {types}"


# ── Test: gap zone levels detected ───────────────────────────────────────────

def test_gap_zone_detected(engine: KeyLevelsEngine, asset: AssetClass) -> None:
    """
    When today's open is well above the prior day's high, ``_find_gap_zones``
    must emit both a gap_zone_top and a gap_zone_bottom level.

    Note: In a gap-up scenario the gap_zone_bottom equals the prior day's high
    (PDH).  After clustering these two identical-price levels merge, so the
    final ``analyze`` output retains the stronger of the two.  We therefore
    test ``_find_gap_zones`` directly (pre-clustering) to confirm both levels
    are emitted, and separately verify that at least one gap-zone type survives
    clustering in the full ``analyze`` output.
    """
    closes = [100.0 + i * 0.1 for i in range(60)]
    df = make_intraday(closes)

    # make_daily: high = close * 1.01, low = close * 0.99, open = close * 0.998
    # Prior day high ≈ 90 * 1.01 = 90.9
    # Today's open  ≈ 150 * 0.998 = 149.7  → well above 90.9 → gap up
    df_daily = make_daily([90.0, 150.0])

    # ── Pre-clustering: both gap-zone levels must be emitted ──
    raw_gap = engine._find_gap_zones(df_daily, asset)
    raw_types = {lvl.level_type for lvl in raw_gap}
    assert "gap_zone_top" in raw_types, f"_find_gap_zones missing 'gap_zone_top': {raw_gap}"
    assert "gap_zone_bottom" in raw_types, f"_find_gap_zones missing 'gap_zone_bottom': {raw_gap}"

    # ── Post-clustering: at least one gap-zone type must survive ──
    result = engine.analyze(df, asset, df_daily=df_daily)
    types = {lvl.level_type for lvl in result}
    assert types & {"gap_zone_top", "gap_zone_bottom"}, (
        f"No gap-zone level survived clustering in {types}"
    )


# ── Test: crypto asset class passes through ──────────────────────────────────

def test_crypto_asset_class(engine: KeyLevelsEngine) -> None:
    """Engine must work correctly with AssetClass.CRYPTO."""
    asset = AssetClass.CRYPTO
    closes = [30000.0 + i * 10.0 for i in range(60)]
    df = make_intraday(closes)

    result = engine.analyze(df, asset)

    assert isinstance(result, list)
    for lvl in result:
        assert lvl.asset_class == AssetClass.CRYPTO


# ── Test: vwap value is in plausible range ───────────────────────────────────

def test_vwap_value_within_price_range(
    engine: KeyLevelsEngine, asset: AssetClass
) -> None:
    """VWAP must fall within the min/max close price range of the DataFrame."""
    closes = [100.0 + i * 0.1 for i in range(60)]
    df = make_intraday(closes)

    result = engine.analyze(df, asset)

    vwap_levels = [lvl for lvl in result if lvl.level_type == "vwap"]
    assert len(vwap_levels) == 1

    min_price = df["low"].min()
    max_price = df["high"].max()
    vwap_price = vwap_levels[0].price
    assert min_price <= vwap_price <= max_price, (
        f"VWAP {vwap_price} outside [{min_price}, {max_price}]"
    )


# ── Test: strength is bounded [0, 1] for all levels ─────────────────────────

def test_strength_bounds(engine: KeyLevelsEngine, asset: AssetClass) -> None:
    """All returned KeyLevel.strength values must be in [0.0, 1.0]."""
    closes = [100.0 + i * 0.05 for i in range(60)]
    df = make_intraday(closes)
    df_daily = make_daily([80.0, 120.0])

    result = engine.analyze(df, asset, df_daily=df_daily)

    for lvl in result:
        assert 0.0 <= lvl.strength <= 1.0, (
            f"Strength {lvl.strength} out of bounds for {lvl.level_type}"
        )
