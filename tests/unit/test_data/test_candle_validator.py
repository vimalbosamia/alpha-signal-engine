"""
Tests for CandleValidator.

Coverage:
  - empty sequence → CRITICAL
  - negative prices → CRITICAL
  - OHLC violations (high<low, high<open, high<close, low>open, low>close) → CRITICAL
  - duplicate timestamps → CRITICAL
  - future timestamps → CRITICAL
  - timezone mismatch (naive timestamps) → CRITICAL
  - non-UTC timezone → CRITICAL
  - all-zero volume → CRITICAL
  - majority zero-volume → CRITICAL
  - minority zero-volume → WARNING
  - stale last candle (is_live=True) → WARNING
  - outlier price jump (stock) → WARNING
  - outlier price jump (crypto) → WARNING
  - crypto continuity gap → WARNING
  - stock intraday gap → WARNING
  - stock overnight gap (expected) → CLEAN
  - clean candles → CLEAN
  - strict=True promotes WARNING to CRITICAL
  - is_valid False on CRITICAL, True on WARNING
  - should_block_signal True on CRITICAL, False on WARNING
  - CandleValidationResult.clean() factory
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
import zoneinfo

import pytest

from libs.core.models.domain import (
    AssetClass,
    Candle,
    CandleSeverity,
    CandleValidationResult,
    Timeframe,
)
from libs.data.quality.candle_validator import CandleValidator

# ── Helpers ────────────────────────────────────────────────────────────────────

_UTC = timezone.utc

def _ts(offset_minutes: int = 0) -> datetime:
    """UTC timestamp relative to a fixed past anchor."""
    anchor = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)
    return anchor + timedelta(minutes=offset_minutes)


def _candle(
    *,
    timestamp: datetime | None = None,
    open: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float = 1000.0,
    asset_class: AssetClass = AssetClass.CRYPTO,
    timeframe: Timeframe = Timeframe.ONE_MIN,
    symbol: str = "BTCUSDT",
) -> Candle:
    """Build a valid Candle for tests."""
    ts = timestamp or _ts()
    return Candle(
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        timestamp=ts,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_sequence(
    n: int = 5,
    *,
    asset_class: AssetClass = AssetClass.CRYPTO,
    timeframe: Timeframe = Timeframe.ONE_MIN,
    symbol: str = "BTCUSDT",
) -> list[Candle]:
    """Build an N-candle sequence with evenly spaced 1-minute timestamps."""
    return [
        _candle(
            timestamp=_ts(i),
            asset_class=asset_class,
            timeframe=timeframe,
            symbol=symbol,
        )
        for i in range(n)
    ]


def _validator() -> CandleValidator:
    return CandleValidator()


# ── 1. Empty sequence ──────────────────────────────────────────────────────────

class TestEmptySequence:
    def test_empty_list_is_critical(self):
        result = _validator().validate([], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL

    def test_empty_list_blocks_signal(self):
        result = _validator().validate([], AssetClass.CRYPTO)
        assert result.should_block_signal is True

    def test_empty_list_not_valid(self):
        result = _validator().validate([], AssetClass.CRYPTO)
        assert result.is_valid is False

    def test_empty_list_has_error_message(self):
        result = _validator().validate([], AssetClass.CRYPTO)
        assert result.errors
        assert result.candles_checked == 0


# ── 2. Negative / zero prices ─────────────────────────────────────────────────

class TestNegativePrices:
    def test_negative_close_is_critical(self):
        # Build via model_construct to bypass Candle's own validators
        bad = Candle.model_construct(
            symbol="X",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.ONE_MIN,
            timestamp=_ts(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=-1.0,  # impossible
            volume=100.0,
            source="",
        )
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert result.should_block_signal is True
        assert any("close" in e.lower() for e in result.errors)

    def test_negative_open_is_critical(self):
        bad = Candle.model_construct(
            symbol="X",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.ONE_MIN,
            timestamp=_ts(),
            open=-5.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=100.0,
            source="",
        )
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert any("open" in e.lower() for e in result.errors)

    def test_zero_open_is_critical(self):
        bad = Candle.model_construct(
            symbol="X",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.ONE_MIN,
            timestamp=_ts(),
            open=0.0,
            high=105.0,
            low=0.0,
            close=102.0,
            volume=100.0,
            source="",
        )
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL

    def test_mixed_valid_and_negative_is_critical(self):
        good = _candle(timestamp=_ts(0))
        bad = Candle.model_construct(
            symbol="X",
            asset_class=AssetClass.CRYPTO,
            timeframe=Timeframe.ONE_MIN,
            timestamp=_ts(1),
            open=100.0, high=105.0, low=95.0, close=-1.0, volume=100.0, source="",
        )
        result = _validator().validate([good, bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL


# ── 3. OHLC integrity ─────────────────────────────────────────────────────────

class TestOHLCIntegrity:
    def _bad(self, **kwargs) -> Candle:
        defaults = dict(
            symbol="X", asset_class=AssetClass.CRYPTO, timeframe=Timeframe.ONE_MIN,
            timestamp=_ts(), open=100.0, high=105.0, low=95.0, close=102.0,
            volume=1000.0, source="",
        )
        defaults.update(kwargs)
        return Candle.model_construct(**defaults)

    def test_high_lt_low_is_critical(self):
        bad = self._bad(high=90.0, low=95.0)
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert any("high < low" in e for e in result.errors)

    def test_high_lt_open_is_critical(self):
        bad = self._bad(open=110.0, high=105.0, low=95.0, close=108.0)
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert any("high < open" in e for e in result.errors)

    def test_high_lt_close_is_critical(self):
        # high=95, close=100, open=90, low=80 — Pydantic allows (high >= open)
        bad = _candle(open=90.0, high=95.0, low=80.0, close=100.0)
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert any("high < close" in e for e in result.errors)

    def test_low_gt_open_is_critical(self):
        bad = self._bad(open=90.0, high=105.0, low=95.0, close=102.0)
        # low(95) > open(90) — bypass Pydantic
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert any("low > open" in e for e in result.errors)

    def test_low_gt_close_is_critical(self):
        # low=105, close=100, open=110, high=115 — Pydantic: low(105) <= open(110) ✓
        bad = _candle(open=110.0, high=115.0, low=105.0, close=100.0)
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert any("low > close" in e for e in result.errors)

    def test_valid_ohlc_no_errors(self):
        good = _candle(open=100.0, high=110.0, low=90.0, close=105.0)
        result = _validator().validate([good], AssetClass.CRYPTO)
        assert not any("high" in e or "low" in e for e in result.errors)


# ── 4. Duplicate timestamps ───────────────────────────────────────────────────

class TestDuplicateTimestamps:
    def test_duplicate_is_critical(self):
        ts = _ts(0)
        c1 = _candle(timestamp=ts)
        c2 = _candle(timestamp=ts)
        result = _validator().validate([c1, c2], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert result.should_block_signal is True

    def test_duplicate_error_message_contains_timestamp(self):
        ts = _ts(0)
        result = _validator().validate([_candle(timestamp=ts), _candle(timestamp=ts)], AssetClass.CRYPTO)
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_unique_timestamps_no_dup_error(self):
        candles = _make_sequence(3)
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert not any("duplicate" in e.lower() for e in result.errors)


# ── 5. Future timestamps ──────────────────────────────────────────────────────

class TestFutureTimestamps:
    def test_far_future_is_critical(self):
        future_ts = datetime.now(_UTC) + timedelta(hours=1)
        c = _candle(timestamp=future_ts)
        result = _validator().validate([c], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert result.should_block_signal is True

    def test_far_future_error_message(self):
        future_ts = datetime.now(_UTC) + timedelta(hours=1)
        result = _validator().validate([_candle(timestamp=future_ts)], AssetClass.CRYPTO)
        assert any("future" in e.lower() for e in result.errors)

    def test_slight_future_within_tolerance_is_clean(self):
        # 3 seconds in the future — within _MAX_FUTURE_TOLERANCE_SEC=10
        near_future = datetime.now(_UTC) + timedelta(seconds=3)
        c = _candle(timestamp=near_future)
        result = _validator().validate([c], AssetClass.CRYPTO)
        assert not any("future" in e.lower() for e in result.errors)

    def test_past_timestamp_allowed(self):
        past = _ts(0)  # fixed past anchor
        c = _candle(timestamp=past)
        result = _validator().validate([c], AssetClass.CRYPTO)
        assert not any("future" in e.lower() for e in result.errors)


# ── 6. Timezone mismatch ──────────────────────────────────────────────────────

class TestTimezoneMismatch:
    def test_naive_timestamp_is_critical(self):
        naive_ts = datetime(2024, 6, 15, 10, 0, 0)  # no tzinfo
        bad = Candle.model_construct(
            symbol="X", asset_class=AssetClass.CRYPTO, timeframe=Timeframe.ONE_MIN,
            timestamp=naive_ts, open=100.0, high=105.0, low=95.0, close=102.0,
            volume=1000.0, source="",
        )
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert result.should_block_signal is True

    def test_naive_timestamp_error_message(self):
        naive_ts = datetime(2024, 6, 15, 10, 0, 0)
        bad = Candle.model_construct(
            symbol="X", asset_class=AssetClass.CRYPTO, timeframe=Timeframe.ONE_MIN,
            timestamp=naive_ts, open=100.0, high=105.0, low=95.0, close=102.0,
            volume=1000.0, source="",
        )
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert any("naive" in e.lower() or "timezone" in e.lower() for e in result.errors)

    def test_non_utc_timezone_is_critical(self):
        ny_tz = zoneinfo.ZoneInfo("America/New_York")
        non_utc_ts = datetime(2024, 6, 15, 10, 0, 0, tzinfo=ny_tz)
        bad = Candle.model_construct(
            symbol="X", asset_class=AssetClass.CRYPTO, timeframe=Timeframe.ONE_MIN,
            timestamp=non_utc_ts, open=100.0, high=105.0, low=95.0, close=102.0,
            volume=1000.0, source="",
        )
        result = _validator().validate([bad], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL

    def test_utc_timestamp_is_fine(self):
        utc_ts = _ts(0)
        c = _candle(timestamp=utc_ts)
        result = _validator().validate([c], AssetClass.CRYPTO)
        assert not any("naive" in e.lower() or "timezone" in e.lower() for e in result.errors)


# ── 7. Volume checks ─────────────────────────────────────────────────────────

class TestVolume:
    def test_all_zero_volume_is_critical(self):
        candles = [_candle(timestamp=_ts(i), volume=0.0) for i in range(5)]
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        assert result.should_block_signal is True
        assert any("zero" in e.lower() for e in result.errors)

    def test_majority_zero_volume_is_critical(self):
        # 4 zero-volume, 1 normal → 80% zero → CRITICAL
        candles = [_candle(timestamp=_ts(i), volume=0.0) for i in range(4)]
        candles.append(_candle(timestamp=_ts(4), volume=1000.0))
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL

    def test_minority_zero_volume_is_warning(self):
        # 1 zero-volume out of 10 → 10% → WARNING
        candles = [_candle(timestamp=_ts(i)) for i in range(9)]
        candles.append(_candle(timestamp=_ts(9), volume=0.0))
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.WARNING
        assert result.should_block_signal is False
        assert any("zero" in w.lower() for w in result.warnings)

    def test_positive_volume_no_volume_issue(self):
        candles = _make_sequence(5)
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert not any("volume" in e.lower() for e in result.errors)
        assert not any("volume" in w.lower() for w in result.warnings)


# ── 8. Stale candle ───────────────────────────────────────────────────────────

class TestStaleness:
    def test_stale_last_candle_is_warning(self):
        # Last candle is 30 minutes old; 1m timeframe → stale limit = 3 min
        old_ts = datetime.now(_UTC) - timedelta(minutes=30)
        c = _candle(timestamp=old_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=True)
        assert result.severity == CandleSeverity.WARNING
        assert any("stale" in w.lower() for w in result.warnings)

    def test_stale_does_not_block(self):
        old_ts = datetime.now(_UTC) - timedelta(minutes=30)
        c = _candle(timestamp=old_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=True)
        assert result.should_block_signal is False

    def test_fresh_candle_no_staleness_warning(self):
        fresh_ts = datetime.now(_UTC) - timedelta(seconds=30)
        c = _candle(timestamp=fresh_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=True)
        assert not any("stale" in w.lower() for w in result.warnings)

    def test_staleness_not_checked_when_not_live(self):
        old_ts = datetime.now(_UTC) - timedelta(hours=10)
        c = _candle(timestamp=old_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=False)
        assert not any("stale" in w.lower() for w in result.warnings)


# ── 9. Outlier candles ────────────────────────────────────────────────────────

class TestOutlierCandles:
    def test_large_price_jump_crypto_is_warning(self):
        # 25% jump → above 20% crypto threshold
        c1 = _candle(timestamp=_ts(0), close=100.0, open=98.0, high=102.0, low=96.0)
        c2 = _candle(timestamp=_ts(1), close=126.0, open=125.0, high=130.0, low=120.0)
        result = _validator().validate([c1, c2], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.WARNING
        assert any("outlier" in w.lower() for w in result.warnings)

    def test_large_price_jump_stock_is_warning(self):
        # 15% jump → above 10% stock threshold
        c1 = _candle(timestamp=_ts(0), close=100.0, open=98.0, high=102.0, low=96.0,
                     asset_class=AssetClass.STOCK)
        c2 = _candle(timestamp=_ts(1), close=116.0, open=115.0, high=120.0, low=110.0,
                     asset_class=AssetClass.STOCK)
        result = _validator().validate([c1, c2], AssetClass.STOCK)
        assert result.severity == CandleSeverity.WARNING
        assert any("outlier" in w.lower() for w in result.warnings)

    def test_small_price_jump_no_outlier_warning(self):
        # 5% jump — below both thresholds
        c1 = _candle(timestamp=_ts(0), close=100.0, open=98.0, high=102.0, low=96.0)
        c2 = _candle(timestamp=_ts(1), close=105.0, open=103.0, high=107.0, low=101.0)
        result = _validator().validate([c1, c2], AssetClass.CRYPTO)
        assert not any("outlier" in w.lower() for w in result.warnings)

    def test_single_candle_no_outlier_check(self):
        c = _candle()
        result = _validator().validate([c], AssetClass.CRYPTO)
        assert not any("outlier" in w.lower() for w in result.warnings)


# ── 10. Gaps in sequence ──────────────────────────────────────────────────────

class TestSequenceGaps:
    def test_crypto_gap_exceeds_threshold_is_warning(self):
        # 1m timeframe → threshold = 3 × 60 = 180s
        # Gap of 10 minutes = 600s → warning
        c1 = _candle(timestamp=_ts(0))
        c2 = _candle(timestamp=_ts(10))  # 10-minute gap
        result = _validator().validate([c1, c2], AssetClass.CRYPTO, timeframe_seconds=60)
        assert result.severity == CandleSeverity.WARNING
        assert any("gap" in w.lower() or "continuity" in w.lower() for w in result.warnings)

    def test_crypto_small_gap_no_warning(self):
        # Gap of 1 minute = 60s (exactly the timeframe) → no warning
        c1 = _candle(timestamp=_ts(0))
        c2 = _candle(timestamp=_ts(1))
        result = _validator().validate([c1, c2], AssetClass.CRYPTO, timeframe_seconds=60)
        assert not any("gap" in w.lower() or "continuity" in w.lower() for w in result.warnings)

    def test_stock_intraday_gap_is_warning(self):
        # Same day, 10-minute gap when expecting 1-minute bars → warning
        base = datetime(2024, 6, 15, 10, 0, 0, tzinfo=_UTC)
        c1 = _candle(timestamp=base, asset_class=AssetClass.STOCK)
        c2 = _candle(timestamp=base + timedelta(minutes=10), asset_class=AssetClass.STOCK)
        result = _validator().validate([c1, c2], AssetClass.STOCK, timeframe_seconds=60)
        assert result.severity == CandleSeverity.WARNING
        assert any("gap" in w.lower() for w in result.warnings)

    def test_stock_overnight_gap_not_flagged(self):
        # Gap spans midnight → different calendar dates → NOT flagged
        day1 = datetime(2024, 6, 14, 15, 59, 0, tzinfo=_UTC)
        day2 = datetime(2024, 6, 15, 13, 30, 0, tzinfo=_UTC)
        c1 = _candle(timestamp=day1, asset_class=AssetClass.STOCK)
        c2 = _candle(timestamp=day2, asset_class=AssetClass.STOCK)
        result = _validator().validate([c1, c2], AssetClass.STOCK, timeframe_seconds=60)
        assert not any("stock session gap" in w.lower() for w in result.warnings)

    def test_no_gap_check_without_timeframe(self):
        # No timeframe_seconds provided and no timeframe inferrable → gaps not checked
        c1 = _candle(timestamp=_ts(0))
        c2 = _candle(timestamp=_ts(60))  # hour gap
        result = _validator().validate([c1, c2], AssetClass.CRYPTO, timeframe_seconds=None)
        # Without a timeframe reference we can't determine what's a gap — no gap warnings
        # (The validator should handle None tf_sec gracefully)
        assert result.candles_checked == 2

    def test_single_candle_no_gap_check(self):
        result = _validator().validate([_candle()], AssetClass.CRYPTO, timeframe_seconds=60)
        assert not any("gap" in w.lower() for w in result.warnings)


# ── 11. Clean data ────────────────────────────────────────────────────────────

class TestCleanData:
    def test_clean_sequence_returns_clean_severity(self):
        candles = _make_sequence(10)
        result = _validator().validate(candles, AssetClass.CRYPTO, timeframe_seconds=60)
        assert result.severity == CandleSeverity.CLEAN

    def test_clean_sequence_is_valid(self):
        candles = _make_sequence(10)
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.is_valid is True

    def test_clean_sequence_does_not_block(self):
        candles = _make_sequence(10)
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.should_block_signal is False

    def test_clean_sequence_no_errors_or_warnings(self):
        candles = _make_sequence(10)
        result = _validator().validate(candles, AssetClass.CRYPTO, timeframe_seconds=60)
        assert result.errors == []
        assert result.warnings == []

    def test_candles_checked_count(self):
        candles = _make_sequence(7)
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.candles_checked == 7


# ── 12. Strict mode ───────────────────────────────────────────────────────────

class TestStrictMode:
    def test_strict_promotes_warning_to_critical(self):
        # Stale candle → WARNING normally → CRITICAL in strict mode
        old_ts = datetime.now(_UTC) - timedelta(minutes=30)
        c = _candle(timestamp=old_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=True, strict=True)
        assert result.severity == CandleSeverity.CRITICAL
        assert result.should_block_signal is True

    def test_strict_promotes_outlier_to_critical(self):
        c1 = _candle(timestamp=_ts(0), close=100.0, open=98.0, high=102.0, low=96.0)
        c2 = _candle(timestamp=_ts(1), close=130.0, open=128.0, high=135.0, low=125.0)
        result = _validator().validate([c1, c2], AssetClass.CRYPTO, strict=True)
        assert result.severity == CandleSeverity.CRITICAL

    def test_strict_clean_data_still_clean(self):
        candles = _make_sequence(5)
        result = _validator().validate(candles, AssetClass.CRYPTO, timeframe_seconds=60, strict=True)
        assert result.severity == CandleSeverity.CLEAN


# ── 13. is_valid and should_block_signal semantics ────────────────────────────

class TestResultSemantics:
    def test_critical_not_valid(self):
        result = _validator().validate([], AssetClass.CRYPTO)
        assert result.is_valid is False

    def test_warning_is_valid(self):
        old_ts = datetime.now(_UTC) - timedelta(minutes=30)
        c = _candle(timestamp=old_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=True)
        assert result.is_valid is True

    def test_clean_is_valid(self):
        result = _validator().validate(_make_sequence(3), AssetClass.CRYPTO)
        assert result.is_valid is True

    def test_critical_blocks_signal(self):
        result = _validator().validate([], AssetClass.CRYPTO)
        assert result.should_block_signal is True

    def test_warning_does_not_block_signal(self):
        old_ts = datetime.now(_UTC) - timedelta(minutes=30)
        c = _candle(timestamp=old_ts)
        result = _validator().validate([c], AssetClass.CRYPTO, is_live=True)
        assert result.should_block_signal is False

    def test_clean_does_not_block_signal(self):
        result = _validator().validate(_make_sequence(3), AssetClass.CRYPTO)
        assert result.should_block_signal is False


# ── 14. CandleValidationResult factory methods ────────────────────────────────

class TestCandleValidationResultFactories:
    def test_clean_factory(self):
        r = CandleValidationResult.clean(candles_checked=5)
        assert r.severity == CandleSeverity.CLEAN
        assert r.is_valid is True
        assert r.should_block_signal is False
        assert r.errors == []
        assert r.warnings == []
        assert r.candles_checked == 5

    def test_from_issues_no_issues_is_clean(self):
        r = CandleValidationResult.from_issues([], [], 3)
        assert r.severity == CandleSeverity.CLEAN

    def test_from_issues_only_warnings_is_warning(self):
        r = CandleValidationResult.from_issues([], ["something off"], 3)
        assert r.severity == CandleSeverity.WARNING
        assert r.is_valid is True
        assert r.should_block_signal is False

    def test_from_issues_with_errors_is_critical(self):
        r = CandleValidationResult.from_issues(["bad data"], [], 3)
        assert r.severity == CandleSeverity.CRITICAL
        assert r.is_valid is False
        assert r.should_block_signal is True

    def test_from_issues_strict_promotes_warnings(self):
        r = CandleValidationResult.from_issues([], ["minor issue"], 3, strict=True)
        assert r.severity == CandleSeverity.CRITICAL
        assert r.should_block_signal is True

    def test_from_issues_is_frozen(self):
        r = CandleValidationResult.from_issues([], [], 3)
        with pytest.raises(Exception):
            r.is_valid = False  # type: ignore[misc]


# ── 15. Multiple issues accumulate correctly ──────────────────────────────────

class TestMultipleIssues:
    def test_multiple_errors_all_reported(self):
        # Negative price + duplicate timestamp
        ts = _ts(0)
        bad1 = Candle.model_construct(
            symbol="X", asset_class=AssetClass.CRYPTO, timeframe=Timeframe.ONE_MIN,
            timestamp=ts, open=-1.0, high=105.0, low=95.0, close=102.0, volume=100.0, source="",
        )
        bad2 = Candle.model_construct(
            symbol="X", asset_class=AssetClass.CRYPTO, timeframe=Timeframe.ONE_MIN,
            timestamp=ts, open=100.0, high=105.0, low=95.0, close=102.0, volume=100.0, source="",
        )
        result = _validator().validate([bad1, bad2], AssetClass.CRYPTO)
        assert result.severity == CandleSeverity.CRITICAL
        # Should have both duplicate AND negative price errors
        assert len(result.errors) >= 2

    def test_candles_checked_correct_with_issues(self):
        candles = _make_sequence(4)
        result = _validator().validate(candles, AssetClass.CRYPTO)
        assert result.candles_checked == 4
