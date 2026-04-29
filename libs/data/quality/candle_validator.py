"""
CandleValidator — validates a sequence of Candle domain objects.

Operates on typed Candle instances (not DataFrames).
Complements DataQualityValidator (which validates raw OHLCV DataFrames).

Severity rules:
  CRITICAL  → should_block_signal=True, is_valid=False
  WARNING   → should_block_signal=False, is_valid=True
  CLEAN     → no issues found

Critical checks (any one present → CRITICAL):
  - empty sequence
  - negative or zero prices
  - OHLC integrity (high < low, high < open/close, low > open/close)
  - duplicate timestamps
  - future timestamps
  - timezone mismatch (naive timestamps mixed with aware, or non-UTC)
  - all candles have zero volume

Warning checks (issues flagged but signal allowed):
  - stale last candle (age > stale_multiplier × timeframe)
  - individual zero-volume bars (< majority threshold)
  - outlier price jumps (close-to-close > outlier_pct)
  - gaps in sequence larger than expected timeframe
  - stock-specific intra-session gaps
  - crypto continuity issues (excessive gaps for 24/7 market)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

from libs.core.logging.logger import get_logger
from libs.core.models.domain import (
    AssetClass,
    Candle,
    CandleValidationResult,
)

log = get_logger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14_400,
    "1d": 86_400,
    "1w": 604_800,
}

# Outlier: close-to-close jump beyond this pct triggers a warning
_OUTLIER_PCT_STOCK = 0.10    # 10 %
_OUTLIER_PCT_CRYPTO = 0.20   # 20 % — crypto is legitimately volatile

# Stale: last bar older than timeframe × this multiplier
_STALE_MULTIPLIER = 3

# Gap: consecutive interval > timeframe × this multiplier triggers a warning
_GAP_MULTIPLIER_CRYPTO = 3   # 3× expected interval
_GAP_MULTIPLIER_STOCK = 2    # stock intra-session gap warning at 2×

# Future: allow tiny clock skew between providers
_MAX_FUTURE_TOLERANCE_SEC = 10

# Zero-volume majority threshold (above = critical, below = warning)
_ZERO_VOL_CRITICAL_RATIO = 0.50


def _tf_seconds(candles: Sequence[Candle], explicit: int | None) -> int | None:
    """Resolve timeframe in seconds: explicit arg beats candle metadata."""
    if explicit is not None:
        return explicit
    if candles:
        return _TIMEFRAME_SECONDS.get(candles[0].timeframe.value)
    return None


class CandleValidator:
    """
    Validates a sequence of Candle domain objects.

    Usage::

        validator = CandleValidator()
        result = validator.validate(candles, asset_class=AssetClass.CRYPTO)
        if result.should_block_signal:
            return SignalOutput(..., action=SignalAction.NO_TRADE)
    """

    def validate(
        self,
        candles: Sequence[Candle],
        asset_class: AssetClass,
        timeframe_seconds: int | None = None,
        is_live: bool = False,
        strict: bool = False,
    ) -> CandleValidationResult:
        """
        Run all validation checks against *candles*.

        Args:
            candles:           Ordered sequence of Candle objects (oldest first).
            asset_class:       STOCK or CRYPTO — governs gap and outlier rules.
            timeframe_seconds: Expected bar interval.  Inferred from candle
                               metadata when omitted.
            is_live:           Enable staleness check (live pipeline only).
            strict:            Promote warnings to CRITICAL (conservative mode).

        Returns:
            CandleValidationResult with severity, errors, warnings,
            is_valid, and should_block_signal.
        """
        errors: list[str] = []
        warnings: list[str] = []
        n = len(candles)

        # ── 1. Empty sequence ──────────────────────────────────────────────
        if n == 0:
            errors.append("No candles provided — cannot validate empty sequence")
            return CandleValidationResult.from_issues(errors, warnings, 0, strict)

        tf_sec = _tf_seconds(candles, timeframe_seconds)

        # ── 2. Timezone check ──────────────────────────────────────────────
        # All timestamps must be timezone-aware UTC.  Naive timestamps cannot
        # be safely compared against now(UTC) or each other.
        tz_errors = self._check_timezones(candles)
        errors.extend(tz_errors)

        # ── 3. Future timestamps ───────────────────────────────────────────
        future_errors = self._check_future_timestamps(candles)
        errors.extend(future_errors)

        # ── 4. Duplicate timestamps ────────────────────────────────────────
        dup_errors = self._check_duplicates(candles)
        errors.extend(dup_errors)

        # ── 5. Negative / zero prices ──────────────────────────────────────
        price_errors = self._check_prices(candles)
        errors.extend(price_errors)

        # ── 6. OHLC integrity ──────────────────────────────────────────────
        ohlc_errors = self._check_ohlc(candles)
        errors.extend(ohlc_errors)

        # ── 7. Volume ──────────────────────────────────────────────────────
        vol_errors, vol_warnings = self._check_volume(candles)
        errors.extend(vol_errors)
        warnings.extend(vol_warnings)

        # ── 8. Staleness (live mode only) ──────────────────────────────────
        if is_live and tf_sec:
            stale_warnings = self._check_staleness(candles, tf_sec)
            warnings.extend(stale_warnings)

        # ── 9. Outlier candles ─────────────────────────────────────────────
        outlier_warnings = self._check_outliers(candles, asset_class)
        warnings.extend(outlier_warnings)

        # ── 10. Sequence gaps ─────────────────────────────────────────────
        if tf_sec and n > 1:
            gap_warnings = self._check_gaps(candles, asset_class, tf_sec)
            warnings.extend(gap_warnings)

        result = CandleValidationResult.from_issues(errors, warnings, n, strict)

        if result.should_block_signal:
            log.warning(
                "candle_validation_blocked",
                asset_class=asset_class.value,
                candles_checked=n,
                errors=errors,
                warnings=warnings,
            )
        elif result.warnings:
            log.debug(
                "candle_validation_warnings",
                asset_class=asset_class.value,
                candles_checked=n,
                warnings=warnings,
            )

        return result

    # ── Private check methods ──────────────────────────────────────────────────

    def _check_timezones(self, candles: Sequence[Candle]) -> list[str]:
        """All timestamps must be UTC-aware.  Mixed naive/aware is CRITICAL."""
        naive_indices: list[int] = []
        non_utc_indices: list[int] = []

        for i, c in enumerate(candles):
            ts = c.timestamp
            if ts.tzinfo is None:
                naive_indices.append(i)
            elif ts.utcoffset() and ts.utcoffset().total_seconds() != 0:
                non_utc_indices.append(i)

        errs: list[str] = []
        if naive_indices:
            errs.append(
                f"Naive (timezone-unaware) timestamps at {len(naive_indices)} candle(s): "
                f"indices {naive_indices[:5]}{'…' if len(naive_indices) > 5 else ''}"
            )
        if non_utc_indices:
            errs.append(
                f"Non-UTC timestamps at {len(non_utc_indices)} candle(s): "
                f"indices {non_utc_indices[:5]}{'…' if len(non_utc_indices) > 5 else ''}"
            )
        return errs

    def _check_future_timestamps(self, candles: Sequence[Candle]) -> list[str]:
        """Timestamps beyond current wall-clock + tolerance are CRITICAL."""
        now = datetime.now(timezone.utc)
        future_limit = now + timedelta(seconds=_MAX_FUTURE_TOLERANCE_SEC)
        future: list[str] = []

        for i, c in enumerate(candles):
            ts = c.timestamp
            if ts.tzinfo is None:
                continue  # already flagged in timezone check
            if ts > future_limit:
                delta = (ts - now).total_seconds()
                future.append(
                    f"Candle[{i}] timestamp {ts.isoformat()} is "
                    f"{delta:.0f}s in the future"
                )

        if future:
            return [f"Future timestamps detected: {'; '.join(future[:3])}"
                    + (f" (and {len(future)-3} more)" if len(future) > 3 else "")]
        return []

    def _check_duplicates(self, candles: Sequence[Candle]) -> list[str]:
        """Duplicate timestamps produce ambiguous data — CRITICAL."""
        seen: set[datetime] = set()
        dupes: list[str] = []

        for i, c in enumerate(candles):
            if c.timestamp in seen:
                dupes.append(c.timestamp.isoformat())
            seen.add(c.timestamp)

        if dupes:
            unique_dupes = list(dict.fromkeys(dupes))
            return [
                f"Duplicate timestamps ({len(dupes)} occurrence(s)): "
                + ", ".join(unique_dupes[:3])
                + (f" (and {len(unique_dupes)-3} more)" if len(unique_dupes) > 3 else "")
            ]
        return []

    def _check_prices(self, candles: Sequence[Candle]) -> list[str]:
        """Negative or zero prices are impossible in real markets — CRITICAL."""
        fields = ("open", "high", "low", "close")
        bad: dict[str, int] = {f: 0 for f in fields}

        for c in candles:
            for field in fields:
                val: float = getattr(c, field)
                if val <= 0:
                    bad[field] += 1

        errs: list[str] = []
        for field, count in bad.items():
            if count > 0:
                errs.append(f"Non-positive {field}: {count} candle(s)")
        return errs

    def _check_ohlc(self, candles: Sequence[Candle]) -> list[str]:
        """
        OHLC integrity:
          high >= max(open, close)
          low  <= min(open, close)
          high >= low
        """
        high_lt_open: list[int] = []
        high_lt_close: list[int] = []
        low_gt_open: list[int] = []
        low_gt_close: list[int] = []
        high_lt_low: list[int] = []

        for i, c in enumerate(candles):
            if c.high < c.low:
                high_lt_low.append(i)
            if c.high < c.open:
                high_lt_open.append(i)
            if c.high < c.close:
                high_lt_close.append(i)
            if c.low > c.open:
                low_gt_open.append(i)
            if c.low > c.close:
                low_gt_close.append(i)

        errs: list[str] = []
        if high_lt_low:
            errs.append(f"high < low at {len(high_lt_low)} candle(s): indices {high_lt_low[:5]}")
        if high_lt_open:
            errs.append(f"high < open at {len(high_lt_open)} candle(s): indices {high_lt_open[:5]}")
        if high_lt_close:
            errs.append(f"high < close at {len(high_lt_close)} candle(s): indices {high_lt_close[:5]}")
        if low_gt_open:
            errs.append(f"low > open at {len(low_gt_open)} candle(s): indices {low_gt_open[:5]}")
        if low_gt_close:
            errs.append(f"low > close at {len(low_gt_close)} candle(s): indices {low_gt_close[:5]}")
        return errs

    def _check_volume(
        self, candles: Sequence[Candle]
    ) -> tuple[list[str], list[str]]:
        """
        Zero-volume rules:
          - All candles zero-volume → CRITICAL (exchange halt / bad feed)
          - Majority (>50%) zero-volume → CRITICAL
          - Some zero-volume bars → WARNING
        """
        n = len(candles)
        zero_count = sum(1 for c in candles if c.volume == 0)

        if zero_count == 0:
            return [], []

        ratio = zero_count / n
        if ratio >= _ZERO_VOL_CRITICAL_RATIO:
            return (
                [f"Zero-volume bars: {zero_count}/{n} ({ratio:.0%}) — "
                 "exchange halt or feed failure suspected"],
                [],
            )

        return (
            [],
            [f"Zero-volume bars: {zero_count}/{n} ({ratio:.0%})"],
        )

    def _check_staleness(
        self, candles: Sequence[Candle], tf_sec: int
    ) -> list[str]:
        """Last candle is stale if age > stale_multiplier × timeframe."""
        last = candles[-1].timestamp
        if last.tzinfo is None:
            return []  # skip: timezone error already flagged
        age = (datetime.now(timezone.utc) - last).total_seconds()
        stale_limit = tf_sec * _STALE_MULTIPLIER
        if age > stale_limit:
            return [
                f"Stale data: last candle is {age:.0f}s old "
                f"(limit {stale_limit:.0f}s = {_STALE_MULTIPLIER}× {tf_sec}s timeframe)"
            ]
        return []

    def _check_outliers(
        self, candles: Sequence[Candle], asset_class: AssetClass
    ) -> list[str]:
        """Close-to-close jumps beyond threshold flag outlier / bad-tick candles."""
        if len(candles) < 2:
            return []

        limit = (
            _OUTLIER_PCT_STOCK if asset_class == AssetClass.STOCK
            else _OUTLIER_PCT_CRYPTO
        )
        outlier_indices: list[int] = []

        for i in range(1, len(candles)):
            prev_close = candles[i - 1].close
            curr_close = candles[i].close
            if prev_close > 0:
                pct_change = abs(curr_close - prev_close) / prev_close
                if pct_change > limit:
                    outlier_indices.append(i)

        if outlier_indices:
            return [
                f"Outlier candle(s) — close-to-close jump >{limit:.0%}: "
                f"{len(outlier_indices)} bar(s) at indices {outlier_indices[:5]}"
                + (f" (and {len(outlier_indices)-5} more)" if len(outlier_indices) > 5 else "")
            ]
        return []

    def _check_gaps(
        self,
        candles: Sequence[Candle],
        asset_class: AssetClass,
        tf_sec: int,
    ) -> list[str]:
        """
        Gap detection between consecutive candles.

        Crypto (24/7): any gap > GAP_MULTIPLIER_CRYPTO × timeframe is unexpected.
        Stocks: overnight / weekend gaps are normal.  Flag intraday gaps
                > GAP_MULTIPLIER_STOCK × timeframe that occur on the same calendar day.
        """
        if asset_class == AssetClass.CRYPTO:
            return self._check_crypto_gaps(candles, tf_sec)
        return self._check_stock_gaps(candles, tf_sec)

    def _check_crypto_gaps(
        self, candles: Sequence[Candle], tf_sec: int
    ) -> list[str]:
        """Crypto markets run 24/7 — gaps indicate downtime or feed issues."""
        threshold = tf_sec * _GAP_MULTIPLIER_CRYPTO
        large_gaps: list[tuple[int, float]] = []

        for i in range(1, len(candles)):
            prev_ts = candles[i - 1].timestamp
            curr_ts = candles[i].timestamp
            if prev_ts.tzinfo is None or curr_ts.tzinfo is None:
                continue  # skip: tz error already flagged
            gap_sec = (curr_ts - prev_ts).total_seconds()
            if gap_sec > threshold:
                large_gaps.append((i, gap_sec))

        if large_gaps:
            max_gap = max(g for _, g in large_gaps)
            return [
                f"Crypto continuity: {len(large_gaps)} gap(s) > {threshold:.0f}s "
                f"(max {max_gap:.0f}s, expected {tf_sec}s); "
                "possible exchange downtime or missing candles"
            ]
        return []

    def _check_stock_gaps(
        self, candles: Sequence[Candle], tf_sec: int
    ) -> list[str]:
        """
        Stock session gaps: flag same-day intraday gaps only.

        Overnight and weekend gaps are intentional and must not fire.
        """
        threshold = tf_sec * _GAP_MULTIPLIER_STOCK
        intraday_gaps: list[tuple[int, float]] = []

        for i in range(1, len(candles)):
            prev_ts = candles[i - 1].timestamp
            curr_ts = candles[i].timestamp
            if prev_ts.tzinfo is None or curr_ts.tzinfo is None:
                continue  # skip: tz error already flagged

            # Only flag gaps that occur on the same UTC calendar day —
            # overnight/weekend transitions are expected.
            if prev_ts.date() != curr_ts.date():
                continue

            gap_sec = (curr_ts - prev_ts).total_seconds()
            if gap_sec > threshold:
                intraday_gaps.append((i, gap_sec))

        if intraday_gaps:
            max_gap = max(g for _, g in intraday_gaps)
            return [
                f"Stock session gap: {len(intraday_gaps)} intraday gap(s) > {threshold:.0f}s "
                f"(max {max_gap:.0f}s, expected {tf_sec}s)"
            ]
        return []
