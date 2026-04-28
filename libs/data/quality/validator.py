"""
Data quality validation layer.

Applies to both stock and crypto data. All incoming bars must
pass through the validator before any analysis runs.

Stock-specific checks:
  - Corporate action adjustment consistency
  - Suspicious gap patterns around known adjustment dates

Crypto-specific checks:
  - Zero or near-zero volume (exchange downtime indicator)
  - Extreme price jumps (flash crashes, bad ticks)

Validation results are stored in DataQualityReport and determine
whether the signal pipeline is allowed to run:
  - CLEAN: all good, proceed
  - WARNING: proceed but attach flags
  - BLOCKED: do not generate signals
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from libs.core.models.domain import AssetClass, DataQualityReport, DataQualityStatus
from libs.core.logging.logger import get_logger

log = get_logger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}
_MAX_PRICE_JUMP_STOCK_PCT = 10.0
_MAX_PRICE_JUMP_CRYPTO_PCT = 20.0   # crypto is more volatile
_MIN_VOLUME_STOCK = 1_000
_MIN_VOLUME_CRYPTO = 0.001          # BTC volume in BTC — tiny lots are fine
_MAX_GAP_STOCK_SECONDS = 120
_MAX_GAP_CRYPTO_SECONDS = 300       # crypto gaps are rarer but longer
_STALE_LIVE_SECONDS = 120
_STALE_BUFFER_MULTIPLIER = 2   # stale if age > timeframe * this multiplier


class DataQualityValidator:
    """
    Validates a DataFrame of OHLCV bars for a given symbol and asset class.

    Returned DataQualityReport drives whether signals are generated.
    """

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_class: AssetClass,
        timeframe_seconds: int | None = None,
        is_live: bool = False,
        strict: bool = False,
    ) -> DataQualityReport:
        """
        Run all quality checks. Returns DataQualityReport.

        Args:
            df:                OHLCV DataFrame with DatetimeIndex
            symbol:            Ticker
            asset_class:       STOCK or CRYPTO
            timeframe_seconds: Expected bar interval in seconds
            is_live:           True in live mode (enables staleness check)
            strict:            True = warnings also cause BLOCKED status
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── 1. Empty ──────────────────────────────────────────────────────
        if df is None or df.empty:
            return DataQualityReport(
                symbol=symbol,
                asset_class=asset_class,
                status=DataQualityStatus.BLOCKED,
                errors=["DataFrame is empty"],
                rows_checked=0,
            )

        rows = len(df)

        # ── 2. Schema ─────────────────────────────────────────────────────
        missing_cols = _REQUIRED_COLS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # ── 3. Null values ────────────────────────────────────────────────
        present = _REQUIRED_COLS & set(df.columns)
        null_counts = df[list(present)].isnull().sum()
        if null_counts.any():
            null_map = null_counts[null_counts > 0].to_dict()
            errors.append(f"Null values: {null_map}")

        # ── 4. OHLC integrity ─────────────────────────────────────────────
        if present >= {"open", "high", "low", "close"}:
            bad = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            )
            if bad.any():
                errors.append(f"OHLC integrity violations: {bad.sum()} rows")

        # ── 5. Positive prices ────────────────────────────────────────────
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                n_bad = (df[col] <= 0).sum()
                if n_bad > 0:
                    errors.append(f"Non-positive {col}: {n_bad} rows")

        # ── 6. Volume ─────────────────────────────────────────────────────
        if "volume" in df.columns:
            min_vol = _MIN_VOLUME_STOCK if asset_class == AssetClass.STOCK else _MIN_VOLUME_CRYPTO
            low_vol = (df["volume"] < min_vol).sum()
            if low_vol > rows * 0.1:
                warnings.append(
                    f"High proportion of low-volume bars: {low_vol}/{rows}"
                )
            zero_vol = (df["volume"] == 0).sum()
            if zero_vol > 0:
                warnings.append(f"Zero-volume bars: {zero_vol}")

        # ── 7. Timestamp monotonicity ─────────────────────────────────────
        if not df.index.is_monotonic_increasing:
            errors.append("Timestamps not monotonically increasing")

        # ── 8. Duplicate timestamps ───────────────────────────────────────
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            warnings.append(f"Duplicate timestamps: {dupes}")

        # ── 9. Gap detection (crypto only) ───────────────────────────────
        # Stocks have regular overnight/weekend gaps that are not data errors.
        # Gap detection would produce constant false-positive warnings for any
        # equity with intraday bars, so we skip it for stocks.
        if timeframe_seconds and len(df) > 1 and asset_class != AssetClass.STOCK:
            diffs = df.index.to_series().diff().dt.total_seconds().dropna()
            gaps = diffs[diffs > _MAX_GAP_CRYPTO_SECONDS]
            if not gaps.empty:
                warnings.append(
                    f"{len(gaps)} gap(s) > {_MAX_GAP_CRYPTO_SECONDS}s: "
                    f"largest={gaps.max():.0f}s"
                )

        # ── 10. Price jump detection ──────────────────────────────────────
        if "close" in df.columns and len(df) > 1:
            max_jump = (
                _MAX_PRICE_JUMP_STOCK_PCT if asset_class == AssetClass.STOCK
                else _MAX_PRICE_JUMP_CRYPTO_PCT
            ) / 100
            pct_change = df["close"].pct_change().abs()
            jumps = pct_change[pct_change > max_jump]
            if not jumps.empty:
                warnings.append(
                    f"Abnormal price jumps: {len(jumps)} bars > {max_jump:.0%}: "
                    f"max={jumps.max():.2%}"
                )

        # ── 11. Staleness (live mode only) ────────────────────────────────
        if is_live and len(df) > 0:
            latest = df.index[-1]
            if latest.tzinfo is None:
                latest = latest.tz_localize("UTC")
            age = (datetime.now(timezone.utc) - latest).total_seconds()
            # Use 2× the timeframe as the stale threshold so a naturally
            # delayed last-closed bar (up to 1 full interval old) never
            # triggers a false block.  Fall back to _STALE_LIVE_SECONDS
            # when timeframe_seconds is unknown.
            stale_limit = (
                timeframe_seconds * _STALE_BUFFER_MULTIPLIER
                if timeframe_seconds
                else _STALE_LIVE_SECONDS
            )
            if age > stale_limit:
                errors.append(f"Stale data: latest bar is {age:.0f}s old (limit {stale_limit:.0f}s)")

        # ── 12. Stock-specific: suspicious round-number anomalies ─────────
        if asset_class == AssetClass.STOCK and "close" in df.columns:
            # Detect potential bad corporate action: sudden 2x or 0.5x price change
            pct_change = df["close"].pct_change().abs()
            suspicious = pct_change[pct_change > 0.40]
            if not suspicious.empty:
                warnings.append(
                    f"Possible unadjusted corporate action at: "
                    f"{suspicious.index[-1].isoformat()}"
                )

        # ── Determine status ──────────────────────────────────────────────
        rows_with_issues = self._count_bad_rows(df)

        if errors:
            status = DataQualityStatus.BLOCKED
        elif warnings and strict:
            status = DataQualityStatus.BLOCKED
        elif warnings:
            status = DataQualityStatus.WARNING
        else:
            status = DataQualityStatus.CLEAN

        report = DataQualityReport(
            symbol=symbol,
            asset_class=asset_class,
            status=status,
            errors=errors,
            warnings=warnings,
            rows_checked=rows,
            rows_with_issues=rows_with_issues,
        )

        if status == DataQualityStatus.BLOCKED:
            log.warning(
                "data_quality_blocked",
                symbol=symbol,
                errors=errors,
                warnings=warnings,
            )
        elif status == DataQualityStatus.WARNING:
            log.debug(
                "data_quality_warning",
                symbol=symbol,
                warnings=warnings,
            )

        return report

    def _count_bad_rows(self, df: pd.DataFrame) -> int:
        if df.empty or not _REQUIRED_COLS.issubset(df.columns):
            return 0
        bad = df.isnull().any(axis=1)
        if "high" in df.columns and "low" in df.columns:
            bad |= df["high"] < df["low"]
        if "close" in df.columns:
            bad |= df["close"] <= 0
        return int(bad.sum())

    def validate_single_candle(
        self,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        symbol: str,
    ) -> bool:
        """Quick single-bar validation for streaming use."""
        try:
            return (
                h >= max(o, c) and
                l <= min(o, c) and
                h >= l and
                o > 0 and h > 0 and l > 0 and c > 0 and
                v >= 0
            )
        except (TypeError, ValueError):
            log.warning("single_candle_validation_error", symbol=symbol)
            return False
