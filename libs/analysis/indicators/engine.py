"""
Indicators Engine.

Computes a comprehensive set of technical indicators from an OHLCV DataFrame
using the pandas-ta library, returning an immutable IndicatorSnapshot.

Design rules:
  - All results are immutable (frozen dataclass)
  - No magic numbers — all periods and thresholds are named class constants
  - Never raises on an empty / malformed DataFrame; returns safe defaults (all None)
  - Each indicator is computed independently; failure of one does not affect others
  - Uses pandas-ta for all calculations (already installed)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

from libs.core.logging.logger import get_logger

log = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IndicatorSnapshot:
    """Immutable snapshot of all technical indicators for the latest bar."""

    # EMAs
    ema_9: float | None
    ema_20: float | None
    ema_50: float | None
    ema_200: float | None
    # SMAs
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    # RSI
    rsi: float | None           # 0-100
    rsi_prev: float | None      # previous bar RSI
    # MACD
    macd_line: float | None
    macd_signal: float | None
    macd_histogram: float | None
    macd_histogram_prev: float | None
    # Bollinger Bands (20-period, 2 std dev)
    bb_upper: float | None
    bb_middle: float | None
    bb_lower: float | None
    bb_width: float | None      # (upper - lower) / middle
    bb_pct_b: float | None      # (close - lower) / (upper - lower)
    # ATR
    atr: float | None
    # ADX
    adx: float | None           # 0-100, trend strength
    # Stochastic RSI
    stoch_rsi_k: float | None   # 0-100
    stoch_rsi_d: float | None   # 0-100
    # VWAP (session, already in df from builder when available)
    vwap: float | None


# ── All-None sentinel ─────────────────────────────────────────────────────────

_EMPTY_SNAPSHOT = IndicatorSnapshot(
    ema_9=None, ema_20=None, ema_50=None, ema_200=None,
    sma_20=None, sma_50=None, sma_200=None,
    rsi=None, rsi_prev=None,
    macd_line=None, macd_signal=None, macd_histogram=None, macd_histogram_prev=None,
    bb_upper=None, bb_middle=None, bb_lower=None, bb_width=None, bb_pct_b=None,
    atr=None,
    adx=None,
    stoch_rsi_k=None, stoch_rsi_d=None,
    vwap=None,
)


# ── Engine ────────────────────────────────────────────────────────────────────

class IndicatorsEngine:
    """
    Compute all technical indicators from an OHLCV DataFrame.

    Usage:
        engine = IndicatorsEngine()
        snapshot = engine.compute(df)  # never raises
    """

    # ── Period constants ──────────────────────────────────────────────────────
    EMA_FAST: int = 9
    EMA_MED: int = 20
    EMA_SLOW: int = 50
    EMA_TREND: int = 200

    SMA_SHORT: int = 20
    SMA_MED: int = 50
    SMA_LONG: int = 200

    RSI_PERIOD: int = 14

    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    BB_PERIOD: int = 20
    BB_STD: float = 2.0

    ATR_PERIOD: int = 14
    ADX_PERIOD: int = 14

    STOCH_RSI_LENGTH: int = 14
    STOCH_RSI_RSI_LENGTH: int = 14
    STOCH_RSI_K: int = 3
    STOCH_RSI_D: int = 3

    # ── Minimum bar thresholds ─────────────────────────────────────────────────
    _MIN_FOR_BASIC: int = 26    # RSI, MACD, BB, EMA-9/20, SMA-20
    _MIN_FOR_EMA50: int = 50
    _MIN_FOR_SMA50: int = 50
    _MIN_FOR_EMA200: int = 200
    _MIN_FOR_SMA200: int = 200
    _MIN_FOR_ADX: int = 14
    _MIN_FOR_ATR: int = 14

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, df: pd.DataFrame) -> IndicatorSnapshot:
        """
        Compute all indicators from OHLCV DataFrame.

        Returns IndicatorSnapshot with None for any indicator that fails
        or has insufficient data.  Never raises.
        """
        try:
            return self._compute(df)
        except Exception as exc:  # noqa: BLE001
            log.debug("indicators_engine_unexpected_error: %s", exc)
            return _EMPTY_SNAPSHOT

    # ── Private implementation ─────────────────────────────────────────────────

    def _compute(self, df: pd.DataFrame) -> IndicatorSnapshot:
        if df is None or df.empty or len(df) < 2:
            return _EMPTY_SNAPSHOT

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            return _EMPTY_SNAPSHOT

        n = len(df)
        close = df["close"]

        return IndicatorSnapshot(
            ema_9=self._ema(close, self.EMA_FAST) if n >= self.EMA_FAST else None,
            ema_20=self._ema(close, self.EMA_MED) if n >= self.EMA_MED else None,
            ema_50=self._ema(close, self.EMA_SLOW) if n >= self._MIN_FOR_EMA50 else None,
            ema_200=self._ema(close, self.EMA_TREND) if n >= self._MIN_FOR_EMA200 else None,

            sma_20=self._sma(close, self.SMA_SHORT) if n >= self.SMA_SHORT else None,
            sma_50=self._sma(close, self.SMA_MED) if n >= self._MIN_FOR_SMA50 else None,
            sma_200=self._sma(close, self.SMA_LONG) if n >= self._MIN_FOR_SMA200 else None,

            **self._rsi_fields(close, n),
            **self._macd_fields(close, n),
            **self._bb_fields(df, n),

            atr=self._atr(df) if n >= self._MIN_FOR_ATR else None,
            adx=self._adx(df, n),
            **self._stoch_rsi_fields(close, n),

            vwap=self._vwap(df),
        )

    # ── Indicator helpers ──────────────────────────────────────────────────────

    def _ema(self, close: pd.Series, span: int) -> float | None:
        try:
            series = ta.ema(close, length=span)
            if series is None or series.empty:
                return None
            val = float(series.iloc[-1])
            return val if _is_finite(val) else None
        except Exception as exc:
            log.debug("ema_%d_failed: %s", span, exc)
            return None

    def _sma(self, close: pd.Series, period: int) -> float | None:
        try:
            series = ta.sma(close, length=period)
            if series is None or series.empty:
                return None
            val = float(series.iloc[-1])
            return val if _is_finite(val) else None
        except Exception as exc:
            log.debug("sma_%d_failed: %s", period, exc)
            return None

    def _rsi_fields(self, close: pd.Series, n: int) -> dict[str, float | None]:
        if n < self._MIN_FOR_BASIC:
            return {"rsi": None, "rsi_prev": None}
        try:
            rsi_series = ta.rsi(close, length=self.RSI_PERIOD)
            if rsi_series is None or rsi_series.dropna().empty:
                return {"rsi": None, "rsi_prev": None}
            rsi_val = float(rsi_series.iloc[-1])
            rsi_prev_val = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else None
            return {
                "rsi": rsi_val if _is_finite(rsi_val) else None,
                "rsi_prev": rsi_prev_val if (rsi_prev_val is not None and _is_finite(rsi_prev_val)) else None,
            }
        except Exception as exc:
            log.debug("rsi_failed: %s", exc)
            return {"rsi": None, "rsi_prev": None}

    def _macd_fields(self, close: pd.Series, n: int) -> dict[str, float | None]:
        empty: dict[str, float | None] = {
            "macd_line": None, "macd_signal": None,
            "macd_histogram": None, "macd_histogram_prev": None,
        }
        if n < self._MIN_FOR_BASIC:
            return empty
        try:
            macd_df = ta.macd(
                close,
                fast=self.MACD_FAST,
                slow=self.MACD_SLOW,
                signal=self.MACD_SIGNAL,
            )
            if macd_df is None or macd_df.empty:
                return empty

            # pandas-ta returns columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            line_col = [c for c in macd_df.columns if c.startswith("MACD_")]
            hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")]
            sig_col = [c for c in macd_df.columns if c.startswith("MACDs_")]
            if not (line_col and hist_col and sig_col):
                return empty

            line_val = float(macd_df[line_col[0]].iloc[-1])
            sig_val = float(macd_df[sig_col[0]].iloc[-1])
            hist_val = float(macd_df[hist_col[0]].iloc[-1])
            hist_prev = float(macd_df[hist_col[0]].iloc[-2]) if len(macd_df) >= 2 else None

            return {
                "macd_line": line_val if _is_finite(line_val) else None,
                "macd_signal": sig_val if _is_finite(sig_val) else None,
                "macd_histogram": hist_val if _is_finite(hist_val) else None,
                "macd_histogram_prev": hist_prev if (hist_prev is not None and _is_finite(hist_prev)) else None,
            }
        except Exception as exc:
            log.debug("macd_failed: %s", exc)
            return empty

    def _bb_fields(self, df: pd.DataFrame, n: int) -> dict[str, float | None]:
        empty: dict[str, float | None] = {
            "bb_upper": None, "bb_middle": None, "bb_lower": None,
            "bb_width": None, "bb_pct_b": None,
        }
        if n < self._MIN_FOR_BASIC:
            return empty
        try:
            bb_df = ta.bbands(df["close"], length=self.BB_PERIOD, std=self.BB_STD)
            if bb_df is None or bb_df.empty:
                return empty

            # Columns: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
            lower_col = [c for c in bb_df.columns if c.startswith("BBL_")]
            mid_col = [c for c in bb_df.columns if c.startswith("BBM_")]
            upper_col = [c for c in bb_df.columns if c.startswith("BBU_")]
            if not (lower_col and mid_col and upper_col):
                return empty

            upper = float(bb_df[upper_col[0]].iloc[-1])
            middle = float(bb_df[mid_col[0]].iloc[-1])
            lower = float(bb_df[lower_col[0]].iloc[-1])

            if not (_is_finite(upper) and _is_finite(middle) and _is_finite(lower)):
                return empty
            if middle == 0.0:
                return empty

            last_close = float(df["close"].iloc[-1])
            band_range = upper - lower
            bb_width = band_range / middle
            bb_pct_b = (last_close - lower) / band_range if band_range != 0.0 else None

            return {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_width": bb_width if _is_finite(bb_width) else None,
                "bb_pct_b": bb_pct_b if (bb_pct_b is not None and _is_finite(bb_pct_b)) else None,
            }
        except Exception as exc:
            log.debug("bbands_failed: %s", exc)
            return empty

    def _atr(self, df: pd.DataFrame) -> float | None:
        try:
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=self.ATR_PERIOD)
            if atr_series is None or atr_series.dropna().empty:
                return None
            val = float(atr_series.iloc[-1])
            return val if _is_finite(val) else None
        except Exception as exc:
            log.debug("atr_failed: %s", exc)
            return None

    def _adx(self, df: pd.DataFrame, n: int) -> float | None:
        if n < self._MIN_FOR_ADX:
            return None
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.ADX_PERIOD)
            if adx_df is None or adx_df.empty:
                return None
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            if not adx_col:
                return None
            val = float(adx_df[adx_col[0]].iloc[-1])
            return val if _is_finite(val) else None
        except Exception as exc:
            log.debug("adx_failed: %s", exc)
            return None

    def _stoch_rsi_fields(self, close: pd.Series, n: int) -> dict[str, float | None]:
        empty: dict[str, float | None] = {"stoch_rsi_k": None, "stoch_rsi_d": None}
        if n < self._MIN_FOR_BASIC:
            return empty
        try:
            srsi_df = ta.stochrsi(
                close,
                length=self.STOCH_RSI_LENGTH,
                rsi_length=self.STOCH_RSI_RSI_LENGTH,
                k=self.STOCH_RSI_K,
                d=self.STOCH_RSI_D,
            )
            if srsi_df is None or srsi_df.empty:
                return empty
            k_col = [c for c in srsi_df.columns if "STOCHRSIk" in c]
            d_col = [c for c in srsi_df.columns if "STOCHRSId" in c]
            if not (k_col and d_col):
                return empty
            k_val = float(srsi_df[k_col[0]].iloc[-1])
            d_val = float(srsi_df[d_col[0]].iloc[-1])
            return {
                "stoch_rsi_k": k_val if _is_finite(k_val) else None,
                "stoch_rsi_d": d_val if _is_finite(d_val) else None,
            }
        except Exception as exc:
            log.debug("stoch_rsi_failed: %s", exc)
            return empty

    def _vwap(self, df: pd.DataFrame) -> float | None:
        if "vwap" not in df.columns:
            return None
        try:
            val = float(df["vwap"].iloc[-1])
            return val if _is_finite(val) else None
        except Exception as exc:
            log.debug("vwap_read_failed: %s", exc)
            return None


# ── Utility ────────────────────────────────────────────────────────────────────

def _is_finite(val: float) -> bool:
    """Return True if val is a real finite number (not NaN or ±Inf)."""
    return math.isfinite(val)
