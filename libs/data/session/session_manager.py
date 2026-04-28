"""
Market session manager.

Handles completely different session rules for stocks vs crypto:

STOCKS (US Equity):
  - Pre-market: 04:00–09:30 ET
  - Regular:    09:30–16:00 ET
  - Post-market: 16:00–20:00 ET
  - Closed outside those windows
  - Holidays and early-close days
  - First-N-minutes and last-N-minutes exclusion zones

CRYPTO:
  - 24/7, always open
  - No regular close
  - Exchange maintenance windows if detected
  - Session tagging for analytics (Asia / Europe / Americas)

Session state affects:
  - Signal generation eligibility
  - Pattern confidence adjustments
  - Volume interpretation
  - Replay and backtesting bar tagging
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from libs.core.models.domain import AssetClass, SessionState, SessionType

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# US equity market holidays (2024–2026)
_US_HOLIDAYS: set[date] = {
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
}

# Early-close dates (1:00 PM ET)
_EARLY_CLOSE_DATES: set[date] = {
    date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
}


class SessionManager(Protocol):
    """Protocol for session managers — one per asset class."""

    def get_state(self, symbol: str, ts: datetime) -> SessionState: ...
    def is_signal_eligible(self, symbol: str, ts: datetime) -> bool: ...
    def quality_score(self, symbol: str, ts: datetime) -> float: ...


class StockSessionManager:
    """
    Session manager for US equity markets.
    """

    REGULAR_OPEN = time(9, 30)
    REGULAR_CLOSE = time(16, 0)
    PRE_MARKET_START = time(4, 0)
    POST_MARKET_END = time(20, 0)
    EARLY_CLOSE_TIME = time(13, 0)
    AVOID_OPEN_MINUTES = 15
    AVOID_CLOSE_MINUTES = 10

    def _to_et(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_UTC)
        return ts.astimezone(_ET)

    def _is_holiday(self, d: date) -> bool:
        return d in _US_HOLIDAYS or d.weekday() >= 5

    def _is_early_close(self, d: date) -> bool:
        return d in _EARLY_CLOSE_DATES

    def get_session_type(self, ts: datetime) -> SessionType:
        et = self._to_et(ts)
        d = et.date()

        if self._is_holiday(d):
            return SessionType.HOLIDAY
        if self._is_early_close(d):
            close = self.EARLY_CLOSE_TIME
        else:
            close = self.REGULAR_CLOSE

        t = et.time()
        if t < self.PRE_MARKET_START or t >= self.POST_MARKET_END:
            return SessionType.CLOSED
        if t < self.REGULAR_OPEN:
            return SessionType.PRE_MARKET
        if t < close:
            return SessionType.REGULAR
        if self._is_early_close(d):
            return SessionType.HALF_DAY
        return SessionType.POST_MARKET

    def is_signal_eligible(self, symbol: str, ts: datetime) -> bool:
        """True only during the tradable regular-session window."""
        st = self.get_session_type(ts)
        if st != SessionType.REGULAR:
            return False

        et = self._to_et(ts)
        t = et.time()

        # Avoid first N minutes after open
        open_dt = datetime.combine(et.date(), self.REGULAR_OPEN, tzinfo=_ET)
        minutes_since_open = (et - open_dt).total_seconds() / 60
        if minutes_since_open < self.AVOID_OPEN_MINUTES:
            return False

        # Avoid last N minutes before close
        close_dt = datetime.combine(et.date(), self.REGULAR_CLOSE, tzinfo=_ET)
        minutes_to_close = (close_dt - et).total_seconds() / 60
        if minutes_to_close < self.AVOID_CLOSE_MINUTES:
            return False

        return True

    def quality_score(self, symbol: str, ts: datetime) -> float:
        """
        Session quality score 0–1.
        1.0 = prime liquidity (10:30–14:30 ET)
        0.7 = normal regular session
        0.3 = fringe (open/close buffer zone)
        0.0 = outside regular session
        """
        st = self.get_session_type(ts)
        if st not in (SessionType.REGULAR, SessionType.HALF_DAY):
            return 0.0

        et = self._to_et(ts)
        t = et.time()

        open_dt = datetime.combine(et.date(), self.REGULAR_OPEN, tzinfo=_ET)
        close_dt = datetime.combine(et.date(), self.REGULAR_CLOSE, tzinfo=_ET)
        minutes_since_open = (et - open_dt).total_seconds() / 60
        minutes_to_close = (close_dt - et).total_seconds() / 60

        if minutes_since_open < self.AVOID_OPEN_MINUTES or minutes_to_close < self.AVOID_CLOSE_MINUTES:
            return 0.3

        prime_start = time(10, 30)
        prime_end = time(14, 30)
        if prime_start <= t <= prime_end:
            return 1.0

        return 0.7

    def get_state(self, symbol: str, ts: datetime) -> SessionState:
        session_type = self.get_session_type(ts)
        score = self.quality_score(symbol, ts)
        eligible = self.is_signal_eligible(symbol, ts)

        et = self._to_et(ts)
        close_dt = datetime.combine(et.date(), self.REGULAR_CLOSE, tzinfo=_ET)
        mins_to_close = max(0.0, (close_dt - et).total_seconds() / 60)

        return SessionState(
            symbol=symbol,
            asset_class=AssetClass.STOCK,
            session_type=session_type,
            is_tradable=eligible,
            quality_score=score,
            minutes_to_close=mins_to_close if session_type == SessionType.REGULAR else None,
        )


class CryptoSessionManager:
    """
    Session manager for 24/7 crypto markets.

    Crypto never closes. We tag candles with session zones for
    analytics (Asia/Europe/Americas) but never block signals on
    session grounds alone — only on data quality or regime fitness.
    """

    # Crypto "session" windows (UTC) — for analytics tagging only
    ASIA_START = time(0, 0)
    ASIA_END = time(8, 0)
    EUROPE_START = time(7, 0)
    EUROPE_END = time(16, 0)
    AMERICAS_START = time(13, 0)
    AMERICAS_END = time(22, 0)

    def _session_note(self, ts: datetime) -> str:
        """Tag which trading region is active."""
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_UTC)
        t = ts.astimezone(_UTC).time()
        regions = []
        if self.ASIA_START <= t < self.ASIA_END:
            regions.append("Asia")
        if self.EUROPE_START <= t < self.EUROPE_END:
            regions.append("Europe")
        if self.AMERICAS_START <= t < self.AMERICAS_END:
            regions.append("Americas")
        return "/".join(regions) if regions else "off-peak"

    def is_signal_eligible(self, symbol: str, ts: datetime) -> bool:
        """Crypto is always eligible from a session standpoint."""
        return True

    def quality_score(self, symbol: str, ts: datetime) -> float:
        """
        Crypto session quality — weighted by liquidity windows.
        Americas overlap with Europe = highest volume → 1.0
        Pure Asia session = lower liquidity → 0.65
        Off-peak = 0.5
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_UTC)
        t = ts.astimezone(_UTC).time()

        # Americas + Europe overlap: 13:00–16:00 UTC (high volume)
        if time(13, 0) <= t < time(16, 0):
            return 1.0
        # Americas active
        if time(13, 0) <= t < time(22, 0):
            return 0.85
        # Europe active
        if time(7, 0) <= t < time(13, 0):
            return 0.75
        # Asia active
        if time(0, 0) <= t < time(8, 0):
            return 0.65
        return 0.50

    def get_state(self, symbol: str, ts: datetime) -> SessionState:
        return SessionState(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            session_type=SessionType.CONTINUOUS,
            is_tradable=True,
            quality_score=self.quality_score(symbol, ts),
            notes=self._session_note(ts),
        )


def get_session_manager(asset_class: AssetClass) -> StockSessionManager | CryptoSessionManager:
    """Factory: return the correct session manager for the asset class."""
    if asset_class == AssetClass.STOCK:
        return StockSessionManager()
    return CryptoSessionManager()
