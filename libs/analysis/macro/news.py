"""
News Impact Engine.

Detects high-impact economic events and news that should block or reduce
confidence in trading signals during dangerous periods.

Static calendar heuristics (no external feed required):
  - FOMC meetings   → critical, block trades
  - CPI release     → high, reduce confidence -15%
  - NFP release     → high, reduce confidence -10%
  - Crypto Sunday   → medium, reduce confidence -5%
  - Stock weekend   → block (market closed)

An optional live CryptoCompare news fetch is available but degrades
gracefully to an empty list on any network failure.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# ── Severity → confidence impact mapping ──────────────────────────────────────

_SEVERITY_IMPACT: dict[str, float] = {
    "critical": 0.0,   # block entirely — impact irrelevant
    "high": -0.15,
    "medium": -0.05,
    "low": -0.02,
}

# ── FOMC schedule heuristic ───────────────────────────────────────────────────
# Eight meetings per year.  Approximate rule: the 3rd Wednesday of
# Jan, Mar, May, Jun, Jul, Sep, Nov, Dec.
_FOMC_MONTHS = {1, 3, 5, 6, 7, 9, 11, 12}

# ── CPI heuristic ─────────────────────────────────────────────────────────────
# Released around the 13th of every month (±2 days buffer → 11–15).
_CPI_DAY_MIN = 11
_CPI_DAY_MAX = 15

# ── Live news endpoint ────────────────────────────────────────────────────────
_CRYPTOCOMPARE_URL = (
    "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC"
)
_FETCH_TIMEOUT = 5.0


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NewsEvent:
    """A single detected high-impact news or calendar event."""

    event_type: str
    """One of: cpi, fomc, earnings, hack, regulation, war, nfp, weekend, black_swan."""

    severity: str
    """One of: low, medium, high, critical."""

    description: str
    """Human-readable description of the event."""

    confidence_impact: float
    """Signed delta applied to signal confidence. Range: -0.20 to 0.0."""

    should_block_trades: bool
    """True when this event alone warrants blocking all trade signals."""


@dataclass(frozen=True)
class NewsAssessment:
    """Aggregated result of all detected news events for a symbol."""

    events: list[NewsEvent]
    """All events detected for this symbol + asset class."""

    total_confidence_impact: float
    """Sum of all event confidence_impact values, clamped to [-0.20, 0.0]."""

    should_block: bool
    """True when at least one event has should_block_trades=True."""

    explanation: str
    """Human-readable summary of the assessment."""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> int:
    """Return the day-of-month for the nth occurrence of weekday (0=Mon … 6=Sun)."""
    first = datetime.date(year, month, 1)
    # days until the target weekday
    delta = (weekday - first.weekday()) % 7
    target = first + datetime.timedelta(days=delta + (n - 1) * 7)
    return target.day


def _is_fomc_day(now: datetime.datetime) -> bool:
    """Return True if today is (approximately) an FOMC meeting day."""
    if now.month not in _FOMC_MONTHS:
        return False
    # 3rd Wednesday of the month
    third_wed_day = _nth_weekday_of_month(now.year, now.month, 2, 3)
    return now.day == third_wed_day


def _is_cpi_day(now: datetime.datetime) -> bool:
    """Return True if today falls in the CPI release window (~13th ± 2 days)."""
    return _CPI_DAY_MIN <= now.day <= _CPI_DAY_MAX


def _is_nfp_day(now: datetime.datetime) -> bool:
    """Return True if today is the first Friday of the month (NFP release)."""
    if now.weekday() != 4:  # not Friday
        return False
    first_fri_day = _nth_weekday_of_month(now.year, now.month, 4, 1)
    return now.day == first_fri_day


def _build_explanation(events: list[NewsEvent], total_impact: float) -> str:
    if not events:
        return "No high-impact events detected; normal trading conditions."

    parts: list[str] = []
    for ev in events:
        parts.append(f"{ev.event_type.upper()} ({ev.severity}): {ev.description}")

    impact_pct = round(total_impact * 100)
    summary = "; ".join(parts)
    if any(ev.should_block_trades for ev in events):
        return f"{summary} — TRADING BLOCKED."
    return f"{summary} — confidence {impact_pct}%."


# ── Engine ────────────────────────────────────────────────────────────────────


class NewsImpactEngine:
    """
    Detects high-impact economic events and returns a NewsAssessment.

    All calendar logic is pure (no I/O) so assess() is fully synchronous
    and trivially testable.  fetch_live_news() is the only async method
    and degrades gracefully on network failure.
    """

    # ── Public synchronous API ─────────────────────────────────────────────────

    def assess(self, symbol: str, asset_class: str) -> NewsAssessment:
        """
        Check for known high-impact events for the given symbol / asset class.

        Parameters
        ----------
        symbol : str
            Trading symbol, e.g. "BTCUSDT" or "AAPL".
        asset_class : str
            One of "crypto" or "stock".

        Returns
        -------
        NewsAssessment
            Immutable snapshot of all detected events and aggregate impact.
        """
        now = datetime.datetime.now()
        events = self._detect_calendar_events(now, symbol, asset_class)
        return self._build_assessment(events)

    # ── Optional async live feed ───────────────────────────────────────────────

    async def fetch_live_news(self, symbol: str) -> list[NewsEvent]:
        """
        Fetch the latest BTC-tagged headlines from CryptoCompare.

        Returns an empty list on any network or parse failure so callers
        can always safely extend their event list without guarding.

        Parameters
        ----------
        symbol : str
            Trading symbol (used for logging context only).

        Returns
        -------
        list[NewsEvent]
            Zero or more NewsEvent objects parsed from the live feed.
        """
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
                response = await client.get(_CRYPTOCOMPARE_URL)
                response.raise_for_status()
                payload = response.json()
                articles = payload.get("Data", [])
                return [
                    NewsEvent(
                        event_type="news",
                        severity="low",
                        description=article.get("title", "Unknown headline")[:120],
                        confidence_impact=_SEVERITY_IMPACT["low"],
                        should_block_trades=False,
                    )
                    for article in articles[:5]
                ]
        except Exception:
            logger.warning("fetch_live_news_failed", exc_info=True)
            return []

    # ── Internal calendar detection ────────────────────────────────────────────

    def _detect_calendar_events(
        self,
        now: datetime.datetime,
        symbol: str,
        asset_class: str,
    ) -> list[NewsEvent]:
        """Return all calendar-driven events that apply right now."""
        events: list[NewsEvent] = []
        weekday = now.weekday()  # 0=Mon … 6=Sun
        is_weekend = weekday >= 5  # Saturday=5, Sunday=6

        # ── Stock-specific: market closed on weekends ──────────────────────────
        if asset_class == "stock" and is_weekend:
            events.append(
                NewsEvent(
                    event_type="weekend",
                    severity="critical",
                    description="Stock market closed on weekends.",
                    confidence_impact=0.0,
                    should_block_trades=True,
                )
            )
            return events  # no point adding more events when already blocked

        # ── Crypto-specific: Sunday low-liquidity warning ──────────────────────
        if asset_class == "crypto" and weekday == 6:  # Sunday
            events.append(
                NewsEvent(
                    event_type="low_liquidity",
                    severity="medium",
                    description="Sunday crypto session — thin liquidity, wider spreads.",
                    confidence_impact=_SEVERITY_IMPACT["medium"],
                    should_block_trades=False,
                )
            )

        # ── FOMC (all asset classes) ───────────────────────────────────────────
        if _is_fomc_day(now):
            events.append(
                NewsEvent(
                    event_type="fomc",
                    severity="critical",
                    description="FOMC decision day — extreme rate/macro uncertainty.",
                    confidence_impact=0.0,
                    should_block_trades=True,
                )
            )

        # ── CPI release (all asset classes) ───────────────────────────────────
        if _is_cpi_day(now):
            events.append(
                NewsEvent(
                    event_type="cpi",
                    severity="high",
                    description="CPI release window (~13th of month) — inflation data risk.",
                    confidence_impact=_SEVERITY_IMPACT["high"],
                    should_block_trades=False,
                )
            )

        # ── NFP — first Friday of the month ───────────────────────────────────
        if _is_nfp_day(now):
            events.append(
                NewsEvent(
                    event_type="nfp",
                    severity="high",
                    description="Non-Farm Payrolls release day — labour market shock risk.",
                    confidence_impact=_SEVERITY_IMPACT["high"],
                    should_block_trades=False,
                )
            )

        return events

    # ── Assessment builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_assessment(events: list[NewsEvent]) -> NewsAssessment:
        raw_impact = sum(ev.confidence_impact for ev in events)
        clamped_impact = max(-0.20, min(0.0, raw_impact))
        should_block = any(ev.should_block_trades for ev in events)
        explanation = _build_explanation(events, clamped_impact)

        return NewsAssessment(
            events=events,
            total_confidence_impact=clamped_impact,
            should_block=should_block,
            explanation=explanation,
        )
