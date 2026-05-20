"""
Unit tests for NewsImpactEngine.

All tests are synchronous and exercise assess() only.
Date/time is controlled via unittest.mock.patch so the suite is
deterministic regardless of when it runs.
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from libs.analysis.macro.news import NewsAssessment, NewsEvent, NewsImpactEngine


# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture()
def engine() -> NewsImpactEngine:
    return NewsImpactEngine()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_now(dt: datetime.datetime):
    """Return a context manager that patches datetime.datetime.now to return *dt*."""
    return patch("libs.analysis.macro.news.datetime.datetime", wraps=datetime.datetime)


def _patch_now(dt: datetime.datetime):
    """Patch datetime.datetime.now() inside the news module to return *dt*."""
    class _FakeDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt

    return patch("libs.analysis.macro.news.datetime.datetime", _FakeDatetime)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestNoEventsNormalDay:
    """On a plain weekday with no calendar events there should be zero impact."""

    def test_no_events_normal_day(self, engine: NewsImpactEngine) -> None:
        # Tuesday 2026-04-07 — not FOMC, not CPI window, not NFP, not weekend
        normal_tuesday = datetime.datetime(2026, 4, 7, 10, 0, 0)
        with _patch_now(normal_tuesday):
            result = engine.assess("BTCUSDT", "crypto")

        assert result.should_block is False
        assert result.total_confidence_impact == 0.0
        assert result.events == []


class TestSundayCryptoReduced:
    """Sunday + crypto asset class → medium-severity low-liquidity event, -5% impact."""

    def test_sunday_crypto_reduced(self, engine: NewsImpactEngine) -> None:
        # 2026-04-05 is a Sunday
        sunday = datetime.datetime(2026, 4, 5, 14, 0, 0)
        with _patch_now(sunday):
            result = engine.assess("ETHUSDT", "crypto")

        assert result.should_block is False
        assert len(result.events) >= 1

        low_liq_events = [e for e in result.events if e.event_type == "low_liquidity"]
        assert len(low_liq_events) == 1

        ev = low_liq_events[0]
        assert ev.severity == "medium"
        assert ev.confidence_impact == pytest.approx(-0.05)
        # Total impact must include the Sunday penalty
        assert result.total_confidence_impact <= -0.05


class TestStockWeekendBlocked:
    """Saturday + stock asset class → critical weekend event, trades blocked."""

    def test_stock_weekend_blocked(self, engine: NewsImpactEngine) -> None:
        # 2026-04-04 is a Saturday
        saturday = datetime.datetime(2026, 4, 4, 9, 30, 0)
        with _patch_now(saturday):
            result = engine.assess("AAPL", "stock")

        assert result.should_block is True

        weekend_events = [e for e in result.events if e.event_type == "weekend"]
        assert len(weekend_events) == 1
        assert weekend_events[0].severity == "critical"


class TestResultFields:
    """NewsAssessment must always expose all required fields with correct types."""

    def test_result_fields(self, engine: NewsImpactEngine) -> None:
        normal_tuesday = datetime.datetime(2026, 4, 7, 10, 0, 0)
        with _patch_now(normal_tuesday):
            result = engine.assess("BTCUSDT", "crypto")

        assert isinstance(result, NewsAssessment)
        assert isinstance(result.events, list)
        assert isinstance(result.total_confidence_impact, float)
        assert isinstance(result.should_block, bool)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0

        # Verify NewsEvent fields when events exist (inject a known day)
        fomc_wednesday = datetime.datetime(2026, 1, 21, 14, 0, 0)  # 3rd Wed of Jan 2026
        with _patch_now(fomc_wednesday):
            fomc_result = engine.assess("BTCUSDT", "crypto")

        assert len(fomc_result.events) >= 1
        ev: NewsEvent = fomc_result.events[0]
        assert isinstance(ev.event_type, str)
        assert isinstance(ev.severity, str)
        assert isinstance(ev.description, str)
        assert isinstance(ev.confidence_impact, float)
        assert isinstance(ev.should_block_trades, bool)


class TestCriticalEventBlocks:
    """A critical-severity event must set should_block=True on the assessment."""

    def test_critical_event_blocks(self, engine: NewsImpactEngine) -> None:
        # 3rd Wednesday of January 2026 → FOMC
        fomc_day = datetime.datetime(2026, 1, 21, 14, 0, 0)
        with _patch_now(fomc_day):
            result = engine.assess("BTCUSDT", "crypto")

        assert result.should_block is True

        fomc_events = [e for e in result.events if e.event_type == "fomc"]
        assert len(fomc_events) == 1
        assert fomc_events[0].severity == "critical"
        assert fomc_events[0].should_block_trades is True


class TestMultipleEventsStack:
    """Two medium-severity events must stack their confidence impacts."""

    def test_multiple_events_stack(self, engine: NewsImpactEngine) -> None:
        # Manufacture a scenario with two events by directly calling _build_assessment
        ev_a = NewsEvent(
            event_type="low_liquidity",
            severity="medium",
            description="Sunday thin liquidity.",
            confidence_impact=-0.05,
            should_block_trades=False,
        )
        ev_b = NewsEvent(
            event_type="cpi",
            severity="medium",
            description="Mid-month CPI window.",
            confidence_impact=-0.05,
            should_block_trades=False,
        )

        result = NewsImpactEngine._build_assessment([ev_a, ev_b])

        assert result.total_confidence_impact == pytest.approx(-0.10)
        assert result.should_block is False
        assert len(result.events) == 2
