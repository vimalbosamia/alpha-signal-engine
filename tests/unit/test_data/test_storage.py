from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from libs.core.models.domain import (
    AssetClass,
    ConfluenceBreakdown,
    DataQualityStatus,
    MarketRegime,
    SessionType,
    SignalAction,
    SignalOutput,
    Timeframe,
    TrendDirection,
)
from libs.data.storage.models import Base
from libs.data.storage.repository import AuditRepository, SignalRepository


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_signal(symbol: str = "AAPL") -> SignalOutput:
    breakdown = ConfluenceBreakdown(
        pattern_score=0.8,
        structure_score=0.7,
        level_score=0.6,
        volume_score=0.7,
        regime_score=0.6,
        session_score=1.0,
        risk_score=0.7,
        data_quality_score=1.0,
        weighted_total=0.72,
        weights={},
        factor_notes={},
        blocked_reasons=[],
        warning_tags=[],
    )
    return SignalOutput(
        symbol=symbol,
        asset_class=AssetClass.STOCK,
        strategy_name="test",
        timeframe=Timeframe.FIVE_MIN,
        action=SignalAction.BUY,
        confidence=0.72,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=100.0,
        entry_zone_high=100.5,
        stop_loss=98.0,
        take_profit_1=105.0,
        estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.REGULAR,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=breakdown,
        generated_at=datetime.utcnow(),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_init_db_creates_tables(session):
    """init_db creates tables without error (session fixture already did it)."""
    # If the fixture yielded without raising, tables were created successfully.
    assert session is not None


async def test_save_and_get_by_id(session):
    """save_signal persists, get_by_id retrieves the same signal_id."""
    repo = SignalRepository(session)
    signal = make_signal()
    await repo.save_signal(signal)

    result = await repo.get_by_id(str(signal.signal_id))
    assert result is not None
    assert result["signal_id"] == str(signal.signal_id)
    assert result["symbol"] == "AAPL"


async def test_get_latest_signals_respects_limit(session):
    """get_latest_signals returns at most `limit` records."""
    repo = SignalRepository(session)
    for _ in range(5):
        await repo.save_signal(make_signal())

    results = await repo.get_latest_signals(limit=3)
    assert len(results) == 3


async def test_get_latest_signals_filters_by_symbol(session):
    """get_latest_signals(symbol=...) filters by symbol."""
    repo = SignalRepository(session)
    for _ in range(3):
        await repo.save_signal(make_signal("AAPL"))
    for _ in range(2):
        await repo.save_signal(make_signal("MSFT"))

    results = await repo.get_latest_signals(symbol="MSFT")
    assert len(results) == 2
    assert all(r["symbol"] == "MSFT" for r in results)


async def test_audit_record_and_trail_order(session):
    """record_event + get_signal_trail returns correct event_type order."""
    repo = AuditRepository(session)
    sid = str(uuid4())

    await repo.record_event(
        event_type="data_fetched",
        symbol="AAPL",
        asset_class="stock",
        payload={"step": 1},
        signal_id=sid,
    )
    await repo.record_event(
        event_type="signal_generated",
        symbol="AAPL",
        asset_class="stock",
        payload={"step": 2},
        signal_id=sid,
    )

    trail = await repo.get_signal_trail(sid)
    assert len(trail) == 2
    assert trail[0]["event_type"] == "data_fetched"
    assert trail[1]["event_type"] == "signal_generated"


async def test_get_by_id_returns_none_for_unknown(session):
    """get_by_id returns None for an unknown signal_id."""
    repo = SignalRepository(session)
    result = await repo.get_by_id("nonexistent-id-12345")
    assert result is None


async def test_audit_trail_isolated_by_signal_id(session):
    """get_signal_trail only returns events for the requested signal_id."""
    repo = AuditRepository(session)
    sid_a = str(uuid4())
    sid_b = str(uuid4())

    await repo.record_event("ev_a", "AAPL", "stock", {}, signal_id=sid_a)
    await repo.record_event("ev_b", "AAPL", "stock", {}, signal_id=sid_b)

    trail_a = await repo.get_signal_trail(sid_a)
    assert len(trail_a) == 1
    assert trail_a[0]["event_type"] == "ev_a"
