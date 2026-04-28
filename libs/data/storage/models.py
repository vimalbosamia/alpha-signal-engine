from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # signal_id UUID str
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    asset_class: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(10), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    strategy_name: Mapped[str] = mapped_column(String(100))
    timeframe: Mapped[str] = mapped_column(String(10))
    regime: Mapped[str] = mapped_column(String(30))
    estimated_rr: Mapped[float] = mapped_column(Float)
    generated_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    payload: Mapped[str] = mapped_column(Text)  # full JSON of SignalOutput


class AuditRecord(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # event_id UUID str
    signal_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    asset_class: Mapped[str] = mapped_column(String(10))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    payload: Mapped[str] = mapped_column(Text)  # JSON blob


class SignalOutcomeRecord(Base):
    """Tracks whether a BUY/SELL signal was directionally correct after N minutes."""

    __tablename__ = "signal_outcomes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    strategy_name: Mapped[str] = mapped_column(String(100), index=True)
    action: Mapped[str] = mapped_column(String(10))        # BUY or SELL
    entry_price: Mapped[float] = mapped_column(Float)
    check_price: Mapped[float] = mapped_column(Float, default=0.0)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit_1: Mapped[float] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(String(10), index=True, default="PENDING")
    # WIN  = price moved in correct direction at check time
    # LOSS = price moved against signal direction
    # PENDING = not yet checked
    # EXPIRED = signal too old, no price data
    direction_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    check_after_minutes: Mapped[int] = mapped_column(Integer, default=30)
    ml_features: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list[float]
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class StrategyPerformanceRecord(Base):
    """Running win/loss tally per strategy — updated after each outcome check."""

    __tablename__ = "strategy_performance"

    strategy_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    total_signals: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    is_muted: Mapped[bool] = mapped_column(Boolean, default=False)
    last_updated: Mapped[datetime] = mapped_column(DateTime)


class ProviderHealthRecord(Base):
    __tablename__ = "provider_health"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(30), index=True)
    is_healthy: Mapped[bool] = mapped_column(Boolean)
    checked_at: Mapped[datetime] = mapped_column(DateTime)
    error_message: Mapped[str] = mapped_column(String(500), default="")
