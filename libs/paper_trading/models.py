"""SQLAlchemy 2.0 async models for paper trading persistence.

Three tables:
- paper_trades       — individual trade records (open / closed / force-closed)
- paper_bot_stats    — per-bot running statistics and phase tracking
- paper_equity_curve — portfolio-level equity snapshots over time
"""
from datetime import datetime

from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from libs.data.storage.models import Base


class PaperTradeRecord(Base):
    """One simulated trade entry, open or closed."""

    __tablename__ = "paper_trades"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    bot_name: Mapped[str] = mapped_column(String(100), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    asset_class: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(10))  # BUY | SELL

    # Pricing
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size_usd: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)

    # P&L
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    hold_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Signal metadata
    strategy_name: Mapped[str] = mapped_column(String(100))
    signal_id: Mapped[str] = mapped_column(String)

    # Risk levels
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_2: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # OPEN | CLOSED | FORCE_CLOSED
    status: Mapped[str] = mapped_column(String(20), default="OPEN")


class PaperBotStatsRecord(Base):
    """Running statistics for a single paper-trading bot."""

    __tablename__ = "paper_bot_stats"

    bot_name: Mapped[str] = mapped_column(String(100), primary_key=True)
    allocated_capital: Mapped[float] = mapped_column(Float)
    current_balance: Mapped[float] = mapped_column(Float)

    # Aggregate P&L counters
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, default=0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)

    # Risk metrics
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    kelly_fraction: Mapped[float] = mapped_column(Float, default=0.0)

    # Adaptive-bot fields
    phase: Mapped[str] = mapped_column(String(30), default="cold_start")
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False)

    def __init__(self, **kwargs):
        # Apply Python-level defaults for fields not passed
        defaults = {
            "total_pnl": 0.0, "win_count": 0, "loss_count": 0, "trade_count": 0,
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
            "kelly_fraction": 0.0, "phase": "cold_start", "is_paused": False,
        }
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        super().__init__(**kwargs)
    # JSON blob used by AdaptiveBot to persist filter state
    adaptive_filters: Mapped[str | None] = mapped_column(Text, nullable=True)

    last_rebalance_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)


class PaperEquityCurveRecord(Base):
    """Portfolio-level equity snapshot for charting and drawdown analysis."""

    __tablename__ = "paper_equity_curve"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    total_balance: Mapped[float] = mapped_column(Float)
    # JSON mapping of bot_name → balance, e.g. '{"MomentumBot": 1667.0}'
    bot_balances_json: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
