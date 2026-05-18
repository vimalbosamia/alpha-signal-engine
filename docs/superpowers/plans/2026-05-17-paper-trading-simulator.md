# Paper Trading Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent paper trading simulator with 6 competing hedge fund bots, Kelly criterion sizing, and a real-time P&L dashboard at `/paper`.

**Architecture:** Signals from existing `SignalPipeline` dispatch via `EventBus` to a `PaperTradingEngine` that fans them to 6 specialized bots. Each bot filters signals by strategy style, sizes via Kelly criterion, and manages virtual positions. A single-page dashboard at `/paper` shows a giant H1 money counter, bot leaderboard, equity curve, open positions, and trade history — all updating in real-time.

**Tech Stack:** Python 3.12+, SQLAlchemy 2.0 async, FastAPI, Pydantic, existing EventBus, existing Binance WebSocket for live prices.

**Spec:** `docs/superpowers/specs/2026-05-17-paper-trading-simulator-design.md`

---

## File Structure

```
libs/paper_trading/
    __init__.py              # Package init, re-exports
    models.py                # SQLAlchemy models: PaperTradeRecord, PaperBotStatsRecord, PaperEquityCurveRecord
    portfolio.py             # PaperPortfolio: balance mgmt, trade open/close, metrics calc
    allocator.py             # CapitalAllocator: Kelly sizing + weekly rebalance
    bot_agent.py             # BotAgent base class: signal filter, position mgmt, P&L tracking
    bots/
        __init__.py          # Bot registry: ALL_BOTS list
        momentum.py          # MomentumBot
        reversal.py          # ReversalBot
        mean_reversion.py    # MeanReversionBot
        scalper.py           # ScalperBot
        swing.py             # SwingBot
        adaptive.py          # AdaptiveBot (self-learning)
    engine.py                # PaperTradingEngine: orchestrator, EventBus subscriber, price monitor

apps/dashboard/
    paper_page.py            # /paper dashboard page + /api/paper/* endpoints

tests/unit/test_paper_trading/
    __init__.py
    test_models.py
    test_portfolio.py
    test_allocator.py
    test_bot_agent.py
    test_bots.py
    test_adaptive.py
    test_engine.py
```

---

## Task 1: DB Models

**Files:**
- Create: `libs/paper_trading/__init__.py`
- Create: `libs/paper_trading/models.py`
- Create: `tests/unit/test_paper_trading/__init__.py`
- Create: `tests/unit/test_paper_trading/test_models.py`
- Modify: `libs/data/storage/db.py` (import new models so `create_all` picks them up)

- [ ] **Step 1: Write failing test for model instantiation**

```python
# tests/unit/test_paper_trading/__init__.py
# (empty)
```

```python
# tests/unit/test_paper_trading/test_models.py
"""Tests for paper trading DB models."""
from __future__ import annotations

from datetime import datetime, timezone


def test_paper_trade_record_defaults():
    from libs.paper_trading.models import PaperTradeRecord

    rec = PaperTradeRecord(
        id="t1",
        bot_name="MomentumBot",
        symbol="BTCUSDT",
        asset_class="crypto",
        action="BUY",
        entry_price=67000.0,
        position_size_usd=200.0,
        fees=0.40,
        strategy_name="ema_crossover",
        signal_id="sig-1",
        opened_at=datetime.now(timezone.utc),
        status="OPEN",
    )
    assert rec.bot_name == "MomentumBot"
    assert rec.exit_price is None
    assert rec.realized_pnl is None
    assert rec.status == "OPEN"


def test_paper_bot_stats_record_defaults():
    from libs.paper_trading.models import PaperBotStatsRecord

    rec = PaperBotStatsRecord(
        bot_name="MomentumBot",
        allocated_capital=1667.0,
        current_balance=1667.0,
    )
    assert rec.total_pnl == 0.0
    assert rec.win_count == 0
    assert rec.phase == "cold_start"
    assert rec.is_paused is False


def test_paper_equity_curve_record():
    from libs.paper_trading.models import PaperEquityCurveRecord

    rec = PaperEquityCurveRecord(
        id="e1",
        total_balance=10000.0,
        bot_balances_json='{"MomentumBot": 1667}',
        timestamp=datetime.now(timezone.utc),
    )
    assert rec.total_balance == 10000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'libs.paper_trading'`

- [ ] **Step 3: Create package init and models**

```python
# libs/paper_trading/__init__.py
"""Paper trading simulator — multi-agent hedge fund."""
```

```python
# libs/paper_trading/models.py
"""SQLAlchemy models for paper trading persistence."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from libs.data.storage.models import Base


class PaperTradeRecord(Base):
    __tablename__ = "paper_trades"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    bot_name: Mapped[str] = mapped_column(String(50), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    asset_class: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(10))  # BUY or SELL
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_size_usd: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    hold_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    strategy_name: Mapped[str] = mapped_column(String(100))
    signal_id: Mapped[str] = mapped_column(String(100))
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")  # OPEN, CLOSED, FORCE_CLOSED


class PaperBotStatsRecord(Base):
    __tablename__ = "paper_bot_stats"

    bot_name: Mapped[str] = mapped_column(String(50), primary_key=True)
    allocated_capital: Mapped[float] = mapped_column(Float)
    current_balance: Mapped[float] = mapped_column(Float)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    loss_count: Mapped[int] = mapped_column(Integer, default=0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    profit_factor: Mapped[float] = mapped_column(Float, default=0.0)
    kelly_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    phase: Mapped[str] = mapped_column(String(20), default="cold_start")
    is_paused: Mapped[bool] = mapped_column(Boolean, default=False)
    adaptive_filters: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON for AdaptiveBot
    last_rebalance_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class PaperEquityCurveRecord(Base):
    __tablename__ = "paper_equity_curve"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    total_balance: Mapped[float] = mapped_column(Float)
    bot_balances_json: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
```

- [ ] **Step 4: Import models in db.py so `create_all` picks them up**

In `libs/data/storage/db.py`, add at the top imports (after existing model imports):

```python
import libs.paper_trading.models  # noqa: F401 — register tables
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_models.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add libs/paper_trading/__init__.py libs/paper_trading/models.py \
  tests/unit/test_paper_trading/__init__.py tests/unit/test_paper_trading/test_models.py \
  libs/data/storage/db.py
git commit -m "feat(paper): add SQLAlchemy models for paper trades, bot stats, equity curve"
```

---

## Task 2: PaperPortfolio — Balance & Trade Management

**Files:**
- Create: `libs/paper_trading/portfolio.py`
- Create: `tests/unit/test_paper_trading/test_portfolio.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_paper_trading/test_portfolio.py
"""Tests for PaperPortfolio trade and balance management."""
from __future__ import annotations

import pytest
from libs.paper_trading.portfolio import PaperPortfolio


def test_initial_balance():
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    assert pf.balance == 1667.0
    assert pf.open_trades == []
    assert pf.total_pnl == 0.0


def test_open_trade_reduces_balance():
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    trade_id = pf.open_trade(
        symbol="BTCUSDT",
        asset_class="crypto",
        action="BUY",
        entry_price=67000.0,
        position_size_usd=200.0,
        stop_loss=66000.0,
        take_profit_1=69000.0,
        take_profit_2=71000.0,
        strategy_name="ema_crossover",
        signal_id="sig-1",
    )
    assert trade_id is not None
    assert pf.balance == pytest.approx(1667.0 - 200.0 - 0.20)  # 0.1% fee on entry
    assert len(pf.open_trades) == 1


def test_close_trade_win():
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    tid = pf.open_trade(
        symbol="BTCUSDT", asset_class="crypto", action="BUY",
        entry_price=100.0, position_size_usd=100.0,
        stop_loss=95.0, take_profit_1=110.0, take_profit_2=None,
        strategy_name="test", signal_id="s1",
    )
    result = pf.close_trade(tid, exit_price=110.0)
    assert result["realized_pnl"] > 0
    assert result["status"] == "CLOSED"
    assert pf.win_count == 1
    assert pf.loss_count == 0
    assert len(pf.open_trades) == 0


def test_close_trade_loss():
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    tid = pf.open_trade(
        symbol="BTCUSDT", asset_class="crypto", action="BUY",
        entry_price=100.0, position_size_usd=100.0,
        stop_loss=95.0, take_profit_1=110.0, take_profit_2=None,
        strategy_name="test", signal_id="s1",
    )
    result = pf.close_trade(tid, exit_price=95.0)
    assert result["realized_pnl"] < 0
    assert pf.win_count == 0
    assert pf.loss_count == 1


def test_close_trade_sell_direction():
    """SELL trade profits when price goes down."""
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    tid = pf.open_trade(
        symbol="BTCUSDT", asset_class="crypto", action="SELL",
        entry_price=100.0, position_size_usd=100.0,
        stop_loss=105.0, take_profit_1=90.0, take_profit_2=None,
        strategy_name="test", signal_id="s1",
    )
    result = pf.close_trade(tid, exit_price=90.0)
    assert result["realized_pnl"] > 0


def test_unrealized_pnl():
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    pf.open_trade(
        symbol="BTCUSDT", asset_class="crypto", action="BUY",
        entry_price=100.0, position_size_usd=100.0,
        stop_loss=95.0, take_profit_1=110.0, take_profit_2=None,
        strategy_name="test", signal_id="s1",
    )
    unrealized = pf.unrealized_pnl({"BTCUSDT": 105.0})
    assert unrealized > 0


def test_skip_trade_if_insufficient_balance():
    pf = PaperPortfolio(initial_capital=50.0, bot_name="TestBot")
    tid = pf.open_trade(
        symbol="BTCUSDT", asset_class="crypto", action="BUY",
        entry_price=100.0, position_size_usd=200.0,
        stop_loss=95.0, take_profit_1=110.0, take_profit_2=None,
        strategy_name="test", signal_id="s1",
    )
    assert tid is None
    assert len(pf.open_trades) == 0


def test_metrics():
    pf = PaperPortfolio(initial_capital=1667.0, bot_name="TestBot")
    # Open and close a winning trade
    tid = pf.open_trade(
        symbol="BTCUSDT", asset_class="crypto", action="BUY",
        entry_price=100.0, position_size_usd=100.0,
        stop_loss=95.0, take_profit_1=110.0, take_profit_2=None,
        strategy_name="test", signal_id="s1",
    )
    pf.close_trade(tid, exit_price=110.0)
    metrics = pf.get_metrics()
    assert metrics["win_rate"] == 1.0
    assert metrics["trade_count"] == 1
    assert metrics["total_pnl"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'libs.paper_trading.portfolio'`

- [ ] **Step 3: Implement PaperPortfolio**

```python
# libs/paper_trading/portfolio.py
"""PaperPortfolio — virtual balance, trade open/close, metrics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


FEE_RATE = 0.001  # 0.1% per side


@dataclass
class VirtualTrade:
    id: str
    bot_name: str
    symbol: str
    asset_class: str
    action: str  # BUY or SELL
    entry_price: float
    position_size_usd: float
    fees_paid: float
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    strategy_name: str
    signal_id: str
    opened_at: datetime


class PaperPortfolio:
    """Manages virtual balance and trades for a single bot."""

    def __init__(self, initial_capital: float, bot_name: str) -> None:
        self._initial_capital = initial_capital
        self._bot_name = bot_name
        self._balance = initial_capital
        self._open: dict[str, VirtualTrade] = {}
        self._closed: list[dict] = []
        self._returns: list[float] = []  # per-trade return % for Sharpe calc
        self._peak_balance = initial_capital
        self._max_drawdown = 0.0
        self._win_count = 0
        self._loss_count = 0
        self._total_pnl = 0.0

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def open_trades(self) -> list[VirtualTrade]:
        return list(self._open.values())

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def win_count(self) -> int:
        return self._win_count

    @property
    def loss_count(self) -> int:
        return self._loss_count

    def open_trade(
        self,
        symbol: str,
        asset_class: str,
        action: str,
        entry_price: float,
        position_size_usd: float,
        stop_loss: float | None,
        take_profit_1: float | None,
        take_profit_2: float | None,
        strategy_name: str,
        signal_id: str,
    ) -> str | None:
        """Open a virtual trade. Returns trade ID, or None if insufficient balance."""
        entry_fee = position_size_usd * FEE_RATE
        total_cost = position_size_usd + entry_fee
        if total_cost > self._balance:
            return None

        trade_id = str(uuid4())
        self._balance -= total_cost
        trade = VirtualTrade(
            id=trade_id,
            bot_name=self._bot_name,
            symbol=symbol,
            asset_class=asset_class,
            action=action,
            entry_price=entry_price,
            position_size_usd=position_size_usd,
            fees_paid=entry_fee,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            strategy_name=strategy_name,
            signal_id=signal_id,
            opened_at=datetime.now(timezone.utc),
        )
        self._open[trade_id] = trade
        return trade_id

    def close_trade(
        self, trade_id: str, exit_price: float, status: str = "CLOSED"
    ) -> dict:
        """Close a virtual trade. Returns result dict."""
        trade = self._open.pop(trade_id)
        exit_fee = trade.position_size_usd * FEE_RATE
        total_fees = trade.fees_paid + exit_fee

        # Calculate P&L
        units = trade.position_size_usd / trade.entry_price
        if trade.action == "BUY":
            raw_pnl = (exit_price - trade.entry_price) * units
        else:  # SELL
            raw_pnl = (trade.entry_price - exit_price) * units

        realized_pnl = raw_pnl - total_fees
        pnl_pct = (realized_pnl / trade.position_size_usd) * 100
        now = datetime.now(timezone.utc)
        hold_seconds = int((now - trade.opened_at).total_seconds())

        # Update balance and stats
        self._balance += trade.position_size_usd + raw_pnl - exit_fee
        self._total_pnl += realized_pnl
        self._returns.append(pnl_pct)

        if realized_pnl >= 0:
            self._win_count += 1
        else:
            self._loss_count += 1

        # Track drawdown
        current_equity = self._balance + self.unrealized_pnl({})
        if current_equity > self._peak_balance:
            self._peak_balance = current_equity
        dd = (self._peak_balance - current_equity) / self._peak_balance if self._peak_balance > 0 else 0
        self._max_drawdown = max(self._max_drawdown, dd)

        result = {
            "trade_id": trade.id,
            "bot_name": trade.bot_name,
            "symbol": trade.symbol,
            "asset_class": trade.asset_class,
            "action": trade.action,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "position_size_usd": trade.position_size_usd,
            "fees": round(total_fees, 4),
            "realized_pnl": round(realized_pnl, 4),
            "pnl_pct": round(pnl_pct, 4),
            "hold_duration_seconds": hold_seconds,
            "strategy_name": trade.strategy_name,
            "signal_id": trade.signal_id,
            "opened_at": trade.opened_at.isoformat(),
            "closed_at": now.isoformat(),
            "status": status,
        }
        self._closed.append(result)
        return result

    def unrealized_pnl(self, live_prices: dict[str, float]) -> float:
        """Calculate total unrealized P&L across open trades."""
        total = 0.0
        for trade in self._open.values():
            price = live_prices.get(trade.symbol)
            if price is None:
                continue
            units = trade.position_size_usd / trade.entry_price
            if trade.action == "BUY":
                total += (price - trade.entry_price) * units
            else:
                total += (trade.entry_price - price) * units
        return total

    def get_metrics(self) -> dict:
        """Return bot performance metrics."""
        total_trades = self._win_count + self._loss_count
        win_rate = self._win_count / total_trades if total_trades > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~250 trading days)
        sharpe = 0.0
        if len(self._returns) >= 2:
            mean_r = sum(self._returns) / len(self._returns)
            std_r = (sum((r - mean_r) ** 2 for r in self._returns) / (len(self._returns) - 1)) ** 0.5
            if std_r > 0:
                sharpe = (mean_r / std_r) * math.sqrt(min(len(self._returns), 250))

        # Profit factor
        gross_wins = sum(r for r in self._returns if r > 0)
        gross_losses = abs(sum(r for r in self._returns if r < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

        return {
            "bot_name": self._bot_name,
            "balance": round(self._balance, 2),
            "initial_capital": self._initial_capital,
            "total_pnl": round(self._total_pnl, 2),
            "pnl_pct": round((self._total_pnl / self._initial_capital) * 100, 2) if self._initial_capital else 0.0,
            "win_count": self._win_count,
            "loss_count": self._loss_count,
            "trade_count": total_trades,
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(self._max_drawdown, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
            "open_positions": len(self._open),
        }

    @property
    def closed_trades(self) -> list[dict]:
        return list(self._closed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_portfolio.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add libs/paper_trading/portfolio.py tests/unit/test_paper_trading/test_portfolio.py
git commit -m "feat(paper): add PaperPortfolio with trade open/close, P&L, and metrics"
```

---

## Task 3: CapitalAllocator — Kelly Sizing + Rebalance

**Files:**
- Create: `libs/paper_trading/allocator.py`
- Create: `tests/unit/test_paper_trading/test_allocator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_paper_trading/test_allocator.py
"""Tests for Kelly criterion sizing and capital rebalance."""
from __future__ import annotations

import pytest
from libs.paper_trading.allocator import CapitalAllocator


def test_kelly_fraction_positive_edge():
    alloc = CapitalAllocator()
    # 60% win rate, avg win = 2x avg loss → kelly = 0.6 - 0.4/2 = 0.4
    frac = alloc.kelly_fraction(win_rate=0.6, avg_win_loss_ratio=2.0)
    assert frac == pytest.approx(0.4)


def test_kelly_fraction_no_edge():
    alloc = CapitalAllocator()
    # 40% win rate, avg win = 1x avg loss → kelly = 0.4 - 0.6/1 = -0.2
    frac = alloc.kelly_fraction(win_rate=0.4, avg_win_loss_ratio=1.0)
    assert frac < 0


def test_kelly_fraction_capped():
    alloc = CapitalAllocator()
    # Extreme edge → capped at 0.25
    frac = alloc.kelly_fraction(win_rate=0.9, avg_win_loss_ratio=5.0)
    assert frac <= 0.25


def test_position_size_cold_start():
    alloc = CapitalAllocator()
    size = alloc.position_size(
        bot_capital=1667.0, trade_count=5, win_rate=0.0, avg_win_loss_ratio=0.0
    )
    # Cold start: 1% of capital
    assert size == pytest.approx(16.67, abs=0.01)


def test_position_size_learning_phase():
    alloc = CapitalAllocator()
    size = alloc.position_size(
        bot_capital=1667.0, trade_count=20, win_rate=0.6, avg_win_loss_ratio=2.0
    )
    # Learning: quarter-Kelly → 0.4 * 0.25 * 1667 = 166.7
    assert 0 < size < 1667 * 0.25


def test_position_size_full_phase():
    alloc = CapitalAllocator()
    size = alloc.position_size(
        bot_capital=1667.0, trade_count=50, win_rate=0.6, avg_win_loss_ratio=2.0
    )
    # Full: half-Kelly → 0.4 * 0.5 * 1667 = 333.4
    assert 0 < size < 1667 * 0.5


def test_position_size_negative_kelly_returns_zero():
    alloc = CapitalAllocator()
    size = alloc.position_size(
        bot_capital=1667.0, trade_count=50, win_rate=0.3, avg_win_loss_ratio=1.0
    )
    assert size == 0.0


def test_position_size_minimum():
    alloc = CapitalAllocator()
    size = alloc.position_size(
        bot_capital=500.0, trade_count=5, win_rate=0.0, avg_win_loss_ratio=0.0
    )
    # 1% of 500 = 5 → below $10 minimum
    assert size == 0.0


def test_rebalance_allocations():
    alloc = CapitalAllocator()
    bot_sharpes = {
        "Bot1": 1.5,
        "Bot2": 1.2,
        "Bot3": 0.8,
        "Bot4": 0.5,
        "Bot5": 0.2,
        "Bot6": -0.1,
    }
    allocations = alloc.rebalance(total_capital=10000.0, bot_sharpes=bot_sharpes)
    assert sum(allocations.values()) == pytest.approx(10000.0, abs=1.0)
    # Best bot gets most
    assert allocations["Bot1"] > allocations["Bot6"]


def test_rebalance_paused_bot():
    alloc = CapitalAllocator()
    bot_sharpes = {
        "Bot1": 1.5,
        "Bot2": 1.2,
        "Bot3": 0.8,
        "Bot4": 0.5,
        "Bot5": 0.2,
        "Bot6": -0.1,
    }
    allocations = alloc.rebalance(
        total_capital=10000.0, bot_sharpes=bot_sharpes, paused_bots={"Bot6"}
    )
    assert allocations["Bot6"] == 0.0
    assert sum(allocations.values()) == pytest.approx(10000.0, abs=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_allocator.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CapitalAllocator**

```python
# libs/paper_trading/allocator.py
"""Capital allocation — Kelly criterion sizing + weekly rebalance."""
from __future__ import annotations

MIN_TRADE_SIZE = 10.0
MAX_KELLY_FRACTION = 0.25
MAX_SINGLE_TRADE_PCT = 0.05  # 5% of bot capital

# Rebalance weights by rank (1st through 6th)
REBALANCE_WEIGHTS = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]


class CapitalAllocator:
    """Kelly criterion position sizing and Darwinian capital rebalance."""

    def kelly_fraction(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """Calculate raw Kelly fraction. Can be negative (no edge)."""
        if avg_win_loss_ratio <= 0:
            return -1.0
        loss_rate = 1.0 - win_rate
        raw = win_rate - (loss_rate / avg_win_loss_ratio)
        return min(raw, MAX_KELLY_FRACTION)

    def position_size(
        self,
        bot_capital: float,
        trade_count: int,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> float:
        """Calculate position size in USD based on phase and Kelly."""
        # Phase 1: Cold Start (trades 1-10) — fixed 1%
        if trade_count < 10:
            size = bot_capital * 0.01
            return size if size >= MIN_TRADE_SIZE else 0.0

        kelly = self.kelly_fraction(win_rate, avg_win_loss_ratio)
        if kelly <= 0:
            return 0.0

        # Phase 2: Learning (trades 11-30) — quarter-Kelly
        if trade_count < 30:
            size = kelly * 0.25 * bot_capital
        # Phase 3: Full (trades 31+) — half-Kelly
        else:
            size = kelly * 0.5 * bot_capital

        # Guardrails
        max_size = bot_capital * MAX_SINGLE_TRADE_PCT
        size = min(size, max_size)
        return size if size >= MIN_TRADE_SIZE else 0.0

    def rebalance(
        self,
        total_capital: float,
        bot_sharpes: dict[str, float],
        paused_bots: set[str] | None = None,
    ) -> dict[str, float]:
        """Rebalance capital across bots by Sharpe ratio ranking."""
        paused = paused_bots or set()
        active_bots = [b for b in bot_sharpes if b not in paused]
        paused_list = [b for b in bot_sharpes if b in paused]

        # Sort active bots by Sharpe descending
        ranked = sorted(active_bots, key=lambda b: bot_sharpes[b], reverse=True)

        # Assign weights — redistribute paused capital proportionally
        weights = REBALANCE_WEIGHTS[: len(ranked)]
        if not weights:
            return {b: 0.0 for b in bot_sharpes}

        weight_sum = sum(weights)
        allocations: dict[str, float] = {}
        for i, bot in enumerate(ranked):
            allocations[bot] = round((weights[i] / weight_sum) * total_capital, 2)

        for bot in paused_list:
            allocations[bot] = 0.0

        # Fix rounding — assign remainder to top bot
        remainder = total_capital - sum(allocations.values())
        if ranked:
            allocations[ranked[0]] = round(allocations[ranked[0]] + remainder, 2)

        return allocations
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_allocator.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add libs/paper_trading/allocator.py tests/unit/test_paper_trading/test_allocator.py
git commit -m "feat(paper): add CapitalAllocator with Kelly sizing and Darwinian rebalance"
```

---

## Task 4: BotAgent Base Class

**Files:**
- Create: `libs/paper_trading/bot_agent.py`
- Create: `tests/unit/test_paper_trading/test_bot_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_paper_trading/test_bot_agent.py
"""Tests for BotAgent base class."""
from __future__ import annotations

from unittest.mock import MagicMock
from libs.paper_trading.bot_agent import BotAgent
from libs.core.models.domain import (
    SignalAction, SignalOutput, AssetClass, Timeframe, MarketRegime,
    ConfluenceBreakdown,
)


def _make_signal(**overrides) -> SignalOutput:
    """Helper to create a test signal."""
    from uuid import uuid4
    defaults = dict(
        signal_id=uuid4(),
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M15,
        action=SignalAction.BUY,
        confidence=0.75,
        entry_zone_low=67000.0,
        entry_zone_high=67200.0,
        stop_loss=66000.0,
        take_profit_1=69000.0,
        take_profit_2=71000.0,
        estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP,
        strategy_name="ema_crossover",
        patterns_detected=["hammer"],
        confluence=ConfluenceBreakdown(
            patterns_score=0.7, indicators_score=0.6, structure_score=0.5,
            volume_score=0.5, regime_score=0.8, higher_tf_score=0.7,
            final_score=0.68,
        ),
        explanation="test",
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


class ConcreteBotForTest(BotAgent):
    """Minimal concrete bot for testing base class."""
    NAME = "TestBot"
    def should_take_signal(self, signal: SignalOutput) -> bool:
        return signal.confidence >= 0.6


def test_bot_name():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    assert bot.name == "TestBot"


def test_bot_takes_signal():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    sig = _make_signal(confidence=0.75)
    assert bot.should_take_signal(sig) is True


def test_bot_skips_no_trade():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    sig = _make_signal(action=SignalAction.NO_TRADE)
    result = bot.on_signal(sig)
    assert result is None


def test_bot_opens_trade_on_valid_signal():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    sig = _make_signal(confidence=0.75)
    result = bot.on_signal(sig)
    assert result is not None
    assert result["trade_id"] is not None
    assert bot.portfolio.balance < 1667.0


def test_bot_skips_low_confidence():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    sig = _make_signal(confidence=0.3)
    result = bot.on_signal(sig)
    assert result is None


def test_bot_check_exits_tp1():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    sig = _make_signal(action=SignalAction.BUY, entry_zone_low=100.0,
                       entry_zone_high=100.0, stop_loss=95.0,
                       take_profit_1=110.0, take_profit_2=None)
    bot.on_signal(sig)
    assert len(bot.portfolio.open_trades) == 1
    results = bot.check_exits({"BTCUSDT": 110.0})
    assert len(results) == 1
    assert results[0]["realized_pnl"] > 0


def test_bot_check_exits_sl():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    sig = _make_signal(action=SignalAction.BUY, entry_zone_low=100.0,
                       entry_zone_high=100.0, stop_loss=95.0,
                       take_profit_1=110.0, take_profit_2=None)
    bot.on_signal(sig)
    results = bot.check_exits({"BTCUSDT": 94.0})
    assert len(results) == 1
    assert results[0]["realized_pnl"] < 0


def test_bot_stats():
    bot = ConcreteBotForTest(initial_capital=1667.0)
    stats = bot.get_stats()
    assert stats["bot_name"] == "TestBot"
    assert stats["phase"] == "cold_start"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_bot_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BotAgent**

```python
# libs/paper_trading/bot_agent.py
"""BotAgent — base class for all paper trading bots."""
from __future__ import annotations

from abc import ABC, abstractmethod

from libs.core.logging.logger import get_logger
from libs.core.models.domain import SignalAction, SignalOutput
from libs.paper_trading.allocator import CapitalAllocator
from libs.paper_trading.portfolio import PaperPortfolio

log = get_logger(__name__)


class BotAgent(ABC):
    """Base class for hedge fund bots. Subclasses implement should_take_signal."""

    NAME: str = "BaseBot"

    def __init__(self, initial_capital: float = 1667.0) -> None:
        self._portfolio = PaperPortfolio(initial_capital=initial_capital, bot_name=self.NAME)
        self._allocator = CapitalAllocator()
        self._is_paused = False
        self._avg_win: float = 0.0
        self._avg_loss: float = 0.0
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def portfolio(self) -> PaperPortfolio:
        return self._portfolio

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False

    @abstractmethod
    def should_take_signal(self, signal: SignalOutput) -> bool:
        """Return True if this bot's strategy matches the signal."""

    def _target_exit(self) -> str:
        """Which TP to target. Override in subclasses."""
        return "tp1"

    def on_signal(self, signal: SignalOutput) -> dict | None:
        """Process a signal. Returns trade info dict or None if skipped."""
        if self._is_paused:
            return None
        if signal.action == SignalAction.NO_TRADE:
            return None
        if not self.should_take_signal(signal):
            return None

        # Calculate position size
        metrics = self._portfolio.get_metrics()
        size = self._allocator.position_size(
            bot_capital=self._portfolio.balance,
            trade_count=metrics["trade_count"],
            win_rate=metrics["win_rate"],
            avg_win_loss_ratio=self._avg_win / self._avg_loss if self._avg_loss > 0 else 2.0,
        )
        if size <= 0:
            return None

        entry = (signal.entry_zone_low + signal.entry_zone_high) / 2.0
        trade_id = self._portfolio.open_trade(
            symbol=signal.symbol,
            asset_class=signal.asset_class.value,
            action=signal.action.value,
            entry_price=entry,
            position_size_usd=size,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            strategy_name=signal.strategy_name,
            signal_id=str(signal.signal_id),
        )
        if trade_id is None:
            return None

        log.info(
            "paper_trade_opened",
            bot=self.NAME,
            symbol=signal.symbol,
            action=signal.action.value,
            size=round(size, 2),
            entry=entry,
        )
        return {"trade_id": trade_id, "bot": self.NAME, "size": size}

    def check_exits(self, live_prices: dict[str, float]) -> list[dict]:
        """Check open trades against live prices. Close those that hit TP/SL."""
        results: list[dict] = []
        for trade in list(self._portfolio.open_trades):
            price = live_prices.get(trade.symbol)
            if price is None:
                continue

            exit_price: float | None = None
            status = "CLOSED"

            if trade.action == "BUY":
                if price <= (trade.stop_loss or 0):
                    exit_price = trade.stop_loss
                elif price >= (trade.take_profit_1 or float("inf")):
                    if self._target_exit() == "tp2" and trade.take_profit_2:
                        if price >= trade.take_profit_2:
                            exit_price = trade.take_profit_2
                        # else hold — not hit TP2 yet
                    else:
                        exit_price = trade.take_profit_1
            else:  # SELL
                if price >= (trade.stop_loss or float("inf")):
                    exit_price = trade.stop_loss
                elif price <= (trade.take_profit_1 or 0):
                    if self._target_exit() == "tp2" and trade.take_profit_2:
                        if price <= trade.take_profit_2:
                            exit_price = trade.take_profit_2
                    else:
                        exit_price = trade.take_profit_1

            if exit_price is not None:
                result = self._portfolio.close_trade(trade.id, exit_price, status)
                self._update_running_stats(result)
                results.append(result)
                log.info(
                    "paper_trade_closed",
                    bot=self.NAME,
                    symbol=trade.symbol,
                    pnl=result["realized_pnl"],
                )
        return results

    def force_close_all(self, live_prices: dict[str, float]) -> list[dict]:
        """Force-close all open trades at current price."""
        results: list[dict] = []
        for trade in list(self._portfolio.open_trades):
            price = live_prices.get(trade.symbol)
            if price is None:
                continue
            result = self._portfolio.close_trade(trade.id, price, "FORCE_CLOSED")
            self._update_running_stats(result)
            results.append(result)
        return results

    def _update_running_stats(self, result: dict) -> None:
        """Update running average win/loss for Kelly calculation."""
        pnl = result["realized_pnl"]
        if pnl >= 0:
            n = self._portfolio.win_count
            self._avg_win = ((self._avg_win * (n - 1)) + abs(pnl)) / n if n > 0 else abs(pnl)
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            n = self._portfolio.loss_count
            self._avg_loss = ((self._avg_loss * (n - 1)) + abs(pnl)) / n if n > 0 else abs(pnl)
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def get_stats(self) -> dict:
        """Return bot stats for dashboard."""
        metrics = self._portfolio.get_metrics()
        trade_count = metrics["trade_count"]
        if trade_count < 10:
            phase = "cold_start"
        elif trade_count < 30:
            phase = "learning"
        else:
            phase = "full"

        kelly = self._allocator.kelly_fraction(
            metrics["win_rate"],
            self._avg_win / self._avg_loss if self._avg_loss > 0 else 2.0,
        )
        if kelly < 0:
            phase = "paused"

        metrics["phase"] = phase
        metrics["kelly_fraction"] = round(kelly, 4)
        metrics["is_paused"] = self._is_paused
        metrics["consecutive_wins"] = self._consecutive_wins
        metrics["consecutive_losses"] = self._consecutive_losses
        return metrics

    def set_capital(self, new_capital: float) -> None:
        """Adjust bot capital after rebalance (cash portion only)."""
        self._portfolio._initial_capital = new_capital
        # Adjust balance by the difference (only free cash, not tied in positions)
        open_value = sum(t.position_size_usd for t in self._portfolio.open_trades)
        self._portfolio._balance = new_capital - open_value
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_bot_agent.py -v`
Expected: 9 passed

- [ ] **Step 5: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: All pass (538+ existing + new)

- [ ] **Step 6: Commit**

```bash
git add libs/paper_trading/bot_agent.py tests/unit/test_paper_trading/test_bot_agent.py
git commit -m "feat(paper): add BotAgent base class with Kelly sizing and TP/SL exit logic"
```

---

## Task 5: Six Bot Implementations

**Files:**
- Create: `libs/paper_trading/bots/__init__.py`
- Create: `libs/paper_trading/bots/momentum.py`
- Create: `libs/paper_trading/bots/reversal.py`
- Create: `libs/paper_trading/bots/mean_reversion.py`
- Create: `libs/paper_trading/bots/scalper.py`
- Create: `libs/paper_trading/bots/swing.py`
- Create: `libs/paper_trading/bots/adaptive.py`
- Create: `tests/unit/test_paper_trading/test_bots.py`
- Create: `tests/unit/test_paper_trading/test_adaptive.py`

- [ ] **Step 1: Write failing tests for the 5 standard bots**

```python
# tests/unit/test_paper_trading/test_bots.py
"""Tests for all 5 standard bot implementations."""
from __future__ import annotations

from uuid import uuid4
from libs.core.models.domain import (
    SignalAction, SignalOutput, AssetClass, Timeframe, MarketRegime,
    ConfluenceBreakdown,
)


def _sig(**overrides) -> SignalOutput:
    defaults = dict(
        signal_id=uuid4(), symbol="BTCUSDT", asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M15, action=SignalAction.BUY, confidence=0.75,
        entry_zone_low=100.0, entry_zone_high=100.0, stop_loss=95.0,
        take_profit_1=110.0, take_profit_2=120.0, estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP, strategy_name="ema_crossover",
        patterns_detected=["hammer"], explanation="test",
        confluence=ConfluenceBreakdown(
            patterns_score=0.7, indicators_score=0.6, structure_score=0.5,
            volume_score=0.5, regime_score=0.8, higher_tf_score=0.7, final_score=0.68,
        ),
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


# ── MomentumBot ──────────────────────────────────────────────────────────────

def test_momentum_takes_trending_ema():
    from libs.paper_trading.bots.momentum import MomentumBot
    bot = MomentumBot(initial_capital=1667.0)
    sig = _sig(strategy_name="ema_crossover", market_regime=MarketRegime.TRENDING_UP, timeframe=Timeframe.M15)
    assert bot.should_take_signal(sig) is True


def test_momentum_skips_ranging():
    from libs.paper_trading.bots.momentum import MomentumBot
    bot = MomentumBot(initial_capital=1667.0)
    sig = _sig(market_regime=MarketRegime.RANGING_LOW_VOL)
    assert bot.should_take_signal(sig) is False


# ── ReversalBot ──────────────────────────────────────────────────────────────

def test_reversal_takes_hammer():
    from libs.paper_trading.bots.reversal import ReversalBot
    bot = ReversalBot(initial_capital=1667.0)
    sig = _sig(strategy_name="hammer_reversal", patterns_detected=["hammer", "pin_bar"])
    assert bot.should_take_signal(sig) is True


def test_reversal_skips_choppy():
    from libs.paper_trading.bots.reversal import ReversalBot
    bot = ReversalBot(initial_capital=1667.0)
    sig = _sig(strategy_name="hammer_reversal", market_regime=MarketRegime.RANGING_HIGH_VOL)
    assert bot.should_take_signal(sig) is False


# ── MeanReversionBot ─────────────────────────────────────────────────────────

def test_meanrev_takes_ranging():
    from libs.paper_trading.bots.mean_reversion import MeanReversionBot
    bot = MeanReversionBot(initial_capital=1667.0)
    sig = _sig(strategy_name="pullback_continuation", market_regime=MarketRegime.RANGING_LOW_VOL)
    assert bot.should_take_signal(sig) is True


def test_meanrev_skips_trending():
    from libs.paper_trading.bots.mean_reversion import MeanReversionBot
    bot = MeanReversionBot(initial_capital=1667.0)
    sig = _sig(strategy_name="pullback_continuation", market_regime=MarketRegime.TRENDING_UP)
    assert bot.should_take_signal(sig) is False


# ── ScalperBot ───────────────────────────────────────────────────────────────

def test_scalper_takes_short_tf():
    from libs.paper_trading.bots.scalper import ScalperBot
    bot = ScalperBot(initial_capital=1667.0)
    sig = _sig(timeframe=Timeframe.M5, estimated_risk_reward=2.0)
    assert bot.should_take_signal(sig) is True


def test_scalper_skips_long_tf():
    from libs.paper_trading.bots.scalper import ScalperBot
    bot = ScalperBot(initial_capital=1667.0)
    sig = _sig(timeframe=Timeframe.H1)
    assert bot.should_take_signal(sig) is False


def test_scalper_skips_low_rr():
    from libs.paper_trading.bots.scalper import ScalperBot
    bot = ScalperBot(initial_capital=1667.0)
    sig = _sig(timeframe=Timeframe.M5, estimated_risk_reward=1.2)
    assert bot.should_take_signal(sig) is False


# ── SwingBot ─────────────────────────────────────────────────────────────────

def test_swing_takes_high_tf():
    from libs.paper_trading.bots.swing import SwingBot
    bot = SwingBot(initial_capital=1667.0)
    sig = _sig(timeframe=Timeframe.H1, estimated_risk_reward=3.0)
    assert bot.should_take_signal(sig) is True


def test_swing_skips_short_tf():
    from libs.paper_trading.bots.swing import SwingBot
    bot = SwingBot(initial_capital=1667.0)
    sig = _sig(timeframe=Timeframe.M5)
    assert bot.should_take_signal(sig) is False


def test_swing_skips_low_rr():
    from libs.paper_trading.bots.swing import SwingBot
    bot = SwingBot(initial_capital=1667.0)
    sig = _sig(timeframe=Timeframe.H4, estimated_risk_reward=1.5)
    assert bot.should_take_signal(sig) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_bots.py -v`
Expected: FAIL

- [ ] **Step 3: Implement bots/__init__.py and all 5 standard bots**

```python
# libs/paper_trading/bots/__init__.py
"""Bot registry."""
from __future__ import annotations

from libs.paper_trading.bots.momentum import MomentumBot
from libs.paper_trading.bots.reversal import ReversalBot
from libs.paper_trading.bots.mean_reversion import MeanReversionBot
from libs.paper_trading.bots.scalper import ScalperBot
from libs.paper_trading.bots.swing import SwingBot
from libs.paper_trading.bots.adaptive import AdaptiveBot

ALL_BOTS = [MomentumBot, ReversalBot, MeanReversionBot, ScalperBot, SwingBot, AdaptiveBot]
```

```python
# libs/paper_trading/bots/momentum.py
"""MomentumBot — trends, breakouts, EMA crossovers."""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

MOMENTUM_STRATEGIES = {"ema_crossover", "resistance_breakout"}
TRENDING_REGIMES = {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT}
ALLOWED_TFS = {Timeframe.M15, Timeframe.M30, Timeframe.H1}


class MomentumBot(BotAgent):
    NAME = "MomentumBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.market_regime not in TRENDING_REGIMES:
            return False
        if signal.timeframe not in ALLOWED_TFS:
            return False
        return signal.strategy_name in MOMENTUM_STRATEGIES or signal.confidence >= 0.7
```

```python
# libs/paper_trading/bots/reversal.py
"""ReversalBot — reversal patterns at key levels."""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

REVERSAL_STRATEGIES = {"hammer_reversal", "shooting_star_reversal", "support_breakdown"}
REVERSAL_PATTERNS = {
    "hammer", "inverted_hammer", "shooting_star", "hanging_man",
    "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star",
    "pin_bar", "rejection_candle", "tweezer_top", "tweezer_bottom",
}
CHOPPY = {MarketRegime.RANGING_HIGH_VOL}


class ReversalBot(BotAgent):
    NAME = "ReversalBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.market_regime in CHOPPY:
            return False
        if signal.timeframe not in {Timeframe.M15, Timeframe.H1}:
            return False
        has_reversal_pattern = bool(set(signal.patterns_detected or []) & REVERSAL_PATTERNS)
        return signal.strategy_name in REVERSAL_STRATEGIES and has_reversal_pattern
```

```python
# libs/paper_trading/bots/mean_reversion.py
"""MeanReversionBot — pullbacks in ranging markets."""
from __future__ import annotations

from libs.core.models.domain import MarketRegime, SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

PULLBACK_STRATEGIES = {"pullback_continuation", "pullback_bear_continuation"}
RANGING_REGIMES = {MarketRegime.RANGING_LOW_VOL, MarketRegime.RANGING_HIGH_VOL}


class MeanReversionBot(BotAgent):
    NAME = "MeanReversionBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.market_regime not in RANGING_REGIMES:
            return False
        if signal.timeframe not in {Timeframe.M15, Timeframe.M30}:
            return False
        return signal.strategy_name in PULLBACK_STRATEGIES
```

```python
# libs/paper_trading/bots/scalper.py
"""ScalperBot — short timeframes, tight stops, TP1-only."""
from __future__ import annotations

from libs.core.models.domain import SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

SHORT_TFS = {Timeframe.M1, Timeframe.M5}
MIN_RR = 1.5


class ScalperBot(BotAgent):
    NAME = "ScalperBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.timeframe not in SHORT_TFS:
            return False
        if signal.estimated_risk_reward < MIN_RR:
            return False
        return True

    def _target_exit(self) -> str:
        return "tp1"
```

```python
# libs/paper_trading/bots/swing.py
"""SwingBot — higher timeframes, wider stops, holds for TP2."""
from __future__ import annotations

from libs.core.models.domain import SignalOutput, Timeframe
from libs.paper_trading.bot_agent import BotAgent

SWING_TFS = {Timeframe.H1, Timeframe.H4}
MIN_RR = 2.5


class SwingBot(BotAgent):
    NAME = "SwingBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        if signal.timeframe not in SWING_TFS:
            return False
        if signal.estimated_risk_reward < MIN_RR:
            return False
        return True

    def _target_exit(self) -> str:
        return "tp2"
```

- [ ] **Step 4: Implement AdaptiveBot**

```python
# libs/paper_trading/bots/adaptive.py
"""AdaptiveBot — self-learning bot that evolves from losses."""
from __future__ import annotations

import json

from libs.core.logging.logger import get_logger
from libs.core.models.domain import SignalOutput
from libs.paper_trading.bot_agent import BotAgent

log = get_logger(__name__)

# Baseline filters
BASE_CONFIDENCE = 0.60
BASE_MIN_RR = 1.5


class AdaptiveBot(BotAgent):
    NAME = "AdaptiveBot"

    def __init__(self, initial_capital: float = 1667.0) -> None:
        super().__init__(initial_capital)
        # Adaptive filter state
        self._strategy_confidence: dict[str, float] = {}  # strategy → min confidence
        self._blocked_regimes: dict[str, int] = {}  # regime → trades remaining to block
        self._blocked_symbols: dict[str, int] = {}  # symbol → trades remaining to block
        self._require_htf_alignment: bool = False
        self._min_rr: float = BASE_MIN_RR
        self._loss_log: list[dict] = []  # recent losses for pattern analysis
        self._trade_counter: int = 0
        self._adaptation_log: list[dict] = []

    def should_take_signal(self, signal: SignalOutput) -> bool:
        # Check blocked symbols
        if signal.symbol in self._blocked_symbols and self._blocked_symbols[signal.symbol] > 0:
            return False

        # Check blocked regimes
        regime_key = signal.market_regime.value
        if regime_key in self._blocked_regimes and self._blocked_regimes[regime_key] > 0:
            return False

        # Check confidence threshold (per-strategy or base)
        min_conf = self._strategy_confidence.get(signal.strategy_name, BASE_CONFIDENCE)
        if signal.confidence < min_conf:
            return False

        # Check R:R
        if signal.estimated_risk_reward < self._min_rr:
            return False

        # Check HTF alignment if required
        if self._require_htf_alignment:
            htf = getattr(signal, "higher_tf_bias", None) or getattr(signal.confluence, "higher_tf_score", 0)
            if isinstance(htf, (int, float)) and htf < 0.5:
                return False

        return True

    def _target_exit(self) -> str:
        return "tp2" if self._portfolio.get_metrics().get("win_rate", 0) > 0.6 else "tp1"

    def on_signal(self, signal: SignalOutput) -> dict | None:
        result = super().on_signal(signal)
        if result is not None:
            self._trade_counter += 1
            # Decrement blocked counters
            for k in list(self._blocked_regimes):
                self._blocked_regimes[k] = max(0, self._blocked_regimes[k] - 1)
            for k in list(self._blocked_symbols):
                self._blocked_symbols[k] = max(0, self._blocked_symbols[k] - 1)
        return result

    def on_trade_closed(self, result: dict, signal: SignalOutput) -> None:
        """Called after a trade closes. Triggers adaptation on loss."""
        self._update_running_stats(result)

        if result["realized_pnl"] < 0:
            self._loss_log.append({
                "strategy": signal.strategy_name,
                "regime": signal.market_regime.value,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe.value,
                "rr": signal.estimated_risk_reward,
                "patterns": signal.patterns_detected or [],
                "htf_score": getattr(signal.confluence, "higher_tf_score", 0),
            })
            self._adapt_from_losses()

        # Recovery: 5 consecutive wins → relax one filter
        if self._consecutive_wins >= 5:
            self._relax_one_filter()
            self._consecutive_wins = 0

    def _adapt_from_losses(self) -> None:
        """Analyze recent losses and tighten filters."""
        if self._trade_counter < 10:
            return  # Safety: min 10 samples

        recent = self._loss_log[-30:]  # Rolling 30-trade window

        # Count losses by strategy
        strategy_losses: dict[str, int] = {}
        regime_losses: dict[str, int] = {}
        symbol_losses: dict[str, int] = {}

        for loss in recent:
            strategy_losses[loss["strategy"]] = strategy_losses.get(loss["strategy"], 0) + 1
            regime_losses[loss["regime"]] = regime_losses.get(loss["regime"], 0) + 1
            symbol_losses[loss["symbol"]] = symbol_losses.get(loss["symbol"], 0) + 1

        # Adapt: 3+ losses on same strategy → raise threshold
        for strat, count in strategy_losses.items():
            if count >= 3:
                old = self._strategy_confidence.get(strat, BASE_CONFIDENCE)
                new = min(old + 0.05, 0.95)
                if new != old:
                    self._strategy_confidence[strat] = new
                    self._log_adaptation(f"Raised {strat} confidence to {new:.0%} after {count} losses")

        # Adapt: 3+ losses in same regime → block for 10 trades
        for regime, count in regime_losses.items():
            if count >= 3:
                self._blocked_regimes[regime] = 10
                self._log_adaptation(f"Blocked regime {regime} for 10 trades after {count} losses")

        # Adapt: 3+ losses on same symbol → block for 10 trades
        for symbol, count in symbol_losses.items():
            if count >= 3:
                self._blocked_symbols[symbol] = 10
                self._log_adaptation(f"Blocked {symbol} for 10 trades after {count} losses")

        # Adapt: loss with low HTF score → require alignment
        htf_losses = [l for l in recent if l.get("htf_score", 1) < 0.5]
        if len(htf_losses) >= 2 and not self._require_htf_alignment:
            self._require_htf_alignment = True
            self._log_adaptation("Now requiring HTF alignment after HTF-conflict losses")

        # Adapt: loss with R:R < 2 → raise min R:R
        low_rr_losses = [l for l in recent if l.get("rr", 99) < 2.0]
        if len(low_rr_losses) >= 3 and self._min_rr < 3.0:
            self._min_rr = round(self._min_rr + 0.2, 1)
            self._log_adaptation(f"Raised min R:R to {self._min_rr} after {len(low_rr_losses)} low-RR losses")

    def _relax_one_filter(self) -> None:
        """After 5 consecutive wins, relax the least impactful filter."""
        # Priority: blocked symbols → blocked regimes → strategy confidence → min R:R
        for symbol in list(self._blocked_symbols):
            if self._blocked_symbols[symbol] > 0:
                self._blocked_symbols[symbol] = 0
                self._log_adaptation(f"Relaxed: unblocked {symbol} after 5 wins")
                return

        for regime in list(self._blocked_regimes):
            if self._blocked_regimes[regime] > 0:
                self._blocked_regimes[regime] = 0
                self._log_adaptation(f"Relaxed: unblocked regime {regime} after 5 wins")
                return

        for strat in list(self._strategy_confidence):
            if self._strategy_confidence[strat] > BASE_CONFIDENCE:
                old = self._strategy_confidence[strat]
                self._strategy_confidence[strat] = max(BASE_CONFIDENCE, old - 0.05)
                self._log_adaptation(f"Relaxed: lowered {strat} confidence from {old:.0%} to {self._strategy_confidence[strat]:.0%}")
                return

        if self._min_rr > BASE_MIN_RR:
            old = self._min_rr
            self._min_rr = max(BASE_MIN_RR, self._min_rr - 0.2)
            self._log_adaptation(f"Relaxed: lowered min R:R from {old} to {self._min_rr}")

    def _log_adaptation(self, message: str) -> None:
        self._adaptation_log.append({"trade_num": self._trade_counter, "message": message})
        log.info("adaptive_bot_filter_change", message=message, trade_num=self._trade_counter)

    def get_adaptive_filters(self) -> dict:
        """Return current filter state as dict (for persistence)."""
        return {
            "strategy_confidence": dict(self._strategy_confidence),
            "blocked_regimes": dict(self._blocked_regimes),
            "blocked_symbols": dict(self._blocked_symbols),
            "require_htf_alignment": self._require_htf_alignment,
            "min_rr": self._min_rr,
            "adaptation_log": self._adaptation_log[-20:],  # Last 20 entries
        }

    def load_adaptive_filters(self, data: dict) -> None:
        """Restore filter state from persistence."""
        self._strategy_confidence = data.get("strategy_confidence", {})
        self._blocked_regimes = data.get("blocked_regimes", {})
        self._blocked_symbols = data.get("blocked_symbols", {})
        self._require_htf_alignment = data.get("require_htf_alignment", False)
        self._min_rr = data.get("min_rr", BASE_MIN_RR)
```

- [ ] **Step 5: Write AdaptiveBot tests**

```python
# tests/unit/test_paper_trading/test_adaptive.py
"""Tests for AdaptiveBot self-learning behavior."""
from __future__ import annotations

from uuid import uuid4
from libs.core.models.domain import (
    SignalAction, SignalOutput, AssetClass, Timeframe, MarketRegime,
    ConfluenceBreakdown,
)
from libs.paper_trading.bots.adaptive import AdaptiveBot, BASE_CONFIDENCE, BASE_MIN_RR


def _sig(**overrides) -> SignalOutput:
    defaults = dict(
        signal_id=uuid4(), symbol="BTCUSDT", asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M15, action=SignalAction.BUY, confidence=0.75,
        entry_zone_low=100.0, entry_zone_high=100.0, stop_loss=95.0,
        take_profit_1=110.0, take_profit_2=120.0, estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP, strategy_name="ema_crossover",
        patterns_detected=["hammer"], explanation="test",
        confluence=ConfluenceBreakdown(
            patterns_score=0.7, indicators_score=0.6, structure_score=0.5,
            volume_score=0.5, regime_score=0.8, higher_tf_score=0.7, final_score=0.68,
        ),
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


def test_adaptive_takes_above_baseline():
    bot = AdaptiveBot(initial_capital=1667.0)
    sig = _sig(confidence=0.65)
    assert bot.should_take_signal(sig) is True


def test_adaptive_skips_below_baseline():
    bot = AdaptiveBot(initial_capital=1667.0)
    sig = _sig(confidence=0.55)
    assert bot.should_take_signal(sig) is False


def test_adaptive_raises_strategy_confidence_after_losses():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._trade_counter = 15  # Past min samples

    sig = _sig(strategy_name="bad_strat")
    # Simulate 3 losses on same strategy
    for _ in range(3):
        bot._loss_log.append({
            "strategy": "bad_strat", "regime": "trending_up",
            "symbol": "BTCUSDT", "timeframe": "15m", "rr": 2.5,
            "patterns": [], "htf_score": 0.8,
        })
    bot._adapt_from_losses()

    assert bot._strategy_confidence["bad_strat"] > BASE_CONFIDENCE


def test_adaptive_blocks_regime_after_losses():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._trade_counter = 15

    for _ in range(3):
        bot._loss_log.append({
            "strategy": "ema_crossover", "regime": "ranging_low_vol",
            "symbol": "ETHUSDT", "timeframe": "15m", "rr": 2.5,
            "patterns": [], "htf_score": 0.8,
        })
    bot._adapt_from_losses()

    assert bot._blocked_regimes["ranging_low_vol"] == 10
    sig = _sig(market_regime=MarketRegime.RANGING_LOW_VOL)
    assert bot.should_take_signal(sig) is False


def test_adaptive_blocks_symbol_after_losses():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._trade_counter = 15

    for _ in range(3):
        bot._loss_log.append({
            "strategy": "ema_crossover", "regime": "trending_up",
            "symbol": "SOLUSDT", "timeframe": "15m", "rr": 2.5,
            "patterns": [], "htf_score": 0.8,
        })
    bot._adapt_from_losses()

    sig = _sig(symbol="SOLUSDT")
    assert bot.should_take_signal(sig) is False


def test_adaptive_relaxes_after_wins():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._blocked_symbols["SOLUSDT"] = 5
    bot._consecutive_wins = 5
    bot._relax_one_filter()
    assert bot._blocked_symbols["SOLUSDT"] == 0


def test_adaptive_never_below_baseline():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._strategy_confidence["test"] = BASE_CONFIDENCE
    bot._consecutive_wins = 5
    bot._relax_one_filter()
    assert bot._strategy_confidence.get("test", BASE_CONFIDENCE) >= BASE_CONFIDENCE
    assert bot._min_rr >= BASE_MIN_RR


def test_adaptive_filters_persist():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._strategy_confidence["bad"] = 0.8
    bot._blocked_regimes["choppy"] = 5
    bot._min_rr = 2.0

    data = bot.get_adaptive_filters()
    bot2 = AdaptiveBot(initial_capital=5000.0)
    bot2.load_adaptive_filters(data)

    assert bot2._strategy_confidence["bad"] == 0.8
    assert bot2._blocked_regimes["choppy"] == 5
    assert bot2._min_rr == 2.0


def test_adaptive_no_adapt_before_min_samples():
    bot = AdaptiveBot(initial_capital=5000.0)
    bot._trade_counter = 5  # Below 10
    for _ in range(5):
        bot._loss_log.append({
            "strategy": "bad", "regime": "trending_up",
            "symbol": "BTCUSDT", "timeframe": "15m", "rr": 2.5,
            "patterns": [], "htf_score": 0.8,
        })
    bot._adapt_from_losses()
    assert "bad" not in bot._strategy_confidence
```

- [ ] **Step 6: Run all bot tests**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_bots.py tests/unit/test_paper_trading/test_adaptive.py -v`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add libs/paper_trading/bots/ tests/unit/test_paper_trading/test_bots.py \
  tests/unit/test_paper_trading/test_adaptive.py
git commit -m "feat(paper): add 6 hedge fund bots — Momentum, Reversal, MeanReversion, Scalper, Swing, Adaptive"
```

---

## Task 6: PaperTradingEngine — Orchestrator

**Files:**
- Create: `libs/paper_trading/engine.py`
- Create: `tests/unit/test_paper_trading/test_engine.py`
- Modify: `apps/signal_agent/runner.py` (~line 168, add engine startup)

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_paper_trading/test_engine.py
"""Tests for PaperTradingEngine orchestrator."""
from __future__ import annotations

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from libs.core.models.domain import (
    SignalAction, SignalOutput, AssetClass, Timeframe, MarketRegime,
    ConfluenceBreakdown,
)
from libs.paper_trading.engine import PaperTradingEngine


def _sig(**overrides) -> SignalOutput:
    defaults = dict(
        signal_id=uuid4(), symbol="BTCUSDT", asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M15, action=SignalAction.BUY, confidence=0.75,
        entry_zone_low=100.0, entry_zone_high=100.0, stop_loss=95.0,
        take_profit_1=110.0, take_profit_2=120.0, estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP, strategy_name="ema_crossover",
        patterns_detected=["hammer"], explanation="test",
        confluence=ConfluenceBreakdown(
            patterns_score=0.7, indicators_score=0.6, structure_score=0.5,
            volume_score=0.5, regime_score=0.8, higher_tf_score=0.7, final_score=0.68,
        ),
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


def test_engine_creates_6_bots():
    engine = PaperTradingEngine(initial_capital=10000.0)
    assert len(engine.bots) == 6


def test_engine_distributes_capital():
    engine = PaperTradingEngine(initial_capital=10000.0)
    total = sum(b.portfolio.balance for b in engine.bots)
    assert total == pytest.approx(10000.0, abs=1.0)


def test_engine_dispatch_signal():
    engine = PaperTradingEngine(initial_capital=10000.0)
    sig = _sig()
    results = engine.dispatch_signal(sig)
    # At least MomentumBot should take a trending ema_crossover
    assert isinstance(results, list)


def test_engine_summary():
    engine = PaperTradingEngine(initial_capital=10000.0)
    summary = engine.get_summary(live_prices={})
    assert summary["total_balance"] == pytest.approx(10000.0, abs=1.0)
    assert summary["total_pnl"] == pytest.approx(0.0, abs=0.01)
    assert len(summary["bots"]) == 6


def test_engine_check_exits():
    engine = PaperTradingEngine(initial_capital=10000.0)
    sig = _sig()
    engine.dispatch_signal(sig)
    results = engine.check_all_exits({"BTCUSDT": 110.0})
    # Any bots that took the trade should have exits
    assert isinstance(results, list)


def test_engine_pause_resume_bot():
    engine = PaperTradingEngine(initial_capital=10000.0)
    engine.pause_bot("MomentumBot")
    bot = next(b for b in engine.bots if b.name == "MomentumBot")
    assert bot.is_paused is True

    engine.resume_bot("MomentumBot")
    assert bot.is_paused is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PaperTradingEngine**

```python
# libs/paper_trading/engine.py
"""PaperTradingEngine — orchestrates 6 hedge fund bots."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from libs.core.events.bus import EventBus
from libs.core.logging.logger import get_logger
from libs.core.models.domain import SignalOutput
from libs.paper_trading.allocator import CapitalAllocator
from libs.paper_trading.bots import ALL_BOTS
from libs.paper_trading.bot_agent import BotAgent

log = get_logger(__name__)

INITIAL_CAPITAL = 10000.0
EQUITY_SNAPSHOT_INTERVAL = 300  # 5 minutes
REBALANCE_CHECK_INTERVAL = 3600  # 1 hour (checks if Sunday)


class PaperTradingEngine:
    """Orchestrates 6 competing hedge fund bots on virtual capital."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self._initial_capital = initial_capital
        self._allocator = CapitalAllocator()
        per_bot = round(initial_capital / len(ALL_BOTS), 2)
        self._bots: list[BotAgent] = [BotCls(initial_capital=per_bot) for BotCls in ALL_BOTS]
        self._started_at = datetime.now(timezone.utc)
        self._equity_snapshots: list[dict] = []
        self._all_closed_trades: list[dict] = []

    @property
    def bots(self) -> list[BotAgent]:
        return list(self._bots)

    def dispatch_signal(self, signal: SignalOutput) -> list[dict]:
        """Fan signal to all bots. Returns list of trade results."""
        results: list[dict] = []
        for bot in self._bots:
            result = bot.on_signal(signal)
            if result is not None:
                results.append(result)
        return results

    def check_all_exits(self, live_prices: dict[str, float]) -> list[dict]:
        """Check all bots for TP/SL exits."""
        all_results: list[dict] = []
        for bot in self._bots:
            results = bot.check_exits(live_prices)
            all_results.extend(results)
            self._all_closed_trades.extend(results)
        return all_results

    def get_summary(self, live_prices: dict[str, float]) -> dict:
        """Portfolio-wide summary for dashboard."""
        bot_stats = []
        total_balance = 0.0
        total_pnl = 0.0
        total_open = 0

        for bot in self._bots:
            stats = bot.get_stats()
            unrealized = bot.portfolio.unrealized_pnl(live_prices)
            stats["unrealized_pnl"] = round(unrealized, 2)
            stats["effective_balance"] = round(stats["balance"] + unrealized, 2)
            bot_stats.append(stats)
            total_balance += stats["effective_balance"]
            total_pnl += stats["total_pnl"] + unrealized
            total_open += stats["open_positions"]

        # Sort by effective balance descending for leaderboard
        bot_stats.sort(key=lambda s: s["effective_balance"], reverse=True)

        uptime = datetime.now(timezone.utc) - self._started_at
        return {
            "total_balance": round(total_balance, 2),
            "initial_capital": self._initial_capital,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / self._initial_capital) * 100, 2),
            "total_open_positions": total_open,
            "uptime_seconds": int(uptime.total_seconds()),
            "started_at": self._started_at.isoformat(),
            "bots": bot_stats,
        }

    def get_all_open_positions(self) -> list[dict]:
        """All open virtual trades across all bots."""
        positions = []
        for bot in self._bots:
            for trade in bot.portfolio.open_trades:
                positions.append({
                    "bot_name": bot.name,
                    "trade_id": trade.id,
                    "symbol": trade.symbol,
                    "asset_class": trade.asset_class,
                    "action": trade.action,
                    "entry_price": trade.entry_price,
                    "position_size_usd": round(trade.position_size_usd, 2),
                    "stop_loss": trade.stop_loss,
                    "take_profit_1": trade.take_profit_1,
                    "take_profit_2": trade.take_profit_2,
                    "strategy_name": trade.strategy_name,
                    "opened_at": trade.opened_at.isoformat(),
                })
        return positions

    def get_closed_trades(self, limit: int = 100) -> list[dict]:
        """All closed trades, newest first."""
        all_trades = []
        for bot in self._bots:
            all_trades.extend(bot.portfolio.closed_trades)
        all_trades.sort(key=lambda t: t.get("closed_at", ""), reverse=True)
        return all_trades[:limit]

    def get_equity_curve(self) -> list[dict]:
        return list(self._equity_snapshots)

    def snapshot_equity(self, live_prices: dict[str, float]) -> None:
        """Take equity snapshot for charting."""
        bot_balances = {}
        total = 0.0
        for bot in self._bots:
            eff = bot.portfolio.balance + bot.portfolio.unrealized_pnl(live_prices)
            bot_balances[bot.name] = round(eff, 2)
            total += eff
        self._equity_snapshots.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_balance": round(total, 2),
            "bots": bot_balances,
        })

    def pause_bot(self, bot_name: str) -> bool:
        for bot in self._bots:
            if bot.name == bot_name:
                bot.pause()
                return True
        return False

    def resume_bot(self, bot_name: str) -> bool:
        for bot in self._bots:
            if bot.name == bot_name:
                bot.resume()
                return True
        return False

    def reset(self) -> None:
        """Reset everything to fresh $10K start."""
        per_bot = round(self._initial_capital / len(ALL_BOTS), 2)
        self._bots = [BotCls(initial_capital=per_bot) for BotCls in ALL_BOTS]
        self._started_at = datetime.now(timezone.utc)
        self._equity_snapshots.clear()
        self._all_closed_trades.clear()
        log.info("paper_trading_reset", capital=self._initial_capital)

    # ── Async lifecycle ──────────────────────────────────────────────────────

    async def on_signal_event(self, payload: dict) -> None:
        """EventBus handler for signal.generated events."""
        signal = payload.get("signal")
        if signal is None:
            return
        self.dispatch_signal(signal)

    async def start(self) -> None:
        """Subscribe to EventBus and start background loops."""
        EventBus.subscribe(EventBus.SIGNAL_GENERATED, self.on_signal_event)
        log.info("paper_trading_engine_started", bots=len(self._bots), capital=self._initial_capital)

    async def run_exit_check_loop(
        self,
        get_prices: callable,
        interval: int = 30,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Periodically check exits and snapshot equity."""
        tick = 0
        while True:
            if stop_event and stop_event.is_set():
                break
            prices = await get_prices()
            self.check_all_exits(prices)
            tick += interval
            if tick >= EQUITY_SNAPSHOT_INTERVAL:
                self.snapshot_equity(prices)
                tick = 0
            await asyncio.sleep(interval)
```

- [ ] **Step 4: Run tests**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_engine.py -v`
Expected: 6 passed

- [ ] **Step 5: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add libs/paper_trading/engine.py tests/unit/test_paper_trading/test_engine.py
git commit -m "feat(paper): add PaperTradingEngine orchestrator with EventBus integration"
```

---

## Task 7: Wire Engine into SignalRunner

**Files:**
- Modify: `apps/signal_agent/runner.py` (~line 168)

- [ ] **Step 1: Read current runner.py to confirm exact integration point**

Read `apps/signal_agent/runner.py` lines 155–180 to find where PositionWatcher and OutcomeTracker start.

- [ ] **Step 2: Add paper trading engine startup**

After the line `outcome_task = asyncio.create_task(self._outcome_tracker.run_loop())` (line 169), add:

```python
        # Start paper trading engine
        from libs.paper_trading.engine import PaperTradingEngine
        self._paper_engine = PaperTradingEngine()
        await self._paper_engine.start()
```

Also in the `run_all` method, after signals are tracked with watcher/tracker, dispatch to paper engine. After line ~160 where signals are enqueued to outcome tracker:

```python
            # Dispatch to paper trading bots
            if hasattr(self, '_paper_engine'):
                self._paper_engine.dispatch_signal(output)
```

- [ ] **Step 3: Run full test suite to verify no breakage**

Run: `uv run python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add apps/signal_agent/runner.py
git commit -m "feat(paper): wire PaperTradingEngine into SignalRunner"
```

---

## Task 8: Dashboard — /paper Page + API Endpoints

**Files:**
- Create: `apps/dashboard/paper_page.py`
- Modify: `apps/dashboard/server.py` (mount new routes)

- [ ] **Step 1: Create paper_page.py with API endpoints and HTML page**

```python
# apps/dashboard/paper_page.py
"""Paper trading dashboard — single page with H1 money counter + bot leaderboard."""
from __future__ import annotations

import html as html_mod
import json
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from libs.core.logging.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

# Engine reference — set by runner on startup
_engine = None


def set_paper_engine(engine) -> None:
    global _engine
    _engine = engine


# ── JSON APIs ────────────────────────────────────────────────────────────────

@router.get("/api/paper/summary")
async def paper_summary() -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"error": "Paper trading not started"}, status_code=503)
    # Get live prices from position data (best-effort)
    summary = _engine.get_summary(live_prices={})
    return JSONResponse(content=summary)


@router.get("/api/paper/positions")
async def paper_positions() -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"positions": []})
    return JSONResponse(content={"positions": _engine.get_all_open_positions()})


@router.get("/api/paper/trades")
async def paper_trades(limit: int = 100) -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"trades": []})
    limit = min(max(limit, 1), 500)
    return JSONResponse(content={"trades": _engine.get_closed_trades(limit=limit)})


@router.get("/api/paper/equity")
async def paper_equity() -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"curve": []})
    return JSONResponse(content={"curve": _engine.get_equity_curve()})


@router.post("/api/paper/bot/{bot_name}/pause")
async def pause_bot(bot_name: str) -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"error": "Not started"}, status_code=503)
    ok = _engine.pause_bot(bot_name)
    return JSONResponse(content={"paused": ok, "bot": bot_name})


@router.post("/api/paper/bot/{bot_name}/resume")
async def resume_bot(bot_name: str) -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"error": "Not started"}, status_code=503)
    ok = _engine.resume_bot(bot_name)
    return JSONResponse(content={"resumed": ok, "bot": bot_name})


@router.post("/api/paper/reset")
async def reset_paper() -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"error": "Not started"}, status_code=503)
    _engine.reset()
    return JSONResponse(content={"reset": True})


# ── HTML Page ────────────────────────────────────────────────────────────────

@router.get("/paper", response_class=HTMLResponse)
async def paper_dashboard() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Paper Trading — Hedge Fund Bots</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --muted: #8b949e; --text: #c9d1d9; --bright: #f0f6fc;
      --blue: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
    }
    body { font-family: 'Courier New', monospace; background: var(--bg); color: var(--text); min-height: 100vh; font-size: 13px; }

    /* H1 Money Counter */
    .money-hero { text-align: center; padding: 40px 20px 20px; }
    .money-hero h1 { font-size: 3.5rem; font-weight: bold; transition: color 0.3s; }
    .money-hero h1.profit { color: var(--green); }
    .money-hero h1.loss { color: var(--red); }
    .money-hero h1.neutral { color: var(--text); }
    .money-hero .sub { color: var(--muted); font-size: 0.85rem; margin-top: 6px; }
    @keyframes pulse-green { 0%,100% { text-shadow: none; } 50% { text-shadow: 0 0 20px rgba(63,185,80,0.4); } }
    .money-hero h1.profit { animation: pulse-green 2s ease infinite; }

    /* Layout */
    main { padding: 14px 20px; max-width: 1400px; margin: 0 auto; }
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; margin-bottom: 14px; }
    .panel-header { padding: 9px 14px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
    .panel-header h2 { font-size: 0.78rem; color: var(--bright); font-weight: bold; }

    /* Tables */
    table { width: 100%; border-collapse: collapse; font-size: 0.73rem; }
    th { padding: 6px 10px; text-align: left; color: var(--muted); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }
    td { padding: 7px 10px; border-bottom: 1px solid #1c2128; white-space: nowrap; }
    tr:hover td { background: #1c2128; }
    .green { color: var(--green) !important; }
    .red { color: var(--red) !important; }
    .yellow { color: var(--yellow) !important; }
    .blue { color: var(--blue) !important; }
    .empty { padding: 28px; text-align: center; color: var(--muted); font-size: 0.78rem; }

    /* Controls */
    .btn { background: #238636; color: #fff; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.72rem; font-family: inherit; }
    .btn:hover { background: #2ea043; }
    .btn-red { background: #da3633; }
    .btn-red:hover { background: #f85149; }
    .btn-sm { padding: 2px 8px; font-size: 0.66rem; border-radius: 3px; background: #21262d; color: var(--muted); border: 1px solid var(--border); cursor: pointer; font-family: inherit; }
    .btn-sm:hover { border-color: var(--blue); color: var(--blue); }

    /* Equity chart placeholder */
    .equity-chart { height: 200px; background: #0a0e17; border: 1px solid var(--border); border-radius: 4px; display: flex; align-items: center; justify-content: center; color: var(--muted); margin: 10px 14px; }

    /* Navbar */
    nav { background: var(--surface); border-bottom: 1px solid var(--border); padding: 8px 20px; display: flex; align-items: center; gap: 16px; }
    nav a { color: var(--blue); text-decoration: none; font-size: 0.78rem; }
    nav a:hover { text-decoration: underline; }
    nav .title { color: var(--bright); font-weight: bold; font-size: 0.9rem; }

    /* Progress bar */
    .prog-wrap { background: #21262d; border-radius: 3px; height: 5px; width: 80px; display: inline-block; vertical-align: middle; overflow: hidden; }
    .prog-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
  </style>
</head>
<body>

<nav>
  <span class="title">Paper Trading</span>
  <a href="/">← Main Dashboard</a>
  <span style="margin-left:auto;color:var(--muted);font-size:0.7rem">VIRTUAL MONEY — NO REAL TRADES</span>
  <button class="btn-red btn" onclick="resetAll()" style="margin-left:10px">Reset $10K</button>
</nav>

<!-- Giant H1 Money Counter -->
<div class="money-hero">
  <h1 id="money-h1" class="neutral">$10,000.00</h1>
  <div class="sub">
    <span id="money-pnl">+$0.00 / +0.00%</span>
    &nbsp;·&nbsp; Started: $10,000 &nbsp;·&nbsp; Running <span id="uptime">0s</span>
  </div>
</div>

<main>

  <!-- Bot Leaderboard -->
  <div class="panel">
    <div class="panel-header">
      <h2>Bot Leaderboard</h2>
      <button class="btn-sm" onclick="loadSummary()">Refresh</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Bot</th><th>Balance</th><th>P&L</th><th>P&L%</th>
          <th>Win Rate</th><th>Trades</th><th>Sharpe</th><th>Phase</th><th>Actions</th>
        </tr>
      </thead>
      <tbody id="bot-tbody">
        <tr><td colspan="10" class="empty">Loading...</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Equity Curve -->
  <div class="panel">
    <div class="panel-header"><h2>Equity Curve</h2></div>
    <div class="equity-chart" id="equity-chart">Chart loads after first equity snapshot (5 min)</div>
  </div>

  <!-- Open Positions -->
  <div class="panel">
    <div class="panel-header">
      <h2>Open Positions</h2>
      <button class="btn-sm" onclick="loadPositions()">Refresh</button>
    </div>
    <table>
      <thead>
        <tr><th>Bot</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Size</th><th>Stop</th><th>TP1</th><th>TP2</th><th>Strategy</th><th>Since</th></tr>
      </thead>
      <tbody id="pos-tbody">
        <tr><td colspan="10" class="empty">No open positions</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Trade History -->
  <div class="panel">
    <div class="panel-header">
      <h2>Trade History</h2>
      <div style="display:flex;gap:6px;align-items:center">
        <select id="filter-bot" onchange="loadTrades()" style="background:#21262d;border:1px solid var(--border);color:var(--text);padding:2px 6px;border-radius:3px;font-family:inherit;font-size:0.72rem">
          <option value="">All Bots</option>
        </select>
        <button class="btn-sm" onclick="loadTrades()">Refresh</button>
      </div>
    </div>
    <table>
      <thead>
        <tr><th>Time</th><th>Bot</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L%</th><th>Hold</th><th>Strategy</th></tr>
      </thead>
      <tbody id="trades-tbody">
        <tr><td colspan="10" class="empty">No closed trades yet</td></tr>
      </tbody>
    </table>
  </div>

</main>

<script>
let summaryData = null;

function fmt(n, d=2) { return n != null ? parseFloat(n).toFixed(d) : '—'; }
function elapsed(s) {
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm ' + (s%60) + 's';
  const h = Math.floor(s/3600);
  const m = Math.floor((s%3600)/60);
  return h + 'h ' + m + 'm';
}
function holdTime(s) {
  if (!s) return '—';
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}

async function loadSummary() {
  try {
    const r = await fetch('/api/paper/summary');
    const d = await r.json();
    if (d.error) return;
    summaryData = d;

    // H1 Money Counter
    const h1 = document.getElementById('money-h1');
    const pnlEl = document.getElementById('money-pnl');
    h1.textContent = '$' + d.total_balance.toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
    const sign = d.total_pnl >= 0 ? '+' : '';
    pnlEl.textContent = sign + '$' + d.total_pnl.toFixed(2) + ' / ' + sign + d.total_pnl_pct.toFixed(2) + '%';
    h1.className = d.total_pnl > 0 ? 'profit' : d.total_pnl < 0 ? 'loss' : 'neutral';

    document.getElementById('uptime').textContent = elapsed(d.uptime_seconds);

    // Bot Leaderboard
    const tbody = document.getElementById('bot-tbody');
    if (!d.bots.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty">No bots running</td></tr>';
      return;
    }
    tbody.innerHTML = d.bots.map((b, i) => {
      const pnlClr = b.total_pnl >= 0 ? 'green' : 'red';
      const phaseClr = b.phase === 'paused' ? 'red' : b.phase === 'full' ? 'green' : 'yellow';
      const pauseBtn = b.is_paused
        ? `<button class="btn-sm" onclick="resumeBot('${b.bot_name}')">Resume</button>`
        : `<button class="btn-sm" onclick="pauseBot('${b.bot_name}')">Pause</button>`;
      return `<tr>
        <td style="color:var(--yellow);font-weight:bold">#${i+1}</td>
        <td style="font-weight:bold">${b.bot_name}</td>
        <td>$${fmt(b.effective_balance || b.balance)}</td>
        <td class="${pnlClr}">$${fmt(b.total_pnl)}</td>
        <td class="${pnlClr}">${b.pnl_pct >= 0 ? '+' : ''}${fmt(b.pnl_pct)}%</td>
        <td>${(b.win_rate * 100).toFixed(0)}% <span style="color:var(--muted)">(${b.win_count}W/${b.loss_count}L)</span></td>
        <td>${b.trade_count}</td>
        <td>${fmt(b.sharpe_ratio, 3)}</td>
        <td class="${phaseClr}" style="font-size:0.68rem">${b.phase.toUpperCase()}</td>
        <td>${pauseBtn}</td>
      </tr>`;
    }).join('');

    // Populate bot filter dropdown
    const sel = document.getElementById('filter-bot');
    const current = sel.value;
    sel.innerHTML = '<option value="">All Bots</option>' + d.bots.map(b =>
      `<option value="${b.bot_name}" ${b.bot_name===current?'selected':''}>${b.bot_name}</option>`
    ).join('');
  } catch(e) { console.error(e); }
}

async function loadPositions() {
  try {
    const r = await fetch('/api/paper/positions');
    const d = await r.json();
    const tbody = document.getElementById('pos-tbody');
    if (!d.positions.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty">No open positions</td></tr>';
      return;
    }
    tbody.innerHTML = d.positions.map(p => {
      const ac = p.action === 'BUY' ? 'green' : 'red';
      const since = Math.round((Date.now() - new Date(p.opened_at)) / 60000);
      return `<tr>
        <td style="color:var(--muted)">${p.bot_name}</td>
        <td style="font-weight:bold">${p.symbol}</td>
        <td class="${ac}">${p.action === 'BUY' ? '▲ BUY' : '▼ SELL'}</td>
        <td>${fmt(p.entry_price, 4)}</td>
        <td>$${fmt(p.position_size_usd)}</td>
        <td class="red">${fmt(p.stop_loss, 4)}</td>
        <td class="green">${fmt(p.take_profit_1, 4)}</td>
        <td class="blue">${p.take_profit_2 ? fmt(p.take_profit_2, 4) : '—'}</td>
        <td style="color:var(--muted);font-size:0.68rem">${p.strategy_name}</td>
        <td style="color:var(--muted)">${since}m ago</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

async function loadTrades() {
  try {
    const r = await fetch('/api/paper/trades?limit=200');
    const d = await r.json();
    const tbody = document.getElementById('trades-tbody');
    let trades = d.trades || [];
    const botFilter = document.getElementById('filter-bot').value;
    if (botFilter) trades = trades.filter(t => t.bot_name === botFilter);
    if (!trades.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty">No closed trades</td></tr>';
      return;
    }
    tbody.innerHTML = trades.map(t => {
      const pnlClr = t.realized_pnl >= 0 ? 'green' : 'red';
      const ac = t.action === 'BUY' ? 'green' : 'red';
      return `<tr>
        <td style="color:var(--muted)">${new Date(t.closed_at).toLocaleString()}</td>
        <td>${t.bot_name}</td>
        <td style="font-weight:bold">${t.symbol}</td>
        <td class="${ac}">${t.action}</td>
        <td>${fmt(t.entry_price, 4)}</td>
        <td>${fmt(t.exit_price, 4)}</td>
        <td class="${pnlClr}">$${fmt(t.realized_pnl)}</td>
        <td class="${pnlClr}">${t.pnl_pct >= 0 ? '+' : ''}${fmt(t.pnl_pct)}%</td>
        <td>${holdTime(t.hold_duration_seconds)}</td>
        <td style="color:var(--muted);font-size:0.68rem">${t.strategy_name}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}

async function pauseBot(name) { await fetch(`/api/paper/bot/${name}/pause`, {method:'POST'}); loadSummary(); }
async function resumeBot(name) { await fetch(`/api/paper/bot/${name}/resume`, {method:'POST'}); loadSummary(); }
async function resetAll() {
  if (!confirm('Reset all paper trading to $10,000? All history will be lost.')) return;
  await fetch('/api/paper/reset', {method:'POST'});
  loadSummary(); loadPositions(); loadTrades();
}

async function refreshAll() {
  await Promise.all([loadSummary(), loadPositions(), loadTrades()]);
}

// Boot
refreshAll();
setInterval(refreshAll, 5000);  // Refresh every 5s
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
```

- [ ] **Step 2: Mount router in dashboard server.py**

In `apps/dashboard/server.py`, after the `app = FastAPI(...)` line (line 63), add:

```python
from apps.dashboard.paper_page import router as paper_router
app.include_router(paper_router)
```

- [ ] **Step 3: Test manually by starting server**

Run: `uv run python -m apps.dashboard.server`
Open: `http://localhost:8000/paper`
Expected: Page loads with H1 counter showing $10,000.00, empty tables

- [ ] **Step 4: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add apps/dashboard/paper_page.py apps/dashboard/server.py
git commit -m "feat(paper): add /paper dashboard page with H1 money counter, bot leaderboard, trade history"
```

---

## Task 9: Integration Test — Full Flow

**Files:**
- Create: `tests/unit/test_paper_trading/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/unit/test_paper_trading/test_integration.py
"""Integration test — full signal → bot → trade → exit flow."""
from __future__ import annotations

import pytest
from uuid import uuid4

from libs.core.models.domain import (
    SignalAction, SignalOutput, AssetClass, Timeframe, MarketRegime,
    ConfluenceBreakdown,
)
from libs.paper_trading.engine import PaperTradingEngine


def _sig(**overrides) -> SignalOutput:
    defaults = dict(
        signal_id=uuid4(), symbol="BTCUSDT", asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.M15, action=SignalAction.BUY, confidence=0.75,
        entry_zone_low=100.0, entry_zone_high=100.0, stop_loss=95.0,
        take_profit_1=110.0, take_profit_2=120.0, estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP, strategy_name="ema_crossover",
        patterns_detected=["hammer"], explanation="test",
        confluence=ConfluenceBreakdown(
            patterns_score=0.7, indicators_score=0.6, structure_score=0.5,
            volume_score=0.5, regime_score=0.8, higher_tf_score=0.7, final_score=0.68,
        ),
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


def test_full_flow_signal_to_profit():
    """Signal → at least one bot opens trade → price hits TP1 → profit."""
    engine = PaperTradingEngine(initial_capital=10000.0)
    initial_summary = engine.get_summary({})

    # Dispatch trending BUY signal
    sig = _sig()
    trades = engine.dispatch_signal(sig)
    assert len(trades) >= 1, "At least MomentumBot should take a trending BUY"

    # Verify positions exist
    positions = engine.get_all_open_positions()
    assert len(positions) >= 1

    # Price hits TP1
    exits = engine.check_all_exits({"BTCUSDT": 110.0})
    assert len(exits) >= 1

    # Summary shows profit
    summary = engine.get_summary({})
    assert summary["total_pnl"] > 0
    assert summary["total_balance"] > initial_summary["total_balance"]


def test_full_flow_signal_to_loss():
    """Signal → bot opens trade → price hits SL → loss."""
    engine = PaperTradingEngine(initial_capital=10000.0)
    sig = _sig()
    engine.dispatch_signal(sig)

    exits = engine.check_all_exits({"BTCUSDT": 94.0})  # Below SL
    assert len(exits) >= 1

    summary = engine.get_summary({})
    assert summary["total_pnl"] < 0


def test_full_flow_multiple_signals():
    """Multiple signals across different styles → multiple bots active."""
    engine = PaperTradingEngine(initial_capital=10000.0)

    # Trending signal for MomentumBot
    engine.dispatch_signal(_sig(
        strategy_name="ema_crossover",
        market_regime=MarketRegime.TRENDING_UP,
        timeframe=Timeframe.M15,
    ))

    # Reversal signal for ReversalBot
    engine.dispatch_signal(_sig(
        strategy_name="hammer_reversal",
        market_regime=MarketRegime.TRENDING_DOWN,
        patterns_detected=["hammer", "pin_bar"],
        symbol="ETHUSDT",
    ))

    positions = engine.get_all_open_positions()
    bot_names = {p["bot_name"] for p in positions}
    assert len(bot_names) >= 1, "Multiple bot types should be active"


def test_engine_reset():
    engine = PaperTradingEngine(initial_capital=10000.0)
    engine.dispatch_signal(_sig())
    engine.check_all_exits({"BTCUSDT": 110.0})

    engine.reset()
    summary = engine.get_summary({})
    assert summary["total_balance"] == pytest.approx(10000.0, abs=1.0)
    assert summary["total_pnl"] == pytest.approx(0.0)
    assert len(engine.get_closed_trades()) == 0
```

- [ ] **Step 2: Run integration tests**

Run: `uv run python -m pytest tests/unit/test_paper_trading/test_integration.py -v`
Expected: 4 passed

- [ ] **Step 3: Run full test suite**

Run: `uv run python -m pytest tests/ -x -q`
Expected: All pass (538+ existing + ~40 new)

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_paper_trading/test_integration.py
git commit -m "test(paper): add integration test for full signal → trade → exit flow"
```

---

## Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | DB Models | 3 new | 3 |
| 2 | PaperPortfolio | 1 new | 8 |
| 3 | CapitalAllocator (Kelly + rebalance) | 1 new | 10 |
| 4 | BotAgent base class | 1 new | 9 |
| 5 | 6 bot implementations | 7 new | 22 |
| 6 | PaperTradingEngine | 1 new | 6 |
| 7 | Wire into SignalRunner | 1 modify | 0 (existing tests) |
| 8 | Dashboard /paper page | 2 new/modify | manual |
| 9 | Integration test | 1 new | 4 |
| **Total** | | **~18 files** | **~62 tests** |
