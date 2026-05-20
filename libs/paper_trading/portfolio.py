"""
PaperPortfolio — pure in-memory virtual portfolio for a single bot.

No database interaction. Tracks virtual balance, open/closed trades,
running P&L, and performance metrics.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

FEE_RATE: float = 0.001  # 0.1% per side


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class VirtualTrade:
    """Represents a single open (or recently opened) virtual trade."""

    id: str
    bot_name: str
    symbol: str
    asset_class: str
    action: str                   # "BUY" | "SELL"
    entry_price: float
    position_size_usd: float
    fees_paid: float
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    strategy_name: str
    signal_id: str
    opened_at: datetime


# ── Portfolio class ───────────────────────────────────────────────────────────

class PaperPortfolio:
    """
    Virtual portfolio for a single bot.

    Manages balance, open trades, closed trades, and performance metrics.
    All monetary values are in the account currency (USD by default).
    All datetimes use UTC.
    """

    def __init__(self, initial_capital: float, bot_name: str) -> None:
        self._initial_capital: float = initial_capital
        self._bot_name: str = bot_name
        self._balance: float = initial_capital

        self._open_trades: list[VirtualTrade] = []
        self._closed_trades: list[dict] = []

        self._total_pnl: float = 0.0
        self._win_count: int = 0
        self._loss_count: int = 0

        # Internal tracking for metrics
        self._returns: list[float] = []      # per-trade return %
        self._peak_balance: float = initial_capital
        self._max_drawdown: float = 0.0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def open_trades(self) -> list[VirtualTrade]:
        return list(self._open_trades)

    @property
    def closed_trades(self) -> list[dict]:
        return list(self._closed_trades)

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def win_count(self) -> int:
        return self._win_count

    @property
    def loss_count(self) -> int:
        return self._loss_count

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def open_trade(
        self,
        symbol: str,
        asset_class: str,
        action: str,
        entry_price: float,
        position_size_usd: float,
        stop_loss: Optional[float],
        take_profit_1: Optional[float],
        take_profit_2: Optional[float],
        strategy_name: str,
        signal_id: str,
    ) -> Optional[str]:
        """
        Open a new virtual trade.

        Deducts position_size_usd + entry fee from the balance.

        Returns:
            Trade ID string on success, or None if balance is insufficient.
        """
        entry_fee = position_size_usd * FEE_RATE
        total_cost = position_size_usd + entry_fee

        if total_cost > self._balance:
            return None

        trade_id = str(uuid.uuid4())
        trade = VirtualTrade(
            id=trade_id,
            bot_name=self._bot_name,
            symbol=symbol,
            asset_class=asset_class,
            action=action.upper(),
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

        self._balance -= total_cost
        self._open_trades.append(trade)
        return trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        status: str = "CLOSED",
    ) -> dict:
        """
        Close an open virtual trade.

        Calculates realized P&L, deducts exit fee, updates metrics.

        P&L calculation:
            units = position_size_usd / entry_price
            BUY:  gross_pnl = (exit_price - entry_price) * units
            SELL: gross_pnl = (entry_price - exit_price) * units
            net_pnl = gross_pnl - exit_fee

        Returns:
            Dict with trade details and realized P&L.
        """
        trade = self._find_open_trade(trade_id)
        if trade is None:
            return {"error": f"Trade {trade_id} not found in open trades"}

        units = trade.position_size_usd / trade.entry_price

        if trade.action == "BUY":
            gross_pnl = (exit_price - trade.entry_price) * units
        else:  # SELL
            gross_pnl = (trade.entry_price - exit_price) * units

        exit_fee = trade.position_size_usd * FEE_RATE
        net_pnl = gross_pnl - exit_fee

        # Return position value + P&L to balance
        self._balance += trade.position_size_usd + net_pnl

        # Update win/loss counts
        if net_pnl >= 0:
            self._win_count += 1
        else:
            self._loss_count += 1

        # Track running return % (relative to position size)
        return_pct = net_pnl / trade.position_size_usd * 100.0
        self._returns.append(return_pct)

        # Update total P&L
        self._total_pnl += net_pnl

        # Update peak balance and max drawdown
        if self._balance > self._peak_balance:
            self._peak_balance = self._balance
        else:
            drawdown = (self._peak_balance - self._balance) / self._peak_balance
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

        closed_at = datetime.now(timezone.utc)

        result = {
            "trade_id": trade_id,
            "symbol": trade.symbol,
            "action": trade.action,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "position_size_usd": trade.position_size_usd,
            "units": units,
            "gross_pnl": gross_pnl,
            "exit_fee": exit_fee,
            "pnl": net_pnl,
            "return_pct": return_pct,
            "status": status,
            "opened_at": trade.opened_at,
            "closed_at": closed_at,
            "strategy_name": trade.strategy_name,
            "signal_id": trade.signal_id,
        }

        self._open_trades = [t for t in self._open_trades if t.id != trade_id]
        self._closed_trades.append(result)
        return result

    # ── Unrealized P&L ────────────────────────────────────────────────────────

    def unrealized_pnl(self, live_prices: dict[str, float]) -> float:
        """
        Calculate total unrealized P&L across all open trades.

        Symbols not present in live_prices contribute 0.
        """
        total = 0.0
        for trade in self._open_trades:
            price = live_prices.get(trade.symbol)
            if price is None:
                continue
            units = trade.position_size_usd / trade.entry_price
            if trade.action == "BUY":
                total += (price - trade.entry_price) * units
            else:
                total += (trade.entry_price - price) * units
        return total

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """
        Return a snapshot of portfolio performance metrics.

        Keys: bot_name, balance, initial_capital, total_pnl, pnl_pct,
              win_count, loss_count, trade_count, win_rate, sharpe_ratio
              (annualized), max_drawdown, profit_factor, open_positions.
        """
        trade_count = self._win_count + self._loss_count
        win_rate = self._win_count / trade_count if trade_count > 0 else 0.0
        pnl_pct = self._total_pnl / self._initial_capital * 100.0

        sharpe_ratio = self._compute_sharpe()
        profit_factor = self._compute_profit_factor()

        return {
            "bot_name": self._bot_name,
            "balance": self._balance,
            "initial_capital": self._initial_capital,
            "total_pnl": self._total_pnl,
            "pnl_pct": pnl_pct,
            "win_count": self._win_count,
            "loss_count": self._loss_count,
            "trade_count": trade_count,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self._max_drawdown,
            "profit_factor": profit_factor,
            "open_positions": len(self._open_trades),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_open_trade(self, trade_id: str) -> Optional[VirtualTrade]:
        for trade in self._open_trades:
            if trade.id == trade_id:
                return trade
        return None

    def _compute_sharpe(self, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Annualized Sharpe ratio using per-trade return percentages.

        Returns NaN when there are fewer than 2 trades (std dev undefined).
        """
        if len(self._returns) < 2:
            return float("nan")

        n = len(self._returns)
        mean_r = sum(self._returns) / n
        variance = sum((r - mean_r) ** 2 for r in self._returns) / (n - 1)
        std_r = math.sqrt(variance)

        if std_r == 0.0:
            return float("nan")

        sharpe = (mean_r - risk_free_rate) / std_r * math.sqrt(periods_per_year)
        return sharpe

    def _compute_profit_factor(self) -> float:
        """
        Profit factor = gross_profits / |gross_losses|.

        Returns 0.0 when there are no closed trades.
        Returns inf when there are only winning trades.
        """
        gross_profit = sum(r for r in self._returns if r > 0)
        gross_loss = abs(sum(r for r in self._returns if r < 0))

        if gross_loss == 0.0:
            return 999.0 if gross_profit > 0 else 0.0  # Avoid Inf for JSON safety

        return gross_profit / gross_loss
