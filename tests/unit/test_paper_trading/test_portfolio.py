"""
Unit tests for libs/paper_trading/portfolio.py.

All tests are isolated — PaperPortfolio is constructed directly,
no I/O or external dependencies.
"""
from __future__ import annotations

import math

import pytest

from libs.paper_trading.portfolio import FEE_RATE, PaperPortfolio, VirtualTrade


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_portfolio(initial_capital: float = 10_000.0, bot_name: str = "test_bot") -> PaperPortfolio:
    return PaperPortfolio(initial_capital=initial_capital, bot_name=bot_name)


def open_buy_trade(
    portfolio: PaperPortfolio,
    symbol: str = "AAPL",
    entry_price: float = 100.0,
    position_size_usd: float = 1_000.0,
    stop_loss: float = 95.0,
    take_profit_1: float = 110.0,
    take_profit_2: float = 120.0,
) -> str | None:
    return portfolio.open_trade(
        symbol=symbol,
        asset_class="STOCK",
        action="BUY",
        entry_price=entry_price,
        position_size_usd=position_size_usd,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        strategy_name="test_strategy",
        signal_id="sig-001",
    )


# ── Test Cases ────────────────────────────────────────────────────────────────


class TestInitialState:
    def test_initial_balance(self):
        """Balance equals initial_capital, no trades, zero P&L."""
        p = make_portfolio(initial_capital=5_000.0)
        assert p.balance == 5_000.0
        assert p.open_trades == []
        assert p.closed_trades == []
        assert p.total_pnl == 0.0
        assert p.win_count == 0
        assert p.loss_count == 0


class TestOpenTrade:
    def test_open_trade_reduces_balance(self):
        """Balance is reduced by position size + 0.1% entry fee."""
        p = make_portfolio(initial_capital=10_000.0)
        position_size = 1_000.0
        trade_id = open_buy_trade(p, position_size_usd=position_size)

        assert trade_id is not None
        expected_fee = position_size * FEE_RATE
        expected_balance = 10_000.0 - position_size - expected_fee
        assert math.isclose(p.balance, expected_balance, rel_tol=1e-9)
        assert len(p.open_trades) == 1

    def test_skip_trade_if_insufficient_balance(self):
        """Returns None and does not open trade when balance is insufficient."""
        p = make_portfolio(initial_capital=50.0)
        trade_id = open_buy_trade(p, position_size_usd=200.0)

        assert trade_id is None
        assert p.open_trades == []
        assert p.balance == 50.0

    def test_open_trade_returns_string_id(self):
        """open_trade returns a non-empty string trade ID."""
        p = make_portfolio()
        trade_id = open_buy_trade(p)
        assert isinstance(trade_id, str)
        assert len(trade_id) > 0


class TestCloseTrade:
    def test_close_trade_win(self):
        """BUY at 100, exit at 110 yields positive P&L, win_count incremented."""
        p = make_portfolio(initial_capital=10_000.0)
        position_size = 1_000.0
        entry_price = 100.0
        exit_price = 110.0

        trade_id = open_buy_trade(p, entry_price=entry_price, position_size_usd=position_size)
        result = p.close_trade(trade_id, exit_price=exit_price)

        # units = 1000 / 100 = 10; gross_pnl = (110 - 100) * 10 = 100
        units = position_size / entry_price
        gross_pnl = (exit_price - entry_price) * units
        exit_fee = position_size * FEE_RATE
        expected_pnl = gross_pnl - exit_fee

        assert result["realized_pnl"] > 0
        assert math.isclose(result["realized_pnl"], expected_pnl, rel_tol=1e-9)
        assert p.win_count == 1
        assert p.loss_count == 0
        assert p.total_pnl > 0
        assert len(p.open_trades) == 0
        assert len(p.closed_trades) == 1

    def test_close_trade_loss(self):
        """BUY at 100, exit at 95 yields negative P&L, loss_count incremented."""
        p = make_portfolio(initial_capital=10_000.0)
        position_size = 1_000.0
        entry_price = 100.0
        exit_price = 95.0

        trade_id = open_buy_trade(p, entry_price=entry_price, position_size_usd=position_size)
        result = p.close_trade(trade_id, exit_price=exit_price)

        assert result["realized_pnl"] < 0
        assert p.loss_count == 1
        assert p.win_count == 0

    def test_close_trade_sell_direction(self):
        """SELL at 100, exit at 90 yields positive P&L (short profit)."""
        p = make_portfolio(initial_capital=10_000.0)
        position_size = 1_000.0
        entry_price = 100.0
        exit_price = 90.0

        trade_id = p.open_trade(
            symbol="AAPL",
            asset_class="STOCK",
            action="SELL",
            entry_price=entry_price,
            position_size_usd=position_size,
            stop_loss=105.0,
            take_profit_1=90.0,
            take_profit_2=80.0,
            strategy_name="test_strategy",
            signal_id="sig-002",
        )
        result = p.close_trade(trade_id, exit_price=exit_price)

        # SELL P&L = (entry - exit) * units = (100 - 90) * 10 = 100
        assert result["realized_pnl"] > 0
        assert p.win_count == 1

    def test_close_trade_result_dict_keys(self):
        """Result dict contains expected keys."""
        p = make_portfolio()
        trade_id = open_buy_trade(p)
        result = p.close_trade(trade_id, exit_price=105.0)

        required_keys = {"trade_id", "realized_pnl", "exit_price", "status"}
        assert required_keys.issubset(result.keys())

    def test_close_trade_updates_balance(self):
        """Closing a winning trade increases balance (net positive after fees)."""
        p = make_portfolio(initial_capital=10_000.0)
        balance_after_open = p.balance  # captured after open

        trade_id = open_buy_trade(p, entry_price=100.0, position_size_usd=1_000.0)
        balance_after_open = p.balance
        p.close_trade(trade_id, exit_price=110.0)

        assert p.balance > balance_after_open


class TestUnrealizedPnl:
    def test_unrealized_pnl_positive_for_buy(self):
        """Open BUY at 100, live price 105 => positive unrealized P&L."""
        p = make_portfolio()
        open_buy_trade(p, symbol="AAPL", entry_price=100.0, position_size_usd=1_000.0)
        unrealized = p.unrealized_pnl({"AAPL": 105.0})
        assert unrealized > 0

    def test_unrealized_pnl_zero_no_open_trades(self):
        """No open trades => zero unrealized P&L."""
        p = make_portfolio()
        assert p.unrealized_pnl({"AAPL": 105.0}) == 0.0

    def test_unrealized_pnl_negative_for_losing_buy(self):
        """Open BUY at 100, live price 90 => negative unrealized P&L."""
        p = make_portfolio()
        open_buy_trade(p, symbol="AAPL", entry_price=100.0, position_size_usd=1_000.0)
        unrealized = p.unrealized_pnl({"AAPL": 90.0})
        assert unrealized < 0

    def test_unrealized_pnl_missing_symbol_skipped(self):
        """Symbol not in live_prices is skipped (contributes 0)."""
        p = make_portfolio()
        open_buy_trade(p, symbol="AAPL", entry_price=100.0, position_size_usd=1_000.0)
        # AAPL not in prices dict — should not raise, should return 0
        unrealized = p.unrealized_pnl({"BTCUSDT": 50_000.0})
        assert unrealized == 0.0


class TestMetrics:
    def test_metrics_keys(self):
        """get_metrics() returns all required keys."""
        p = make_portfolio()
        metrics = p.get_metrics()
        required = {
            "bot_name", "balance", "initial_capital", "total_pnl",
            "pnl_pct", "win_count", "loss_count", "trade_count",
            "win_rate", "sharpe_ratio", "max_drawdown", "profit_factor",
            "open_positions",
        }
        assert required.issubset(metrics.keys())

    def test_metrics_after_win(self):
        """Open + close a winning trade; metrics reflect correct values."""
        initial = 10_000.0
        p = make_portfolio(initial_capital=initial, bot_name="my_bot")

        trade_id = open_buy_trade(p, entry_price=100.0, position_size_usd=1_000.0)
        p.close_trade(trade_id, exit_price=110.0)

        metrics = p.get_metrics()

        assert metrics["bot_name"] == "my_bot"
        assert metrics["initial_capital"] == initial
        assert metrics["win_count"] == 1
        assert metrics["loss_count"] == 0
        assert metrics["trade_count"] == 1
        assert metrics["win_rate"] == 1.0
        assert metrics["total_pnl"] > 0
        assert metrics["pnl_pct"] > 0
        assert metrics["open_positions"] == 0
        assert metrics["profit_factor"] > 0
        # sharpe_ratio is finite (or NaN if only 1 trade)
        assert isinstance(metrics["sharpe_ratio"], float)

    def test_metrics_initial_state(self):
        """Fresh portfolio metrics have sane zero values."""
        p = make_portfolio(initial_capital=5_000.0)
        metrics = p.get_metrics()

        assert metrics["trade_count"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["total_pnl"] == 0.0
        assert metrics["open_positions"] == 0
        assert metrics["max_drawdown"] == 0.0
