"""Unit tests for paper trading SQLAlchemy models."""
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
