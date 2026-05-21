"""
State Persistence — saves/loads paper trading state to disk.

Persists across server restarts:
  - Open trades per bot
  - Closed trades per bot
  - Bot stats (balance, win/loss, returns, peak, drawdown)
  - SharedLossMemory (losses + wins)
  - Equity snapshots
  - Engine metadata (started_at, initial_capital)

Storage: JSON files in data/paper_state/
  - engine.json          — engine metadata + equity snapshots
  - bots/{bot_name}.json — per-bot state (balance, trades, stats)
  - shared_memory.json   — shared loss/win records
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from libs.core.logging.logger import get_logger

log = get_logger(__name__)

STATE_DIR = "data/paper_state"


def _ensure_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(os.path.join(STATE_DIR, "bots"), exist_ok=True)


def _write_json(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ── Save functions ───────────────────────────────────────────────────────────

def save_engine_state(engine) -> None:
    """Save engine metadata + equity snapshots."""
    _ensure_dir()
    data = {
        "initial_capital": engine._initial_capital,
        "started_at": engine._started_at.isoformat(),
        "equity_snapshots": engine._equity_snapshots[-500:],  # keep last 500
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(os.path.join(STATE_DIR, "engine.json"), data)


def save_bot_state(bot) -> None:
    """Save a single bot's portfolio state."""
    _ensure_dir()
    portfolio = bot.portfolio

    # Serialize open trades
    open_trades = []
    for t in portfolio.open_trades:
        trade_dict = {}
        for field_name in t.__dataclass_fields__:
            val = getattr(t, field_name)
            if isinstance(val, datetime):
                trade_dict[field_name] = val.isoformat()
            else:
                trade_dict[field_name] = val
        open_trades.append(trade_dict)

    data = {
        "bot_name": bot.name,
        "balance": portfolio.balance,
        "initial_capital": portfolio._initial_capital,
        "total_pnl": portfolio._total_pnl,
        "win_count": portfolio._win_count,
        "loss_count": portfolio._loss_count,
        "returns": portfolio._returns[-200:],  # keep last 200
        "peak_balance": portfolio._peak_balance,
        "max_drawdown": portfolio._max_drawdown,
        "open_trades": open_trades,
        "closed_trades": portfolio._closed_trades[-500:],  # keep last 500
        # Bot-level stats
        "avg_win": bot._avg_win,
        "avg_loss": bot._avg_loss,
        "consecutive_wins": bot._consecutive_wins,
        "consecutive_losses": bot._consecutive_losses,
        "is_paused": bot._is_paused,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(os.path.join(STATE_DIR, "bots", f"{bot.name}.json"), data)


def save_shared_memory(memory) -> None:
    """Save SharedLossMemory to disk."""
    _ensure_dir()
    losses = []
    for rec in memory._losses[-100:]:
        losses.append({
            "bot_name": rec.bot_name, "symbol": rec.symbol,
            "action": rec.action, "strategy": rec.strategy,
            "regime": rec.regime, "patterns": rec.patterns,
            "risk_reward": rec.risk_reward, "loss_amount": rec.loss_amount,
            "timestamp": rec.timestamp.isoformat(),
        })

    wins = []
    for rec in memory._wins[-100:]:
        wins.append({
            "bot_name": rec.bot_name, "symbol": rec.symbol,
            "action": rec.action, "strategy": rec.strategy,
            "regime": rec.regime,
            "timestamp": rec.timestamp.isoformat(),
        })

    _write_json(os.path.join(STATE_DIR, "shared_memory.json"), {
        "losses": losses,
        "wins": wins,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })


def save_all(engine) -> None:
    """Save everything — call periodically and on shutdown."""
    try:
        save_engine_state(engine)
        for bot in engine._bots:
            save_bot_state(bot)
        from libs.paper_trading.shared_memory import get_shared_memory
        save_shared_memory(get_shared_memory())
        log.info("state_saved", bots=len(engine._bots),
                 path=STATE_DIR)
    except Exception as exc:
        log.warning("state_save_failed", error=str(exc))


# ── Load functions ───────────────────────────────────────────────────────────

def load_engine_state() -> dict | None:
    """Load engine metadata."""
    return _read_json(os.path.join(STATE_DIR, "engine.json"))


def load_bot_state(bot_name: str) -> dict | None:
    """Load a bot's saved state."""
    return _read_json(os.path.join(STATE_DIR, "bots", f"{bot_name}.json"))


def load_shared_memory_state() -> dict | None:
    """Load shared memory state."""
    return _read_json(os.path.join(STATE_DIR, "shared_memory.json"))


def restore_bot(bot, state: dict) -> None:
    """Restore a bot's portfolio from saved state."""
    from libs.paper_trading.portfolio import VirtualTrade

    portfolio = bot.portfolio

    # Restore balance and stats
    portfolio._balance = state.get("balance", portfolio._initial_capital)
    portfolio._total_pnl = state.get("total_pnl", 0.0)
    portfolio._win_count = state.get("win_count", 0)
    portfolio._loss_count = state.get("loss_count", 0)
    portfolio._returns = state.get("returns", [])
    portfolio._peak_balance = state.get("peak_balance", portfolio._initial_capital)
    portfolio._max_drawdown = state.get("max_drawdown", 0.0)

    # Restore closed trades
    portfolio._closed_trades = state.get("closed_trades", [])

    # Restore open trades
    portfolio._open_trades = []
    for td in state.get("open_trades", []):
        try:
            opened_at = datetime.fromisoformat(td["opened_at"]) if isinstance(td["opened_at"], str) else td["opened_at"]
            trade = VirtualTrade(
                id=td["id"],
                bot_name=td["bot_name"],
                symbol=td["symbol"],
                asset_class=td.get("asset_class", "crypto"),
                action=td["action"],
                entry_price=td["entry_price"],
                position_size_usd=td["position_size_usd"],
                fees_paid=td.get("fees_paid", 0),
                stop_loss=td.get("stop_loss"),
                take_profit_1=td.get("take_profit_1"),
                take_profit_2=td.get("take_profit_2"),
                strategy_name=td.get("strategy_name", ""),
                signal_id=td.get("signal_id", ""),
                opened_at=opened_at,
                entry_bias=td.get("entry_bias", "unknown"),
                entry_rsi=td.get("entry_rsi", 0.0),
                entry_regime=td.get("entry_regime", "unknown"),
                trading_mode=td.get("trading_mode", "spot"),
                leverage=td.get("leverage", 1.0),
                liquidation_price=td.get("liquidation_price", 0.0),
                bias_flip_count=td.get("bias_flip_count", 0),
                market_mode=td.get("market_mode", "SPOT"),
                position_intent=td.get("position_intent", "OPEN_LONG"),
                direction=td.get("direction", "LONG"),
                margin_mode=td.get("margin_mode"),
                notional_size=td.get("notional_size", 0.0),
                liquidation_buffer_percent=td.get("liquidation_buffer_percent", 0.0),
                bias_adverse_count=td.get("bias_adverse_count", 0),
                current_bias=td.get("current_bias", "unknown"),
                bias_status=td.get("bias_status", "unknown"),
            )
            portfolio._open_trades.append(trade)
        except Exception as exc:
            log.warning("restore_trade_failed", trade_id=td.get("id"), error=str(exc))

    # Restore bot-level stats
    bot._avg_win = state.get("avg_win", 1.0)
    bot._avg_loss = state.get("avg_loss", 1.0)
    bot._consecutive_wins = state.get("consecutive_wins", 0)
    bot._consecutive_losses = state.get("consecutive_losses", 0)
    bot._is_paused = state.get("is_paused", False)

    log.info("bot_restored", bot=bot.name,
             balance=round(portfolio._balance, 2),
             open_trades=len(portfolio._open_trades),
             closed_trades=len(portfolio._closed_trades),
             wins=portfolio._win_count, losses=portfolio._loss_count)


def restore_shared_memory(memory, state: dict) -> None:
    """Restore SharedLossMemory from saved state."""
    from libs.paper_trading.shared_memory import LossRecord, WinRecord

    for ld in state.get("losses", []):
        try:
            memory._losses.append(LossRecord(
                bot_name=ld["bot_name"], symbol=ld["symbol"],
                action=ld["action"], strategy=ld["strategy"],
                regime=ld["regime"], patterns=ld.get("patterns", []),
                risk_reward=ld.get("risk_reward", 0),
                loss_amount=ld.get("loss_amount", 0),
                timestamp=datetime.fromisoformat(ld["timestamp"]),
            ))
        except Exception:
            pass

    for wd in state.get("wins", []):
        try:
            memory._wins.append(WinRecord(
                bot_name=wd["bot_name"], symbol=wd["symbol"],
                action=wd["action"], strategy=wd["strategy"],
                regime=wd["regime"],
                timestamp=datetime.fromisoformat(wd["timestamp"]),
            ))
        except Exception:
            pass

    log.info("shared_memory_restored",
             losses=len(memory._losses), wins=len(memory._wins))


def restore_all(engine) -> bool:
    """
    Restore full engine state from disk. Returns True if state was found.

    Call after engine.__init__() but before start().
    """
    engine_state = load_engine_state()
    if engine_state is None:
        log.info("no_saved_state", path=STATE_DIR)
        return False

    # Restore engine metadata
    try:
        engine._started_at = datetime.fromisoformat(engine_state["started_at"])
        engine._equity_snapshots = engine_state.get("equity_snapshots", [])
    except Exception:
        pass

    # Restore each bot
    restored_bots = 0
    for bot in engine._bots:
        bot_state = load_bot_state(bot.name)
        if bot_state:
            restore_bot(bot, bot_state)
            restored_bots += 1

    # Restore shared memory
    mem_state = load_shared_memory_state()
    if mem_state:
        from libs.paper_trading.shared_memory import get_shared_memory
        restore_shared_memory(get_shared_memory(), mem_state)

    log.info("state_restored", bots=restored_bots,
             equity_points=len(engine._equity_snapshots),
             saved_at=engine_state.get("saved_at", "unknown"))
    return True


def has_saved_state() -> bool:
    """Check if saved state exists on disk."""
    return os.path.exists(os.path.join(STATE_DIR, "engine.json"))
