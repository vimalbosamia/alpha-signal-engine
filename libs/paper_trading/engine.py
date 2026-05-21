"""
PaperTradingEngine — orchestrator for the multi-agent paper trading simulator.

Responsibilities:
  - Instantiate all 6 hedge-fund bot subclasses from ALL_BOTS
  - Fan signals to every bot via dispatch_signal()
  - Aggregate portfolio stats across all bots
  - Subscribe to EventBus.SIGNAL_GENERATED
  - Run a periodic exit-check / equity-snapshot loop
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Awaitable

from libs.core.events.bus import EventBus
from libs.core.logging.logger import get_logger
from libs.core.models.domain import SignalOutput
from libs.paper_trading.bot_agent import BotAgent
from libs.paper_trading.bots import ALL_BOTS

log = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

INITIAL_CAPITAL: float = 10_000.0
EQUITY_SNAPSHOT_INTERVAL: int = 300  # seconds — 5 minutes


# ── Engine ─────────────────────────────────────────────────────────────────────

class PaperTradingEngine:
    """
    Orchestrates all paper-trading bots.

    Lifecycle:
      1. Construct with optional initial_capital.
      2. Call await engine.start() to subscribe to EventBus.
      3. Optionally call await engine.run_exit_check_loop(...) in a background task.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self._initial_capital = initial_capital
        self._started_at: datetime = datetime.now(timezone.utc)
        self._equity_snapshots: list[dict] = []
        self._bots: list[BotAgent] = self._create_bots(initial_capital)

        # Restore saved state if available
        from libs.paper_trading.state_persistence import restore_all, has_saved_state
        if has_saved_state():
            restore_all(self)
            log.info("paper_trading_engine_restored", bots=len(self._bots))
        else:
            log.info("paper_trading_engine_init_fresh",
                     bots=len(self._bots), initial_capital=initial_capital)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _create_bots(self, initial_capital: float) -> list[BotAgent]:
        """Instantiate all bots, splitting capital equally."""
        per_bot = initial_capital / len(ALL_BOTS)
        return [BotCls(initial_capital=per_bot) for BotCls in ALL_BOTS]

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def bots(self) -> list[BotAgent]:
        return list(self._bots)

    # ── Signal dispatch ────────────────────────────────────────────────────────

    # ── Exposure limits ──────────────────────────────────────────────────────
    MAX_BOTS_PER_SYMBOL: int = 3       # max bots holding same symbol+direction
    MAX_DIRECTIONAL_PCT: float = 0.65  # max 65% exposure in one direction

    def _directional_exposure(self) -> dict[str, int]:
        """Count open trades by direction across all bots."""
        counts: dict[str, int] = {"BUY": 0, "SELL": 0}
        for bot in self._bots:
            for trade in bot.portfolio.open_trades:
                action = getattr(trade, "action", "BUY")
                key = action if isinstance(action, str) else str(action)
                if "BUY" in key.upper():
                    counts["BUY"] += 1
                else:
                    counts["SELL"] += 1
        return counts

    def _bots_holding_symbol_direction(self, symbol: str, action: str) -> int:
        """Count how many bots already hold this symbol in this direction."""
        count = 0
        for bot in self._bots:
            for trade in bot.portfolio.open_trades:
                t_action = getattr(trade, "action", "")
                t_action_str = t_action if isinstance(t_action, str) else str(t_action)
                if trade.symbol == symbol and action.upper() in t_action_str.upper():
                    count += 1
        return count

    def dispatch_signal(self, signal: SignalOutput) -> list[dict]:
        """Fan signal to bots with exposure guards.

        Guards:
        - Max 3 bots per symbol+direction (prevents herding)
        - Max 65% of total positions in one direction (prevents directional bias)
        """
        action_str = signal.action.value if hasattr(signal.action, "value") else str(signal.action)

        # Guard 1: per-symbol bot limit
        held = self._bots_holding_symbol_direction(signal.symbol, action_str)
        if held >= self.MAX_BOTS_PER_SYMBOL:
            log.info("exposure_guard_symbol_limit",
                     symbol=signal.symbol, action=action_str,
                     held=held, limit=self.MAX_BOTS_PER_SYMBOL)
            return []

        # Guard 2: directional exposure limit
        dir_counts = self._directional_exposure()
        total_open = dir_counts["BUY"] + dir_counts["SELL"]
        if total_open > 5:  # only enforce after portfolio has some positions
            direction_key = "BUY" if "BUY" in action_str.upper() else "SELL"
            dir_pct = dir_counts[direction_key] / total_open
            if dir_pct >= self.MAX_DIRECTIONAL_PCT:
                log.info("exposure_guard_directional_limit",
                         symbol=signal.symbol, action=action_str,
                         direction_pct=round(dir_pct, 2),
                         counts=dir_counts, limit=self.MAX_DIRECTIONAL_PCT)
                return []

        # Guard 3: limit bots that can take this signal
        remaining_slots = self.MAX_BOTS_PER_SYMBOL - held
        results: list[dict] = []
        for bot in self._bots:
            if len(results) >= remaining_slots:
                break
            result = bot.on_signal(signal)
            if result is not None:
                results.append(result)
        return results

    # ── Exit checking ──────────────────────────────────────────────────────────

    def check_all_exits(self, live_prices: dict[str, float]) -> list[dict]:
        """Check every bot for TP/SL exits.

        Returns all closed-trade result dicts across all bots.
        """
        all_closed: list[dict] = []
        for bot in self._bots:
            closed = bot.check_exits(live_prices)
            all_closed.extend(closed)
        # Save state after any trade closes
        if all_closed:
            try:
                from libs.paper_trading.state_persistence import save_all
                save_all(self)
            except Exception:
                pass
        return all_closed

    @staticmethod
    def _sanitize(d: dict) -> dict:
        """Replace NaN/Inf floats with 0.0 for JSON safety."""
        import math
        return {
            k: (0.0 if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
            for k, v in d.items()
        }

    # ── Summary ────────────────────────────────────────────────────────────────

    def get_summary(self, live_prices: dict[str, float]) -> dict:
        """Return a portfolio-wide performance summary.

        Bot entries are sorted by effective_balance descending so the
        best-performing bot appears first.
        """
        total_balance = 0.0
        total_open_positions = 0
        bot_entries: list[dict] = []

        for bot in self._bots:
            stats = bot.get_stats()
            unrealized = bot.portfolio.unrealized_pnl(live_prices)
            effective_balance = bot.portfolio.balance + bot.portfolio.invested_capital + unrealized

            total_balance += effective_balance
            total_open_positions += stats.get("open_positions", 0)

            bot_entries.append(self._sanitize({
                **stats,
                "unrealized_pnl": unrealized,
                "effective_balance": effective_balance,
            }))

        bot_entries.sort(key=lambda b: b["effective_balance"], reverse=True)

        total_pnl = total_balance - self._initial_capital
        total_pnl_pct = (total_pnl / self._initial_capital * 100.0) if self._initial_capital else 0.0
        uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        return {
            "total_balance": total_balance,
            "initial_capital": self._initial_capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_open_positions": total_open_positions,
            "uptime_seconds": uptime,
            "started_at": self._started_at.isoformat(),
            "bots": bot_entries,
        }

    # ── Open positions ─────────────────────────────────────────────────────────

    def get_all_open_positions(self) -> list[dict]:
        """Return all open trades across every bot as plain dicts."""
        positions: list[dict] = []
        for bot in self._bots:
            for trade in bot.portfolio.open_trades:
                positions.append({
                    "bot_name": bot.name,
                    "trade_id": trade.id,
                    "symbol": trade.symbol,
                    "asset_class": trade.asset_class,
                    "action": trade.action,
                    "entry_price": trade.entry_price,
                    "position_size_usd": trade.position_size_usd,
                    "stop_loss": trade.stop_loss,
                    "take_profit_1": trade.take_profit_1,
                    "take_profit_2": trade.take_profit_2,
                    "strategy_name": trade.strategy_name,
                    "opened_at": trade.opened_at.isoformat(),
                    "entry_bias": getattr(trade, 'entry_bias', 'unknown'),
                    "entry_regime": getattr(trade, 'entry_regime', 'unknown'),
                    "trading_mode": getattr(trade, 'trading_mode', 'spot'),
                    "leverage": getattr(trade, 'leverage', 1.0),
                    "liquidation_price": getattr(trade, 'liquidation_price', 0.0),
                    # Document (4) fields
                    "market_mode": getattr(trade, 'market_mode', 'SPOT'),
                    "position_intent": getattr(trade, 'position_intent', 'OPEN_LONG'),
                    "direction": getattr(trade, 'direction', 'LONG'),
                    "margin_mode": getattr(trade, 'margin_mode', None),
                    "notional_size": getattr(trade, 'notional_size', 0.0),
                    "liquidation_buffer_percent": getattr(trade, 'liquidation_buffer_percent', 0.0),
                    "bias_status": getattr(trade, 'bias_status', 'unknown'),
                    "current_bias": getattr(trade, 'current_bias', 'unknown'),
                    "bias_adverse_count": getattr(trade, 'bias_adverse_count', 0),
                })
        return positions

    # ── Closed trades ──────────────────────────────────────────────────────────

    def get_closed_trades(self, limit: int = 100) -> list[dict]:
        """Return closed trades across all bots, newest first, up to limit."""
        all_closed: list[dict] = []
        for bot in self._bots:
            all_closed.extend(bot.portfolio.closed_trades)

        all_closed.sort(
            key=lambda t: str(t.get("closed_at", "")),
            reverse=True,
        )
        return all_closed[:limit]

    # ── Equity curve ───────────────────────────────────────────────────────────

    def get_equity_curve(self) -> list[dict]:
        """Return all recorded equity snapshots in chronological order."""
        return list(self._equity_snapshots)

    def snapshot_equity(self, live_prices: dict[str, float]) -> None:
        """Record an equity snapshot — total balance plus per-bot balances.

        effective = cash_balance + invested_capital + unrealized_pnl
        This ensures positions without live prices still count at entry value.
        """
        per_bot: dict[str, float] = {}
        total = 0.0
        for bot in self._bots:
            unrealized = bot.portfolio.unrealized_pnl(live_prices)
            # balance = remaining cash (after position costs deducted)
            # invested_capital = sum of position_size_usd + fees in open trades
            # unrealized = P&L on positions WITH live prices (0 for unpriced)
            effective = bot.portfolio.balance + bot.portfolio.invested_capital + unrealized
            per_bot[bot.name] = effective
            total += effective

        self._equity_snapshots.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_balance": total,
            "bots": per_bot,
        })
        # Cap at 2000 snapshots to prevent unbounded memory growth
        if len(self._equity_snapshots) > 2000:
            self._equity_snapshots = self._equity_snapshots[-2000:]

        # Auto-save state to disk every equity snapshot
        try:
            from libs.paper_trading.state_persistence import save_all
            save_all(self)
        except Exception:
            pass

    # ── Pause / resume ─────────────────────────────────────────────────────────

    def pause_bot(self, bot_name: str) -> bool:
        """Pause a bot by name.  Returns True if found and paused, False otherwise."""
        for bot in self._bots:
            if bot.name == bot_name:
                bot.pause()
                log.info("bot_paused", bot=bot_name)
                return True
        log.warning("pause_bot_not_found", bot=bot_name)
        return False

    def resume_bot(self, bot_name: str) -> bool:
        """Resume a paused bot by name.  Returns True if found and resumed."""
        for bot in self._bots:
            if bot.name == bot_name:
                bot.resume()
                log.info("bot_resumed", bot=bot_name)
                return True
        log.warning("resume_bot_not_found", bot=bot_name)
        return False

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the engine — recreate all bots with fresh capital and clear snapshots."""
        self._bots = self._create_bots(self._initial_capital)
        self._equity_snapshots = []
        self._started_at = datetime.now(timezone.utc)
        # Clear saved state — fresh start
        try:
            import shutil
            shutil.rmtree("data/paper_state", ignore_errors=True)
        except Exception:
            pass
        log.info("paper_trading_engine_reset", initial_capital=self._initial_capital)

    # ── Async EventBus integration ─────────────────────────────────────────────

    async def on_signal_event(self, payload: dict) -> None:
        """EventBus handler — extract SignalOutput from payload and dispatch."""
        signal: SignalOutput | None = payload.get("signal")
        if signal is None:
            log.warning("on_signal_event_missing_signal", payload_keys=list(payload.keys()))
            return

        results = self.dispatch_signal(signal)
        if results:
            log.info(
                "signal_dispatched",
                signal_id=str(signal.signal_id),
                symbol=signal.symbol,
                trades_opened=len(results),
            )

    async def start(self, bus: EventBus) -> None:
        """Subscribe to raw paper signals (before ML/guard filters)."""
        bus.subscribe("paper.signal.raw", self.on_signal_event)
        log.info("paper_trading_engine_started")

    async def run_exit_check_loop(
        self,
        get_prices: Callable[[], Awaitable[dict[str, float]] | dict[str, float]],
        interval: int = 30,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Periodically check exits and take equity snapshots.

        Args:
            get_prices: Async or sync callable returning {symbol: price} dict.
            interval:   Seconds between each cycle. Default 30.
            stop_event: Optional asyncio.Event; loop exits when set.
        """
        snapshot_elapsed: float = 0.0

        while True:
            if stop_event is not None and stop_event.is_set():
                break

            # Fetch live prices (support both sync and async callables)
            try:
                prices_result = get_prices()
                if asyncio.isfuture(prices_result) or asyncio.iscoroutine(prices_result):
                    live_prices = await prices_result
                else:
                    live_prices = prices_result
            except Exception as exc:
                log.warning("exit_check_loop_price_fetch_error", error=str(exc))
                live_prices = {}

            # Check exits
            try:
                closed = self.check_all_exits(live_prices)
                if closed:
                    log.info("exit_check_closed_trades", count=len(closed))
            except Exception as exc:
                log.warning("exit_check_loop_error", error=str(exc))

            # Snapshot equity every EQUITY_SNAPSHOT_INTERVAL seconds
            snapshot_elapsed += interval
            if snapshot_elapsed >= EQUITY_SNAPSHOT_INTERVAL:
                try:
                    self.snapshot_equity(live_prices)
                except Exception as exc:
                    log.warning("equity_snapshot_error", error=str(exc))
                snapshot_elapsed = 0.0

            await asyncio.sleep(interval)
