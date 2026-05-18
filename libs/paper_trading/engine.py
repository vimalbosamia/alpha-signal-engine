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

        log.info(
            "paper_trading_engine_init",
            bots=len(self._bots),
            initial_capital=initial_capital,
        )

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

    def dispatch_signal(self, signal: SignalOutput) -> list[dict]:
        """Fan signal out to all bots.

        Returns a list of trade-result dicts for every bot that opened a trade.
        Bots that skip the signal return None from on_signal and are excluded.
        """
        results: list[dict] = []
        for bot in self._bots:
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
            effective_balance = bot.portfolio.balance + unrealized

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
                    "bot": bot.name,
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
                    "opened_at": trade.opened_at,
                })
        return positions

    # ── Closed trades ──────────────────────────────────────────────────────────

    def get_closed_trades(self, limit: int = 100) -> list[dict]:
        """Return closed trades across all bots, newest first, up to limit."""
        all_closed: list[dict] = []
        for bot in self._bots:
            all_closed.extend(bot.portfolio.closed_trades)

        all_closed.sort(
            key=lambda t: t.get("closed_at", datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )
        return all_closed[:limit]

    # ── Equity curve ───────────────────────────────────────────────────────────

    def get_equity_curve(self) -> list[dict]:
        """Return all recorded equity snapshots in chronological order."""
        return list(self._equity_snapshots)

    def snapshot_equity(self, live_prices: dict[str, float]) -> None:
        """Record an equity snapshot — total balance plus per-bot balances."""
        per_bot: dict[str, float] = {}
        total = 0.0
        for bot in self._bots:
            unrealized = bot.portfolio.unrealized_pnl(live_prices)
            effective = bot.portfolio.balance + unrealized
            per_bot[bot.name] = effective
            total += effective

        self._equity_snapshots.append({
            "timestamp": datetime.now(timezone.utc),
            "total_balance": total,
            "bots": per_bot,
        })

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
