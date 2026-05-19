"""
BotAgent — abstract base class for all hedge-fund paper-trading bots.

Responsibilities:
  - Own a PaperPortfolio and CapitalAllocator
  - Decide whether to act on a SignalOutput (delegated to subclasses)
  - Size positions via Kelly criterion
  - Monitor open trades and close them at TP or SL
  - Track running win/loss statistics
  - Report phase-aware performance stats
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from libs.core.models.domain import SignalAction, SignalOutput
from libs.paper_trading.allocator import CapitalAllocator
from libs.paper_trading.portfolio import PaperPortfolio

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_CAPITAL: float = 1_667.0
_COLD_START_THRESHOLD: int = 10
_LEARNING_THRESHOLD: int = 30


# ── BotAgent ───────────────────────────────────────────────────────────────────

class BotAgent(ABC):
    """
    Abstract base class for all paper-trading bots.

    Subclasses must:
      - Set a class-level NAME attribute.
      - Implement should_take_signal() with strategy-specific filtering.

    Optionally override:
      - _target_exit() to target tp2 instead of tp1.
    """

    NAME: str = "BaseBot"

    # ── Construction ───────────────────────────────────────────────────────────

    def __init__(self, initial_capital: float = _DEFAULT_CAPITAL) -> None:
        self._portfolio = PaperPortfolio(
            initial_capital=initial_capital,
            bot_name=self.NAME,
        )
        self._allocator = CapitalAllocator()

        self._is_paused: bool = False

        # Running averages — used to compute Kelly fraction dynamically
        self._avg_win: float = 1.0    # average winning trade return (absolute)
        self._avg_loss: float = 1.0   # average losing trade return  (absolute)

        # Streak counters
        self._consecutive_wins: int = 0
        self._consecutive_losses: int = 0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def portfolio(self) -> PaperPortfolio:
        return self._portfolio

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    # ── Pause / resume ─────────────────────────────────────────────────────────

    def pause(self) -> None:
        """Pause the bot — on_signal will skip all signals until resumed."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume the bot from paused state."""
        self._is_paused = False

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def should_take_signal(self, signal: SignalOutput) -> bool:
        """Return True if the bot wants to act on this signal.

        Subclasses implement their own strategy-specific filter here
        (e.g. minimum confidence, required regime, pattern checks, etc.).
        """

    # ── Exit target override ───────────────────────────────────────────────────

    def _target_exit(self) -> str:
        """Return 'tp1' or 'tp2'.

        Override in a subclass to hold through tp1 and target tp2 instead.
        Default is 'tp1'.
        """
        return "tp1"

    # ── Signal handler ─────────────────────────────────────────────────────────

    def on_signal(self, signal: SignalOutput) -> dict | None:
        """Process an incoming signal.

        Returns a dict ``{"trade_id": str, "bot": str, "size": float}`` when a
        trade is opened, or ``None`` when the signal is skipped.

        Skips when:
          - The bot is paused
          - Action is NO_TRADE
          - should_take_signal() returns False
          - Position size computes to 0 (insufficient capital / Kelly ≤ 0)
          - Portfolio cannot open the trade (balance too low)
        """
        if self._is_paused:
            return None

        if signal.action == SignalAction.NO_TRADE:
            return None

        if not self.should_take_signal(signal):
            return None

        metrics = self._portfolio.get_metrics()
        trade_count = metrics["trade_count"]
        win_rate = metrics["win_rate"]

        size = self._allocator.position_size(
            bot_capital=self._portfolio.balance,
            trade_count=trade_count,
            win_rate=win_rate,
            avg_win_loss_ratio=self._avg_win_loss_ratio(),
        )

        if size <= 0:
            return None

        # Use mid-point of entry zone as entry price
        entry_price = (signal.entry_zone_low + signal.entry_zone_high) / 2.0

        trade_id = self._portfolio.open_trade(
            symbol=signal.symbol,
            asset_class=signal.asset_class.value,
            action=signal.action.value,
            entry_price=entry_price,
            position_size_usd=size,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            strategy_name=signal.strategy_name,
            signal_id=str(signal.signal_id),
        )

        if trade_id is None:
            return None

        return {"trade_id": trade_id, "bot": self.NAME, "size": size}

    # ── Exit checking ──────────────────────────────────────────────────────────

    def check_exits(self, live_prices: dict[str, float]) -> list[dict]:
        """Check all open trades and close those hitting TP or SL.

        Exit rules:
          BUY  trade: stop-loss if price <= stop_loss;
                      take-profit if price >= target_price.
          SELL trade: stop-loss if price >= stop_loss;
                      take-profit if price <= target_price.

        When _target_exit() == 'tp2' and tp2 is set, hold through tp1 and
        only close at tp2.  Falls back to tp1 when tp2 is None.

        Returns list of close-result dicts for every trade that was closed.
        """
        closed: list[dict] = []

        for trade in self._portfolio.open_trades:
            price = live_prices.get(trade.symbol)
            if price is None:
                continue

            target_exit = self._target_exit()
            if target_exit == "tp2" and trade.take_profit_2 is not None:
                take_profit = trade.take_profit_2
            else:
                take_profit = trade.take_profit_1

            hit_sl = False
            hit_tp = False
            hit_early_profit = False

            # Calculate unrealized % for early profit exit
            if trade.action == "BUY":
                unrealized_pct = (price - trade.entry_price) / trade.entry_price * 100
            else:
                unrealized_pct = (trade.entry_price - price) / trade.entry_price * 100

            if trade.action == "BUY":
                if trade.stop_loss is not None and price <= trade.stop_loss:
                    hit_sl = True
                elif take_profit is not None and price >= take_profit:
                    hit_tp = True
                elif unrealized_pct >= 0.3:  # Early profit: 1.5%+ gain → close
                    hit_early_profit = True
            else:  # SELL
                if trade.stop_loss is not None and price >= trade.stop_loss:
                    hit_sl = True
                elif take_profit is not None and price <= take_profit:
                    hit_tp = True
                elif unrealized_pct >= 0.3:  # Early profit: 1.5%+ gain → close
                    hit_early_profit = True

            if hit_sl:
                status = "STOPPED_OUT"
                exit_price = price
            elif hit_tp:
                status = "TAKE_PROFIT"
                exit_price = price
            elif hit_early_profit:
                status = "EARLY_PROFIT"
                exit_price = price
            else:
                continue

            result = self._portfolio.close_trade(
                trade_id=trade.id,
                exit_price=exit_price,
                status=status,
            )
            self._update_running_stats(result)
            closed.append(result)

        return closed

    # ── Force-close all ────────────────────────────────────────────────────────

    def force_close_all(self, live_prices: dict[str, float]) -> list[dict]:
        """Force-close all open positions at the current market price.

        Uses the live price when available; falls back to entry price when
        the symbol is not in live_prices.
        """
        closed: list[dict] = []

        for trade in self._portfolio.open_trades:
            exit_price = live_prices.get(trade.symbol, trade.entry_price)
            result = self._portfolio.close_trade(
                trade_id=trade.id,
                exit_price=exit_price,
                status="FORCE_CLOSED",
            )
            self._update_running_stats(result)
            closed.append(result)

        return closed

    # ── Running stats ──────────────────────────────────────────────────────────

    def _update_running_stats(self, result: dict) -> None:
        """Update running averages and consecutive win/loss counters.

        Uses exponential moving average (alpha=0.3) so recent trades are
        weighted more heavily than old ones.
        """
        if "pnl" not in result:
            return

        pnl = result["pnl"]
        alpha = 0.3

        if pnl >= 0:
            self._avg_win = (1 - alpha) * self._avg_win + alpha * max(pnl, 0.01)
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._avg_loss = (1 - alpha) * self._avg_loss + alpha * abs(pnl)
            self._consecutive_losses += 1
            self._consecutive_wins = 0

    def _avg_win_loss_ratio(self) -> float:
        """Return average win / average loss ratio (always > 0)."""
        if self._avg_loss <= 0:
            return 1.0
        return self._avg_win / self._avg_loss

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return a comprehensive performance snapshot.

        Keys include all portfolio metrics plus:
          phase            — cold_start / learning / full / paused
          kelly_fraction   — current Kelly fraction
          is_paused        — whether bot is paused
          consecutive_wins — current winning streak
          consecutive_losses — current losing streak
        """
        metrics = self._portfolio.get_metrics()
        trade_count = metrics["trade_count"]
        win_rate = metrics["win_rate"]
        kelly = self._allocator.kelly_fraction(
            win_rate=win_rate,
            avg_win_loss_ratio=self._avg_win_loss_ratio(),
        )

        if self._is_paused:
            phase = "paused"
        elif trade_count < _COLD_START_THRESHOLD:
            phase = "cold_start"
        elif trade_count < _LEARNING_THRESHOLD:
            phase = "learning"
        elif kelly < 0:
            phase = "paused"
        else:
            phase = "full"

        return {
            **metrics,
            "phase": phase,
            "kelly_fraction": kelly,
            "is_paused": self._is_paused,
            "consecutive_wins": self._consecutive_wins,
            "consecutive_losses": self._consecutive_losses,
        }

    # ── Capital adjustment ─────────────────────────────────────────────────────

    def set_capital(self, new_capital: float) -> None:
        """Adjust the bot's available capital after a Darwinian rebalance.

        This directly sets the portfolio balance to ``new_capital``.
        Existing open trades are not affected.
        """
        # PaperPortfolio exposes _balance directly; update it carefully
        self._portfolio._balance = new_capital
