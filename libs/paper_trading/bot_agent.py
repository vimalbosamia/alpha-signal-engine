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

from datetime import datetime, timezone

from libs.core.models.domain import SignalAction, SignalOutput, TradingMode
from libs.paper_trading.allocator import CapitalAllocator
from libs.paper_trading.market_mode_validator import validate_signal
from libs.paper_trading.portfolio import PaperPortfolio
from libs.paper_trading.shared_memory import get_shared_memory

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

        # Skip if already holding a position on this symbol
        open_symbols = {t.symbol for t in self._portfolio.open_trades}
        if signal.symbol in open_symbols:
            return None

        # Check shared loss memory — avoid patterns that failed for ANY bot
        memory = get_shared_memory()
        regime_str = signal.market_regime.value if hasattr(signal.market_regime, "value") else str(signal.market_regime)
        avoid, reason = memory.should_avoid(
            symbol=signal.symbol,
            action=signal.action.value,
            strategy=signal.strategy_name,
            regime=regime_str,
        )
        if avoid:
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

        # Extract entry-time bias snapshot
        bias_str = "unknown"
        rsi_val = 0.0
        regime_str = "unknown"
        try:
            bias_str = signal.market_regime.value if hasattr(signal, 'setup_grade') else "unknown"
            if hasattr(signal, 'confluence') and signal.confluence:
                bias_str = "bullish" if signal.action.value == "BUY" else "bearish"
            regime_str = signal.market_regime.value if hasattr(signal.market_regime, 'value') else str(signal.market_regime)
        except Exception:
            pass

        # ── Determine market mode ──
        # Stocks = EQUITY (no spot/futures concept), Crypto = SPOT or FUTURES
        is_crypto = signal.asset_class.value == "crypto"
        if not is_crypto:
            market_mode = "EQUITY"
            leverage = 1.0
        else:
            trading_mode_raw = getattr(signal, 'trading_mode', TradingMode.SPOT)
            if isinstance(trading_mode_raw, TradingMode):
                market_mode = trading_mode_raw.value.upper()
            else:
                market_mode = "FUTURES" if "futures" in str(trading_mode_raw) else "SPOT"
            leverage = 3.0 if market_mode == "FUTURES" else 1.0

        # ── Spot/Futures validation (document 4) ──
        existing_longs = {t.symbol for t in self._portfolio.open_trades if t.direction == "LONG"}
        existing_shorts = {t.symbol for t in self._portfolio.open_trades if t.direction == "SHORT"}

        validation = validate_signal(
            side=signal.action.value,
            symbol=signal.symbol,
            market_mode=market_mode,
            position_size_usd=size,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit_1,
            leverage=leverage,
            existing_long_symbols=existing_longs,
            existing_short_symbols=existing_shorts,
            available_cash=self._portfolio.balance,
        )

        if not validation.approved:
            from libs.core.logging.logger import get_logger
            _log = get_logger(__name__)
            _log.info("signal_rejected",
                       bot=self.NAME, symbol=signal.symbol,
                       reason=validation.reason, market_mode=market_mode)
            return None

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
            entry_bias=bias_str,
            entry_rsi=rsi_val,
            entry_regime=regime_str,
            trading_mode=market_mode.lower(),
            leverage=leverage,
            market_mode=market_mode,
            position_intent=validation.position_intent,
            direction=validation.direction,
            margin_mode=validation.margin_mode,
            notional_size=validation.notional_size,
            liquidation_buffer_percent=validation.liquidation_buffer_percent,
        )

        if trade_id is None:
            return None

        from libs.core.logging.logger import get_logger
        _log = get_logger(__name__)
        _log.info("paper_trade_opened",
                   bot=self.NAME, symbol=signal.symbol, side=signal.action.value,
                   entry_bias=bias_str, regime=regime_str, strategy=signal.strategy_name,
                   confidence=round(signal.confidence, 3), size=round(size, 2),
                   entry_price=entry_price)

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
            hit_management = False
            mgmt_reason = ""

            # Calculate unrealized P&L %
            if trade.action == "BUY":
                unrealized_pct = (price - trade.entry_price) / trade.entry_price * 100
            else:
                unrealized_pct = (trade.entry_price - price) / trade.entry_price * 100

            # Hold duration in minutes
            hold_minutes = (datetime.now(timezone.utc) - trade.opened_at).total_seconds() / 60

            # ── Standard TP/SL check ──
            if trade.action == "BUY":
                if trade.stop_loss is not None and price <= trade.stop_loss:
                    hit_sl = True
                elif take_profit is not None and price >= take_profit:
                    hit_tp = True
            else:  # SELL
                if trade.stop_loss is not None and price >= trade.stop_loss:
                    hit_sl = True
                elif take_profit is not None and price <= take_profit:
                    hit_tp = True

            # ── Active Trade Management ──
            if not hit_sl and not hit_tp:
                # Rule 1: Max hold time — close stale trades
                max_hold = 240  # 4 hours max — gives trades room to develop
                if hold_minutes > max_hold:
                    hit_management = True
                    mgmt_reason = f"Max hold exceeded ({hold_minutes:.0f}m > {max_hold}m)"

                # Rule 2: Trailing stop — if was profitable but now losing
                # If price moved 50%+ toward TP then reversed back past entry → exit
                if not hit_management and take_profit is not None:
                    if trade.action == "BUY":
                        max_progress = (price - trade.entry_price) / (take_profit - trade.entry_price) if take_profit != trade.entry_price else 0
                    else:
                        max_progress = (trade.entry_price - price) / (trade.entry_price - take_profit) if trade.entry_price != take_profit else 0

                # Rule 3: Deteriorating loss — cut losing trades
                # Crypto needs room: -1.5% after 20+ min means trend is against us
                if not hit_management and unrealized_pct < -1.5 and hold_minutes > 20:
                    hit_management = True
                    mgmt_reason = f"Cutting loss: {unrealized_pct:.2f}% after {hold_minutes:.0f}m"

                # Rule 4: Break-even exit — was profitable, now losing after 40min
                if not hit_management and unrealized_pct < -0.3 and hold_minutes > 40:
                    hit_management = True
                    mgmt_reason = f"Break-even exit: {unrealized_pct:.2f}% after {hold_minutes:.0f}m"

                # Rule 5: Sustained adverse movement — consistent directional loss
                if not hit_management:
                    if unrealized_pct < -0.5:  # Meaningful adverse move
                        trade.bias_flip_count += 1
                    elif unrealized_pct > 0:  # In profit — reset counter
                        trade.bias_flip_count = 0
                    else:
                        trade.bias_flip_count = max(0, trade.bias_flip_count - 1)

                    if trade.bias_flip_count >= 30:  # ~5 min of consistent adverse (30 × 10s)
                        hit_management = True
                        mgmt_reason = f"Sustained adverse movement ({trade.bias_flip_count} checks)"

            if hit_sl:
                status = "STOPPED_OUT"
                exit_price = price
            elif hit_tp:
                status = "TAKE_PROFIT"
                exit_price = price
            elif hit_management:
                status = f"MANAGED: {mgmt_reason}"
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

            from libs.core.logging.logger import get_logger
            _log = get_logger(__name__)
            _log.info("paper_trade_closed",
                       bot=self.NAME, symbol=trade.symbol, side=trade.action,
                       entry_bias=getattr(trade, 'entry_bias', '?'),
                       pnl=round(result.get('realized_pnl', 0), 4),
                       status=status, strategy=trade.strategy_name,
                       exit_reason=mgmt_reason or status)

            # Report to shared memory so ALL bots learn
            memory = get_shared_memory()
            if result.get("realized_pnl", 0) < 0:
                memory.record_loss(
                    bot_name=self.NAME,
                    symbol=trade.symbol,
                    action=trade.action,
                    strategy=trade.strategy_name,
                    regime=result.get("regime", "unknown"),
                    patterns=result.get("patterns", []),
                    risk_reward=0.0,
                    loss_amount=abs(result.get("realized_pnl", 0)),
                )
            else:
                memory.record_win(
                    bot_name=self.NAME,
                    symbol=trade.symbol,
                    action=trade.action,
                    strategy=trade.strategy_name,
                    regime=result.get("regime", "unknown"),
                )

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
        pnl = result.get("realized_pnl", result.get("pnl"))
        if pnl is None:
            return
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
