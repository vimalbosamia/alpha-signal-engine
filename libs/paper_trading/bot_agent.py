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
from libs.paper_trading.market_context import MarketContextCollector
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

        # ── Collect market context for self-training ──
        market_ctx = MarketContextCollector.from_signal(signal)

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
            market_context=market_ctx.to_dict(),
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

    # ── Strategy-specific expected move durations (minutes) ──
    _STRATEGY_MAX_HOLD: dict[str, int] = {
        "candle_direction_flip": 30,
        "candle_momentum": 30,
        "resistance_breakout": 60,
        "support_breakdown": 60,
        "volume_breakout": 60,
        "atr_breakout": 60,
        "range_breakout": 60,
        "volatility_squeeze": 45,
        "opening_range_breakout": 45,
        "hammer_reversal": 120,
        "shooting_star_reversal": 120,
        "engulfing_reversal": 120,
        "ema_crossover": 180,
        "macd_crossover": 180,
        "trend_following": 180,
        "momentum_continuation": 120,
        "mtf_alignment": 180,
        "pullback_continuation": 120,
        "break_and_retest": 90,
        "bollinger_mean_reversion": 60,
        "rsi_mean_reversion": 60,
    }
    _DEFAULT_MAX_HOLD: int = 240  # fallback: 4 hours

    def check_exits(self, live_prices: dict[str, float]) -> list[dict]:
        """Check all open trades for exit conditions.

        Exit hierarchy (first match wins):
          1. Standard TP/SL hit
          2. Momentum decay — MFE was good but now gave back >60%
          3. Time-decay — exceeded strategy-specific max hold
          4. Deteriorating loss — -1.5% after 20+ min
          5. Breakout failure — breakout strategy didn't move 0.5% in 15 min
          6. Break-even protection — -0.3% after 40 min
          7. Sustained adverse — 30 consecutive checks below -0.5%

        Also updates MFE/MAE on every check for analytics.
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

            # ── Update MFE/MAE tracking ──
            if unrealized_pct > trade.mfe:
                trade.mfe = unrealized_pct
                trade.mfe_price = price
            if unrealized_pct < trade.mae:
                trade.mae = unrealized_pct
                trade.mae_price = price

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

                # Rule 1: Momentum decay exit — was profitable, gave back >60%
                # If MFE was >0.5% but now gave back most of it → momentum dead
                if not hit_management and trade.mfe > 0.5:
                    giveback = trade.mfe - unrealized_pct
                    if giveback > trade.mfe * 0.6:
                        hit_management = True
                        mgmt_reason = (
                            f"Momentum decay: MFE={trade.mfe:.2f}% → "
                            f"now={unrealized_pct:.2f}% (gave back {giveback:.2f}%)"
                        )

                # Rule 2: Time-decay exit — strategy-specific max hold
                strategy_max = self._STRATEGY_MAX_HOLD.get(
                    trade.strategy_name, self._DEFAULT_MAX_HOLD
                )
                if not hit_management and hold_minutes > strategy_max:
                    hit_management = True
                    mgmt_reason = (
                        f"Time decay: {trade.strategy_name} held "
                        f"{hold_minutes:.0f}m > {strategy_max}m limit"
                    )

                # Rule 3: Deteriorating loss — cut losing trades
                if not hit_management and unrealized_pct < -1.5 and hold_minutes > 20:
                    hit_management = True
                    mgmt_reason = f"Cutting loss: {unrealized_pct:.2f}% after {hold_minutes:.0f}m"

                # Rule 4: Breakout failure — didn't move 0.5% in 15 min
                _BREAKOUT_STRATEGIES = {
                    "resistance_breakout", "support_breakdown", "volume_breakout",
                    "atr_breakout", "range_breakout", "opening_range_breakout",
                }
                if (not hit_management
                        and trade.strategy_name in _BREAKOUT_STRATEGIES
                        and hold_minutes > 15
                        and trade.mfe < 0.5):
                    hit_management = True
                    mgmt_reason = (
                        f"Breakout failure: {trade.strategy_name} only "
                        f"{trade.mfe:.2f}% MFE in {hold_minutes:.0f}m"
                    )

                # Rule 5: Break-even exit — was profitable, now losing after 40min
                if not hit_management and unrealized_pct < -0.3 and hold_minutes > 40:
                    hit_management = True
                    mgmt_reason = f"Break-even exit: {unrealized_pct:.2f}% after {hold_minutes:.0f}m"

                # Rule 6: Sustained adverse movement — consistent directional loss
                if not hit_management:
                    if unrealized_pct < -0.5:
                        trade.bias_flip_count += 1
                    elif unrealized_pct > 0:
                        trade.bias_flip_count = 0
                    else:
                        trade.bias_flip_count = max(0, trade.bias_flip_count - 1)

                    if trade.bias_flip_count >= 30:
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
            trade_regime = result.get("regime", getattr(trade, "entry_regime", "unknown"))
            trade_won = result.get("realized_pnl", 0) >= 0
            trade_pnl = result.get("realized_pnl", 0)

            if trade_pnl < 0:
                memory.record_loss(
                    bot_name=self.NAME,
                    symbol=trade.symbol,
                    action=trade.action,
                    strategy=trade.strategy_name,
                    regime=trade_regime,
                    patterns=result.get("patterns", []),
                    risk_reward=0.0,
                    loss_amount=abs(trade_pnl),
                )
            else:
                memory.record_win(
                    bot_name=self.NAME,
                    symbol=trade.symbol,
                    action=trade.action,
                    strategy=trade.strategy_name,
                    regime=trade_regime,
                )

            # ── Report to pattern scorer (Task 5) ──
            try:
                from libs.learning.pattern_scorer import get_pattern_store
                pattern_store = get_pattern_store()
                trade_patterns = getattr(trade, "entry_patterns", "") or ""
                trade_rr = 0.0
                if trade.stop_loss and trade.entry_price and trade.stop_loss != trade.entry_price:
                    risk = abs(trade.entry_price - trade.stop_loss)
                    if risk > 0 and trade.position_size_usd:
                        reward = trade_pnl / trade.position_size_usd * trade.entry_price
                        trade_rr = reward / risk
                for pname in trade_patterns.split(","):
                    pname = pname.strip()
                    if pname:
                        pattern_store.record(pname, regime=trade_regime, won=trade_won, rr=trade_rr)
            except Exception:
                pass

            # ── Report to self-training coordinator (Task 10) ──
            try:
                from libs.learning.coordinator import get_coordinator
                coordinator = get_coordinator()
                trade_patterns_list = [p.strip() for p in (getattr(trade, "entry_patterns", "") or "").split(",") if p.strip()]
                coordinator.on_trade_close(
                    strategy=trade.strategy_name,
                    won=trade_won,
                    pnl=trade_pnl,
                    rr=trade_rr if 'trade_rr' in dir() else 0.0,
                    confidence=getattr(trade, "entry_confidence", 0.5) or 0.5,
                    patterns=trade_patterns_list,
                    regime=trade_regime,
                )
            except Exception:
                pass

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
