"""
AdaptiveBot — self-tuning bot that learns from its own loss patterns.

Baseline behaviour:
  - Accept any signal with confidence >= 0.60
  - Minimum R:R of 1.5

Adaptive behaviour (activated after 10 trades, rolling 30-trade window):
  - 3+ losses on the same strategy  → raise that strategy's confidence by 5%
  - 3+ losses in the same regime    → block that regime for 10 trades
  - 3+ losses on the same symbol    → block that symbol for 10 trades
  - 2+ losses with low HTF score    → require HTF alignment
  - 3+ losses with R:R < 2.0        → raise min_rr by 0.2

Recovery (after 5 consecutive wins):
  - Relax the lowest-priority filter in order:
      blocked_symbols → blocked_regimes → strategy confidence → min_rr
  - Never relax below baseline values.
"""
from __future__ import annotations

import logging
from collections import deque

from libs.core.models.domain import MarketRegime, SignalOutput, TrendDirection
from libs.paper_trading.bot_agent import BotAgent

logger = logging.getLogger(__name__)

# ── Baseline constants ─────────────────────────────────────────────────────────

BASE_CONFIDENCE: float = 0.40
BASE_MIN_RR: float = 1.0

# Adaptation thresholds
_MIN_TRADES_BEFORE_ADAPT: int = 10
_ROLLING_WINDOW: int = 30
_BLOCK_DURATION: int = 10

# Trigger counts
_LOSS_TRIGGER_STRATEGY: int = 3
_LOSS_TRIGGER_REGIME: int = 3
_LOSS_TRIGGER_SYMBOL: int = 3
_LOSS_TRIGGER_HTF: int = 2
_LOSS_TRIGGER_RR: int = 3

_LOW_HTF_THRESHOLD: float = 0.5   # confluence.structure_score used as HTF proxy
_LOW_RR_THRESHOLD: float = 2.0

_MIN_RR_INCREMENT: float = 0.2
_CONFIDENCE_INCREMENT: float = 0.05
_WIN_STREAK_FOR_RECOVERY: int = 5


class AdaptiveBot(BotAgent):
    """Self-adjusting bot that adapts its filters based on recent losses."""

    NAME = "AdaptiveBot"

    def __init__(self, initial_capital: float = 1_667.0) -> None:
        super().__init__(initial_capital=initial_capital)

        # Rolling trade history (capped at _ROLLING_WINDOW)
        self._trade_history: deque[dict] = deque(maxlen=_ROLLING_WINDOW)

        # Counter of all trades ever processed by on_trade_closed
        self._trade_counter: int = 0

        # Per-strategy confidence overrides (strategy_name → float)
        self._strategy_confidence: dict[str, float] = {}

        # Blocked regimes: regime_value → trades_remaining_in_block
        self._blocked_regimes: dict[str, int] = {}

        # Blocked symbols: symbol → trades_remaining_in_block
        self._blocked_symbols: dict[str, int] = {}

        # Whether HTF alignment is currently required
        self._require_htf_alignment: bool = False

        # Current minimum R:R (may grow above baseline)
        self._min_rr: float = BASE_MIN_RR

    # ── Core filter ────────────────────────────────────────────────────────────

    def should_take_signal(self, signal: SignalOutput) -> bool:
        # 1. Symbol block check
        if signal.symbol in self._blocked_symbols:
            return False

        # 2. Regime block check
        regime_val = signal.market_regime.value
        if regime_val in self._blocked_regimes:
            return False

        # 3. Confidence check (strategy-specific or baseline)
        min_conf = self._strategy_confidence.get(signal.strategy_name, BASE_CONFIDENCE)
        if signal.confidence < min_conf:
            return False

        # 4. R:R check
        if signal.estimated_risk_reward < self._min_rr:
            return False

        # 5. HTF alignment check
        if self._require_htf_alignment:
            # Use structure_score from confluence as HTF alignment proxy
            htf_score = signal.confluence.structure_score
            if htf_score < _LOW_HTF_THRESHOLD:
                return False

        return True

    # ── Exit target ─────────────────────────────────────────────────────────────

    def _target_exit(self) -> str:
        metrics = self._portfolio.get_metrics()
        win_rate = metrics.get("win_rate", 0.0)
        return "tp2" if win_rate > 0.6 else "tp1"

    # ── Trade feedback ─────────────────────────────────────────────────────────

    def on_trade_closed(self, result: dict, signal: SignalOutput) -> None:
        """Feed a closed trade result back into the adaptive engine.

        Args:
            result: dict from PaperPortfolio.close_trade() — must contain 'pnl'.
            signal: the SignalOutput that generated the trade.
        """
        self._trade_counter += 1

        record = {
            "pnl": result.get("pnl", 0.0),
            "strategy": signal.strategy_name,
            "regime": signal.market_regime.value,
            "symbol": signal.symbol,
            "rr": signal.estimated_risk_reward,
            "htf_score": signal.confluence.structure_score,
            "is_loss": result.get("pnl", 0.0) < 0,
        }
        self._trade_history.append(record)

        # Decrement block counters on every trade (blocking is time-gated)
        self._tick_blocks()

        # Only adapt after minimum samples
        if self._trade_counter >= _MIN_TRADES_BEFORE_ADAPT:
            self._adapt_from_losses()

        # Recovery after win streak
        if self._consecutive_wins >= _WIN_STREAK_FOR_RECOVERY:
            self._relax_lowest_priority_filter()

    def _tick_blocks(self) -> None:
        """Decrement block counters; remove expired blocks."""
        expired_regimes = [k for k, v in self._blocked_regimes.items() if v <= 1]
        for k in expired_regimes:
            del self._blocked_regimes[k]
            logger.info("AdaptiveBot: regime block expired — %s unblocked", k)

        for k in self._blocked_regimes:
            self._blocked_regimes[k] -= 1

        expired_symbols = [k for k, v in self._blocked_symbols.items() if v <= 1]
        for k in expired_symbols:
            del self._blocked_symbols[k]
            logger.info("AdaptiveBot: symbol block expired — %s unblocked", k)

        for k in self._blocked_symbols:
            self._blocked_symbols[k] -= 1

    def _adapt_from_losses(self) -> None:
        """Analyse rolling window and tighten filters where loss clusters appear."""
        losses = [t for t in self._trade_history if t["is_loss"]]

        # -- Strategy confidence ------------------------------------------------
        strategy_loss_counts: dict[str, int] = {}
        for t in losses:
            strategy_loss_counts[t["strategy"]] = strategy_loss_counts.get(t["strategy"], 0) + 1

        for strat, count in strategy_loss_counts.items():
            if count >= _LOSS_TRIGGER_STRATEGY:
                current = self._strategy_confidence.get(strat, BASE_CONFIDENCE)
                new_val = round(current + _CONFIDENCE_INCREMENT, 4)
                if new_val != current:
                    self._strategy_confidence[strat] = new_val
                    logger.info(
                        "AdaptiveBot: strategy '%s' confidence raised to %.2f (%d losses)",
                        strat, new_val, count,
                    )

        # -- Regime blocks -------------------------------------------------------
        regime_loss_counts: dict[str, int] = {}
        for t in losses:
            regime_loss_counts[t["regime"]] = regime_loss_counts.get(t["regime"], 0) + 1

        for regime, count in regime_loss_counts.items():
            if count >= _LOSS_TRIGGER_REGIME and regime not in self._blocked_regimes:
                self._blocked_regimes[regime] = _BLOCK_DURATION
                logger.info(
                    "AdaptiveBot: regime '%s' blocked for %d trades (%d losses)",
                    regime, _BLOCK_DURATION, count,
                )

        # -- Symbol blocks -------------------------------------------------------
        symbol_loss_counts: dict[str, int] = {}
        for t in losses:
            symbol_loss_counts[t["symbol"]] = symbol_loss_counts.get(t["symbol"], 0) + 1

        for symbol, count in symbol_loss_counts.items():
            if count >= _LOSS_TRIGGER_SYMBOL and symbol not in self._blocked_symbols:
                self._blocked_symbols[symbol] = _BLOCK_DURATION
                logger.info(
                    "AdaptiveBot: symbol '%s' blocked for %d trades (%d losses)",
                    symbol, _BLOCK_DURATION, count,
                )

        # -- HTF alignment requirement -------------------------------------------
        low_htf_losses = [t for t in losses if t["htf_score"] < _LOW_HTF_THRESHOLD]
        if len(low_htf_losses) >= _LOSS_TRIGGER_HTF and not self._require_htf_alignment:
            self._require_htf_alignment = True
            logger.info(
                "AdaptiveBot: HTF alignment now required (%d low-HTF losses)",
                len(low_htf_losses),
            )

        # -- Min R:R tightening --------------------------------------------------
        low_rr_losses = [t for t in losses if t["rr"] < _LOW_RR_THRESHOLD]
        if len(low_rr_losses) >= _LOSS_TRIGGER_RR:
            new_min_rr = round(self._min_rr + _MIN_RR_INCREMENT, 4)
            if new_min_rr != self._min_rr:
                self._min_rr = new_min_rr
                logger.info(
                    "AdaptiveBot: min_rr raised to %.2f (%d low-RR losses)",
                    self._min_rr, len(low_rr_losses),
                )

    def _relax_lowest_priority_filter(self) -> None:
        """After sustained wins, relax one filter (lowest priority first)."""
        # Priority order (easiest to relax → hardest):
        #   1. blocked_symbols (most tactical, first to unblock)
        #   2. blocked_regimes
        #   3. strategy confidence overrides (reduce back toward baseline)
        #   4. min_rr (reduce back toward baseline)

        if self._blocked_symbols:
            symbol = next(iter(self._blocked_symbols))
            del self._blocked_symbols[symbol]
            logger.info("AdaptiveBot: recovery — unblocked symbol '%s'", symbol)
            return

        if self._blocked_regimes:
            regime = next(iter(self._blocked_regimes))
            del self._blocked_regimes[regime]
            logger.info("AdaptiveBot: recovery — unblocked regime '%s'", regime)
            return

        # Relax the strategy confidence with the highest override (if above baseline)
        if self._strategy_confidence:
            highest_strat = max(self._strategy_confidence, key=lambda k: self._strategy_confidence[k])
            current = self._strategy_confidence[highest_strat]
            relaxed = round(current - _CONFIDENCE_INCREMENT, 4)
            if relaxed <= BASE_CONFIDENCE:
                del self._strategy_confidence[highest_strat]
                logger.info(
                    "AdaptiveBot: recovery — strategy '%s' confidence reset to baseline",
                    highest_strat,
                )
            else:
                self._strategy_confidence[highest_strat] = relaxed
                logger.info(
                    "AdaptiveBot: recovery — strategy '%s' confidence relaxed to %.2f",
                    highest_strat, relaxed,
                )
            return

        # Finally relax min_rr toward baseline
        if self._min_rr > BASE_MIN_RR:
            new_min_rr = round(self._min_rr - _MIN_RR_INCREMENT, 4)
            self._min_rr = max(new_min_rr, BASE_MIN_RR)
            logger.info("AdaptiveBot: recovery — min_rr relaxed to %.2f", self._min_rr)

    # ── State serialisation ────────────────────────────────────────────────────

    def get_adaptive_filters(self) -> dict:
        """Return a serialisable snapshot of all adaptive filter state."""
        return {
            "strategy_confidence": dict(self._strategy_confidence),
            "blocked_regimes": dict(self._blocked_regimes),
            "blocked_symbols": dict(self._blocked_symbols),
            "require_htf_alignment": self._require_htf_alignment,
            "min_rr": self._min_rr,
            "trade_counter": self._trade_counter,
        }

    def load_adaptive_filters(self, data: dict) -> None:
        """Restore adaptive filter state from a previously serialised snapshot."""
        self._strategy_confidence = dict(data.get("strategy_confidence", {}))
        self._blocked_regimes = dict(data.get("blocked_regimes", {}))
        self._blocked_symbols = dict(data.get("blocked_symbols", {}))
        self._require_htf_alignment = bool(data.get("require_htf_alignment", False))
        self._min_rr = float(data.get("min_rr", BASE_MIN_RR))
        self._trade_counter = int(data.get("trade_counter", 0))
