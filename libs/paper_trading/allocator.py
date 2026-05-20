"""
Capital allocation logic for paper-trading bots.

Implements:
- Kelly-criterion based position sizing with three-phase ramp-up
- Darwinian weekly rebalance — top bots by Sharpe ratio get more capital
"""
from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_TRADE_SIZE: float = 10.0
MAX_KELLY_FRACTION: float = 0.25
MAX_SINGLE_TRADE_PCT: float = 0.15
REBALANCE_WEIGHTS: list[float] = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]


# ── CapitalAllocator ──────────────────────────────────────────────────────────

class CapitalAllocator:
    """Allocates and sizes capital for paper-trading bots.

    All methods are pure (no internal state) — safe for concurrent use.
    """

    # ── Kelly fraction ────────────────────────────────────────────────────────

    def kelly_fraction(self, win_rate: float, avg_win_loss_ratio: float) -> float:
        """Compute the Kelly criterion fraction.

        Formula: f* = win_rate - (loss_rate / avg_win_loss_ratio)

        Args:
            win_rate: Probability of a winning trade (0–1).
            avg_win_loss_ratio: Ratio of average win to average loss. Must be > 0.

        Returns:
            Kelly fraction capped at MAX_KELLY_FRACTION (0.25), or -1.0 when
            avg_win_loss_ratio is non-positive (undefined / invalid input).
        """
        if avg_win_loss_ratio <= 0:
            return -1.0

        loss_rate = 1.0 - win_rate
        raw_kelly = win_rate - (loss_rate / avg_win_loss_ratio)
        return min(raw_kelly, MAX_KELLY_FRACTION)

    # ── Position size ─────────────────────────────────────────────────────────

    def position_size(
        self,
        bot_capital: float,
        trade_count: int,
        win_rate: float,
        avg_win_loss_ratio: float,
        volatility_pct: float = 1.0,
    ) -> float:
        """Compute the dollar position size for a single trade.

        Three-phase ramp:
          Phase 1 (trade_count < 10):   1% of capital (cold-start / no track record).
          Phase 2 (10 ≤ trade_count < 30): quarter-Kelly (kelly × 0.25 × capital).
          Phase 3 (trade_count ≥ 30):   half-Kelly (kelly × 0.5 × capital).

        In all phases:
          - If computed Kelly ≤ 0 return 0.0 (Phase 2/3).
          - Cap at MAX_SINGLE_TRADE_PCT (5%) of capital.
          - Return 0.0 if result < MIN_TRADE_SIZE ($10).

        Args:
            bot_capital: Total capital assigned to this bot.
            trade_count: Number of completed trades so far.
            win_rate: Historical win rate (0–1).
            avg_win_loss_ratio: Average win / average loss.

        Returns:
            Dollar amount to risk on the next trade, or 0.0 when sizing
            conditions are not met.
        """
        max_trade = bot_capital * MAX_SINGLE_TRADE_PCT

        if trade_count < 10:
            # Phase 1 — cold start: fixed 5%
            raw = bot_capital * 0.05
            sized = min(raw, max_trade)
        else:
            kelly = self.kelly_fraction(win_rate, avg_win_loss_ratio)
            if kelly <= 0:
                return 0.0

            if trade_count < 30:
                # Phase 2 — learning: quarter-Kelly
                raw = kelly * 0.25 * bot_capital
            else:
                # Phase 3 — full: half-Kelly
                raw = kelly * 0.5 * bot_capital

            sized = min(raw, max_trade)

        # Volatility adjustment: reduce size in high-ATR environments
        if volatility_pct > 5.0:
            sized *= 0.5  # Cut size in half for extreme volatility
        elif volatility_pct > 3.0:
            sized *= 0.75  # Reduce 25% for high volatility

        return sized if sized >= MIN_TRADE_SIZE else 0.0

    # ── Rebalance ─────────────────────────────────────────────────────────────

    def rebalance(
        self,
        total_capital: float,
        bot_sharpes: dict[str, float],
        paused_bots: set[str] | None,
    ) -> dict[str, float]:
        """Distribute total_capital across bots using Darwinian (Sharpe-ranked) weights.

        Algorithm:
        1. Paused bots receive 0.0 and are excluded from the pool.
        2. Active bots are sorted descending by Sharpe ratio.
        3. REBALANCE_WEIGHTS are assigned positionally; if there are fewer active bots
           than weight slots, the surplus weight is redistributed proportionally to
           those that do exist.
        4. Rounding remainder (floating-point dust) is awarded to the top bot.

        Args:
            total_capital: Total capital to distribute.
            bot_sharpes: Mapping of bot name → Sharpe ratio.
            paused_bots: Set of bot names that are currently paused (receive 0).
                         Pass None to treat all bots as active.

        Returns:
            Dict mapping bot name → allocated capital (floats).
        """
        paused: set[str] = paused_bots if paused_bots is not None else set()

        active_bots = sorted(
            [name for name in bot_sharpes if name not in paused],
            key=lambda n: bot_sharpes[n],
            reverse=True,
        )

        allocations: dict[str, float] = {name: 0.0 for name in bot_sharpes}

        if not active_bots:
            return allocations

        n_active = len(active_bots)

        if n_active <= len(REBALANCE_WEIGHTS):
            # Use the first n_active weights; redistribute remaining weight proportionally
            used_weights = REBALANCE_WEIGHTS[:n_active]
            weight_sum = sum(used_weights)
            # Normalise so they sum to 1.0
            normalised = [w / weight_sum for w in used_weights]
        else:
            # More bots than weight slots — distribute equally among overflow bots
            base_weights = list(REBALANCE_WEIGHTS)
            overflow_count = n_active - len(REBALANCE_WEIGHTS)
            remaining_weight = 1.0 - sum(base_weights)
            overflow_weight = remaining_weight / overflow_count if overflow_count else 0.0
            extended = base_weights + [overflow_weight] * overflow_count
            total_w = sum(extended)
            normalised = [w / total_w for w in extended]

        # Assign capital
        assigned_total = 0.0
        for idx, bot_name in enumerate(active_bots):
            amount = normalised[idx] * total_capital
            allocations[bot_name] = amount
            assigned_total += amount

        # Fix rounding — give remainder to top bot
        remainder = total_capital - assigned_total
        if active_bots and abs(remainder) > 0:
            allocations[active_bots[0]] += remainder

        return allocations
