"""LeakageGuard — prevents feature leakage by enforcing indicator warmup periods."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageCheck:
    indicator_name: str
    warmup_bars: int       # bars needed before indicator is valid
    is_safe: bool
    explanation: str


@dataclass(frozen=True)
class LeakageAudit:
    checks: list[LeakageCheck]
    total_indicators: int
    safe_count: int
    unsafe_count: int
    min_warmup_bars: int   # minimum bars before ANY trading is safe
    explanation: str


class LeakageGuard:
    """Prevents feature leakage by ensuring indicator warmup periods are respected."""

    # Known warmup periods (bars needed before indicator is valid)
    WARMUP: dict[str, int] = {
        "ema_9": 9,
        "ema_20": 20,
        "ema_50": 50,
        "ema_200": 200,
        "sma_20": 20,
        "sma_50": 50,
        "sma_200": 200,
        "rsi": 14,
        "macd_line": 26,
        "macd_signal": 35,   # 26 + 9 signal
        "bb_upper": 20,
        "bb_lower": 20,
        "bb_width": 20,
        "atr": 14,
        "adx": 28,           # 14 + 14 DX smoothing
        "stoch_rsi_k": 14,
        "stoch_rsi_d": 17,   # 14 + 3
        "vwap": 1,
    }

    def audit(self, available_bars: int) -> LeakageAudit:
        """Check which indicators are safe given available bars."""
        checks: list[LeakageCheck] = []
        for name, warmup in self.WARMUP.items():
            safe = available_bars >= warmup
            checks.append(LeakageCheck(
                indicator_name=name,
                warmup_bars=warmup,
                is_safe=safe,
                explanation="OK" if safe else f"Need {warmup} bars, have {available_bars}",
            ))

        safe_count = sum(1 for c in checks if c.is_safe)
        unsafe_count = len(checks) - safe_count
        min_warmup = max(self.WARMUP.values())

        return LeakageAudit(
            checks=checks,
            total_indicators=len(checks),
            safe_count=safe_count,
            unsafe_count=unsafe_count,
            min_warmup_bars=min_warmup,
            explanation=(
                f"{safe_count}/{len(checks)} indicators safe with {available_bars} bars. "
                f"Need {min_warmup} for all."
            ),
        )

    def is_safe_for_trading(
        self,
        available_bars: int,
        required_indicators: list[str] | None = None,
    ) -> bool:
        """True if all required indicators have enough warmup bars."""
        if required_indicators is None:
            required_indicators = list(self.WARMUP.keys())
        return all(available_bars >= self.WARMUP.get(ind, 0) for ind in required_indicators)
