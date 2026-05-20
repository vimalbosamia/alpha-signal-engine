"""
Bias flip auto-exit logic — document (4) section 9.

Rules:
  SPOT LONG + bearish for 2 checks → CLOSE_LONG
  FUTURES LONG + bearish for 2 checks → CLOSE_LONG
  FUTURES SHORT + bullish for 2 checks → CLOSE_SHORT
  Neutral does not increment — only hard flips count.
  Bias re-alignment resets counter to 0.
"""
from __future__ import annotations

BIAS_ADVERSE_THRESHOLD: int = 2


def compute_bias_status(direction: str, current_bias: str) -> str:
    """
    Compute bias status for a trade.

    Returns: ALIGNED | WARNING | CONFLICT | NO_POSITION
    """
    direction = direction.upper()
    current_bias = current_bias.lower()

    if direction == "FLAT":
        return "NO_POSITION"

    if direction == "LONG":
        if current_bias == "bullish":
            return "ALIGNED"
        if current_bias == "neutral":
            return "WARNING"
        return "CONFLICT"

    if direction == "SHORT":
        if current_bias == "bearish":
            return "ALIGNED"
        if current_bias == "neutral":
            return "WARNING"
        return "CONFLICT"

    return "NO_POSITION"


def _is_adverse(direction: str, current_bias: str) -> bool:
    """Check if bias is directly opposed to direction (not neutral)."""
    direction = direction.upper()
    bias = current_bias.lower()

    if direction == "LONG" and bias == "bearish":
        return True
    if direction == "SHORT" and bias == "bullish":
        return True
    return False


def _is_aligned(direction: str, current_bias: str) -> bool:
    """Check if bias supports the direction."""
    direction = direction.upper()
    bias = current_bias.lower()

    if direction == "LONG" and bias == "bullish":
        return True
    if direction == "SHORT" and bias == "bearish":
        return True
    return False


def should_exit_on_bias_flip(
    direction: str,
    market_mode: str,
    current_bias: str,
    bias_adverse_count: int,
) -> tuple[bool, int]:
    """
    Determine if a trade should be closed due to bias flip.

    Args:
        direction: LONG | SHORT | FLAT
        market_mode: SPOT | FUTURES
        current_bias: bullish | bearish | neutral
        bias_adverse_count: how many consecutive adverse checks so far

    Returns:
        (should_exit, new_bias_adverse_count)

    Rules per document (4) section 9:
      - Only hard flips (bearish for long, bullish for short) increment
      - Neutral does NOT increment — manage only, no forced exit
      - Aligned bias resets counter to 0
      - Exit after BIAS_ADVERSE_THRESHOLD consecutive adverse checks
    """
    if direction.upper() == "FLAT":
        return False, 0

    if _is_adverse(direction, current_bias):
        new_count = bias_adverse_count + 1
        if new_count >= BIAS_ADVERSE_THRESHOLD:
            return True, new_count
        return False, new_count

    if _is_aligned(direction, current_bias):
        return False, 0

    # Neutral — don't increment, don't reset
    return False, bias_adverse_count
