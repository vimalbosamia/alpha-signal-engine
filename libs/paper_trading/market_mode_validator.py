"""
Market Mode Validator — enforces spot vs futures rules per document (4).

Validates every signal before position opening:
  - SPOT: no shorts, no leverage, BUY=OPEN_LONG, SELL=CLOSE_LONG only
  - FUTURES: leverage limits, liquidation buffer, TP/SL direction
  - Duplicate position blocking per symbol per mode
  - TP/SL direction validation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from libs.core.logging.logger import get_logger

log = get_logger(__name__)


# ── Config defaults ──────────────────────────────────────────────────────────

SPOT_CONFIG = {
    "allow_short": False,
    "max_positions_per_symbol": 1,
    "min_cash_buffer_percent": 20,
    "use_margin": False,
}

FUTURES_CONFIG = {
    "default_leverage": 3,
    "max_leverage": 5,
    "margin_mode": "isolated",
    "min_liquidation_buffer_percent": 5,
    "max_account_risk_per_trade_percent": 2.0,
    "allow_short": True,
    "allow_reverse_position": False,
    "max_positions_per_symbol": 1,
}


# ── Validation result ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    approved: bool
    reason: str = ""
    position_intent: str = "OPEN_LONG"
    direction: str = "LONG"
    market_mode: str = "SPOT"
    leverage: float = 1.0
    margin_mode: Optional[str] = None
    notional_size: float = 0.0
    liquidation_buffer_percent: float = 0.0


# ── Intent resolution ────────────────────────────────────────────────────────

def resolve_intent(
    side: str,
    market_mode: str,
    has_existing_long: bool = False,
    has_existing_short: bool = False,
) -> tuple[str, str]:
    """
    Resolve BUY/SELL into explicit (position_intent, direction).

    SPOT:
      BUY  → (OPEN_LONG, LONG)
      SELL → (CLOSE_LONG, FLAT)

    FUTURES:
      BUY + no existing short  → (OPEN_LONG, LONG)
      BUY + existing short     → (CLOSE_SHORT, FLAT)
      SELL + no existing long  → (OPEN_SHORT, SHORT)
      SELL + existing long     → (CLOSE_LONG, FLAT)
    """
    side = side.upper()

    if market_mode == "SPOT":
        if side == "BUY":
            return "OPEN_LONG", "LONG"
        return "CLOSE_LONG", "FLAT"

    # FUTURES
    if side == "BUY":
        if has_existing_short:
            return "CLOSE_SHORT", "FLAT"
        return "OPEN_LONG", "LONG"

    # SELL
    if has_existing_long:
        return "CLOSE_LONG", "FLAT"
    return "OPEN_SHORT", "SHORT"


# ── Spot validation ──────────────────────────────────────────────────────────

def validate_spot_signal(
    side: str,
    symbol: str,
    position_size_usd: float,
    entry_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    existing_long_symbols: set[str],
    available_cash: float,
) -> ValidationResult:
    """Validate a SPOT signal per document (4) section 6 + 12."""

    intent, direction = resolve_intent(side, "SPOT")

    # SPOT must not allow OPEN_SHORT
    if intent == "OPEN_SHORT":
        return ValidationResult(
            approved=False,
            reason="SPOT_SHORT_NOT_ALLOWED",
            market_mode="SPOT",
        )

    # SELL without existing long
    if side.upper() == "SELL" and symbol not in existing_long_symbols:
        return ValidationResult(
            approved=False,
            reason="SPOT_SELL_WITHOUT_LONG",
            market_mode="SPOT",
        )

    # Duplicate long
    if intent == "OPEN_LONG" and symbol in existing_long_symbols:
        return ValidationResult(
            approved=False,
            reason="DUPLICATE_SPOT_LONG",
            market_mode="SPOT",
        )

    # Insufficient cash
    if intent == "OPEN_LONG" and position_size_usd > available_cash:
        return ValidationResult(
            approved=False,
            reason="INSUFFICIENT_SPOT_CASH",
            market_mode="SPOT",
        )

    # TP/SL direction validation for OPEN_LONG
    if intent == "OPEN_LONG":
        if stop_loss is not None and stop_loss >= entry_price:
            return ValidationResult(
                approved=False,
                reason="SPOT_LONG_SL_ABOVE_ENTRY",
                market_mode="SPOT",
            )
        if take_profit is not None and take_profit <= entry_price:
            return ValidationResult(
                approved=False,
                reason="SPOT_LONG_TP_BELOW_ENTRY",
                market_mode="SPOT",
            )

    return ValidationResult(
        approved=True,
        position_intent=intent,
        direction=direction,
        market_mode="SPOT",
        leverage=1.0,
    )


# ── Futures validation ───────────────────────────────────────────────────────

def validate_futures_signal(
    side: str,
    symbol: str,
    position_size_usd: float,
    entry_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    leverage: float,
    existing_long_symbols: set[str],
    existing_short_symbols: set[str],
    available_cash: float,
    config: dict | None = None,
) -> ValidationResult:
    """Validate a FUTURES signal per document (4) section 6 + 11."""

    cfg = config or FUTURES_CONFIG
    max_lev = cfg.get("max_leverage", 5)
    min_liq_buffer = cfg.get("min_liquidation_buffer_percent", 5)

    has_long = symbol in existing_long_symbols
    has_short = symbol in existing_short_symbols
    intent, direction = resolve_intent(side, "FUTURES", has_long, has_short)

    # Leverage check
    if leverage > max_lev:
        return ValidationResult(
            approved=False,
            reason="LEVERAGE_TOO_HIGH",
            market_mode="FUTURES",
        )

    # Duplicate position check
    if intent == "OPEN_LONG" and has_long:
        return ValidationResult(
            approved=False,
            reason="DUPLICATE_FUTURES_LONG",
            market_mode="FUTURES",
        )

    if intent == "OPEN_SHORT" and has_short:
        return ValidationResult(
            approved=False,
            reason="DUPLICATE_FUTURES_SHORT",
            market_mode="FUTURES",
        )

    # Opposite position check
    if intent == "OPEN_LONG" and has_short:
        if not cfg.get("allow_reverse_position", False):
            return ValidationResult(
                approved=False,
                reason="OPPOSITE_POSITION_EXISTS",
                market_mode="FUTURES",
            )

    if intent == "OPEN_SHORT" and has_long:
        if not cfg.get("allow_reverse_position", False):
            return ValidationResult(
                approved=False,
                reason="OPPOSITE_POSITION_EXISTS",
                market_mode="FUTURES",
            )

    # Liquidation buffer check for new positions
    notional = position_size_usd * leverage
    if intent in ("OPEN_LONG", "OPEN_SHORT") and entry_price > 0:
        mmr = 0.004
        if intent == "OPEN_LONG":
            liq_price = entry_price * (1 - (1 / leverage) + mmr)
            liq_buffer = ((entry_price - liq_price) / entry_price) * 100
        else:
            liq_price = entry_price * (1 + (1 / leverage) - mmr)
            liq_buffer = ((liq_price - entry_price) / entry_price) * 100

        if liq_buffer < min_liq_buffer:
            return ValidationResult(
                approved=False,
                reason="LIQUIDATION_BUFFER_TOO_SMALL",
                market_mode="FUTURES",
            )
    else:
        liq_buffer = 0.0

    # TP/SL direction validation
    if intent == "OPEN_LONG":
        if stop_loss is not None and stop_loss >= entry_price:
            return ValidationResult(
                approved=False,
                reason="FUTURES_LONG_SL_ABOVE_ENTRY",
                market_mode="FUTURES",
            )
        if take_profit is not None and take_profit <= entry_price:
            return ValidationResult(
                approved=False,
                reason="FUTURES_LONG_TP_BELOW_ENTRY",
                market_mode="FUTURES",
            )

    if intent == "OPEN_SHORT":
        if stop_loss is not None and stop_loss <= entry_price:
            return ValidationResult(
                approved=False,
                reason="FUTURES_SHORT_SL_BELOW_ENTRY",
                market_mode="FUTURES",
            )
        if take_profit is not None and take_profit >= entry_price:
            return ValidationResult(
                approved=False,
                reason="FUTURES_SHORT_TP_ABOVE_ENTRY",
                market_mode="FUTURES",
            )

    return ValidationResult(
        approved=True,
        position_intent=intent,
        direction=direction,
        market_mode="FUTURES",
        leverage=leverage,
        margin_mode=cfg.get("margin_mode", "isolated"),
        notional_size=notional,
        liquidation_buffer_percent=round(liq_buffer, 2),
    )


# ── Unified validator ────────────────────────────────────────────────────────

def validate_signal(
    side: str,
    symbol: str,
    market_mode: str,
    position_size_usd: float,
    entry_price: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    leverage: float = 1.0,
    existing_long_symbols: set[str] | None = None,
    existing_short_symbols: set[str] | None = None,
    available_cash: float = 0.0,
    futures_config: dict | None = None,
) -> ValidationResult:
    """
    Unified entry point — routes to spot or futures validation.

    Hard rule: if market_mode is missing/empty, reject.
    """
    existing_long_symbols = existing_long_symbols or set()
    existing_short_symbols = existing_short_symbols or set()

    if not market_mode:
        return ValidationResult(approved=False, reason="MARKET_MODE_MISSING")

    market_mode = market_mode.upper()

    if market_mode in ("SPOT", "EQUITY"):
        result = validate_spot_signal(
            side=side,
            symbol=symbol,
            position_size_usd=position_size_usd,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            existing_long_symbols=existing_long_symbols,
            available_cash=available_cash,
        )
        result.market_mode = market_mode  # preserve EQUITY vs SPOT label
        return result

    if market_mode == "FUTURES":
        return validate_futures_signal(
            side=side,
            symbol=symbol,
            position_size_usd=position_size_usd,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            existing_long_symbols=existing_long_symbols,
            existing_short_symbols=existing_short_symbols,
            available_cash=available_cash,
            config=futures_config,
        )

    return ValidationResult(approved=False, reason=f"UNKNOWN_MARKET_MODE:{market_mode}")
