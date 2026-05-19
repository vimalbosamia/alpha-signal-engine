"""
BTC / Market Correlation Filter.

Checks whether the crypto or equity market leader (BTC for crypto,
SPY for stocks) supports or conflicts with a proposed trade direction.

Rules
-----
* Crypto trades:
  - If action=BUY  and BTC is bearish → conflicts=True   (block altcoin long)
  - If action=SELL and BTC is bullish → conflicts=True   (block altcoin short)
  - BTCUSDT itself is exempt — it trades its own signal.
* Stock trades:
  - Same logic but uses SPY as the leader.

Design principles
-----------------
* All results are immutable (frozen dataclass).
* _calc_trend is pure — fully testable without mocks.
* Network I/O is isolated in _fetch_crypto_prices / _fetch_stock_prices
  so tests can patch a single coroutine.
* No magic numbers — all thresholds are named class constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from libs.core.logging.logger import get_logger
from libs.core.models.domain import Timeframe

log = get_logger(__name__)

# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrelationCheck:
    """Immutable result of a single leader-vs-trade correlation check."""

    leader_symbol: str       # "BTCUSDT" or "SPY"
    leader_trend: str        # "bullish", "bearish", "neutral"
    leader_change_pct: float # 24 h % change of the leader
    correlation_strength: str  # "strong", "moderate", "weak"
    conflicts_with_trade: bool
    explanation: str


# ── Filter ─────────────────────────────────────────────────────────────────────


class MarketCorrelationFilter:
    """
    Filter trades based on BTC (crypto) or SPY (stocks) macro trend.

    Usage
    -----
    filter = MarketCorrelationFilter()
    check  = await filter.check_crypto("ETHUSDT", "BUY")
    if check.conflicts_with_trade:
        ...
    """

    # Leader symbols
    CRYPTO_LEADER: str = "BTCUSDT"
    STOCK_LEADER: str = "SPY"

    # EMA periods for trend detection
    EMA_FAST_PERIOD: int = 9
    EMA_SLOW_PERIOD: int = 21

    # Minimum bars required before we attempt a trend call
    MIN_BARS: int = 22  # at least EMA_SLOW_PERIOD + 1

    # Flat-change threshold — below this absolute % → neutral
    NEUTRAL_CHANGE_THRESHOLD: float = 0.10  # 0.10 %

    # Correlation strength thresholds (absolute 24 h % change of leader)
    STRONG_CHANGE_THRESHOLD: float = 2.0    # |change| >= 2 % → strong
    MODERATE_CHANGE_THRESHOLD: float = 0.5  # |change| >= 0.5 % → moderate

    # Number of 1-hour candles to fetch for a ~24-hour window
    _CANDLE_COUNT: int = 25  # slight overshoot so EMA warms up

    # ── Public API ─────────────────────────────────────────────────────────────

    async def check_crypto(self, symbol: str, action: str) -> CorrelationCheck:
        """
        Check whether BTC's current trend supports the proposed trade.

        Parameters
        ----------
        symbol : str
            The altcoin being traded (e.g. "ETHUSDT").
        action : str
            "BUY" or "SELL".

        Returns
        -------
        CorrelationCheck
            conflicts_with_trade=True when BTC trend opposes the action
            AND the symbol is not BTCUSDT itself.
        """
        prices = await self._fetch_crypto_prices(self.CRYPTO_LEADER)
        return self._build_check(
            leader_symbol=self.CRYPTO_LEADER,
            traded_symbol=symbol,
            action=action.upper(),
            prices=prices,
        )

    async def check_stock(self, symbol: str, action: str) -> CorrelationCheck:
        """
        Check whether SPY's current trend supports the proposed trade.

        Parameters
        ----------
        symbol : str
            The stock being traded (e.g. "AAPL").
        action : str
            "BUY" or "SELL".

        Returns
        -------
        CorrelationCheck
            conflicts_with_trade=True when SPY trend opposes the action.
        """
        prices = await self._fetch_stock_prices(self.STOCK_LEADER)
        return self._build_check(
            leader_symbol=self.STOCK_LEADER,
            traded_symbol=symbol,
            action=action.upper(),
            prices=prices,
        )

    # ── Pure trend logic ───────────────────────────────────────────────────────

    def _calc_trend(self, prices: list[float]) -> tuple[str, float]:
        """
        Determine trend direction and 24-hour % change from a close-price list.

        Algorithm
        ---------
        1. Compute EMA-9 and EMA-21 over the price series.
        2. If EMA-9 > EMA-21 at the last bar → "bullish".
           If EMA-9 < EMA-21 at the last bar → "bearish".
        3. If the absolute 24 h % change is below NEUTRAL_CHANGE_THRESHOLD
           → override to "neutral" regardless of EMA crossover.

        Parameters
        ----------
        prices : list[float]
            Ordered list of close prices, oldest first.

        Returns
        -------
        (direction, change_pct) where direction ∈ {"bullish","bearish","neutral"}
        and change_pct is the % change from the first to the last price.
        """
        if len(prices) < 2:
            return "neutral", 0.0

        series = pd.Series(prices, dtype=float)

        # 24 h % change (first → last price in the window)
        first_price = series.iloc[0]
        last_price = series.iloc[-1]
        change_pct = (
            ((last_price - first_price) / first_price) * 100.0
            if first_price != 0.0
            else 0.0
        )

        # EMA crossover — only meaningful with enough bars
        if len(prices) < self.MIN_BARS:
            # Fall back to raw change direction
            if abs(change_pct) < self.NEUTRAL_CHANGE_THRESHOLD:
                return "neutral", change_pct
            return ("bullish" if change_pct > 0.0 else "bearish"), change_pct

        ema_fast = series.ewm(span=self.EMA_FAST_PERIOD, adjust=False).mean()
        ema_slow = series.ewm(span=self.EMA_SLOW_PERIOD, adjust=False).mean()

        last_ema_fast = float(ema_fast.iloc[-1])
        last_ema_slow = float(ema_slow.iloc[-1])

        if abs(change_pct) < self.NEUTRAL_CHANGE_THRESHOLD:
            return "neutral", change_pct

        if last_ema_fast > last_ema_slow:
            return "bullish", change_pct

        if last_ema_fast < last_ema_slow:
            return "bearish", change_pct

        return "neutral", change_pct

    # ── Internal builders ──────────────────────────────────────────────────────

    def _build_check(
        self,
        leader_symbol: str,
        traded_symbol: str,
        action: str,
        prices: list[float],
    ) -> CorrelationCheck:
        """Construct a CorrelationCheck from leader prices and trade parameters."""
        direction, change_pct = self._calc_trend(prices)
        strength = self._correlation_strength(change_pct)

        # BTC/SPY exemption: the leader always trades its own signal
        is_leader_itself = traded_symbol.upper() == leader_symbol.upper()

        conflicts = False
        if not is_leader_itself:
            if action == "BUY" and direction == "bearish":
                conflicts = True
            elif action == "SELL" and direction == "bullish":
                conflicts = True

        explanation = self._build_explanation(
            leader_symbol=leader_symbol,
            direction=direction,
            change_pct=change_pct,
            action=action,
            traded_symbol=traded_symbol,
            conflicts=conflicts,
            is_leader_itself=is_leader_itself,
        )

        return CorrelationCheck(
            leader_symbol=leader_symbol,
            leader_trend=direction,
            leader_change_pct=round(change_pct, 4),
            correlation_strength=strength,
            conflicts_with_trade=conflicts,
            explanation=explanation,
        )

    def _correlation_strength(self, change_pct: float) -> str:
        """Map absolute % change to a qualitative strength label."""
        abs_change = abs(change_pct)
        if abs_change >= self.STRONG_CHANGE_THRESHOLD:
            return "strong"
        if abs_change >= self.MODERATE_CHANGE_THRESHOLD:
            return "moderate"
        return "weak"

    @staticmethod
    def _build_explanation(
        leader_symbol: str,
        direction: str,
        change_pct: float,
        action: str,
        traded_symbol: str,
        conflicts: bool,
        is_leader_itself: bool,
    ) -> str:
        sign = "+" if change_pct >= 0.0 else ""
        base = (
            f"{leader_symbol} is {direction} ({sign}{change_pct:.2f}% 24h). "
            f"Trade: {action} {traded_symbol}. "
        )
        if is_leader_itself:
            return base + "Leader trades its own signal — no conflict check applied."
        if conflicts:
            return base + f"CONFLICT: {leader_symbol} trend opposes {action} direction."
        return base + "Trade aligned with macro trend."

    # ── Data-fetch helpers (isolated for easy mocking in tests) ───────────────

    async def _fetch_crypto_prices(self, symbol: str) -> list[float]:
        """
        Fetch the last ~25 hourly close prices for *symbol* from Binance.

        Returns an empty list on any failure so callers degrade gracefully.
        """
        from libs.data.providers.binance.provider import BinanceDataProvider

        provider = BinanceDataProvider()
        try:
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(hours=self._CANDLE_COUNT + 2)
            df = await provider.get_candles(
                symbol=symbol,
                timeframe=Timeframe.ONE_HOUR,
                start=start,
                end=end,
                limit=self._CANDLE_COUNT,
            )
            return df["close"].tolist() if not df.empty else []
        except Exception as exc:
            log.warning("correlation_fetch_crypto_failed", symbol=symbol, error=str(exc))
            return []
        finally:
            await provider.close()

    async def _fetch_stock_prices(self, symbol: str) -> list[float]:
        """
        Fetch the last ~25 hourly close prices for *symbol* from Alpaca.

        Returns an empty list on any failure so callers degrade gracefully.
        """
        from libs.data.providers.alpaca.provider import AlpacaDataProvider

        provider = AlpacaDataProvider()
        try:
            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(hours=self._CANDLE_COUNT + 2)
            df = await provider.get_candles(
                symbol=symbol,
                timeframe=Timeframe.ONE_HOUR,
                start=start,
                end=end,
                limit=self._CANDLE_COUNT,
            )
            return df["close"].tolist() if not df.empty else []
        except Exception as exc:
            log.warning("correlation_fetch_stock_failed", symbol=symbol, error=str(exc))
            return []
