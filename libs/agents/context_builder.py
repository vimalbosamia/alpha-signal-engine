"""
libs.agents.context_builder — Builds agent context dict from all data sources.

Bridges MarketContext, live sentiment feeds, portfolio state, and learning
subsystem data into the unified context dict consumed by all 15 agents.
"""
from __future__ import annotations

from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


async def fetch_live_sentiment() -> dict[str, Any]:
    """Fetch live sentiment data from multiple sources.

    Gathers: Fear/Greed index, BTC funding rate, long/short ratio,
    BTC dominance change. All best-effort with safe defaults.
    """
    import httpx

    result: dict[str, Any] = {
        "fear_greed_index": 50,
        "funding_rate": 0.0,
        "long_short_ratio": 1.0,
        "btc_dominance_change": 0.0,
        "social_sentiment": 0.0,
    }

    async with httpx.AsyncClient(timeout=6.0) as client:
        # ── Fear & Greed Index ──────────────────────────────────────
        try:
            resp = await client.get("https://api.alternative.me/fng/?limit=1")
            data = resp.json().get("data", [{}])[0]
            result["fear_greed_index"] = int(data.get("value", 50))
        except Exception as exc:
            _log.debug("sentiment.fng_failed", error=str(exc))

        # ── BTC Funding Rate (Binance perpetuals) ───────────────────
        try:
            from libs.core.config.settings import get_settings
            settings = get_settings()
            base = "https://fapi.binance.com"
            if getattr(settings.binance, "region", "global") == "us":
                base = ""  # Binance.US has no futures
            if base:
                resp = await client.get(
                    f"{base}/fapi/v1/fundingRate",
                    params={"symbol": "BTCUSDT", "limit": "1"},
                )
                rates = resp.json()
                if rates:
                    result["funding_rate"] = float(rates[-1].get("fundingRate", 0.0))
        except Exception as exc:
            _log.debug("sentiment.funding_failed", error=str(exc))

        # ── Long/Short Ratio (Binance top trader accounts) ──────────
        try:
            if base:
                resp = await client.get(
                    f"{base}/futures/data/topLongShortAccountRatio",
                    params={"symbol": "BTCUSDT", "period": "1h", "limit": "1"},
                )
                ls_data = resp.json()
                if ls_data:
                    result["long_short_ratio"] = float(
                        ls_data[-1].get("longShortRatio", 1.0)
                    )
        except Exception as exc:
            _log.debug("sentiment.ls_ratio_failed", error=str(exc))

        # ── BTC Dominance Change (CoinGecko) ────────────────────────
        try:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/global",
            )
            global_data = resp.json().get("data", {})
            btc_dom = global_data.get("market_cap_percentage", {}).get("btc", 0)
            btc_dom_change = global_data.get(
                "market_cap_change_percentage_24h_usd", 0
            )
            result["btc_dominance"] = round(btc_dom, 2)
            result["btc_dominance_change"] = round(btc_dom_change, 2)
        except Exception as exc:
            _log.debug("sentiment.dominance_failed", error=str(exc))

    _log.info(
        "sentiment.fetched",
        fng=result["fear_greed_index"],
        funding=result["funding_rate"],
        ls_ratio=result["long_short_ratio"],
    )
    return result


def build_agent_context(
    market_context: dict[str, Any] | None = None,
    signal: Any | None = None,
    portfolio_state: dict[str, Any] | None = None,
    live_sentiment: dict[str, Any] | None = None,
    regime_analysis: Any | None = None,
    mtf_alignment: Any | None = None,
) -> dict[str, Any]:
    """Build unified context dict for all 15 agents.

    Merges data from multiple sources into a single flat dict.
    Agents extract what they need from this shared context.

    Parameters
    ----------
    market_context: MarketContext.to_dict() or raw dict from trade entry.
    signal: SignalOutput or similar object with signal fields.
    portfolio_state: Dict with open_positions, balance, exposure, etc.
    live_sentiment: Dict from fetch_live_sentiment().
    regime_analysis: RegimeAnalysis dataclass from regime engine.
    mtf_alignment: MTFAlignment dataclass from multi-timeframe analysis.
    """
    ctx: dict[str, Any] = {}

    # ── Market context (indicators, regime, volume, macro) ───────────────
    if market_context:
        # Map entry_ prefixed fields to unprefixed keys for agents
        for key, value in market_context.items():
            if value is None:
                continue
            # Strip "entry_" prefix for agent consumption
            clean_key = key.replace("entry_", "") if key.startswith("entry_") else key
            ctx[clean_key] = value

    # Map market_regime to regime for agent consumption
    if "market_regime" in ctx and "regime" not in ctx:
        ctx["regime"] = ctx["market_regime"]

    # ── Signal data ──────────────────────────────────────────────────────
    if signal is not None:
        ctx.setdefault("confidence", getattr(signal, "confidence", 0.5))
        ctx.setdefault("strategy", getattr(signal, "strategy_name", ""))
        ctx.setdefault("action", _enum_val(getattr(signal, "action", "")))
        ctx.setdefault("asset_class", _enum_val(getattr(signal, "asset_class", "crypto")))
        ctx.setdefault("risk_reward_ratio", getattr(signal, "estimated_risk_reward", 0.0))

        patterns = getattr(signal, "patterns_detected", [])
        ctx.setdefault("patterns", patterns or [])
        ctx.setdefault("pattern_count", len(patterns or []))

        ctx.setdefault("stop_loss", getattr(signal, "stop_loss", 0.0))
        ctx.setdefault("take_profit_1", getattr(signal, "take_profit_1", 0.0))

        # Session
        session = getattr(signal, "session_status", None)
        if session:
            ctx.setdefault("session", _enum_val(session))

    # ── Regime analysis ──────────────────────────────────────────────────
    if regime_analysis is not None:
        ctx.setdefault("regime", _enum_val(getattr(regime_analysis, "regime", "unknown")))
        ctx.setdefault("atr_pct", getattr(regime_analysis, "atr_pct", 0.0))
        ctx.setdefault("vol_score", getattr(regime_analysis, "vol_score", 0.5))
        ctx.setdefault("is_expanding", getattr(regime_analysis, "is_expanding", False))
        ctx.setdefault("is_contracting", getattr(regime_analysis, "is_contracting", False))

        ema_fast = getattr(regime_analysis, "ema_fast", None)
        ema_slow = getattr(regime_analysis, "ema_slow", None)
        if ema_fast and ema_slow and ema_slow > 0:
            ctx.setdefault("ema_alignment", ema_fast / ema_slow)

    # ── Multi-timeframe alignment ────────────────────────────────────────
    if mtf_alignment is not None:
        ctx["htf_alignment"] = getattr(mtf_alignment, "htf_agrees", False)
        ctx["mtf_alignment_score"] = getattr(mtf_alignment, "alignment_score", 0.0)
        ctx["mtf_direction_score"] = getattr(mtf_alignment, "direction_score", 0.0)
        ctx["stacked_bullish"] = getattr(mtf_alignment, "stacked_bullish", False)
        ctx["stacked_bearish"] = getattr(mtf_alignment, "stacked_bearish", False)
        ctx["mtf_divergence"] = getattr(mtf_alignment, "divergence_detected", False)

    # ── Portfolio state ──────────────────────────────────────────────────
    if portfolio_state:
        ctx.setdefault("open_positions", portfolio_state.get("open_positions", 0))
        ctx.setdefault("max_positions", portfolio_state.get("max_positions", 10))
        ctx.setdefault("portfolio_exposure_pct", portfolio_state.get("exposure_pct", 0.0))
        ctx.setdefault("max_drawdown_pct", portfolio_state.get("max_drawdown_pct", 0.0))
        ctx.setdefault("unrealized_pnl_pct", portfolio_state.get("unrealized_pnl_pct", 0.0))
        ctx.setdefault("consecutive_losses", portfolio_state.get("consecutive_losses", 0))
        ctx.setdefault("total_trades", portfolio_state.get("total_trades", 0))
        ctx.setdefault("win_rate", portfolio_state.get("win_rate", 0.5))
        ctx.setdefault("leverage", portfolio_state.get("leverage", 1.0))
        ctx.setdefault("correlation_risk", portfolio_state.get("correlation_risk", 0.0))
        ctx.setdefault("directional_bias_pct", portfolio_state.get("directional_bias_pct", 0.0))

    # ── Live sentiment feeds ─────────────────────────────────────────────
    if live_sentiment:
        ctx.setdefault("fear_greed_index", live_sentiment.get("fear_greed_index", 50))
        ctx.setdefault("funding_rate", live_sentiment.get("funding_rate", 0.0))
        ctx.setdefault("btc_dominance_change", live_sentiment.get("btc_dominance_change", 0.0))
        ctx.setdefault("long_short_ratio", live_sentiment.get("long_short_ratio", 1.0))
        ctx.setdefault("social_sentiment", live_sentiment.get("social_sentiment", 0.0))

    # ── Learning subsystem stats (best-effort) ───────────────────────────
    try:
        from libs.learning.coordinator import get_coordinator
        progress = get_coordinator().get_training_progress()
        ctx.setdefault("total_trades", progress.get("total_trades", 0))
        ctx.setdefault("win_rate", progress.get("win_rate", 0.5))

        # Strategy win rates for strategy selection agent
        tuner_stats = progress.get("strategy_tuner_stats", {})
        if tuner_stats:
            strategy_win_rates = {}
            for name, stats in tuner_stats.items():
                total = stats.get("total_outcomes", 0)
                wins = stats.get("wins", 0)
                if total > 0:
                    strategy_win_rates[name] = wins / total
            ctx.setdefault("strategy_win_rates", strategy_win_rates)
    except Exception:
        pass

    # ── Defaults for required keys ───────────────────────────────────────
    ctx.setdefault("regime", "unknown")
    ctx.setdefault("rsi", 50.0)
    ctx.setdefault("macd_histogram", 0.0)
    ctx.setdefault("atr_pct", 0.0)
    ctx.setdefault("volume_ratio", 1.0)
    ctx.setdefault("ema_alignment", 1.0)
    ctx.setdefault("bb_position", 0.5)
    ctx.setdefault("confidence", 0.5)
    ctx.setdefault("spread_pct", 0.0)
    ctx.setdefault("hour", 12)
    ctx.setdefault("day_of_week", 0)

    return ctx


def _enum_val(obj: Any) -> Any:
    """Extract .value from enum or return as-is."""
    return obj.value if hasattr(obj, "value") else obj
