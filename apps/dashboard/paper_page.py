"""
Paper trading dashboard page and API endpoints.

Mounted at /paper (HTML) and /api/paper/* (JSON APIs).

The engine is injected at runtime via set_paper_engine() by the runner.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from libs.core.logging.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

# ── Engine reference ──────────────────────────────────────────────────────────

_engine = None


def set_paper_engine(engine) -> None:
    global _engine
    _engine = engine


# ── JSON API endpoints ────────────────────────────────────────────────────────

@router.get("/api/paper/summary")
async def paper_summary() -> JSONResponse:
    if _engine is None:
        return JSONResponse(
            content={"error": "Paper trading not started"},
            status_code=503,
        )
    try:
        # Fetch live prices for accurate unrealized P&L
        prices: dict[str, float] = {}
        try:
            positions = _engine.get_all_open_positions()
            symbols = {p["symbol"] for p in positions}
            for sym in symbols:
                try:
                    if sym.endswith("USDT"):
                        from libs.data.providers.binance.provider import BinanceDataProvider
                        price = await BinanceDataProvider().get_latest_price(sym)
                    else:
                        from libs.data.providers.alpaca.provider import AlpacaDataProvider
                        price = await AlpacaDataProvider().get_latest_price(sym)
                    if price:
                        prices[sym] = price
                except Exception:
                    pass
        except Exception:
            pass
        summary = _engine.get_summary(live_prices=prices)
        return JSONResponse(content=summary)
    except Exception as exc:
        log.warning("paper_summary_failed", error=str(exc))
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.get("/api/paper/positions")
async def paper_positions() -> JSONResponse:
    if _engine is None:
        return JSONResponse(content={"positions": []})
    try:
        positions = _engine.get_all_open_positions()

        # Fetch live prices — use futures endpoint for futures positions
        futures_syms = {p["symbol"] for p in positions if "futures" in str(p.get("trading_mode", ""))}
        spot_syms = {p["symbol"] for p in positions if p["symbol"] not in futures_syms}
        prices: dict[str, float] = {}
        for sym in spot_syms:
            try:
                if sym.endswith("USDT"):
                    from libs.data.providers.binance.provider import BinanceDataProvider
                    price = await BinanceDataProvider().get_latest_price(sym)
                else:
                    from libs.data.providers.alpaca.provider import AlpacaDataProvider
                    price = await AlpacaDataProvider().get_latest_price(sym)
                if price:
                    prices[sym] = price
            except Exception:
                pass
        for sym in futures_syms:
            try:
                from libs.data.providers.binance_futures.provider import BinanceFuturesProvider
                price = await BinanceFuturesProvider().get_latest_price(sym)
                if price:
                    prices[sym] = price
            except Exception:
                pass

        for p in positions:
            live = prices.get(p["symbol"])
            if live:
                p["live_price"] = live
                entry = p["entry_price"]
                size = p["position_size_usd"]
                lev = p.get("leverage", 1.0) or 1.0
                units = size / entry if entry else 0
                if p["action"] == "BUY":
                    pnl = (live - entry) * units * lev
                else:
                    pnl = (entry - live) * units * lev
                pnl_pct = (pnl / size * 100) if size else 0
                p["unrealized_pnl"] = round(pnl, 4)
                p["unrealized_pnl_pct"] = round(pnl_pct, 2)
            else:
                p["live_price"] = None
                p["unrealized_pnl"] = 0
                p["unrealized_pnl_pct"] = 0

        return JSONResponse(content={"positions": positions})
    except Exception as exc:
        log.warning("paper_positions_failed", error=str(exc))
        return JSONResponse(content={"positions": []})


@router.get("/api/paper/trades")
async def paper_trades(limit: int = 100) -> JSONResponse:
    limit = min(max(limit, 1), 500)
    if _engine is None:
        return JSONResponse(content={"trades": []})
    try:
        trades = _engine.get_closed_trades(limit)
        return JSONResponse(content={"trades": trades})
    except Exception as exc:
        log.warning("paper_trades_failed", error=str(exc))
        return JSONResponse(content={"trades": []})


@router.get("/api/paper/equity")
async def paper_equity() -> JSONResponse:
    if _engine is None:
        return JSONResponse(content=[])
    try:
        curve = _engine.get_equity_curve()
        return JSONResponse(content=curve)
    except Exception as exc:
        log.warning("paper_equity_failed", error=str(exc))
        return JSONResponse(content=[])


@router.post("/api/paper/bot/{bot_name}/pause")
async def paper_pause_bot(bot_name: str) -> JSONResponse:
    if _engine is None:
        return JSONResponse(
            content={"error": "Paper trading not started"},
            status_code=503,
        )
    try:
        _engine.pause_bot(bot_name)
        return JSONResponse(content={"ok": True, "bot": bot_name, "state": "paused"})
    except Exception as exc:
        log.warning("paper_pause_bot_failed", bot=bot_name, error=str(exc))
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post("/api/paper/bot/{bot_name}/resume")
async def paper_resume_bot(bot_name: str) -> JSONResponse:
    if _engine is None:
        return JSONResponse(
            content={"error": "Paper trading not started"},
            status_code=503,
        )
    try:
        _engine.resume_bot(bot_name)
        return JSONResponse(content={"ok": True, "bot": bot_name, "state": "running"})
    except Exception as exc:
        log.warning("paper_resume_bot_failed", bot=bot_name, error=str(exc))
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post("/api/paper/reset")
async def paper_reset() -> JSONResponse:
    if _engine is None:
        return JSONResponse(
            content={"error": "Paper trading not started"},
            status_code=503,
        )
    try:
        _engine.reset()
        return JSONResponse(content={"ok": True, "message": "Paper engine reset to $10,000"})
    except Exception as exc:
        log.warning("paper_reset_failed", error=str(exc))
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.get("/api/paper/preview")
async def signal_preview() -> JSONResponse:
    """Scan all watchlist symbols and show bias + upcoming trade potential."""
    from datetime import timedelta
    from libs.core.config.settings import get_settings
    from libs.core.models.domain import Timeframe
    from libs.data.providers.cache import ProviderCache
    from libs.data.candles.builder import CandleBuilder
    from libs.analysis.indicators.engine import IndicatorsEngine
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    from libs.analysis.regime.engine import RegimeEngine
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    import math

    try:
        settings = get_settings()
        symbols = settings.signal.crypto_symbols[:20]  # Top 20
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=3)
        previews = []

        for sym in symbols:
            try:
                provider = ProviderCache.binance() if sym.endswith("USDT") else ProviderCache.alpaca()
                df = await provider.get_candles(sym, Timeframe.FIFTEEN_MIN, start, now)
                df = CandleBuilder().enrich(df)
                if len(df) < 20:
                    continue

                indicators = IndicatorsEngine().compute(df)
                regime = RegimeEngine().analyze(df)
                structure = MarketStructureAnalyzer().analyze(df)

                last_close = float(df["close"].iloc[-1])
                rel_vol = float(df["relative_volume"].iloc[-1]) if "relative_volume" in df else 1.0
                is_bull = bool(df["is_bullish"].iloc[-1]) if "is_bullish" in df else True

                indicator_bias = IndicatorBiasAnalyzer().analyze_all(
                    rsi=indicators.rsi or 50, rsi_prev=indicators.rsi_prev or 50,
                    macd_line=indicators.macd_line or 0, macd_signal=indicators.macd_signal or 0,
                    histogram=indicators.macd_histogram or 0, histogram_prev=indicators.macd_histogram_prev or 0,
                    close=last_close, bb_upper=indicators.bb_upper or last_close + 1,
                    bb_lower=indicators.bb_lower or last_close - 1, bb_pct_b=indicators.bb_pct_b or 0.5,
                    ema_9=indicators.ema_9 or last_close, ema_20=indicators.ema_20 or last_close,
                    ema_50=indicators.ema_50 or last_close,
                    adx=indicators.adx or 0,
                    relative_volume=rel_vol, is_bullish_candle=is_bull,
                )

                bias = BullBearBiasEngine().score(BiasInput(
                    indicator_bullish=indicator_bias.bullish_score,
                    indicator_bearish=indicator_bias.bearish_score,
                    structure_bias=structure.trend_bias,
                    structure_strength=structure.strength,
                    candle_bullish_count=0, candle_bearish_count=0, candle_total=1,
                    regime_supports_direction=True,
                    volume_confirms=rel_vol > 1.2,
                    htf_bias="neutral",
                ))

                def _s(v):
                    if v is None: return None
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
                    return round(v, 4)

                # Readiness: match pipeline logic exactly
                adx_val = indicators.adx if indicators.adx else 0
                gap = abs(bias.bullish_score - bias.bearish_score)

                # Check if symbol already held
                held_symbols = set()
                if _engine:
                    for pos in _engine.get_all_open_positions():
                        held_symbols.add(pos["symbol"])

                if sym in held_symbols:
                    readiness = "HELD"
                elif adx_val < 20:
                    readiness = "NO TREND"  # ADX too low — pipeline blocks
                elif bias.net_bias != "neutral":
                    readiness = "READY" if gap > 0.15 else "ALMOST"
                else:
                    readiness = "WAITING"

                # Participation matrix
                from libs.analysis.participation.matrix import get_participation
                part = get_participation(
                    net_bias=bias.net_bias,
                    bullish_score=bias.bullish_score,
                    bearish_score=bias.bearish_score,
                    conflict_score=bias.conflict_score,
                    adx=indicators.adx or 0,
                )

                previews.append({
                    "symbol": sym,
                    "bias": bias.net_bias,
                    "bullish": _s(bias.bullish_score),
                    "bearish": _s(bias.bearish_score),
                    "conflict": _s(bias.conflict_score),
                    "regime": regime.regime.value if hasattr(regime.regime, "value") else str(regime.regime),
                    "structure": structure.trend_bias,
                    "rsi": _s(indicators.rsi),
                    "adx": _s(indicators.adx),
                    "volume": _s(rel_vol),
                    "readiness": readiness,
                    "price": _s(last_close),
                    "market_state": part.market_state.value,
                    "spot_ok": part.spot_allowed,
                    "fut_long_ok": part.futures_long_allowed,
                    "fut_short_ok": part.futures_short_allowed,
                    "part_reason": part.reason,
                })
            except Exception:
                continue

        # Sort: READY first, then ALMOST, then WAITING
        order = {"READY": 0, "ALMOST": 1, "HELD": 2, "WAITING": 3, "NO TREND": 4}
        previews.sort(key=lambda p: order.get(p["readiness"], 3))

        return JSONResponse(content={"previews": previews})
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.get("/api/paper/chart/{symbol}")
async def paper_chart_data(symbol: str, timeframe: str = "15m") -> JSONResponse:
    """Return OHLCV candles + indicators for charting."""
    from datetime import datetime, timedelta, timezone
    from libs.core.models.domain import Timeframe

    tf_map = {"1s": Timeframe.ONE_MIN, "1m": Timeframe.ONE_MIN, "5m": Timeframe.FIVE_MIN,
              "15m": Timeframe.FIFTEEN_MIN, "30m": Timeframe.THIRTY_MIN,
              "1h": Timeframe.ONE_HOUR, "4h": Timeframe.FOUR_HOUR,
              "1d": Timeframe.ONE_DAY, "1w": Timeframe.ONE_WEEK}
    # For 1s, we'll use Binance's 1s kline endpoint separately
    tf = tf_map.get(timeframe, Timeframe.FIFTEEN_MIN)

    try:
        now = datetime.now(timezone.utc)
        # Adjust lookback based on timeframe to get 500 bars
        lookback_map = {"1s": 1, "1m": 1, "5m": 3, "15m": 7, "30m": 14,
                        "1h": 25, "4h": 90, "1d": 550, "1w": 3650}
        days = lookback_map.get(timeframe, 7)
        start = now - timedelta(days=days)

        from libs.data.candles.builder import CandleBuilder

        if timeframe == "1s" and symbol.endswith("USDT"):
            # Binance 1s klines — fetch 500 seconds of data via REST
            import httpx
            import pandas as pd
            end_ms = int(now.timestamp() * 1000)
            start_ms = end_ms - (500 * 1000)  # 500 seconds back
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1s&startTime={start_ms}&endTime={end_ms}&limit=500"
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                raw = resp.json()
            if not raw:
                return JSONResponse(content={"error": "No 1s data"}, status_code=404)
            df = pd.DataFrame(raw, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trade_count",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ])
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df.set_index("timestamp")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = CandleBuilder().enrich(df)
        else:
            if symbol.endswith("USDT"):
                from libs.data.providers.binance.provider import BinanceDataProvider
                provider = BinanceDataProvider()
            else:
                from libs.data.providers.alpaca.provider import AlpacaDataProvider
                provider = AlpacaDataProvider()
            df = await provider.get_candles(symbol, tf, start, now)
            df = CandleBuilder().enrich(df)

        # Compute indicators
        from libs.analysis.indicators.engine import IndicatorsEngine
        from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
        from libs.analysis.structure.market_structure import MarketStructureAnalyzer
        from libs.analysis.regime.engine import RegimeEngine

        indicators = IndicatorsEngine().compute(df)
        regime = RegimeEngine().analyze(df)
        structure = MarketStructureAnalyzer().analyze(df)

        last_close = float(df["close"].iloc[-1])
        rel_vol = float(df["relative_volume"].iloc[-1]) if "relative_volume" in df else 1.0
        is_bull = bool(df["is_bullish"].iloc[-1]) if "is_bullish" in df else True

        bias_report = IndicatorBiasAnalyzer().analyze_all(
            rsi=indicators.rsi or 50, rsi_prev=indicators.rsi_prev or 50,
            macd_line=indicators.macd_line or 0, macd_signal=indicators.macd_signal or 0,
            histogram=indicators.macd_histogram or 0, histogram_prev=indicators.macd_histogram_prev or 0,
            close=last_close, bb_upper=indicators.bb_upper or last_close + 1,
            bb_lower=indicators.bb_lower or last_close - 1, bb_pct_b=indicators.bb_pct_b or 0.5,
            ema_9=indicators.ema_9 or last_close, ema_20=indicators.ema_20 or last_close,
            ema_50=indicators.ema_50 or last_close,
            adx=indicators.adx or 0,
            relative_volume=rel_vol, is_bullish_candle=is_bull,
        )

        # Build candle array for Lightweight Charts
        candles = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp()) if hasattr(idx, 'timestamp') else 0
            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 6),
                "high": round(float(row["high"]), 6),
                "low": round(float(row["low"]), 6),
                "close": round(float(row["close"]), 6),
                "volume": round(float(row["volume"]), 2) if "volume" in row else 0,
            })

        # Get position info for this symbol
        positions = []
        if _engine:
            for p in _engine.get_all_open_positions():
                if p["symbol"] == symbol:
                    positions.append(p)

        import math
        def _safe(v):
            """Replace NaN/Inf with None for JSON safety."""
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v

        resp_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles[-500:],
            "indicators": {
                "rsi": _safe(round(indicators.rsi, 1) if indicators.rsi else None),
                "macd_line": _safe(round(indicators.macd_line, 6) if indicators.macd_line else None),
                "macd_signal": _safe(round(indicators.macd_signal, 6) if indicators.macd_signal else None),
                "macd_histogram": _safe(round(indicators.macd_histogram, 6) if indicators.macd_histogram else None),
                "ema_9": _safe(round(indicators.ema_9, 6) if indicators.ema_9 else None),
                "ema_20": _safe(round(indicators.ema_20, 6) if indicators.ema_20 else None),
                "ema_50": _safe(round(indicators.ema_50, 6) if indicators.ema_50 else None),
                "bb_upper": _safe(round(indicators.bb_upper, 6) if indicators.bb_upper else None),
                "bb_lower": _safe(round(indicators.bb_lower, 6) if indicators.bb_lower else None),
                "adx": _safe(round(indicators.adx, 1) if indicators.adx else None),
                "atr": _safe(round(indicators.atr, 6) if indicators.atr else None),
                "volume_relative": _safe(round(rel_vol, 2)),
            },
            "bias": {
                "net": bias_report.net_bias,
                "bullish": _safe(round(bias_report.bullish_score, 3)),
                "bearish": _safe(round(bias_report.bearish_score, 3)),
            },
            "structure": {
                "trend": structure.trend_bias,
                "strength": _safe(round(structure.strength, 3)),
                "last_swing_high": structure.last_swing_high,
                "last_swing_low": structure.last_swing_low,
                "events": [{"kind": e.kind, "direction": e.direction, "price": e.price} for e in structure.events[-5:]],
            },
            "regime": {
                "name": regime.regime.value if hasattr(regime.regime, 'value') else str(regime.regime),
                "vol_score": _safe(round(regime.vol_score, 3) if hasattr(regime, 'vol_score') else 0),
            },
            "positions": positions,
            "patterns": [],
            "strategies": [],
        }

        # Enrich with pattern detection + strategy signals
        try:
            from apps.signal_agent.pipeline import DEFAULT_DETECTORS
            from libs.core.models.domain import AssetClass
            for det in DEFAULT_DETECTORS:
                try:
                    result = det.detect(df)
                    if result and hasattr(result, 'detected') and result.detected:
                        resp_data["patterns"].append({
                            "name": getattr(result, 'pattern_name', det.__class__.__name__),
                            "bias": getattr(result, 'bias', 'neutral'),
                            "category": getattr(result, 'category', 'unknown'),
                            "strength": round(getattr(result, 'strength_score', 0), 2),
                            "explanation": getattr(result, 'explanation', ''),
                        })
                except Exception:
                    pass
        except Exception:
            pass

        try:
            from apps.signal_agent.runner import DEFAULT_STRATEGIES
            from libs.core.models.domain import AssetClass
            from libs.analysis.structure.engine import MarketStructureEngine
            from libs.analysis.levels.engine import KeyLevelsEngine
            struct = MarketStructureEngine().analyze(df)
            ac = AssetClass.CRYPTO if symbol.endswith('USDT') else AssetClass.STOCK
            levels = KeyLevelsEngine().analyze(df, ac)
            for strat in DEFAULT_STRATEGIES:
                try:
                    if len(df) < strat.min_bars_required:
                        continue
                    candidate = strat.generate_candidate(
                        symbol=symbol, asset_class=ac,
                        df=df, df_htf=None, session=None, quality=None,
                        structure=struct, levels=levels, volume=None,
                        regime=regime, indicators=indicators,
                    )
                    if candidate and candidate.proposed_action.value != 'NO_TRADE':
                        resp_data["strategies"].append({
                            "name": strat.name,
                            "action": candidate.proposed_action.value,
                            "entry": round((candidate.entry_zone_low + candidate.entry_zone_high) / 2, 6),
                            "stop": round(candidate.stop_loss, 6) if candidate.stop_loss else None,
                            "tp1": round(candidate.take_profit_1, 6) if candidate.take_profit_1 else None,
                        })
                except Exception:
                    pass
        except Exception:
            pass

        return JSONResponse(content=resp_data)
    except Exception as exc:
        log.warning("chart_data_failed", symbol=symbol, error=str(exc))
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ── HTML Dashboard ────────────────────────────────────────────────────────────

@router.get("/paper", response_class=HTMLResponse)
async def paper_dashboard() -> HTMLResponse:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Paper Trading — AI Signal Agent</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;600;700&family=Orbitron:wght@400;500;600;700&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #0B0F1A; --surface: rgba(17,24,42,0.75); --surface-solid: #111828;
      --border: rgba(255,255,255,0.08); --border-hover: rgba(255,255,255,0.15);
      --muted: #64748B; --text: #E2E8F0; --bright: #F8FAFC;
      --blue: #38BDF8; --green: #10B981; --red: #EF4444; --yellow: #F59E0B;
      --cyan: #22D3EE; --accent: #F59E0B; --surface2: rgba(24,36,64,0.6);
      --glass-blur: 16px; --glass-border: 1px solid rgba(255,255,255,0.1);
      --spring: cubic-bezier(0.34, 1.56, 0.64, 1);
      --smooth: cubic-bezier(0.16, 1, 0.3, 1);
      --ease-out: cubic-bezier(0, 0, 0.2, 1);
    }

    body {
      font-family: 'Exo 2', system-ui, sans-serif;
      background: var(--bg); color: var(--text);
      min-height: 100vh; font-size: 13px;
      background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(56,189,248,0.08), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(16,185,129,0.05), transparent);
      background-attachment: fixed;
    }

    /* ── Framer Motion-style keyframes ── */
    @keyframes fm-fade-up {
      from { opacity: 0; transform: translateY(16px) scale(0.98); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes fm-slide-up {
      from { opacity: 0; transform: translateY(30px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fm-scale-in {
      from { opacity: 0; transform: scale(0.95); }
      to   { opacity: 1; transform: scale(1); }
    }
    @keyframes fm-glow-pulse {
      0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.0); }
      50%      { box-shadow: 0 0 40px rgba(16,185,129,0.15); }
    }
    @keyframes shimmer {
      0%   { background-position: -200% 0; }
      100% { background-position: 200% 0; }
    }
    @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }

    /* Staggered panel reveals */
    .panel { animation: fm-fade-up 0.5s var(--smooth) both; }
    .panel:nth-child(1) { animation-delay: 0.05s; }
    .panel:nth-child(2) { animation-delay: 0.12s; }
    .panel:nth-child(3) { animation-delay: 0.19s; }
    .panel:nth-child(4) { animation-delay: 0.26s; }
    .panel:nth-child(5) { animation-delay: 0.33s; }
    tr { transition: background 0.2s var(--ease-out), transform 0.2s var(--ease-out); }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
    }

    /* ── Floating Nav ── */
    header {
      background: rgba(17,24,42,0.85); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
      border: var(--glass-border); border-radius: 14px;
      padding: 10px 20px; display: flex; align-items: center;
      justify-content: space-between; position: sticky; top: 10px; z-index: 100;
      margin: 10px 16px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      animation: fm-fade-up 0.4s var(--smooth) both;
    }
    header h1 { font-family: 'Orbitron', sans-serif; font-size: 0.85rem; color: var(--cyan); font-weight: 600; letter-spacing: 1px; }
    .header-right { display: flex; align-items: center; gap: 10px; font-size: 0.72rem; color: var(--muted); }
    .back-link { color: var(--blue); text-decoration: none; font-size: 0.78rem; transition: color 0.2s; }
    .back-link:hover { color: var(--cyan); text-decoration: underline; }
    .virtual-badge {
      padding: 3px 12px; border-radius: 20px; font-weight: 700;
      font-size: 0.62rem; color: var(--yellow);
      background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.25);
      letter-spacing: 0.8px; text-transform: uppercase;
    }

    /* ── Hero money counter ── */
    .hero {
      background: linear-gradient(160deg, rgba(17,24,42,0.9) 0%, rgba(24,36,64,0.7) 100%);
      backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 44px 20px 32px; text-align: center;
      animation: fm-slide-up 0.6s var(--smooth) both;
      animation-delay: 0.1s;
      position: relative; overflow: hidden;
    }
    .hero::before {
      content: ''; position: absolute; top: -50%; left: -50%;
      width: 200%; height: 200%;
      background: radial-gradient(circle at 50% 120%, rgba(16,185,129,0.06), transparent 60%);
      pointer-events: none;
    }
    #money-h1 {
      font-family: 'Orbitron', sans-serif;
      font-size: 4.5rem; font-weight: 700; color: var(--bright);
      letter-spacing: -1px; transition: color 0.4s var(--smooth), text-shadow 0.4s var(--smooth);
      font-variant-numeric: tabular-nums;
      text-shadow: 0 2px 20px rgba(0,0,0,0.4);
      position: relative; z-index: 1;
    }
    #money-h1.profit { color: var(--green); text-shadow: 0 0 40px rgba(16,185,129,0.3), 0 0 80px rgba(16,185,129,0.1); }
    #money-h1.loss   { color: var(--red); text-shadow: 0 0 40px rgba(239,68,68,0.3), 0 0 80px rgba(239,68,68,0.1); }

    @keyframes pulse-green { 0%,100% { text-shadow: 0 0 20px rgba(16,185,129,0.15); } 50% { text-shadow: 0 0 60px rgba(16,185,129,0.35); } }
    @keyframes pulse-red   { 0%,100% { text-shadow: 0 0 20px rgba(239,68,68,0.15); } 50% { text-shadow: 0 0 60px rgba(239,68,68,0.35); } }
    #money-h1.profit { animation: pulse-green 3s ease infinite; }
    #money-h1.loss   { animation: pulse-red 3s ease infinite; }

    .hero-sub {
      margin-top: 16px; display: flex; justify-content: center;
      gap: 28px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted);
      position: relative; z-index: 1;
    }
    .hero-sub span { white-space: nowrap; }
    .hero-sub .val { color: var(--text); font-weight: 600; }

    /* ── Layout ── */
    main { padding: 16px 20px; max-width: 1700px; margin: 0 auto; }

    /* ── Glass Panel ── */
    .panel {
      background: var(--surface); backdrop-filter: blur(var(--glass-blur)); -webkit-backdrop-filter: blur(var(--glass-blur));
      border: var(--glass-border); border-radius: 14px;
      overflow: hidden; margin-bottom: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.04);
      transition: box-shadow 0.3s var(--ease-out), transform 0.3s var(--ease-out), border-color 0.3s;
    }
    .panel:hover { box-shadow: 0 8px 40px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.06); border-color: var(--border-hover); }
    .panel-header {
      padding: 12px 18px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
      background: rgba(24,36,64,0.4);
    }
    .panel-header h2 { font-family: 'Orbitron', sans-serif; font-size: 0.7rem; color: var(--bright); font-weight: 500; letter-spacing: 0.8px; text-transform: uppercase; }

    /* ── Tables ── */
    table { width: 100%; border-collapse: collapse; font-size: 0.73rem; }
    th {
      padding: 8px 12px; text-align: left; color: var(--muted);
      font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.8px;
      border-bottom: 1px solid var(--border); white-space: nowrap;
      font-family: 'Orbitron', sans-serif; font-weight: 400;
    }
    td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); white-space: nowrap; vertical-align: middle; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(56,189,248,0.04); }
    .empty { padding: 32px; text-align: center; color: var(--muted); font-size: 0.78rem; }

    /* ── Colors ── */
    .green  { color: var(--green)  !important; }
    .red    { color: var(--red)    !important; }
    .yellow { color: var(--yellow) !important; }
    .blue   { color: var(--blue)   !important; }
    .muted  { color: var(--muted)  !important; }

    /* ── Rank badge ── */
    .rank { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 8px; font-weight: bold; color: var(--muted); font-size: 0.72rem; }
    .rank.gold   { color: #ffd700; background: rgba(255,215,0,0.1); }
    .rank.silver { color: #c0c0c0; background: rgba(192,192,192,0.08); }
    .rank.bronze { color: #cd7f32; background: rgba(205,127,50,0.08); }

    /* ── Bot action buttons ── */
    .bot-btn {
      background: none; border: 1px solid var(--border); border-radius: 8px;
      padding: 3px 10px; font-size: 0.65rem; color: var(--muted);
      cursor: pointer; font-family: inherit;
      transition: all 0.2s var(--spring);
    }
    .bot-btn:hover { transform: translateY(-1px); }
    .bot-btn.pause:hover { border-color: var(--yellow); color: var(--yellow); background: rgba(245,158,11,0.08); }
    .bot-btn.resume:hover { border-color: var(--green); color: var(--green); background: rgba(16,185,129,0.08); }

    /* ── Equity placeholder ── */
    .equity-placeholder {
      padding: 32px; text-align: center; color: var(--muted);
      font-size: 0.82rem; border: 1px dashed var(--border);
      border-radius: 8px; margin: 12px 14px;
    }

    /* ── Filter bar ── */
    .filter-bar { padding: 8px 14px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
    select, input {
      background: rgba(15,23,42,0.6); border: 1px solid var(--border); color: var(--text);
      padding: 5px 10px; border-radius: 8px; font-family: inherit; font-size: 0.76rem;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    select:focus, input:focus { outline: none; border-color: var(--blue); box-shadow: 0 0 0 3px rgba(56,189,248,0.15); }

    /* ── Refresh controls ── */
    .rbtn {
      background: none; border: 1px solid var(--border); border-radius: 10px;
      padding: 3px 10px; font-size: 0.66rem; color: var(--muted);
      cursor: pointer; font-family: inherit;
      transition: all 0.2s var(--spring);
    }
    .rbtn:hover { border-color: var(--cyan); color: var(--cyan); transform: translateY(-1px); }

    /* ── Phase badge ── */
    .phase-badge {
      display: inline-block; padding: 2px 8px; border-radius: 6px;
      font-size: 0.6rem; font-weight: 600; letter-spacing: 0.5px;
      background: rgba(30,40,60,0.5); color: var(--muted);
    }
    .phase-badge.running { background: rgba(16,185,129,0.12); color: var(--green); border: 1px solid rgba(16,185,129,0.2); }
    .phase-badge.paused  { background: rgba(245,158,11,0.12); color: var(--yellow); border: 1px solid rgba(245,158,11,0.2); }
    .phase-badge.stopped { background: rgba(239,68,68,0.12); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }

    #last-refresh { color: var(--muted); font-size: 0.65rem; }

    /* ── Buttons ── */
    .btn-sm {
      padding: 4px 12px; font-size: 0.7rem; border-radius: 8px;
      background: rgba(15,23,42,0.5); color: var(--muted);
      border: 1px solid var(--border); cursor: pointer; font-family: inherit;
      transition: all 0.2s var(--spring);
    }
    .btn-sm:hover { border-color: var(--cyan); color: var(--cyan); background: rgba(34,211,238,0.06); transform: translateY(-1px); }
    .btn-action {
      padding: 6px 16px; border-radius: 10px; background: linear-gradient(135deg, var(--green), #059669);
      color: #fff; border: none; cursor: pointer; font-family: inherit;
      font-size: 0.75rem; font-weight: 600;
      transition: all 0.25s var(--spring);
      box-shadow: 0 2px 12px rgba(16,185,129,0.2);
    }
    .btn-action:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16,185,129,0.35); }
    .btn-reset {
      padding: 5px 14px; border-radius: 10px;
      background: rgba(239,68,68,0.15); color: var(--red);
      border: 1px solid rgba(239,68,68,0.3); cursor: pointer;
      font-family: inherit; font-size: 0.72rem;
      transition: all 0.2s var(--spring);
    }
    .btn-reset:hover { background: rgba(239,68,68,0.25); transform: translateY(-1px); }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* ── Selection ── */
    ::selection { background: rgba(56,189,248,0.25); color: var(--bright); }

    /* ── Chart Modal ── */
    .chart-overlay { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); z-index:1000; }
    .chart-overlay.open { display:flex; align-items:center; justify-content:center; animation: fm-scale-in 0.3s var(--smooth); }
    .chart-modal { background:var(--surface-solid); border:var(--glass-border); border-radius:16px; width:92%; max-width:1200px; max-height:90vh; overflow:auto; box-shadow: 0 24px 64px rgba(0,0,0,0.5); }
    .chart-header { display:flex; align-items:center; justify-content:space-between; padding:12px 18px; border-bottom:1px solid var(--border); }
    .chart-header h3 { font-family:'Orbitron',sans-serif; font-size:0.85rem; color:var(--bright); font-weight:500; letter-spacing:0.5px; }
    .chart-close { background:none; border:none; color:var(--muted); font-size:1.2rem; cursor:pointer; transition: color 0.2s, transform 0.2s var(--spring); }
    .chart-close:hover { color:var(--red); transform:scale(1.15); }
    .chart-body { display:flex; gap:0; }
    .chart-candles { flex:1; min-height:500px; }
    .chart-indicators { width:290px; padding:14px; border-left:1px solid var(--border); font-size:0.72rem; overflow-y:auto; }
    .ind-row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
    .ind-label { color:var(--muted); font-size:0.68rem; }
    .ind-val { font-weight:600; font-variant-numeric:tabular-nums; }
    .ind-section { margin-top:12px; padding-top:8px; border-top:1px solid var(--border); }
    .ind-section h4 { font-family:'Orbitron',sans-serif; font-size:0.6rem; color:var(--cyan); text-transform:uppercase; letter-spacing:1.2px; margin-bottom:8px; font-weight:400; }
    .bias-badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.65rem; font-weight:600; }
    .bias-badge.bullish { background:rgba(16,185,129,0.12); color:var(--green); border:1px solid rgba(16,185,129,0.2); }
    .bias-badge.bearish { background:rgba(239,68,68,0.12); color:var(--red); border:1px solid rgba(239,68,68,0.2); }
    .bias-badge.neutral { background:rgba(100,116,139,0.12); color:var(--muted); border:1px solid rgba(100,116,139,0.2); }
    .struct-event { font-size:0.65rem; padding:2px 0; }
  </style>
</head>
<body>

<!-- Navigation -->
<header>
  <h1>ALPHA SIGNAL ENGINE</h1>
  <div class="header-right">
    <a href="/" class="back-link">Main Dashboard</a>
    <span class="virtual-badge">PAPER TRADING</span>
    <span id="last-refresh">--</span>
    <button class="rbtn" onclick="refreshAll()">Refresh</button>
    <button class="btn-reset" onclick="resetAll()">Reset $10K</button>
  </div>
</header>

<!-- Hero: Giant money counter -->
<div class="hero">
  <h1 id="money-h1">$10,000.00</h1>
  <div class="hero-sub">
    <span>P&amp;L: <span class="val" id="hero-pnl">$0.00</span></span>
    <span>P&amp;L%: <span class="val" id="hero-pnl-pct">0.00%</span></span>
    <span>Started: <span class="val" id="hero-start">$10,000.00</span></span>
    <span>Uptime: <span class="val" id="hero-uptime">—</span></span>
    <span>Bots: <span class="val" id="hero-bots">—</span></span>
    <span>Open Pos: <span class="val" id="hero-open-pos">—</span></span>
  </div>
</div>

<main>

  <!-- Bot Leaderboard -->
  <div class="panel">
    <div class="panel-header">
      <h2>Bot Leaderboard</h2>
      <button class="rbtn" onclick="loadSummary()">↻</button>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Bot</th>
            <th>Balance</th>
            <th>P&amp;L</th>
            <th>P&amp;L%</th>
            <th>Win Rate</th>
            <th>Trades</th>
            <th>Sharpe</th>
            <th>Phase</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="leaderboard-body">
          <tr><td colspan="10" class="empty">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Signal Preview -->
  <div class="panel">
    <div class="panel-header">
      <h2>Signal Preview</h2>
      <button class="btn-sm" onclick="loadPreview()">Scan Now</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>Status</th><th>Symbol</th><th>Market State</th><th>SPOT</th><th>FUT L</th><th>FUT S</th>
          <th>Bias</th><th>Bull%</th><th>Bear%</th>
          <th>Conflict</th><th>RSI</th><th>ADX</th><th>Price</th>
        </tr>
      </thead>
      <tbody id="preview-body">
        <tr><td colspan="13" class="empty">Click "Scan Now" to preview signals</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Equity Curve -->
  <div class="panel">
    <div class="panel-header">
      <h2>Equity Curve</h2>
      <button class="btn-sm" onclick="loadEquity()">Refresh</button>
    </div>
    <div id="equity-container" style="height:200px;background:#0a0e17;"></div>
  </div>

  <!-- Open Positions -->
  <div class="panel">
    <div class="panel-header">
      <h2>Open Positions</h2>
      <button class="rbtn" onclick="loadPositions()">↻</button>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Bot</th>
            <th>Symbol</th>
            <th>Mode</th>
            <th>Side</th>
            <th>Intent</th>
            <th>Direction</th>
            <th>Entry</th>
            <th>Live Price</th>
            <th>P&L</th>
            <th>P&L%</th>
            <th>Size</th>
            <th>Leverage</th>
            <th>Entry Bias</th>
            <th>Bias Status</th>
            <th>Stop</th>
            <th>TP1</th>
            <th>Strategy</th>
            <th>Since</th>
          </tr>
        </thead>
        <tbody id="positions-body">
          <tr><td colspan="10" class="empty">No open positions</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Trade History -->
  <div class="panel">
    <div class="panel-header">
      <h2>Trade History</h2>
      <div style="display:flex;gap:8px;align-items:center">
        <label style="color:var(--muted);font-size:0.7rem">Bot</label>
        <select id="trade-bot-filter" onchange="filterTrades()" style="width:130px">
          <option value="">All bots</option>
        </select>
        <button class="rbtn" onclick="loadTrades()">↻</button>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Bot</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>P&amp;L</th>
            <th>P&amp;L%</th>
            <th>Hold</th>
            <th>Strategy</th>
          </tr>
        </thead>
        <tbody id="trades-body">
          <tr><td colspan="10" class="empty">No closed trades yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</main>

<script>
// ── State ──────────────────────────────────────────────────────────────────────
let allTrades = [];
let botNames = new Set();

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(n, d = 2) {
  if (n == null) return '—';
  return parseFloat(n).toFixed(d);
}

function fmtMoney(n) {
  if (n == null) return '—';
  return '$' + parseFloat(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPct(n) {
  if (n == null) return '—';
  const v = parseFloat(n);
  const sign = v >= 0 ? '+' : '';
  return sign + v.toFixed(2) + '%';
}

function pnlColor(n) {
  if (n == null) return 'var(--muted)';
  return parseFloat(n) >= 0 ? 'var(--green)' : 'var(--red)';
}

function elapsed(iso) {
  if (!iso) return '—';
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h + 'h ' + m + 'm';
}

function holdDuration(openedAt, closedAt) {
  if (!openedAt || !closedAt) return '—';
  const s = Math.floor((new Date(closedAt) - new Date(openedAt)) / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
}

// ── Summary + Leaderboard ─────────────────────────────────────────────────────
async function loadSummary() {
  try {
    const r = await fetch('/api/paper/summary');
    if (r.status === 503) {
      document.getElementById('leaderboard-body').innerHTML =
        '<tr><td colspan="10" class="empty muted">Paper trading engine not started</td></tr>';
      return;
    }
    const d = await r.json();
    updateHero(d);
    renderLeaderboard(d.bots || []);
  } catch (e) {
    console.error('loadSummary error', e);
  }
}

function updateHero(d) {
  const balance = d.total_balance ?? 10000;
  const startBalance = d.initial_capital ?? 10000;
  const pnl = d.total_pnl ?? (balance - startBalance);
  const pnlPct = d.total_pnl_pct ?? (startBalance > 0 ? (pnl / startBalance) * 100 : 0);

  const h1 = document.getElementById('money-h1');
  h1.textContent = fmtMoney(balance);
  h1.className = pnl > 0.005 ? 'profit' : pnl < -0.005 ? 'loss' : '';

  const pnlEl = document.getElementById('hero-pnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + fmtMoney(pnl);
  pnlEl.style.color = pnlColor(pnl);

  const pnlPctEl = document.getElementById('hero-pnl-pct');
  pnlPctEl.textContent = fmtPct(pnlPct);
  pnlPctEl.style.color = pnlColor(pnlPct);

  document.getElementById('hero-start').textContent = fmtMoney(startBalance);
  const secs = d.uptime_seconds ?? 0;
  const hrs = Math.floor(secs / 3600); const mins = Math.floor((secs % 3600) / 60);
  document.getElementById('hero-uptime').textContent = hrs > 0 ? hrs + 'h ' + mins + 'm' : mins + 'm';
  document.getElementById('hero-bots').textContent = (d.bots ?? []).length;
  document.getElementById('hero-open-pos').textContent = d.total_open_positions ?? 0;
}

function renderLeaderboard(bots) {
  const tbody = document.getElementById('leaderboard-body');
  if (!bots.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty muted">No bots registered yet</td></tr>';
    return;
  }

  // Sort by effective_balance descending
  const sorted = [...bots].sort((a, b) => (b.effective_balance ?? 0) - (a.effective_balance ?? 0));

  tbody.innerHTML = sorted.map((bot, i) => {
    const rank = i + 1;
    const rankClass = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
    const name = bot.bot_name ?? bot.name ?? '—';
    const initCap = bot.initial_capital ?? 1666.67;
    const effBal = bot.effective_balance ?? bot.balance ?? initCap;
    const pnl = bot.total_pnl ?? (effBal - initCap);
    const pnlPct = bot.pnl_pct ?? (initCap > 0 ? (pnl / initCap) * 100 : 0);
    const wr = bot.win_rate != null ? (bot.win_rate * 100).toFixed(0) + '%' : '0%';
    const trades = bot.trade_count ?? bot.total_trades ?? 0;
    const sharpe = bot.sharpe_ratio != null ? bot.sharpe_ratio.toFixed(2) : '0.00';
    const phase = (bot.phase ?? 'cold_start').toUpperCase();
    const phaseCls = phase === 'FULL' ? 'running' : phase === 'PAUSED' ? 'paused' : 'stopped';

    return `<tr>
      <td><span class="rank ${rankClass}">${rank}</span></td>
      <td style="font-weight:bold;color:var(--bright)">${name}</td>
      <td style="font-variant-numeric:tabular-nums">${fmtMoney(effBal)}</td>
      <td style="color:${pnlColor(pnl)}">${pnl >= 0 ? '+' : ''}${fmtMoney(pnl)}</td>
      <td style="color:${pnlColor(pnlPct)}">${fmtPct(pnlPct)}</td>
      <td>${wr} (${bot.win_count ?? 0}W/${bot.loss_count ?? 0}L)</td>
      <td>${trades}</td>
      <td style="color:var(--muted)">${sharpe}</td>
      <td><span class="phase-badge ${phaseCls}">${phase}</span></td>
      <td>
        <button class="bot-btn pause" onclick="pauseBot('${name}')">Pause</button>
        <button class="bot-btn resume" onclick="resumeBot('${name}')">Resume</button>
      </td>
    </tr>`;
  }).join('');
}

// ── Positions ─────────────────────────────────────────────────────────────────
async function loadPositions() {
  try {
    const r = await fetch('/api/paper/positions');
    const d = await r.json();
    renderPositions(d.positions || []);
  } catch (e) {
    console.error('loadPositions error', e);
  }
}

function renderPositions(positions) {
  const tbody = document.getElementById('positions-body');
  if (!positions.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty">No open positions</td></tr>';
    return;
  }
  tbody.innerHTML = positions.map(p => {
    const sideClr = p.action === 'BUY' ? 'var(--green)' : 'var(--red)';
    const sideLabel = p.action === 'BUY' ? '▲ BUY' : '▼ SELL';
    const pnl = p.unrealized_pnl ?? 0;
    const pnlPct = p.unrealized_pnl_pct ?? 0;
    const pnlClr = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    const pnlSign = pnl >= 0 ? '+' : '';
    const livePrice = p.live_price ? fmt(p.live_price, 4) : '<span style="color:var(--muted)">—</span>';

    // Market mode display
    const mode = (p.market_mode || 'SPOT').toUpperCase();
    const modeClr = mode === 'FUTURES' ? 'var(--yellow)' : mode === 'EQUITY' ? 'var(--blue)' : 'var(--cyan)';
    const modeLabel = mode;

    // Intent + Direction
    const intent = p.position_intent || 'OPEN_LONG';
    const intentClr = intent.startsWith('OPEN') ? 'var(--green)' : intent.startsWith('CLOSE') ? 'var(--red)' : 'var(--muted)';
    const dir = p.direction || 'LONG';
    const dirClr = dir === 'LONG' ? 'var(--green)' : dir === 'SHORT' ? 'var(--red)' : 'var(--muted)';

    // Leverage display
    const lev = p.leverage || 1;
    const levLabel = mode === 'FUTURES' ? lev + 'x' : '1x';

    // Bias status
    const biasStatus = p.bias_status || 'unknown';
    const bsClr = biasStatus === 'ALIGNED' ? 'var(--green)' : biasStatus === 'WARNING' ? 'var(--yellow)' : biasStatus === 'CONFLICT' || biasStatus === 'EXIT' ? 'var(--red)' : 'var(--muted)';

    const entryBiasClr = p.entry_bias === 'bullish' ? 'var(--green)' : p.entry_bias === 'bearish' ? 'var(--red)' : 'var(--muted)';

    return `<tr style="cursor:pointer" onclick="openChart('${p.symbol}',${p.entry_price},${p.stop_loss||0},${p.take_profit_1||0},'${p.action}','${p.strategy_name||""}')" title="Click to view chart">
      <td style="color:var(--blue)">${p.bot_name ?? '—'}</td>
      <td style="font-weight:bold;color:var(--bright)">${p.symbol ?? '—'}</td>
      <td style="color:${modeClr};font-weight:600;font-size:0.68rem">${modeLabel}</td>
      <td style="color:${sideClr};font-weight:bold">${sideLabel}</td>
      <td style="color:${intentClr};font-size:0.66rem;font-weight:600">${intent}</td>
      <td style="color:${dirClr};font-weight:bold">${dir}</td>
      <td>${fmt(p.entry_price, 4)}</td>
      <td style="font-weight:bold">${livePrice}</td>
      <td style="color:${pnlClr};font-weight:bold">${pnlSign}$${fmt(pnl, 2)}</td>
      <td style="color:${pnlClr}">${pnlSign}${fmt(pnlPct, 2)}%</td>
      <td style="color:var(--muted)">$${fmt(p.position_size_usd, 2)}</td>
      <td style="color:${mode === 'FUTURES' ? 'var(--yellow)' : 'var(--muted)'}">${levLabel}</td>
      <td style="font-size:0.68rem;color:${entryBiasClr}">${(p.entry_bias||'?').toUpperCase()}</td>
      <td style="font-size:0.68rem;font-weight:bold;color:${bsClr}" title="Adverse checks: ${p.bias_adverse_count || 0}/2">${biasStatus.toUpperCase()}${(p.bias_adverse_count || 0) > 0 ? ' (' + p.bias_adverse_count + '/2)' : ''}</td>
      <td style="color:var(--red)">${fmt(p.stop_loss, 4)}</td>
      <td style="color:var(--green)">${fmt(p.take_profit_1, 4)}</td>
      <td style="color:var(--muted);font-size:0.66rem">${p.strategy_name ?? '—'}</td>
      <td style="color:var(--muted)">${elapsed(p.opened_at)} ago</td>
    </tr>`;
  }).join('');
}

// ── Trades ────────────────────────────────────────────────────────────────────
async function loadTrades() {
  try {
    const r = await fetch('/api/paper/trades?limit=500');
    const d = await r.json();
    allTrades = d.trades || [];

    // Rebuild bot filter options
    botNames = new Set(allTrades.map(t => t.bot_name ?? t.bot).filter(Boolean));
    const sel = document.getElementById('trade-bot-filter');
    const cur = sel.value;
    sel.innerHTML = '<option value="">All bots</option>' +
      [...botNames].sort().map(n => `<option value="${n}">${n}</option>`).join('');
    if (cur && botNames.has(cur)) sel.value = cur;

    filterTrades();
  } catch (e) {
    console.error('loadTrades error', e);
  }
}

function filterTrades() {
  const bot = document.getElementById('trade-bot-filter').value;
  const list = bot ? allTrades.filter(t => (t.bot_name ?? t.bot) === bot) : allTrades;
  renderTrades(list);
}

function renderTrades(trades) {
  const tbody = document.getElementById('trades-body');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty">No closed trades yet</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(t => {
    const side = t.side ?? t.action ?? '—';
    const sideClr = side === 'BUY' ? 'var(--green)' : 'var(--red)';
    const sideLabel = side === 'BUY' ? '▲ BUY' : side === 'SELL' ? '▼ SELL' : side;
    const pnl = t.pnl ?? t.realized_pnl;
    const pnlPct = t.pnl_pct ?? t.pnl_percent;
    const hold = holdDuration(t.opened_at ?? t.entered_at, t.closed_at ?? t.exited_at);
    return `<tr>
      <td style="color:var(--muted)">${fmtTime(t.closed_at ?? t.exited_at ?? t.opened_at)}</td>
      <td style="color:var(--blue)">${t.bot_name ?? t.bot ?? '—'}</td>
      <td style="font-weight:bold">${t.symbol ?? '—'}</td>
      <td style="color:${sideClr};font-weight:bold">${sideLabel}</td>
      <td>${fmt(t.entry_price ?? t.entry, 4)}</td>
      <td>${fmt(t.exit_price ?? t.exit, 4)}</td>
      <td style="color:${pnlColor(pnl)}">${pnl != null ? (pnl >= 0 ? '+' : '') + fmtMoney(pnl) : '—'}</td>
      <td style="color:${pnlColor(pnlPct)}">${pnlPct != null ? fmtPct(pnlPct) : '—'}</td>
      <td style="color:var(--muted)">${hold}</td>
      <td style="color:var(--muted);font-size:0.68rem">${t.strategy ?? t.strategy_name ?? '—'}</td>
    </tr>`;
  }).join('');
}

// ── Bot controls ──────────────────────────────────────────────────────────────
async function pauseBot(name) {
  try {
    await fetch(`/api/paper/bot/${encodeURIComponent(name)}/pause`, { method: 'POST' });
    await loadSummary();
  } catch (e) {
    console.error('pauseBot error', e);
  }
}

async function resumeBot(name) {
  try {
    await fetch(`/api/paper/bot/${encodeURIComponent(name)}/resume`, { method: 'POST' });
    await loadSummary();
  } catch (e) {
    console.error('resumeBot error', e);
  }
}

async function resetAll() {
  const ok = confirm('Reset paper trading engine to $10,000? All positions and trade history will be cleared.');
  if (!ok) return;
  try {
    await fetch('/api/paper/reset', { method: 'POST' });
    await refreshAll();
  } catch (e) {
    console.error('resetAll error', e);
  }
}

// ── Refresh orchestration ─────────────────────────────────────────────────────
async function refreshAll() {
  document.getElementById('last-refresh').textContent =
    'refreshed: ' + new Date().toLocaleTimeString();
  await Promise.all([loadSummary(), loadPositions(), loadTrades()]);
}

// ── Chart Modal ──────────────────────────────────────────────────────────────
let chartInstance = null;
let chartCandleSeries = null;
let chartRefreshInterval = null;
let chartSymbol = null;
let chartEntry = null;
let chartSL = null;
let chartTP1 = null;
let chartAction = null;

function openChart(symbol, entryPrice, stopLoss, tp1, action, strategy) {
  const overlay = document.getElementById('chart-overlay');
  overlay.classList.add('open');
  document.getElementById('chart-title').textContent = symbol + ' — ' + action + ' via ' + strategy;
  document.getElementById('chart-container').innerHTML = '<div style="color:var(--muted);text-align:center;padding:40px">Loading chart...</div>';
  document.getElementById('chart-indicators').innerHTML = '<div style="color:var(--muted)">Loading...</div>';

  chartSymbol = symbol;
  chartEntry = entryPrice;
  chartSL = stopLoss;
  chartTP1 = tp1;
  chartAction = action;

  loadChartData(symbol, entryPrice, stopLoss, tp1, action, true);

  // Auto-refresh: 1s for 1s timeframe, 5s for others
  if (chartRefreshInterval) clearInterval(chartRefreshInterval);
  const refreshMs = chartTimeframe === '1s' ? 1000 : 5000;
  chartRefreshInterval = setInterval(() => {
    if (!document.getElementById('chart-overlay').classList.contains('open')) return;
    loadChartData(chartSymbol, chartEntry, chartSL, chartTP1, chartAction, false);
  }, refreshMs);
}

function loadChartData(symbol, entryPrice, stopLoss, tp1, action, firstLoad) {
  fetch('/api/paper/chart/' + symbol + '?timeframe=' + chartTimeframe)
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        if (firstLoad) document.getElementById('chart-container').innerHTML = '<div style="color:var(--red);padding:20px">' + data.error + '</div>';
        return;
      }
      if (firstLoad) {
        renderChart(data, entryPrice, stopLoss, tp1, action);
      } else {
        // Update existing chart with new candle data
        if (chartCandleSeries && data.candles.length) {
          chartCandleSeries.setData(data.candles);
        }
      }
      renderIndicatorPanel(data);
    })
    .catch(e => {
      if (firstLoad) document.getElementById('chart-container').innerHTML = '<div style="color:var(--red);padding:20px">Failed: ' + e.message + '</div>';
    });
}

let chartTimeframe = '15m';

function switchTF(tf) {
  chartTimeframe = tf;
  // Highlight selected button
  document.querySelectorAll('.chart-header .btn-sm').forEach(b => {
    b.style.borderColor = b.textContent.trim() === tf ? 'var(--blue)' : 'var(--border)';
    b.style.color = b.textContent.trim() === tf ? 'var(--blue)' : 'var(--muted)';
  });
  // Reload chart with new timeframe
  if (chartSymbol) {
    if (chartInstance) { chartInstance.remove(); chartInstance = null; chartCandleSeries = null; }
    loadChartData(chartSymbol, chartEntry, chartSL, chartTP1, chartAction, true);
  }
}

function closeChart() {
  document.getElementById('chart-overlay').classList.remove('open');
  if (chartRefreshInterval) { clearInterval(chartRefreshInterval); chartRefreshInterval = null; }
  if (chartInstance) { chartInstance.remove(); chartInstance = null; }
  chartCandleSeries = null;
}

function renderChart(data, entryPrice, stopLoss, tp1, action) {
  const container = document.getElementById('chart-container');
  container.innerHTML = '';

  chartInstance = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 500,
    layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
    grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { timeVisible: true, secondsVisible: true, barSpacing: chartTimeframe === '1s' ? 5 : 8, rightOffset: 5 },
    rightPriceScale: { autoScale: true, borderColor: '#30363d' },
  });

  // Candlesticks — wider bars
  chartCandleSeries = chartInstance.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149',
    borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor: '#3fb950', wickDownColor: '#f85149',
  });
  chartCandleSeries.setData(data.candles);

  // Volume bars at bottom
  const volumeSeries = chartInstance.addHistogramSeries({
    color: '#26a69a',
    priceFormat: { type: 'volume' },
    priceScaleId: 'volume',
  });
  chartInstance.priceScale('volume').applyOptions({
    scaleMargins: { top: 0.85, bottom: 0 },
  });
  if (data.candles.length > 0 && data.candles[0].volume !== undefined) {
    volumeSeries.setData(data.candles.map(c => ({
      time: c.time,
      value: c.volume || 0,
      color: c.close >= c.open ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)',
    })));
  }

  // ── Price lines: Entry, SL, TP1 ──
  if (entryPrice) {
    chartCandleSeries.createPriceLine({
      price: entryPrice, color: '#58a6ff', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: '► Entry',
    });
  }
  if (stopLoss && stopLoss > 0) {
    chartCandleSeries.createPriceLine({
      price: stopLoss, color: '#f85149', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: '✕ SL',
    });
  }
  if (tp1 && tp1 > 0) {
    chartCandleSeries.createPriceLine({
      price: tp1, color: '#3fb950', lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true, title: '✓ TP1',
    });
  }

  // ── EMA overlay lines (9, 20, 50) ──
  const ind = data.indicators;
  if (ind.ema_9 && ind.ema_20) {
    // Draw as horizontal reference lines across visible area
    const ema9Line = chartInstance.addLineSeries({ color: '#d29922', lineWidth: 1, title: 'EMA9', lastValueVisible: true, priceLineVisible: false });
    const ema20Line = chartInstance.addLineSeries({ color: '#58a6ff', lineWidth: 1, title: 'EMA20', lastValueVisible: true, priceLineVisible: false });
    const lastT = data.candles[data.candles.length - 1].time;
    const firstT = data.candles[0].time;
    ema9Line.setData([{time: firstT, value: ind.ema_9}, {time: lastT, value: ind.ema_9}]);
    ema20Line.setData([{time: firstT, value: ind.ema_20}, {time: lastT, value: ind.ema_20}]);
  }
  if (ind.ema_50) {
    const ema50Line = chartInstance.addLineSeries({ color: '#f0883e', lineWidth: 1, title: 'EMA50', lastValueVisible: true, priceLineVisible: false });
    const lastT = data.candles[data.candles.length - 1].time;
    const firstT = data.candles[0].time;
    ema50Line.setData([{time: firstT, value: ind.ema_50}, {time: lastT, value: ind.ema_50}]);
  }

  // ── Bollinger Bands ──
  if (ind.bb_upper && ind.bb_lower) {
    const lastT = data.candles[data.candles.length - 1].time;
    const firstT = data.candles[0].time;
    const bbUp = chartInstance.addLineSeries({ color: 'rgba(88,166,255,0.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, lastValueVisible: true, priceLineVisible: false, title: 'BB↑' });
    const bbLo = chartInstance.addLineSeries({ color: 'rgba(88,166,255,0.4)', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, lastValueVisible: true, priceLineVisible: false, title: 'BB↓' });
    bbUp.setData([{time: firstT, value: ind.bb_upper}, {time: lastT, value: ind.bb_upper}]);
    bbLo.setData([{time: firstT, value: ind.bb_lower}, {time: lastT, value: ind.bb_lower}]);
  }

  // ── Zoom: show recent candles, scroll left for history ──
  chartInstance.timeScale().scrollToPosition(0, false);
  const visibleBars = chartTimeframe === '1s' ? 120 : 60;
  if (data.candles.length > visibleBars) {
    chartInstance.timeScale().setVisibleLogicalRange({
      from: data.candles.length - visibleBars,
      to: data.candles.length,
    });
  } else {
    chartInstance.timeScale().fitContent();
  }
}

function renderIndicatorPanel(data) {
  const ind = data.indicators;
  const bias = data.bias;
  const struct = data.structure;
  const regime = data.regime;

  // ── Entry vs Current bias conflict detection ──────────────────────────────
  const pos = (data.positions || []).find(p => p.symbol === data.symbol);
  const entryBias = pos ? pos.entry_bias : null;
  const entrySide = pos ? pos.action : null;

  let biasStatus = 'no_position';
  let statusColor = 'var(--muted)';
  let statusLabel = 'No Position';

  if (entryBias && entrySide) {
    const currentBias = data.bias.net;
    if (entrySide === 'BUY' && currentBias === 'bullish') {
      biasStatus = 'aligned'; statusColor = 'var(--green)'; statusLabel = '✓ ALIGNED';
    } else if (entrySide === 'SELL' && currentBias === 'bearish') {
      biasStatus = 'aligned'; statusColor = 'var(--green)'; statusLabel = '✓ ALIGNED';
    } else if (currentBias === 'neutral') {
      biasStatus = 'warning'; statusColor = 'var(--yellow)'; statusLabel = '⚠ WARNING — bias now neutral';
    } else {
      biasStatus = 'conflict'; statusColor = 'var(--red)'; statusLabel = '🚨 EXIT NOW — bias flipped';
    }
  }

  const entrySideColor = entrySide === 'BUY' ? 'var(--green)' : entrySide === 'SELL' ? 'var(--red)' : 'var(--muted)';
  const entryBiasColor = entryBias === 'bullish' ? 'var(--green)' : entryBias === 'bearish' ? 'var(--red)' : 'var(--muted)';
  const currentBiasColor = bias.net === 'bullish' ? 'var(--green)' : bias.net === 'bearish' ? 'var(--red)' : 'var(--muted)';
  const bullPct = bias.bullish != null ? (bias.bullish * 100).toFixed(0) : '—';
  const bearPct = bias.bearish != null ? (bias.bearish * 100).toFixed(0) : '—';

  const rsiColor = (ind.rsi || 50) <= 30 ? 'var(--green)' : (ind.rsi || 50) >= 70 ? 'var(--red)' : 'var(--text)';
  const macdColor = (ind.macd_histogram || 0) > 0 ? 'var(--green)' : 'var(--red)';

  document.getElementById('chart-indicators').innerHTML = `
    <div class="ind-section">
      <h4>Trade State</h4>
      <div class="ind-row"><span class="ind-label">Entry Side</span><span class="ind-val" style="color:${entrySideColor}">${entrySide || '—'}</span></div>
      <div class="ind-row"><span class="ind-label">Entry Bias</span><span class="ind-val" style="color:${entryBiasColor}">${entryBias ? entryBias.toUpperCase() : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">Current Bias</span><span class="ind-val" style="color:${currentBiasColor}">${bias.net.toUpperCase()}</span></div>
      <div class="ind-row"><span class="ind-label">Bull Score</span><span class="ind-val green">${bullPct}%</span></div>
      <div class="ind-row"><span class="ind-label">Bear Score</span><span class="ind-val red">${bearPct}%</span></div>
      <div style="text-align:center;margin:8px 0;padding:6px;border-radius:6px;background:rgba(0,0,0,0.3);font-weight:bold;color:${statusColor}">${statusLabel}</div>
    </div>

    <div class="ind-section">
      <h4>Indicators</h4>
      <div class="ind-row"><span class="ind-label">RSI</span><span class="ind-val" style="color:${rsiColor}">${ind.rsi ?? '—'}</span></div>
      <div class="ind-row"><span class="ind-label">MACD</span><span class="ind-val" style="color:${macdColor}">${ind.macd_histogram != null ? ind.macd_histogram.toFixed(4) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">EMA 9</span><span class="ind-val">${ind.ema_9 ? ind.ema_9.toFixed(2) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">EMA 20</span><span class="ind-val">${ind.ema_20 ? ind.ema_20.toFixed(2) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">EMA 50</span><span class="ind-val">${ind.ema_50 ? ind.ema_50.toFixed(2) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">ADX</span><span class="ind-val">${ind.adx ?? '—'}</span></div>
      <div class="ind-row"><span class="ind-label">ATR</span><span class="ind-val">${ind.atr ? ind.atr.toFixed(4) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">BB Upper</span><span class="ind-val">${ind.bb_upper ? ind.bb_upper.toFixed(2) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">BB Lower</span><span class="ind-val">${ind.bb_lower ? ind.bb_lower.toFixed(2) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">Volume</span><span class="ind-val">${ind.volume_relative}x avg</span></div>
    </div>

    <div class="ind-section">
      <h4>Market Structure</h4>
      <div class="ind-row"><span class="ind-label">Trend</span><span class="ind-val" style="color:${struct.trend==='bullish'?'var(--green)':struct.trend==='bearish'?'var(--red)':'var(--muted)'}">${struct.trend.toUpperCase()}</span></div>
      <div class="ind-row"><span class="ind-label">Strength</span><span class="ind-val">${(struct.strength*100).toFixed(0)}%</span></div>
      <div class="ind-row"><span class="ind-label">Swing High</span><span class="ind-val">${struct.last_swing_high ? struct.last_swing_high.toFixed(2) : '—'}</span></div>
      <div class="ind-row"><span class="ind-label">Swing Low</span><span class="ind-val">${struct.last_swing_low ? struct.last_swing_low.toFixed(2) : '—'}</span></div>
      ${(struct.events || []).map(e => '<div class="struct-event">' + e.kind + ' ' + e.direction + ' @ ' + e.price.toFixed(2) + '</div>').join('')}
    </div>

    <div class="ind-section">
      <h4>Regime</h4>
      <div class="ind-row"><span class="ind-label">Type</span><span class="ind-val">${regime.name}</span></div>
      <div class="ind-row"><span class="ind-label">Vol Score</span><span class="ind-val">${(regime.vol_score*100).toFixed(0)}%</span></div>
    </div>

    <div class="ind-section">
      <h4>Candle Patterns (${(data.patterns||[]).length})</h4>
      ${(data.patterns||[]).length ? (data.patterns||[]).map(p => {
        const clr = p.bias === 'bullish' ? 'var(--green)' : p.bias === 'bearish' ? 'var(--red)' : 'var(--muted)';
        const icon = p.bias === 'bullish' ? '▲' : p.bias === 'bearish' ? '▼' : '─';
        return '<div style="padding:2px 0;border-bottom:1px solid #1c2128">' +
          '<span style="color:' + clr + ';font-weight:bold">' + icon + '</span> ' +
          '<span style="color:var(--text)">' + p.name + '</span>' +
          (p.strength ? ' <span style="color:var(--muted)">str=' + p.strength + '</span>' : '') +
          (p.explanation ? '<div style="color:var(--muted);font-size:0.62rem;margin-left:12px">' + p.explanation.slice(0,60) + '</div>' : '') +
          '</div>';
      }).join('') : '<div style="color:var(--muted)">No patterns detected</div>'}
    </div>

    <div class="ind-section">
      <h4>Strategy Signals (${(data.strategies||[]).length})</h4>
      ${(data.strategies||[]).length ? (data.strategies||[]).map(s => {
        const clr = s.action === 'BUY' ? 'var(--green)' : 'var(--red)';
        const icon = s.action === 'BUY' ? '▲ BUY' : '▼ SELL';
        return '<div style="padding:3px 0;border-bottom:1px solid #1c2128">' +
          '<span style="color:' + clr + ';font-weight:bold">' + icon + '</span> ' +
          '<span style="color:var(--blue)">' + s.name + '</span>' +
          '<div style="font-size:0.62rem;color:var(--muted);margin-left:12px">' +
          'Entry: ' + (s.entry||'—') + ' SL: ' + (s.stop||'—') + ' TP: ' + (s.tp1||'—') +
          '</div></div>';
      }).join('') : '<div style="color:var(--muted)">No strategies triggering</div>'}
    </div>
  `;
}

// ── Signal Preview ──────────────────────────────────────────────────────────
async function loadPreview() {
  const tbody = document.getElementById('preview-body');
  tbody.innerHTML = '<tr><td colspan="13" class="empty">Scanning 20 symbols...</td></tr>';
  try {
    const r = await fetch('/api/paper/preview');
    const d = await r.json();
    const previews = d.previews || [];
    if (!previews.length) {
      tbody.innerHTML = '<tr><td colspan="13" class="empty">No symbols scanned</td></tr>';
      return;
    }
    tbody.innerHTML = previews.map(p => {
      const statusIcon = {'READY':'●','ALMOST':'◐','HELD':'◆','NO TREND':'✕','WAITING':'○'}[p.readiness] || '○';
      const clrMap = {'READY':'var(--green)','ALMOST':'var(--yellow)','HELD':'var(--blue)','NO TREND':'var(--red)','WAITING':'var(--muted)'};
      const statusClr = clrMap[p.readiness] || 'var(--muted)';
      const biasClr = p.bias === 'bullish' ? 'var(--green)' : p.bias === 'bearish' ? 'var(--red)' : 'var(--muted)';
      const biasLabel = p.bias === 'bullish' ? '▲ BULL' : p.bias === 'bearish' ? '▼ BEAR' : '— NEUTRAL';
      const rsiClr = (p.rsi||50) <= 30 ? 'var(--green)' : (p.rsi||50) >= 70 ? 'var(--red)' : 'var(--text)';

      // Market state color
      const ms = p.market_state || 'NEUTRAL';
      const msClr = ms.includes('BULL') ? 'var(--green)' : ms.includes('BEAR') ? 'var(--red)' : ms === 'NEWS_LOCKDOWN' ? 'var(--yellow)' : 'var(--muted)';
      const okBadge = (v) => v ? '<span style="color:var(--green);font-weight:bold">ON</span>' : '<span style="color:var(--muted)">OFF</span>';

      return `<tr title="${p.part_reason || ''}">
        <td style="color:${statusClr};font-weight:bold">${statusIcon} ${p.readiness}</td>
        <td style="font-weight:bold;color:var(--bright);cursor:pointer" onclick="openChart('${p.symbol}',0,0,0,'${p.bias}','preview')">${p.symbol}</td>
        <td style="color:${msClr};font-size:0.66rem;font-weight:600">${ms.replace('_',' ')}</td>
        <td style="font-size:0.66rem">${okBadge(p.spot_ok)}</td>
        <td style="font-size:0.66rem">${okBadge(p.fut_long_ok)}</td>
        <td style="font-size:0.66rem">${okBadge(p.fut_short_ok)}</td>
        <td style="color:${biasClr};font-weight:bold">${biasLabel}</td>
        <td class="green">${((p.bullish||0)*100).toFixed(0)}%</td>
        <td class="red">${((p.bearish||0)*100).toFixed(0)}%</td>
        <td style="color:${p.conflict>0.8?'var(--red)':p.conflict>0.5?'var(--yellow)':'var(--green)'}">${((p.conflict||0)*100).toFixed(0)}%</td>
        <td style="color:${rsiClr}">${p.rsi||'—'}</td>
        <td>${p.adx||'—'}</td>
        <td>\$${p.price||'—'}</td>
      </tr>`;
    }).join('');
  } catch(e) {
    tbody.innerHTML = '<tr><td colspan="13" class="empty">Scan failed</td></tr>';
  }
}

// ── Equity Curve ────────────────────────────────────────────────────────────
let equityChart = null;
let equityLineSeries = null;

async function loadEquity() {
  try {
    const r = await fetch('/api/paper/equity');
    const data = await r.json();
    const curve = Array.isArray(data) ? data : (data.curve || []);
    if (!curve.length) return;

    const container = document.getElementById('equity-container');
    if (!equityChart) {
      equityChart = LightweightCharts.createChart(container, {
        width: container.clientWidth, height: 200,
        layout: { background: { color: '#0a0e17' }, textColor: '#c9d1d9' },
        grid: { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
        timeScale: { timeVisible: true },
        rightPriceScale: { borderColor: '#30363d' },
      });
      equityLineSeries = equityChart.addAreaSeries({
        topColor: 'rgba(63,185,80,0.4)', bottomColor: 'rgba(63,185,80,0.0)',
        lineColor: '#3fb950', lineWidth: 2,
      });
      // Add $10K baseline
      equityLineSeries.createPriceLine({
        price: 10000, color: '#8b949e', lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true, title: 'Start',
      });
    }

    const points = curve.map(p => {
      const ts = typeof p.timestamp === 'string' ? Math.floor(new Date(p.timestamp).getTime() / 1000) : p.timestamp;
      return { time: ts, value: p.total_balance };
    }).filter(p => p.time && p.value);

    if (points.length) {
      equityLineSeries.setData(points);
      // Color based on profit/loss
      const latest = points[points.length - 1].value;
      const clr = latest >= 10000 ? '#3fb950' : '#f85149';
      equityLineSeries.applyOptions({
        topColor: latest >= 10000 ? 'rgba(63,185,80,0.4)' : 'rgba(248,81,73,0.4)',
        bottomColor: latest >= 10000 ? 'rgba(63,185,80,0.0)' : 'rgba(248,81,73,0.0)',
        lineColor: clr,
      });
      equityChart.timeScale().fitContent();
    }
  } catch(e) { console.error('equity error', e); }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
refreshAll();
loadEquity();
loadPreview();
setInterval(refreshAll, 5000);
setInterval(loadEquity, 10000);
setInterval(loadPreview, 60000);  // Refresh preview every 60s  // Update equity every 10s
</script>

<!-- Chart Modal Overlay -->
<div class="chart-overlay" id="chart-overlay" onclick="if(event.target===this)closeChart()">
  <div class="chart-modal">
    <div class="chart-header">
      <h3 id="chart-title">Loading...</h3>
      <div style="display:flex;gap:4px;align-items:center">
        <button class="btn-sm" onclick="switchTF('1s')">1s</button>
        <button class="btn-sm" onclick="switchTF('1m')">1m</button>
        <button class="btn-sm" onclick="switchTF('5m')">5m</button>
        <button class="btn-sm" onclick="switchTF('15m')" style="border-color:var(--blue);color:var(--blue)">15m</button>
        <button class="btn-sm" onclick="switchTF('30m')">30m</button>
        <button class="btn-sm" onclick="switchTF('1h')">1h</button>
        <button class="btn-sm" onclick="switchTF('4h')">4h</button>
        <button class="btn-sm" onclick="switchTF('1d')">1d</button>
        <button class="btn-sm" onclick="switchTF('1w')">1w</button>
        <button class="chart-close" onclick="closeChart()">✕</button>
      </div>
    </div>
    <div class="chart-body">
      <div class="chart-candles" id="chart-container"></div>
      <div class="chart-indicators" id="chart-indicators"></div>
    </div>
  </div>
</div>

<!-- Lightweight Charts Library -->
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
</body>
</html>"""
    return HTMLResponse(content=html)
