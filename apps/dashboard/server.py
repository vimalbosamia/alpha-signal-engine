"""
Localhost dashboard for the AI Trading Signal Agent.

Single-page dashboard — everything visible at http://localhost:8000
No need to navigate to separate URLs.

Sections on the page:
  - System health bar (DB, Binance, Alpaca)
  - Stats cards (total signals, BUY/SELL count, open positions)
  - Open positions panel (signals being watched for TP/SL)
  - Manual scan panel
  - Recent signals table (entry, stop, TP1/TP2, R:R, patterns, regime)

JSON APIs (consumed by the page, not for manual browsing):
  GET /api/signals   — recent signals from DB
  GET /api/health    — system health
  GET /api/positions — open tracked positions
  GET /api/scan      — trigger a manual scan

Run:
    uv run python -m apps.dashboard.server

Then open: http://localhost:8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from libs.core.config.settings import get_settings
from libs.core.logging.logger import configure_logging, get_logger
from libs.core.models.domain import AssetClass, Timeframe
from libs.data.storage.db import get_session_factory, init_db
from libs.data.storage.repository import SignalRepository
from libs.monitoring.position_watcher import PositionWatcher

log = get_logger(__name__)

# Shared watcher — set by the serve command when running combined mode
_shared_watcher: PositionWatcher | None = None


def set_shared_watcher(watcher: PositionWatcher) -> None:
    global _shared_watcher
    _shared_watcher = watcher


@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    configure_logging(settings.observability.log_level)
    await init_db()
    log.info("dashboard_started", port=8000)
    yield


app = FastAPI(title="AI Trading Signal Agent Dashboard", docs_url="/docs", lifespan=lifespan)


# ── JSON API endpoints ────────────────────────────────────────────────────────

@app.get("/api/signals")
async def get_signals(symbol: str | None = None, limit: int = 100) -> JSONResponse:
    async with get_session_factory()() as session:
        repo = SignalRepository(session)
        signals = await repo.get_latest_signals(symbol=symbol, limit=limit)
    return JSONResponse(content={"signals": signals, "count": len(signals)})


@app.get("/api/health")
async def health() -> JSONResponse:
    import httpx
    checks: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": get_settings().agent_mode.value,
        "binance": False,
        "alpaca": False,
        "database": False,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get("https://api.binance.com/api/v3/ping")
            checks["binance"] = r.status_code == 200
    except Exception:
        pass
    try:
        settings = get_settings()
        if settings.alpaca.api_key.get_secret_value():
            checks["alpaca"] = True   # key present = configured
        else:
            checks["alpaca"] = None   # not configured
    except Exception:
        pass
    try:
        async with get_session_factory()() as session:
            repo = SignalRepository(session)
            await repo.get_latest_signals(limit=1)
            checks["database"] = True
    except Exception:
        pass
    ok = checks["database"] is True
    return JSONResponse(content=checks, status_code=200 if ok else 503)


@app.get("/api/positions")
async def get_positions() -> JSONResponse:
    if _shared_watcher is None:
        return JSONResponse(content={"positions": [], "count": 0})
    positions = _shared_watcher.active_positions()
    data = [
        {
            "signal_id": str(p.signal_id),
            "symbol": p.symbol,
            "asset_class": p.asset_class,
            "action": p.action.value,
            "entry": round(p.entry_mid, 6),
            "stop_loss": round(p.stop_loss, 6),
            "take_profit_1": round(p.take_profit_1, 6),
            "take_profit_2": round(p.take_profit_2, 6) if p.take_profit_2 else None,
            "strategy": p.strategy_name,
            "confidence": round(p.confidence, 3),
            "tp1_hit": p.tp1_hit,
            "opened_at": p.opened_at.isoformat(),
        }
        for p in positions
    ]
    return JSONResponse(content={"positions": data, "count": len(data)})


@app.get("/api/prices")
async def get_prices(symbols: str = "") -> JSONResponse:
    """Return current prices for a comma-separated list of symbols.
    Each symbol may be prefixed with asset class: 'crypto:BTCUSDT' or 'stock:AAPL'.
    Plain symbols without prefix are treated as crypto.
    """
    import asyncio
    from libs.data.providers.binance.provider import BinanceDataProvider
    from libs.data.providers.alpaca.provider import AlpacaDataProvider

    if not symbols:
        return JSONResponse(content={"prices": {}, "timestamp": datetime.now(timezone.utc).isoformat()})

    _binance: BinanceDataProvider | None = None
    _alpaca: AlpacaDataProvider | None = None
    prices: dict[str, float | None] = {}

    async def fetch_one(symbol: str, asset_class: str) -> None:
        nonlocal _binance, _alpaca
        try:
            if asset_class == "stock":
                if _alpaca is None:
                    _alpaca = AlpacaDataProvider()
                price = await _alpaca.get_latest_price(symbol)
            else:
                if _binance is None:
                    _binance = BinanceDataProvider()
                price = await _binance.get_latest_price(symbol)
            prices[symbol] = price
        except Exception:
            prices[symbol] = None

    tasks = []
    for entry in symbols.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" in entry:
            ac, sym = entry.split(":", 1)
        else:
            ac, sym = "crypto", entry
        tasks.append(fetch_one(sym, ac))

    await asyncio.gather(*tasks)
    return JSONResponse(content={"prices": prices, "timestamp": datetime.now(timezone.utc).isoformat()})


@app.get("/api/accuracy")
async def get_accuracy() -> JSONResponse:
    """Return per-strategy win rates from outcome tracking."""
    from libs.data.storage.repository import OutcomeRepository
    try:
        async with get_session_factory()() as db_session:
            repo = OutcomeRepository(db_session)
            stats = await repo.get_strategy_stats()
        return JSONResponse(content={"strategies": stats})
    except Exception as exc:
        return JSONResponse(content={"strategies": [], "error": str(exc)})


@app.get("/api/ml-stats")
async def ml_stats() -> JSONResponse:
    """Return ML classifier stats: accuracy, samples, feature importances."""
    try:
        from libs.ml.signal_classifier import get_classifier
        clf = get_classifier()
        s = clf.stats
        return JSONResponse(content={
            "trained": s.trained,
            "samples": s.samples,
            "cv_accuracy": round(s.accuracy * 100, 1),
            "precision": round(s.precision * 100, 1),
            "win_rate": round(s.win_rate * 100, 1),
            "last_trained_at": s.last_trained_at,
            "outcomes_since_retrain": s.outcomes_since_retrain,
            "blocked_last_hour": s.blocked_last_hour,
            "feature_importances": s.feature_importances,
            "block_threshold": 85,   # 85% win prob threshold
        })
    except Exception as exc:
        return JSONResponse(content={"trained": False, "error": str(exc)})


@app.get("/api/portfolio-guard")
async def portfolio_guard_stats() -> JSONResponse:
    """Return current portfolio guard exposure summary."""
    try:
        from libs.monitoring.portfolio_guard import get_portfolio_guard
        guard = get_portfolio_guard()
        summary = guard.exposure_summary()
        cfg = guard._config
        active = [
            {
                "symbol": s.symbol,
                "asset_class": s.asset_class,
                "action": s.action,
                "accepted_at": s.accepted_at.isoformat(),
            }
            for s in guard.active_signals()
        ]
        return JSONResponse(content={
            "summary": summary,
            "active_signals": active,
            "limits": {
                "max_concurrent": cfg.max_concurrent_signals,
                "max_per_symbol": cfg.max_per_symbol,
                "max_same_direction": cfg.max_same_direction,
                "max_per_asset_class": cfg.max_per_asset_class,
            },
        })
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)})


@app.get("/api/scan")
async def manual_scan(
    symbol: str = "BTCUSDT",
    asset_class: str = "crypto",
    timeframe: str = "15m",
) -> JSONResponse:
    from libs.data.providers.binance.provider import BinanceDataProvider
    from libs.data.providers.alpaca.provider import AlpacaDataProvider
    from libs.strategies.reversal.hammer_reversal import HammerReversalStrategy
    from libs.strategies.breakout.resistance_breakout import ResistanceBreakoutStrategy
    from libs.strategies.continuation.pullback import PullbackContinuationStrategy
    from apps.signal_agent.pipeline import SignalPipeline

    ac = AssetClass(asset_class)
    tf = Timeframe(timeframe)
    provider = BinanceDataProvider() if ac == AssetClass.CRYPTO else AlpacaDataProvider()

    pipeline = SignalPipeline(
        provider=provider,
        strategies=[
            HammerReversalStrategy(),
            ResistanceBreakoutStrategy(),
            PullbackContinuationStrategy(),
        ],
    )
    outputs = await pipeline.run_once(symbol, ac, tf)

    # Track scanned signals in watcher if available
    if _shared_watcher:
        for out in outputs:
            _shared_watcher.track(out)

    return JSONResponse(content={
        "symbol": symbol,
        "signals_found": len(outputs),
        "signals": [json.loads(o.model_dump_json()) for o in outputs],
    })


# ── Single-page HTML Dashboard ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    settings = get_settings()
    crypto_symbols = settings.signal.crypto_symbols
    stock_symbols = settings.signal.stock_symbols
    futures_symbols = settings.signal.futures_symbols if settings.signal.enable_futures else []
    mode_badge = "paper" if settings.agent_mode.value == "paper" else "live"
    mode_label = settings.agent_mode.value.upper()

    crypto_pills = "".join(
        f'<button class="pill" onclick="setScan(\'{s}\',\'crypto\')">{s}</button>'
        for s in crypto_symbols
    )
    futures_pills = "".join(
        f'<button class="pill" style="border-color:var(--yellow);color:var(--yellow)" onclick="setScan(\'{s}\',\'crypto\',\'futures\')">{s} ⚡</button>'
        for s in futures_symbols
    )
    stock_pills = "".join(
        f'<button class="pill" onclick="setScan(\'{s}\',\'stock\')">{s} ★</button>'
        for s in stock_symbols
    )
    watchlist_html = crypto_pills + futures_pills + stock_pills

    crypto_options = "".join(
        f'<option value="{s}" data-market="spot" {"selected" if i == 0 else ""}>{s.replace("USDT","")}/USDT (Spot)</option>'
        for i, s in enumerate(crypto_symbols)
    )
    futures_options = "".join(
        f'<option value="{s}" data-market="futures">{s.replace("USDT","")}/USDT (Futures)</option>'
        for s in futures_symbols
    )
    stock_options = "".join(
        f'<option value="{s}">{s}</option>'
        for s in stock_symbols
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Trading Signal Agent</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --muted: #8b949e; --text: #c9d1d9; --bright: #f0f6fc;
      --blue: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
    }}
    body {{ font-family: 'Courier New', monospace; background: var(--bg); color: var(--text); min-height: 100vh; font-size: 13px; }}

    /* ── Header ── */
    header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }}
    header h1 {{ font-size: 1rem; color: var(--blue); }}
    .header-right {{ display: flex; align-items: center; gap: 10px; font-size: 0.72rem; color: var(--muted); }}
    .badge {{ padding: 2px 8px; border-radius: 10px; font-weight: bold; font-size: 0.68rem; color: #fff; }}
    .badge.paper {{ background: #9a6700; }}
    .badge.live  {{ background: #da3633; }}
    #live-clock {{ color: var(--blue); font-size: 0.78rem; min-width: 80px; }}

    /* ── Health bar ── */
    #health-bar {{ background: #0a0e17; border-bottom: 1px solid var(--border); padding: 5px 20px; display: flex; align-items: center; gap: 18px; font-size: 0.7rem; }}
    .hitem {{ display: flex; align-items: center; gap: 5px; }}
    .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--muted); display: inline-block; transition: background 0.4s; }}
    .dot.ok   {{ background: var(--green); box-shadow: 0 0 5px var(--green); }}
    .dot.fail {{ background: var(--red); }}
    .dot.warn {{ background: var(--yellow); }}
    #last-updated {{ margin-left: auto; color: var(--muted); font-size: 0.68rem; }}

    /* ── Layout ── */
    main {{ padding: 14px 20px; max-width: 1700px; margin: 0 auto; }}

    /* ── Stats row ── */
    .stats-row {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 14px; }}
    .stat-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; transition: border-color 0.3s; }}
    .stat-card.flash {{ border-color: var(--blue); }}
    .stat-card .label {{ font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }}
    .stat-card .val {{ font-size: 1.4rem; font-weight: bold; color: var(--bright); }}
    .stat-card .sub {{ font-size: 0.62rem; color: var(--muted); margin-top: 2px; }}
    .green {{ color: var(--green) !important; }}
    .red   {{ color: var(--red)   !important; }}
    .yellow {{ color: var(--yellow) !important; }}
    .blue  {{ color: var(--blue)  !important; }}

    /* ── Two-col row ── */
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }}
    @media (max-width: 960px) {{ .two-col {{ grid-template-columns: 1fr; }} .stats-row {{ grid-template-columns: repeat(3,1fr); }} }}

    /* ── Panel ── */
    .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
    .panel-header {{ padding: 9px 14px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }}
    .panel-header h2 {{ font-size: 0.78rem; color: var(--bright); font-weight: bold; }}
    .panel-body {{ padding: 12px 14px; }}

    /* ── Tables ── */
    table {{ width: 100%; border-collapse: collapse; font-size: 0.73rem; }}
    th {{ padding: 6px 10px; text-align: left; color: var(--muted); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #1c2128; white-space: nowrap; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #1c2128; }}
    .action-buy  {{ color: var(--green); font-weight: bold; }}
    .action-sell {{ color: var(--red);   font-weight: bold; }}
    .action-nt   {{ color: var(--muted); }}
    .empty {{ padding: 28px; text-align: center; color: var(--muted); font-size: 0.78rem; }}

    /* ── Controls ── */
    .pill {{ background: #21262d; border: 1px solid var(--border); border-radius: 4px; padding: 3px 8px; font-size: 0.7rem; cursor: pointer; color: var(--text); font-family: inherit; }}
    .pill:hover {{ border-color: var(--blue); color: var(--blue); }}
    input, select {{ background: #21262d; border: 1px solid var(--border); color: var(--text); padding: 4px 8px; border-radius: 4px; font-family: inherit; font-size: 0.76rem; }}
    input:focus, select:focus {{ outline: 1px solid var(--blue); border-color: var(--blue); }}
    .btn {{ background: #238636; color: #fff; border: none; padding: 5px 13px; border-radius: 4px; cursor: pointer; font-size: 0.76rem; font-family: inherit; }}
    .btn:hover {{ background: #2ea043; }}
    .btn:disabled {{ background: #21262d; color: var(--muted); cursor: not-allowed; }}
    .rbtn {{ background: none; border: 1px solid var(--border); border-radius: 10px; padding: 2px 8px; font-size: 0.66rem; color: var(--muted); cursor: pointer; font-family: inherit; }}
    .rbtn:hover {{ border-color: var(--blue); color: var(--blue); }}
    #countdown {{ color: var(--blue); font-weight: bold; }}

    /* ── Progress bar ── */
    .prog-wrap {{ background: #21262d; border-radius: 3px; height: 5px; width: 90px; display: inline-block; vertical-align: middle; overflow: hidden; }}
    .prog-fill  {{ height: 100%; border-radius: 3px; transition: width 0.5s ease; }}

    /* ── Confidence bar ── */
    .conf-bar {{ display: inline-block; height: 4px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }}

    /* ── Price flash ── */
    @keyframes flash-up   {{ 0%,100% {{ background: transparent; }} 30% {{ background: rgba(63,185,80,0.18); }} }}
    @keyframes flash-down {{ 0%,100% {{ background: transparent; }} 30% {{ background: rgba(248,81,73,0.18); }} }}
    .flash-up   {{ animation: flash-up   0.6s ease; }}
    .flash-down {{ animation: flash-down 0.6s ease; }}

    /* ── Spinner ── */
    .spin {{ display: none; animation: rot 0.7s linear infinite; }}
    .spin.on {{ display: inline; }}
    @keyframes rot {{ to {{ transform: rotate(360deg); }} }}

    /* ── Scan result ── */
    #scan-result {{ font-size: 0.76rem; min-height: 28px; }}

    /* ── Price cell ── */
    .price-cell {{ font-variant-numeric: tabular-nums; }}
    .price-up   {{ color: var(--green); }}
    .price-down {{ color: var(--red); }}
    .price-flat {{ color: var(--text); }}
  </style>
</head>
<body>

<header>
  <h1>⚡ AI Trading Signal Agent</h1>
  <div class="header-right">
    <span id="live-clock">--:--:--</span>
    <span>· next refresh <span id="countdown">15</span>s</span>
    <button class="rbtn" onclick="refreshAll()">↻ Now</button>
    <span class="badge {mode_badge}">{mode_label}</span>
    <span style="color:#555;">SIGNAL ONLY — NO TRADES</span>
  </div>
</header>

<!-- Health bar -->
<div id="health-bar">
  <span style="color:var(--muted);font-size:0.64rem;text-transform:uppercase;letter-spacing:1px;">System</span>
  <span class="hitem"><span class="dot" id="h-db"></span><span id="h-db-lbl">Database</span></span>
  <span class="hitem"><span class="dot" id="h-bn"></span><span id="h-bn-lbl">Binance</span></span>
  <span class="hitem"><span class="dot" id="h-ap"></span><span id="h-ap-lbl">Alpaca</span></span>
  <span id="last-updated">last updated: —</span>
</div>

<!-- Live price ticker strip (1-second updates via Binance public API) -->
<div id="ticker-bar" style="background:#0a0e17;border-bottom:1px solid var(--border);padding:5px 20px;display:flex;gap:20px;align-items:center;overflow-x:auto;font-size:0.72rem;min-height:30px;">
  <span style="color:var(--muted);font-size:0.64rem;text-transform:uppercase;letter-spacing:1px;white-space:nowrap;">Live Prices</span>
  <span id="ticker-coins" style="display:flex;gap:18px;"></span>
  <span style="margin-left:auto;color:#333;font-size:0.6rem" id="ticker-ts">—</span>
</div>

<main>

  <!-- Stats row -->
  <div class="stats-row">
    <div class="stat-card" id="sc-total">
      <div class="label">Total Signals</div>
      <div class="val" id="s-total">—</div>
      <div class="sub">all time</div>
    </div>
    <div class="stat-card" id="sc-buy">
      <div class="label">BUY Signals</div>
      <div class="val green" id="s-buy">—</div>
      <div class="sub">actionable</div>
    </div>
    <div class="stat-card" id="sc-sell">
      <div class="label">SELL Signals</div>
      <div class="val red" id="s-sell">—</div>
      <div class="sub">actionable</div>
    </div>
    <div class="stat-card" id="sc-nt">
      <div class="label">No Trade</div>
      <div class="val" id="s-nt" style="color:var(--muted)">—</div>
      <div class="sub">filtered out</div>
    </div>
    <div class="stat-card" id="sc-pos">
      <div class="label">Open Positions</div>
      <div class="val blue" id="s-pos">—</div>
      <div class="sub">watching TP/SL</div>
    </div>
    <div class="stat-card" id="sc-conf">
      <div class="label">Avg Confidence</div>
      <div class="val" id="s-conf">—</div>
      <div class="sub">BUY + SELL only</div>
    </div>
  </div>

  <!-- Strategy Accuracy -->
  <div class="panel" style="margin-bottom:14px">
    <div class="panel-header">
      <h2>🎓 Strategy Accuracy <span style="font-weight:normal;color:var(--muted);font-size:0.68rem"> — 30-min directional win rate · auto-mutes below 40%</span></h2>
      <button class="rbtn" onclick="loadAccuracy()">↻</button>
    </div>
    <div id="accuracy-body" style="padding:10px 0">
      <span style="color:var(--muted);font-size:0.8rem">Loading accuracy data… (populates after first signals resolve)</span>
    </div>
  </div>

  <!-- ML Classifier -->
  <div class="panel" style="margin-bottom:14px">
    <div class="panel-header">
      <h2>🤖 ML Signal Filter <span style="font-weight:normal;color:var(--muted);font-size:0.68rem"> — learns from outcomes · blocks low-probability signals</span></h2>
      <button class="rbtn" onclick="loadMlStats()">↻</button>
    </div>
    <div id="ml-body" style="padding:10px 0">
      <span style="color:var(--muted);font-size:0.8rem">Loading ML stats…</span>
    </div>
  </div>

  <!-- Portfolio Guard -->
  <div class="panel" style="margin-bottom:14px">
    <div class="panel-header">
      <h2>🛡️ Portfolio Guard <span style="font-weight:normal;color:var(--muted);font-size:0.68rem"> — exposure limits · blocks over-concentrated signals</span></h2>
      <button class="rbtn" onclick="loadGuard()">↻</button>
    </div>
    <div id="guard-body" style="padding:10px 0">
      <span style="color:var(--muted);font-size:0.8rem">Loading portfolio guard…</span>
    </div>
  </div>

  <!-- Positions + Scan -->
  <div class="two-col">

    <!-- Open Positions with live price -->
    <div class="panel">
      <div class="panel-header">
        <h2>🎯 Open Positions <span style="font-weight:normal;color:var(--muted);font-size:0.68rem"> — live price vs TP/SL</span></h2>
        <button class="rbtn" onclick="loadPositions()">↻</button>
      </div>
      <div style="overflow-x:auto">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Side</th><th>Entry</th>
              <th>Live Price</th><th>Chg%</th>
              <th>Stop</th><th>TP1</th><th>TP2</th>
              <th>Progress→TP1</th><th>Strategy</th><th>Since</th>
            </tr>
          </thead>
          <tbody id="pos-table-body">
            <tr><td colspan="11" class="empty">No open positions</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Manual Scan -->
    <div class="panel">
      <div class="panel-header">
        <h2>🔍 Manual Scan</h2>
        <button class="btn" id="scan-btn" onclick="runScan()">▶ Scan</button>
      </div>
      <div class="panel-body">
        <div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:10px;">
          {watchlist_html}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px;">
          <label style="color:var(--muted);font-size:0.7rem">Symbol</label>
          <select id="scan-symbol" onchange="syncAssetClass(this.value)" style="width:160px">
            <optgroup label="Crypto Spot">
              {crypto_options}
            </optgroup>
            <optgroup label="Crypto Futures ⚡">
              {futures_options}
            </optgroup>
            <optgroup label="Stocks">
              {stock_options}
            </optgroup>
          </select>
          <label style="color:var(--muted);font-size:0.7rem">Class</label>
          <select id="scan-class">
            <option value="crypto">crypto</option>
            <option value="stock">stock</option>
          </select>
          <span id="market-badge" style="font-size:0.68rem;font-weight:bold;color:var(--blue)">SPOT</span>
          <label style="color:var(--muted);font-size:0.7rem">TF</label>
          <select id="scan-tf">
            <option value="1m">1m</option>
            <option value="5m">5m</option>
            <option value="15m" selected>15m</option>
            <option value="30m">30m</option>
            <option value="1h">1h</option>
            <option value="4h">4h</option>
            <option value="1d">1d</option>
          </select>
          <span class="spin" id="scan-spin">⟳</span>
        </div>
        <div id="scan-result" style="color:var(--muted)"></div>
      </div>
    </div>

  </div>

  <!-- Recent Signals -->
  <div class="panel" style="margin-bottom:14px">
    <div class="panel-header">
      <h2>📋 Recent Signals</h2>
      <div style="display:flex;gap:8px;align-items:center">
        <input id="filter-sym" placeholder="filter symbol…" style="width:120px" oninput="filterSignals()">
        <select id="filter-act" onchange="filterSignals()" style="width:96px">
          <option value="BUY_SELL" selected>BUY + SELL</option>
          <option value="BUY">BUY only</option>
          <option value="SELL">SELL only</option>
          <option value="">All (incl NO_TRADE)</option>
        </select>
        <button class="rbtn" onclick="loadSignals()">↻</button>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>Time</th><th>Symbol</th><th>Asset</th><th>Action</th>
            <th>Confidence</th><th>R:R</th>
            <th>Entry Zone</th><th>Stop Loss</th><th>Stop Limit</th><th>TP1</th><th>TP2</th>
            <th>Strategy</th><th>Regime</th><th>Patterns</th><th>Warnings</th>
          </tr>
        </thead>
        <tbody id="sig-tbody">
          <tr><td colspan="14" class="empty">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

</main>

<script>
// ── State ──────────────────────────────────────────────────────────────────────
let allSignals = [];
let lastPositions = [];
let livePrices = {{}};   // symbol → price
let prevPrices = {{}};   // symbol → prev price (for flash direction)
let countdown = 15;

// ── Clock ──────────────────────────────────────────────────────────────────────
function tickClock() {{
  document.getElementById('live-clock').textContent = new Date().toLocaleTimeString();
}}
setInterval(tickClock, 1000);
tickClock();

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmt(n, d=2) {{ return n != null ? parseFloat(n).toFixed(d) : '—'; }}
function fmtTime(iso) {{
  const d = new Date(iso);
  return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
}}
function elapsed(iso) {{
  const s = Math.floor((Date.now() - new Date(iso)) / 1000);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s/60) + 'm';
  return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'm';
}}
function flashCard(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 600);
}}

// ── Health ────────────────────────────────────────────────────────────────────
async function loadHealth() {{
  try {{
    const r = await fetch('/api/health');
    const d = await r.json();
    setDot('h-db', d.database === true, 'Database');
    setDot('h-bn', d.binance  === true, 'Binance');
    if (d.alpaca === null) {{
      document.getElementById('h-ap').className = 'dot warn';
      document.getElementById('h-ap-lbl').textContent = 'Alpaca (no key)';
    }} else {{
      setDot('h-ap', d.alpaca === true, 'Alpaca');
    }}
  }} catch(e) {{
    ['h-db','h-bn','h-ap'].forEach(id => document.getElementById(id).className = 'dot fail');
  }}
}}
function setDot(id, ok, label) {{
  document.getElementById(id).className = 'dot ' + (ok ? 'ok' : 'fail');
  document.getElementById(id + '-lbl').textContent = label + (ok ? ' ✓' : ' ✗');
}}

// ── Live prices ───────────────────────────────────────────────────────────────
async function loadPrices() {{
  // Build symbol list from open positions + watchlist
  const posSyms = lastPositions.map(p => p.asset_class + ':' + p.symbol);
  if (!posSyms.length) return;
  try {{
    const r = await fetch('/api/prices?symbols=' + encodeURIComponent(posSyms.join(',')));
    const d = await r.json();
    prevPrices = {{ ...livePrices }};
    livePrices = d.prices;
    renderPositions(lastPositions);  // re-render with new prices
  }} catch(e) {{}}
}}

// ── Positions ─────────────────────────────────────────────────────────────────
async function loadPositions() {{
  try {{
    const r = await fetch('/api/positions');
    const d = await r.json();
    lastPositions = d.positions;
    document.getElementById('s-pos').textContent = d.count;
    if (d.count !== parseInt(document.getElementById('s-pos').dataset.prev || '0')) {{
      flashCard('sc-pos');
      document.getElementById('s-pos').dataset.prev = d.count;
    }}
    await loadPrices();
    renderPositions(d.positions);
  }} catch(e) {{ console.error(e); }}
}}

function renderPositions(positions) {{
  const tbody = document.getElementById('pos-table-body');
  if (!positions.length) {{
    tbody.innerHTML = '<tr><td colspan="11" class="empty">No open positions being tracked</td></tr>';
    return;
  }}
  tbody.innerHTML = positions.map(p => {{
    const ac = p.action === 'BUY' ? 'action-buy' : 'action-sell';
    const tp2cell = p.take_profit_2 ? fmt(p.take_profit_2) : '<span style="color:var(--muted)">—</span>';
    const tp1cell = p.tp1_hit ? '<span style="color:var(--green)">✓ hit</span>' : fmt(p.take_profit_1);

    // Live price
    const cur = livePrices[p.symbol];
    const prev = prevPrices[p.symbol];
    let priceCell = '<span style="color:var(--muted)">fetching…</span>';
    let changeCell = '—';
    let rowFlash = '';
    let progressPct = 0;

    if (cur != null) {{
      const chg = ((cur - p.entry) / p.entry) * 100;
      const isUp = cur > (prev || cur);
      const isDown = cur < (prev || cur);
      const priceClr = cur > p.entry ? 'price-up' : cur < p.entry ? 'price-down' : 'price-flat';
      const chgSign = chg >= 0 ? '+' : '';
      priceCell = `<span class="price-cell ${{priceClr}}">${{fmt(cur, 4)}}</span>`;
      changeCell = `<span style="color:${{chg>=0?'var(--green)':'var(--red)'}}">${{chgSign}}${{chg.toFixed(2)}}%</span>`;
      rowFlash = (prev != null && isUp) ? 'flash-up' : (prev != null && isDown) ? 'flash-down' : '';

      // Real progress: % of the way from entry to TP1
      if (p.action === 'BUY') {{
        const range = p.take_profit_1 - p.entry;
        progressPct = range > 0 ? Math.max(0, Math.min(100, ((cur - p.entry) / range) * 100)) : 0;
      }} else {{
        const range = p.entry - p.take_profit_1;
        progressPct = range > 0 ? Math.max(0, Math.min(100, ((p.entry - cur) / range) * 100)) : 0;
      }}
    }}

    const barClr = p.action === 'BUY' ? '#3fb950' : '#f85149';
    const progressBar = `<div class="prog-wrap"><div class="prog-fill" style="width:${{progressPct.toFixed(0)}}%;background:${{barClr}}"></div></div> <span style="font-size:0.7rem;color:var(--muted)">${{progressPct.toFixed(0)}}%</span>`;

    return `<tr class="${{rowFlash}}">
      <td style="font-weight:bold">${{p.symbol}}</td>
      <td class="${{ac}}">${{p.action === 'BUY' ? '▲ BUY' : '▼ SELL'}}</td>
      <td>${{fmt(p.entry, 4)}}</td>
      <td>${{priceCell}}</td>
      <td>${{changeCell}}</td>
      <td style="color:var(--red)">${{fmt(p.stop_loss, 4)}}</td>
      <td style="color:var(--green)">${{tp1cell}}</td>
      <td style="color:var(--blue)">${{tp2cell}}</td>
      <td>${{progressBar}}</td>
      <td style="color:var(--muted);font-size:0.68rem">${{p.strategy}}</td>
      <td style="color:var(--muted)">${{elapsed(p.opened_at)}} ago</td>
    </tr>`;
  }}).join('');
}}

// ── Strategy Accuracy ─────────────────────────────────────────────────────────
async function loadAccuracy() {{
  try {{
    const r = await fetch('/api/accuracy');
    const d = await r.json();
    const container = document.getElementById('accuracy-body');
    if (!d.strategies || !d.strategies.length) {{
      container.innerHTML = '<span style="color:var(--muted);font-size:0.8rem">No outcomes yet — check back after 30 min from first signal</span>';
      return;
    }}
    const rows = d.strategies.map(s => {{
      const pct = s.win_rate;
      const barW = Math.max(0, Math.min(100, pct));
      const clr = pct >= 60 ? 'var(--green)' : pct >= 40 ? '#e3b341' : 'var(--red)';
      const badge = s.muted ? ' <span style="background:var(--red);color:#fff;padding:1px 5px;border-radius:3px;font-size:0.6rem">MUTED</span>' : '';
      return `<div style="display:flex;align-items:center;gap:10px;padding:5px 10px;border-bottom:1px solid var(--border)">
        <div style="width:160px;font-size:0.75rem;color:var(--text)">${{s.strategy}}${{badge}}</div>
        <div style="flex:1;background:var(--border);border-radius:3px;height:8px;overflow:hidden">
          <div style="width:${{barW}}%;background:${{clr}};height:100%;transition:width 0.4s"></div>
        </div>
        <div style="width:45px;text-align:right;color:${{clr}};font-weight:bold;font-size:0.78rem">${{pct}}%</div>
        <div style="width:80px;text-align:right;color:var(--muted);font-size:0.68rem">${{s.wins}}W / ${{s.losses}}L / ${{s.total}} total</div>
      </div>`;
    }}).join('');
    container.innerHTML = rows;
  }} catch(e) {{
    document.getElementById('accuracy-body').innerHTML = '<span style="color:var(--red)">Failed to load accuracy data</span>';
  }}
}}

// ── ML Classifier Stats ───────────────────────────────────────────────────────
async function loadMlStats() {{
  try {{
    const r = await fetch('/api/ml-stats');
    const d = await r.json();
    const el = document.getElementById('ml-body');

    if (!d.trained) {{
      const needed = 20;
      const have = d.samples || 0;
      const pct = Math.min(100, Math.round(have / needed * 100));
      el.innerHTML = `
        <div style="padding:8px 10px">
          <div style="color:var(--muted);font-size:0.78rem;margin-bottom:6px">
            🔄 Training in progress — need <strong>20 outcomes</strong> to activate model.
            Collecting outcome data (30 min after each signal)…
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <div style="flex:1;background:var(--border);border-radius:3px;height:8px;overflow:hidden">
              <div style="width:${{pct}}%;background:var(--blue);height:100%;transition:width 0.5s"></div>
            </div>
            <span style="color:var(--blue);font-size:0.75rem">${{have}} / 20 outcomes</span>
          </div>
          <div style="color:var(--muted);font-size:0.68rem;margin-top:4px">
            ⚡ Once trained: ML filter auto-blocks signals predicted below 85% win probability
          </div>
        </div>`;
      return;
    }}

    // Model is trained — show stats
    const accClr = d.cv_accuracy >= 65 ? 'var(--green)' : d.cv_accuracy >= 50 ? 'var(--yellow)' : 'var(--red)';
    const topFeats = Object.entries(d.feature_importances || {{}})
      .sort((a,b) => b[1]-a[1]).slice(0,5);
    const featBars = topFeats.map(([k,v]) => {{
      const w = Math.round(v * 400);
      return `<div style="display:flex;align-items:center;gap:6px;margin:2px 0">
        <span style="width:160px;font-size:0.68rem;color:var(--muted)">${{k}}</span>
        <div style="background:var(--blue);height:6px;width:${{w}}px;border-radius:2px"></div>
        <span style="font-size:0.68rem;color:var(--blue)">${{(v*100).toFixed(1)}}%</span>
      </div>`;
    }}).join('');

    const trained = d.last_trained_at ? new Date(d.last_trained_at).toLocaleString() : '—';
    el.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:20px;padding:8px 10px">
        <div>
          <div style="color:var(--muted);font-size:0.68rem">CV Accuracy</div>
          <div style="font-size:1.4rem;font-weight:bold;color:${{accClr}}">${{d.cv_accuracy}}%</div>
        </div>
        <div>
          <div style="color:var(--muted);font-size:0.68rem">Training Samples</div>
          <div style="font-size:1.4rem;font-weight:bold;color:var(--blue)">${{d.samples}}</div>
        </div>
        <div>
          <div style="color:var(--muted);font-size:0.68rem">Training Win Rate</div>
          <div style="font-size:1.4rem;font-weight:bold;color:var(--text)">${{d.win_rate}}%</div>
        </div>
        <div>
          <div style="color:var(--muted);font-size:0.68rem">Blocked (session)</div>
          <div style="font-size:1.4rem;font-weight:bold;color:var(--red)">${{d.blocked_last_hour}}</div>
        </div>
        <div>
          <div style="color:var(--muted);font-size:0.68rem">Next Retrain in</div>
          <div style="font-size:1.4rem;font-weight:bold;color:var(--yellow)">${{10 - (d.outcomes_since_retrain||0)}} outcomes</div>
        </div>
      </div>
      <div style="padding:4px 10px 8px">
        <div style="color:var(--muted);font-size:0.68rem;margin-bottom:4px">Top predictive features:</div>
        ${{featBars}}
      </div>
      <div style="padding:2px 10px;color:var(--muted);font-size:0.65rem">Last trained: ${{trained}}</div>`;
  }} catch(e) {{
    document.getElementById('ml-body').innerHTML = '<span style="color:var(--red)">Failed to load ML stats</span>';
  }}
}}

// ── Portfolio Guard ───────────────────────────────────────────────────────────
async function loadGuard() {{
  try {{
    const r = await fetch('/api/portfolio-guard');
    const d = await r.json();
    if (d.error) {{
      document.getElementById('guard-body').innerHTML = `<span style="color:var(--muted);font-size:0.8rem">${{d.error}}</span>`;
      return;
    }}
    const s = d.summary;
    const lim = d.limits;
    const totalClr = s.total >= lim.max_concurrent * 0.8 ? 'var(--yellow)' : 'var(--green)';
    const buyClr   = s.buy  >= lim.max_same_direction * 0.8 ? 'var(--yellow)' : 'var(--text)';
    const sellClr  = s.sell >= lim.max_same_direction * 0.8 ? 'var(--yellow)' : 'var(--text)';
    const cryptoClr = s.crypto >= lim.max_per_asset_class * 0.8 ? 'var(--yellow)' : 'var(--text)';
    const stockClr  = s.stock  >= lim.max_per_asset_class * 0.8 ? 'var(--yellow)' : 'var(--text)';

    const statCell = (label, val, max, clr) =>
      `<div style="text-align:center;padding:6px 12px;border-right:1px solid var(--border)">
        <div style="font-size:1.3rem;font-weight:bold;color:${{clr}}">${{val}}<span style="font-size:0.7rem;color:var(--muted)">/${{max}}</span></div>
        <div style="font-size:0.68rem;color:var(--muted)">${{label}}</div>
      </div>`;

    const rows = (d.active_signals || []).map(sig => {{
      const ac = sig.action === 'BUY' ? 'var(--green)' : 'var(--red)';
      const ago = Math.round((Date.now() - new Date(sig.accepted_at)) / 60000);
      return `<tr>
        <td>${{sig.symbol}}</td>
        <td style="color:${{ac}}">${{sig.action}}</td>
        <td style="color:var(--muted)">${{sig.asset_class}}</td>
        <td style="color:var(--muted)">${{ago}}m ago</td>
      </tr>`;
    }}).join('');

    document.getElementById('guard-body').innerHTML = `
      <div style="display:flex;border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:10px">
        ${{statCell('Total Open', s.total, lim.max_concurrent, totalClr)}}
        ${{statCell('BUY', s.buy, lim.max_same_direction, buyClr)}}
        ${{statCell('SELL', s.sell, lim.max_same_direction, sellClr)}}
        ${{statCell('Crypto', s.crypto, lim.max_per_asset_class, cryptoClr)}}
        <div style="text-align:center;padding:6px 12px">
          <div style="font-size:1.3rem;font-weight:bold;color:var(--text)">${{s.stock}}<span style="font-size:0.7rem;color:var(--muted)">/${{lim.max_per_asset_class}}</span></div>
          <div style="font-size:0.68rem;color:var(--muted)">Stock</div>
        </div>
      </div>
      ${{rows ? `<table><thead><tr><th>Symbol</th><th>Side</th><th>Class</th><th>Since</th></tr></thead><tbody>${{rows}}</tbody></table>` : '<span style="color:var(--muted);font-size:0.8rem">No signals currently tracked by guard</span>'}}`;
  }} catch(e) {{
    document.getElementById('guard-body').innerHTML = '<span style="color:var(--muted);font-size:0.8rem">Guard stats unavailable</span>';
  }}
}}

// ── Signals ───────────────────────────────────────────────────────────────────
async function loadSignals() {{
  try {{
    const r = await fetch('/api/signals?limit=200');
    const d = await r.json();
    const prev = allSignals.length;
    allSignals = d.signals;
    if (d.signals.length !== prev) flashCard('sc-total');
    filterSignals();
    updateStats(d.signals);
    document.getElementById('last-updated').textContent = 'last updated: ' + new Date().toLocaleTimeString();
  }} catch(e) {{ console.error(e); }}
}}

function filterSignals() {{
  const sym = document.getElementById('filter-sym').value.trim().toUpperCase();
  const act = document.getElementById('filter-act').value;
  let list = allSignals;
  if (sym) list = list.filter(s => s.symbol.includes(sym));
  if (act === 'BUY_SELL') {{
    list = list.filter(s => s.action === 'BUY' || s.action === 'SELL');
  }} else if (act) {{
    list = list.filter(s => s.action === act);
  }}
  renderSignals(list);
}}

function renderSignals(signals) {{
  const tbody = document.getElementById('sig-tbody');
  if (!signals.length) {{
    tbody.innerHTML = '<tr><td colspan="15" class="empty">No signals match the filter.</td></tr>';
    return;
  }}
  tbody.innerHTML = signals.map(s => {{
    const ac = s.action === 'BUY' ? 'action-buy' : s.action === 'SELL' ? 'action-sell' : 'action-nt';
    const conf = Math.round((s.confidence || 0) * 100);
    const barClr = s.action === 'BUY' ? '#3fb950' : s.action === 'SELL' ? '#f85149' : '#8b949e';
    const patterns = (s.patterns_detected || []).slice(0,3).join(', ') || '—';
    const warns = (s.warnings || []).length
      ? `<span class="yellow" title="${{(s.warnings||[]).join('&#10;')}}">⚠ ${{s.warnings.length}}</span>` : '—';
    const entry = (s.entry_zone_low && s.entry_zone_high)
      ? `${{fmt(s.entry_zone_low)}} – ${{fmt(s.entry_zone_high)}}` : '—';
    const rr = s.estimated_risk_reward ? fmt(s.estimated_risk_reward,1) + 'x' : '—';
    const rrClr = parseFloat(rr) >= 2 ? 'var(--green)' : 'var(--yellow)';
    const slCell = s.stop_loss ? fmt(s.stop_loss) : '—';
    const slLimitCell = s.stop_limit_price
      ? `<span title="Stop-Limit: trigger at ${{fmt(s.stop_loss)}}, fill at ${{fmt(s.stop_limit_price)}}">${{fmt(s.stop_limit_price)}}</span>`
      : '<span style="color:var(--muted)">—</span>';
    return `<tr>
      <td style="color:var(--muted)">${{fmtTime(s.generated_at)}}</td>
      <td style="font-weight:bold">${{s.symbol}}</td>
      <td style="color:var(--muted)">${{s.asset_class}}${{s.data_provider === 'binance_futures' ? ' <span style="color:var(--yellow);font-size:0.65rem">⚡F</span>' : ''}}</td>
      <td class="${{ac}}">${{s.action === 'BUY' ? '▲ BUY' : s.action === 'SELL' ? '▼ SELL' : '— NO TRADE'}}</td>
      <td><span class="conf-bar" style="width:${{conf/3}}px;background:${{barClr}}"></span>${{conf}}%</td>
      <td style="color:${{rrClr}}">${{rr}}</td>
      <td style="color:var(--muted)">${{entry}}</td>
      <td style="color:var(--red)">${{slCell}}</td>
      <td style="color:#e0604a;font-size:0.7rem">${{slLimitCell}}</td>
      <td style="color:var(--green)">${{s.take_profit_1 ? fmt(s.take_profit_1) : '—'}}</td>
      <td style="color:var(--blue)">${{s.take_profit_2 ? fmt(s.take_profit_2) : '—'}}</td>
      <td style="color:var(--muted)">${{s.strategy_name || '—'}}</td>
      <td style="color:var(--muted)">${{s.market_regime || '—'}}</td>
      <td style="color:var(--muted);font-size:0.68rem">${{patterns}}</td>
      <td>${{warns}}</td>
    </tr>`;
  }}).join('');
}}

function updateStats(signals) {{
  const buy  = signals.filter(s => s.action === 'BUY').length;
  const sell = signals.filter(s => s.action === 'SELL').length;
  const nt   = signals.filter(s => s.action === 'NO_TRADE').length;
  const confs = signals.filter(s => s.action !== 'NO_TRADE').map(s => s.confidence).filter(Boolean);
  const avg = confs.length ? Math.round(confs.reduce((a,b)=>a+b,0)/confs.length*100) : null;
  document.getElementById('s-total').textContent = signals.length;
  document.getElementById('s-buy').textContent   = buy;
  document.getElementById('s-sell').textContent  = sell;
  document.getElementById('s-nt').textContent    = nt;
  document.getElementById('s-conf').textContent  = avg != null ? avg + '%' : '—';
}}

// ── Manual scan ───────────────────────────────────────────────────────────────
const CRYPTO_SYMBOLS_SET   = new Set({json.dumps(crypto_symbols)});
const FUTURES_SYMBOLS_SET  = new Set({json.dumps(futures_symbols)});
const STOCK_SYMBOLS_SET    = new Set({json.dumps(stock_symbols)});

function syncAssetClass(sym) {{
  const sel = document.getElementById('scan-symbol');
  const opt = [...sel.options].find(o => o.value === sym);
  const market = opt ? opt.getAttribute('data-market') : null;
  const cls = STOCK_SYMBOLS_SET.has(sym) ? 'stock' : 'crypto';
  document.getElementById('scan-class').value = cls;
  // Show futures badge next to class selector
  const mktBadge = document.getElementById('market-badge');
  if (mktBadge) {{
    if (market === 'futures' || FUTURES_SYMBOLS_SET.has(sym)) {{
      mktBadge.textContent = '⚡ FUTURES';
      mktBadge.style.color = 'var(--yellow)';
    }} else if (cls === 'crypto') {{
      mktBadge.textContent = 'SPOT';
      mktBadge.style.color = 'var(--blue)';
    }} else {{
      mktBadge.textContent = 'STOCK';
      mktBadge.style.color = 'var(--green)';
    }}
  }}
}}

function setScan(sym, cls, market) {{
  const sel = document.getElementById('scan-symbol');
  const opt = [...sel.options].find(o => {{
    if (market === 'futures') return o.value === sym && o.getAttribute('data-market') === 'futures';
    return o.value === sym && o.getAttribute('data-market') !== 'futures';
  }});
  if (opt) {{ sel.value = sym; opt.selected = true; }}
  document.getElementById('scan-class').value = cls;
  syncAssetClass(sym);
}}

async function runScan() {{
  const btn    = document.getElementById('scan-btn');
  const spin   = document.getElementById('scan-spin');
  const result = document.getElementById('scan-result');
  const symbol = document.getElementById('scan-symbol').value.trim().toUpperCase();
  const cls    = document.getElementById('scan-class').value;
  const tf     = document.getElementById('scan-tf').value;

  btn.disabled = true;
  spin.className = 'spin on';
  result.innerHTML = `<span style="color:var(--muted)">Scanning ${{symbol}} (${{cls}}, ${{tf}})…</span>`;

  try {{
    const r = await fetch(`/api/scan?symbol=${{encodeURIComponent(symbol)}}&asset_class=${{cls}}&timeframe=${{tf}}`);
    const d = await r.json();
    if (!d.signals_found) {{
      result.innerHTML = `<span style="color:var(--muted)">No signals found for ${{symbol}} on ${{tf}}.</span>`;
    }} else {{
      result.innerHTML = d.signals.map(s => {{
        const clr  = s.action === 'BUY' ? 'var(--green)' : s.action === 'SELL' ? 'var(--red)' : 'var(--muted)';
        const icon = s.action === 'BUY' ? '▲' : s.action === 'SELL' ? '▼' : '─';
        return `<div style="color:${{clr}};margin-bottom:6px;padding:6px 10px;background:#1c2128;border-radius:4px;border-left:3px solid ${{clr}}">
          ${{icon}} <strong>${{s.action}}</strong> ${{s.symbol}}
          &nbsp;·&nbsp; Score: <strong>${{Math.round((s.confidence||0)*100)}}%</strong>
          &nbsp;·&nbsp; R:R <strong>${{(s.estimated_risk_reward||0).toFixed(1)}}x</strong>
          &nbsp;·&nbsp; Entry: ${{fmt(s.entry_zone_low)}}–${{fmt(s.entry_zone_high)}}
          &nbsp;·&nbsp; Stop: <span style="color:var(--red)">${{fmt(s.stop_loss)}}</span>
          &nbsp;·&nbsp; TP1: <span style="color:var(--green)">${{fmt(s.take_profit_1)}}</span>
          ${{s.take_profit_2 ? '&nbsp;·&nbsp; TP2: <span style="color:var(--blue)">' + fmt(s.take_profit_2) + '</span>' : ''}}
          &nbsp;·&nbsp; <span style="color:var(--muted)">${{s.strategy_name}}</span>
          ${{(s.warnings||[]).length ? ' <span style="color:var(--yellow)">⚠ ' + s.warnings.length + ' warn</span>' : ''}}
        </div>`;
      }}).join('');
      await Promise.all([loadSignals(), loadPositions()]);
    }}
  }} catch(e) {{
    result.innerHTML = `<span style="color:var(--red)">Error: ${{e.message}}</span>`;
  }}

  btn.disabled = false;
  spin.className = 'spin';
}}

// ── Live ticker via Binance WebSocket (true push, ~1s updates) ───────────────
const SPOT_TICKER_SYMBOLS    = {json.dumps(crypto_symbols)};
const FUTURES_TICKER_SYMBOLS = {json.dumps(futures_symbols)};
// Combined for display — spot symbols drive the visible bar; futures prices stored in livePrices too
const TICKER_SYMBOLS = SPOT_TICKER_SYMBOLS;
const TICKER_PREV = {{}};   // symbol → previous price for flash direction
let spotTickerWs    = null;
let futuresTickerWs = null;

function renderTickerPrice(sym, price) {{
  const prev  = TICKER_PREV[sym];
  const up    = prev != null && price > prev;
  const down  = prev != null && price < prev;
  const clr   = up ? 'var(--green)' : down ? 'var(--red)' : 'var(--text)';
  const arrow = up ? '▲' : down ? '▼' : '';
  TICKER_PREV[sym] = price;

  return `<span style="white-space:nowrap">
    <span style="color:var(--muted);font-size:0.68rem">${{sym.replace('USDT','')}} </span>
    <span style="color:${{clr}};font-weight:bold;transition:color 0.3s">${{arrow}} ${{price.toLocaleString(undefined, {{minimumFractionDigits:2,maximumFractionDigits:6}})}}</span>
  </span>`;
}}

function _handleTickerMsg(event, symbolSet) {{
  try {{
    const msg = JSON.parse(event.data);
    const d = msg.data || msg;   // combined stream wraps in .data; single stream is direct
    if (!d || !d.s || !d.c) return;
    const sym   = d.s;
    const price = parseFloat(d.c);

    if (!symbolSet.has(sym)) return;
    livePrices[sym] = price;

    // Only re-render ticker bar for spot symbols (keep bar uncluttered)
    if (SPOT_TICKER_SYMBOLS.includes(sym)) {{
      const container = document.getElementById('ticker-coins');
      const parts = SPOT_TICKER_SYMBOLS.map(s => {{
        const p = livePrices[s];
        if (p == null) return '';
        return renderTickerPrice(s, p);
      }});
      container.innerHTML = parts.join('');
      document.getElementById('ticker-ts').textContent = new Date().toLocaleTimeString();
    }}

    // Re-render positions with fresh prices regardless of market
    if (lastPositions.length) renderPositions(lastPositions);
  }} catch(e) {{}}
}}

function startTickerWs() {{
  // ── Spot WebSocket ────────────────────────────────────────────────────────
  if (SPOT_TICKER_SYMBOLS.length) {{
    if (spotTickerWs) {{ try {{ spotTickerWs.close(); }} catch(e) {{}} }}
    const spotStreams = SPOT_TICKER_SYMBOLS.map(s => s.toLowerCase() + '@miniTicker').join('/');
    spotTickerWs = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${{spotStreams}}`);
    const spotSet = new Set(SPOT_TICKER_SYMBOLS);
    spotTickerWs.onmessage = (ev) => _handleTickerMsg(ev, spotSet);
    spotTickerWs.onerror   = () => {{}};
    spotTickerWs.onclose   = () => setTimeout(startSpotWs, 3000);
  }}

  // ── Futures WebSocket ─────────────────────────────────────────────────────
  if (FUTURES_TICKER_SYMBOLS.length) {{
    if (futuresTickerWs) {{ try {{ futuresTickerWs.close(); }} catch(e) {{}} }}
    const futStreams = FUTURES_TICKER_SYMBOLS.map(s => s.toLowerCase() + '@miniTicker').join('/');
    futuresTickerWs = new WebSocket(`wss://fstream.binance.com/stream?streams=${{futStreams}}`);
    const futSet = new Set(FUTURES_TICKER_SYMBOLS);
    futuresTickerWs.onmessage = (ev) => _handleTickerMsg(ev, futSet);
    futuresTickerWs.onerror   = () => {{}};
    futuresTickerWs.onclose   = () => setTimeout(startFuturesWs, 3000);
  }}
}}

// Named restart helpers for reconnect callbacks
function startSpotWs() {{
  if (!SPOT_TICKER_SYMBOLS.length) return;
  if (spotTickerWs) {{ try {{ spotTickerWs.close(); }} catch(e) {{}} }}
  const streams = SPOT_TICKER_SYMBOLS.map(s => s.toLowerCase() + '@miniTicker').join('/');
  spotTickerWs = new WebSocket(`wss://stream.binance.com:9443/stream?streams=${{streams}}`);
  const spotSet = new Set(SPOT_TICKER_SYMBOLS);
  spotTickerWs.onmessage = (ev) => _handleTickerMsg(ev, spotSet);
  spotTickerWs.onerror   = () => {{}};
  spotTickerWs.onclose   = () => setTimeout(startSpotWs, 3000);
}}

function startFuturesWs() {{
  if (!FUTURES_TICKER_SYMBOLS.length) return;
  if (futuresTickerWs) {{ try {{ futuresTickerWs.close(); }} catch(e) {{}} }}
  const streams = FUTURES_TICKER_SYMBOLS.map(s => s.toLowerCase() + '@miniTicker').join('/');
  futuresTickerWs = new WebSocket(`wss://fstream.binance.com/stream?streams=${{streams}}`);
  const futSet = new Set(FUTURES_TICKER_SYMBOLS);
  futuresTickerWs.onmessage = (ev) => _handleTickerMsg(ev, futSet);
  futuresTickerWs.onerror   = () => {{}};
  futuresTickerWs.onclose   = () => setTimeout(startFuturesWs, 3000);
}}

// ── Refresh orchestration ─────────────────────────────────────────────────────
async function refreshAll() {{
  countdown = 15;
  document.getElementById('countdown').textContent = countdown;
  await Promise.all([loadHealth(), loadSignals(), loadPositions(), loadAccuracy(), loadMlStats(), loadGuard()]);
}}

function tick() {{
  countdown--;
  if (countdown < 0) countdown = 0;
  document.getElementById('countdown').textContent = countdown;
  if (countdown <= 0) refreshAll();
}}

// ── Boot ──────────────────────────────────────────────────────────────────────
refreshAll();
startTickerWs();                         // open WebSocket for live prices
setInterval(tick, 1000);                 // countdown + 15s full refresh
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_dashboard(host: str = "0.0.0.0", port: int = 8000) -> None:
    settings = get_settings()
    configure_logging(settings.observability.log_level)
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_dashboard()
