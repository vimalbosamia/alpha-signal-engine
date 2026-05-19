"""
Paper trading dashboard page and API endpoints.

Mounted at /paper (HTML) and /api/paper/* (JSON APIs).

The engine is injected at runtime via set_paper_engine() by the runner.
"""
from __future__ import annotations

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

        # Fetch live prices and calculate unrealized P&L per position
        symbols = {p["symbol"] for p in positions}
        prices: dict[str, float] = {}
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

        for p in positions:
            live = prices.get(p["symbol"])
            if live:
                p["live_price"] = live
                entry = p["entry_price"]
                size = p["position_size_usd"]
                units = size / entry if entry else 0
                if p["action"] == "BUY":
                    pnl = (live - entry) * units
                else:
                    pnl = (entry - live) * units
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
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --muted: #8b949e; --text: #c9d1d9; --bright: #f0f6fc;
      --blue: #58a6ff; --green: #3fb950; --red: #f85149; --yellow: #d29922;
    }
    body { font-family: 'Courier New', monospace; background: var(--bg); color: var(--text); min-height: 100vh; font-size: 13px; }

    /* ── Nav ── */
    header {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 10px 20px; display: flex; align-items: center;
      justify-content: space-between; position: sticky; top: 0; z-index: 100;
    }
    header h1 { font-size: 1rem; color: var(--blue); }
    .header-right { display: flex; align-items: center; gap: 10px; font-size: 0.72rem; color: var(--muted); }
    .back-link { color: var(--blue); text-decoration: none; font-size: 0.78rem; }
    .back-link:hover { text-decoration: underline; }
    .virtual-badge {
      padding: 2px 10px; border-radius: 10px; font-weight: bold;
      font-size: 0.68rem; color: #fff; background: #9a6700;
      letter-spacing: 0.5px;
    }
    .btn-reset {
      background: var(--red); color: #fff; border: none;
      padding: 5px 13px; border-radius: 4px; cursor: pointer;
      font-size: 0.76rem; font-family: inherit;
    }
    .btn-reset:hover { background: #c93e38; }

    /* ── Hero money counter ── */
    .hero {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 32px 20px 24px; text-align: center;
    }
    #money-h1 {
      font-size: 4rem; font-weight: bold; color: var(--bright);
      letter-spacing: -1px; transition: color 0.4s;
      font-variant-numeric: tabular-nums;
    }
    #money-h1.profit { color: var(--green); }
    #money-h1.loss   { color: var(--red); }

    @keyframes pulse-green { 0%,100% { text-shadow: none; } 50% { text-shadow: 0 0 20px rgba(63,185,80,0.5); } }
    @keyframes pulse-red   { 0%,100% { text-shadow: none; } 50% { text-shadow: 0 0 20px rgba(248,81,73,0.5); } }
    #money-h1.profit { animation: pulse-green 2s ease infinite; }
    #money-h1.loss   { animation: pulse-red 2s ease infinite; }

    .hero-sub {
      margin-top: 10px; display: flex; justify-content: center;
      gap: 24px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted);
    }
    .hero-sub span { white-space: nowrap; }
    .hero-sub .val { color: var(--text); font-weight: bold; }

    /* ── Layout ── */
    main { padding: 14px 20px; max-width: 1700px; margin: 0 auto; }

    /* ── Panel ── */
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; margin-bottom: 14px; }
    .panel-header {
      padding: 9px 14px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
    }
    .panel-header h2 { font-size: 0.78rem; color: var(--bright); font-weight: bold; }

    /* ── Tables ── */
    table { width: 100%; border-collapse: collapse; font-size: 0.73rem; }
    th {
      padding: 6px 10px; text-align: left; color: var(--muted);
      font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border); white-space: nowrap;
    }
    td { padding: 7px 10px; border-bottom: 1px solid #1c2128; white-space: nowrap; vertical-align: middle; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #1c2128; }
    .empty { padding: 28px; text-align: center; color: var(--muted); font-size: 0.78rem; }

    /* ── Colors ── */
    .green  { color: var(--green)  !important; }
    .red    { color: var(--red)    !important; }
    .yellow { color: var(--yellow) !important; }
    .blue   { color: var(--blue)   !important; }
    .muted  { color: var(--muted)  !important; }

    /* ── Rank badge ── */
    .rank { display: inline-block; width: 20px; text-align: center; font-weight: bold; color: var(--muted); }
    .rank.gold   { color: #ffd700; }
    .rank.silver { color: #c0c0c0; }
    .rank.bronze { color: #cd7f32; }

    /* ── Bot action buttons ── */
    .bot-btn {
      background: none; border: 1px solid var(--border); border-radius: 4px;
      padding: 2px 7px; font-size: 0.65rem; color: var(--muted);
      cursor: pointer; font-family: inherit;
    }
    .bot-btn:hover { border-color: var(--blue); color: var(--blue); }
    .bot-btn.pause:hover { border-color: var(--yellow); color: var(--yellow); }
    .bot-btn.resume:hover { border-color: var(--green); color: var(--green); }

    /* ── Equity placeholder ── */
    .equity-placeholder {
      padding: 32px; text-align: center; color: var(--muted);
      font-size: 0.82rem; border: 1px dashed var(--border);
      border-radius: 4px; margin: 12px 14px;
    }

    /* ── Filter bar ── */
    .filter-bar { padding: 8px 14px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
    select, input {
      background: #21262d; border: 1px solid var(--border); color: var(--text);
      padding: 4px 8px; border-radius: 4px; font-family: inherit; font-size: 0.76rem;
    }
    select:focus, input:focus { outline: 1px solid var(--blue); border-color: var(--blue); }

    /* ── Refresh controls ── */
    .rbtn {
      background: none; border: 1px solid var(--border); border-radius: 10px;
      padding: 2px 8px; font-size: 0.66rem; color: var(--muted);
      cursor: pointer; font-family: inherit;
    }
    .rbtn:hover { border-color: var(--blue); color: var(--blue); }

    /* ── Phase badge ── */
    .phase-badge {
      display: inline-block; padding: 1px 6px; border-radius: 3px;
      font-size: 0.62rem; font-weight: bold;
      background: #21262d; color: var(--muted);
    }
    .phase-badge.running { background: rgba(63,185,80,0.15); color: var(--green); }
    .phase-badge.paused  { background: rgba(210,153,34,0.15); color: var(--yellow); }
    .phase-badge.stopped { background: rgba(248,81,73,0.15); color: var(--red); }

    #last-refresh { color: var(--muted); font-size: 0.65rem; }
  </style>
</head>
<body>

<!-- Navigation -->
<header>
  <h1>Paper Trading</h1>
  <div class="header-right">
    <a href="/" class="back-link">← Main Dashboard</a>
    <span class="virtual-badge">VIRTUAL MONEY — NO REAL TRADES</span>
    <span id="last-refresh">—</span>
    <button class="rbtn" onclick="refreshAll()">↻ Refresh</button>
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

  <!-- Equity Curve -->
  <div class="panel">
    <div class="panel-header">
      <h2>Equity Curve</h2>
    </div>
    <div class="equity-placeholder" id="equity-placeholder">
      Chart loads after first equity snapshot (5 min)
    </div>
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
            <th>Side</th>
            <th>Entry</th>
            <th>Live Price</th>
            <th>P&L</th>
            <th>P&L%</th>
            <th>Size</th>
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
  const startBalance = d.start_balance ?? 10000;
  const pnl = balance - startBalance;
  const pnlPct = startBalance > 0 ? (pnl / startBalance) * 100 : 0;

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
  document.getElementById('hero-uptime').textContent = d.uptime ?? '—';
  document.getElementById('hero-bots').textContent = (d.bots ?? []).length;
  document.getElementById('hero-open-pos').textContent = d.open_positions ?? 0;
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
    const pnl = (bot.effective_balance ?? 0) - (bot.start_balance ?? 10000);
    const pnlPct = (bot.start_balance ?? 10000) > 0
      ? (pnl / (bot.start_balance ?? 10000)) * 100 : 0;
    const winRate = bot.win_rate != null ? bot.win_rate.toFixed(1) + '%' : '—';
    const sharpe  = bot.sharpe_ratio != null ? bot.sharpe_ratio.toFixed(2) : '—';
    const phase = (bot.phase ?? 'running').toLowerCase();
    const phaseCls = phase === 'running' ? 'running' : phase === 'paused' ? 'paused' : 'stopped';

    return `<tr>
      <td><span class="rank ${rankClass}">${rank}</span></td>
      <td style="font-weight:bold;color:var(--bright)">${bot.name ?? '—'}</td>
      <td style="font-variant-numeric:tabular-nums">${fmtMoney(bot.effective_balance)}</td>
      <td style="color:${pnlColor(pnl)}">${pnl >= 0 ? '+' : ''}${fmtMoney(pnl)}</td>
      <td style="color:${pnlColor(pnlPct)}">${fmtPct(pnlPct)}</td>
      <td>${winRate}</td>
      <td>${bot.total_trades ?? 0}</td>
      <td style="color:var(--muted)">${sharpe}</td>
      <td><span class="phase-badge ${phaseCls}">${phase.toUpperCase()}</span></td>
      <td>
        <button class="bot-btn pause" onclick="pauseBot('${bot.name}')">Pause</button>
        <button class="bot-btn resume" onclick="resumeBot('${bot.name}')">Resume</button>
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
    return `<tr>
      <td style="color:var(--blue)">${p.bot_name ?? '—'}</td>
      <td style="font-weight:bold">${p.symbol ?? '—'}</td>
      <td style="color:${sideClr};font-weight:bold">${sideLabel}</td>
      <td>${fmt(p.entry_price, 4)}</td>
      <td style="font-weight:bold">${livePrice}</td>
      <td style="color:${pnlClr};font-weight:bold">${pnlSign}$${fmt(pnl, 2)}</td>
      <td style="color:${pnlClr}">${pnlSign}${fmt(pnlPct, 2)}%</td>
      <td style="color:var(--muted)">$${fmt(p.position_size_usd, 2)}</td>
      <td style="color:var(--red)">${fmt(p.stop_loss, 4)}</td>
      <td style="color:var(--green)">${fmt(p.take_profit_1, 4)}</td>
      <td style="color:var(--muted);font-size:0.68rem">${p.strategy_name ?? '—'}</td>
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

// ── Boot ──────────────────────────────────────────────────────────────────────
refreshAll();
setInterval(refreshAll, 5000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
