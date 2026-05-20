# Alpha Signal Engine

Institutional-grade AI trading signal intelligence engine with 6 competing hedge fund bots, 27 strategies, 40 candle pattern detectors, 32 analysis engines, and a real-time glassmorphism paper trading dashboard.

**Signal-only system — NEVER places real trades.**

> **WARNING — IMPORTANT DISCLAIMER**
>
> This software is for **educational and research purposes only**. It is NOT financial advice.
>
> - **No guarantee of profit.** Past performance does not indicate future results.
> - **Trading involves substantial risk of loss.** You can lose some or all of your invested capital.
> - **The developers are NOT responsible** for any financial losses, damages, or consequences arising from the use of this software.
> - **Do NOT trade with money you cannot afford to lose.**
> - **This system generates signals only** — it does not place, modify, or cancel any real orders.
> - **Paper trading uses virtual money ($10,000)** with real market prices. Trades are executed against live data but no real money is at risk. Real execution may differ due to slippage, liquidity, and order book depth.
> - **Always consult a licensed financial advisor** before making any investment decisions.
> - By using this software, you acknowledge that you are solely responsible for your own trading decisions.
>
> **USE AT YOUR OWN RISK.**

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (Alpaca for stocks, Binance needs no key)

# 3. Start server (30s scan interval for active trading)
uv run python -m apps.signal_agent.main serve --interval 30

# 4. Open dashboard
open http://localhost:8000/paper
```

## Server Commands

```bash
# Start server (default: 15m timeframe, 300s scan interval)
uv run python -m apps.signal_agent.main serve

# Start with custom settings
uv run python -m apps.signal_agent.main serve --timeframe 5m --interval 30 --port 8000

# Restart server (kill existing + start fresh)
lsof -ti :8000 | xargs kill -9; sleep 2; uv run python -m apps.signal_agent.main serve

# Reset paper trading to $10K
curl -X POST http://localhost:8000/api/paper/reset

# Run tests
uv run python -m pytest tests/ -q
```

## Dashboard URLs

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Main dashboard (signals overview) |
| `http://localhost:8000/paper` | Paper trading dashboard (glassmorphism UI) |
| `http://localhost:8000/docs` | API documentation (Swagger) |

---

## Universe Coverage

| Category | Count | Symbols |
|---|---|---|
| **US Stocks** | 32 | AAPL, MSFT, NVDA, AMD, AMZN, GOOGL, META, TSLA, AVGO, SMCI, MU, ARM, PLTR, TSM, JPM, GS, BAC, MS, NFLX, SHOP, SNOW, CRM, UBER, SPY, QQQ, IWM, DIA, SMH, XLF, XLK, TLT, GLD |
| **Crypto (Spot)** | 29 | BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, DOTUSDT, LINKUSDT, NEARUSDT, SUIUSDT, ATOMUSDT, UNIUSDT, AAVEUSDT, INJUSDT, ARBUSDT, OPUSDT, IMXUSDT, PEPEUSDT, SHIBUSDT, LTCUSDT, APTUSDT, STXUSDT, FETUSDT, RENDERUSDT, TONUSDT, BCHUSDT, POLUSDT |
| **Crypto (Futures)** | 20 | BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, LINKUSDT, DOTUSDT, AVAXUSDT, POLUSDT, LTCUSDT, NEARUSDT, SUIUSDT, APTUSDT, ARBUSDT, INJUSDT, FETUSDT, RENDERUSDT, TONUSDT |
| **Macro** | 15 | SPY, QQQ, DIA, IWM, TLT, IEF, GLD, SLV, USO, VIX, UVXY, SMH, XLF, XLK, DXY |
| **Market Leaders** | 7 | SPY, QQQ, AAPL, MSFT, NVDA, BTCUSDT, ETHUSDT |
| **Total Unique** | **61** | |

---

## What It Does

| Capability | Detail |
|---|---|
| **Data ingestion** | Alpaca (stocks/ETFs), Binance spot + futures (crypto), real-time OHLCV |
| **Data quality** | 14-check candle validator — blocks signals on critical data issues |
| **Pattern detection** | 40 candlestick pattern detectors (reversal, continuation, context-aware) |
| **Technical analysis** | RSI, MACD, EMA, ADX, ATR, Bollinger Bands, volume profile, regime detection |
| **Market structure** | BOS/CHoCH events with decay, swing highs/lows, trend classification |
| **Bias engine** | Composite directional scoring: indicators 40%, structure 15%, regime 15%, HTF 15%, volume 10%, candles 5% |
| **Participation matrix** | 8 market states decide which modes (SPOT/FUTURES LONG/SHORT) are allowed |
| **Spot/Futures separation** | EQUITY (stocks), SPOT (crypto), FUTURES (leveraged) — each with distinct rules |
| **Paper trading** | 6 competing hedge fund bots, $10K virtual capital, Kelly criterion sizing |
| **Active trade management** | TP/SL, trailing stop, max hold (4hr), bias flip auto-exit (2-check rule) |
| **Risk guardrails** | Leverage caps, liquidation buffer, duplicate position blocking, TP/SL validation |
| **Confluence scoring** | Weighted combination of patterns + indicators + structure → signal strength 0-1 |
| **ML classifier** | Online learner — improves signal filtering from outcome history |
| **Shared loss memory** | 3+ losses on same pattern → all bots avoid; 2 wins reset |
| **Dashboard** | Glassmorphism UI with Orbitron headings, interactive Lightweight Charts |
| **Observability** | Structured logs (structlog), Prometheus metrics, audit trail |

---

## Paper Trading Bots

6 competing hedge fund bots, each with distinct strategy preferences:

| Bot | Style | Focus |
|---|---|---|
| **MomentumBot** | Trend follower | EMA crossovers, MTF alignment, strong momentum |
| **ReversalBot** | Mean reversion | Exhaustion patterns, overbought/oversold RSI |
| **MeanReversionBot** | Statistical | Bollinger Band extremes, Z-score reversion |
| **ScalperBot** | Quick trades | Fast entries, tight stops, small targets |
| **SwingBot** | Breakout hunter | Support/resistance breaks, volume confirmation |
| **AdaptiveBot** | Self-tuning | Adjusts strategy mix based on recent win rate |

All bots share loss memory — if one bot loses 3+ times on a pattern, ALL bots avoid it until 2 wins reset confidence.

---

## Market Participation Matrix

The system classifies each symbol into a market state and decides which trading modes are permitted:

| Market State | SPOT | Futures LONG | Futures SHORT | Confidence | Size |
|---|---|---|---|---|---|
| **STRONG_BULL** | ON | ON | OFF | 100% | 100% |
| **WEAK_BULL** | ON | ON | OFF | 85% | 80% |
| **NEUTRAL** | OFF | OFF | OFF | 0% | 0% |
| **WEAK_BEAR** | OFF | OFF | ON | 85% | 70% |
| **STRONG_BEAR** | OFF | OFF | ON | 100% | 80% |
| **HIGH_VOLATILITY** | OFF | OFF | OFF | 0% | 0% |
| **NEWS_LOCKDOWN** | OFF | OFF | OFF | 0% | 0% |
| **TREND_TRANSITION** | OFF | OFF | OFF | 50% | 50% |

Example (live FOMC day):

| Symbol | State | SPOT | FUT L | FUT S | Result |
|---|---|---|---|---|---|
| AAPL | STRONG_BULL | ON | ON | OFF | Trade opened (BUY) |
| MSFT | WEAK_BULL | ON | ON | OFF | Trade opened (BUY) |
| NVDA | WEAK_BEAR | OFF | OFF | ON | SPOT blocked, futures short allowed |
| BTCUSDT | NEUTRAL | OFF | OFF | OFF | All blocked (conflict 0.97) |
| ETHUSDT | NEUTRAL | OFF | OFF | OFF | All blocked |

Classification uses: BullBearBias scores, ADX trend strength, conflict score, macro event calendar, volatility flags.

---

## Spot vs Futures Separation

| Concept | SPOT / EQUITY | FUTURES |
|---|---|---|
| **Shorting** | Not allowed | Allowed (OPEN_SHORT) |
| **Leverage** | 1x always | 1-5x (default 3x, dev cap) |
| **Liquidation** | N/A | Calculated + buffer check (min 5%) |
| **TP/SL validation** | SL < entry, TP > entry | Direction-aware (short: SL > entry, TP < entry) |
| **Market mode** | SPOT (crypto) / EQUITY (stocks) | FUTURES |
| **Position intent** | OPEN_LONG / CLOSE_LONG only | All intents (OPEN_LONG/CLOSE_LONG/OPEN_SHORT/CLOSE_SHORT) |
| **Duplicate blocking** | 1 long per symbol | 1 position per symbol per direction |
| **P&L calculation** | `(exit - entry) * units` | `(exit - entry) * units * leverage` |

**Bias flip auto-exit**: if live bias opposes trade direction for 2 consecutive 5-minute reanalysis checks, position is auto-closed. Aligned bias resets counter. Neutral does not increment.

---

## Active Trade Management

Every open position is monitored every 10 seconds:

| Rule | Condition | Action |
|---|---|---|
| **TP/SL** | Price hits stop loss or take profit | Close immediately |
| **Max hold** | Position held > 4 hours | Close (day trading mode) |
| **Cut loss** | Unrealized P&L < -1.5% after 20 min | Close |
| **Break-even exit** | P&L < -0.3% after 40 min | Close |
| **Sustained adverse** | Price moves against for 30 consecutive checks (~5 min) | Close |
| **Bias flip** | Bias opposes direction for 2 reanalysis cycles (10 min) | Close |

---

## Architecture

```
apps/
  signal_agent/     — main agent loop, pipeline orchestrator, 27 strategies
  cli/              — backtest CLI
  dashboard/        — FastAPI monitoring UI (glassmorphism)
libs/
  analysis/
    bias/           — BullBearBiasEngine (composite directional scoring)
    indicators/     — RSI, MACD, EMA, ADX, ATR, Bollinger Bands
    structure/      — BOS/CHoCH events, market structure, Wyckoff
    regime/         — Market regime classification (trending/ranging/breakout)
    participation/  — MarketParticipationMatrix (8 market states)
    macro/          — News, sentiment, correlation, central bank, global risk
    patterns/       — 40 candlestick pattern detectors
    filters/        — Universe manager, dynamic symbol ranking
    levels/         — Support/resistance key levels
  core/             — Domain models, config, logging
  data/             — Providers (Alpaca/Binance/Binance Futures), SQLite, quality
  ml/               — Feature extraction, online signal classifier
  learning/         — Performance tracker, model registry, shared loss memory
  monitoring/       — Outcome tracker, portfolio guard, metrics, alerts
  paper_trading/    — 6 bots, portfolio, market mode validator, bias exit logic
  risk/             — Futures risk engine, VaR, trailing stop, position sizing
  signals/          — Confluence scorer, confidence calibrator, explainability
  strategies/       — breakout, momentum, reversal, trend, continuation (27 total)
  validation/       — Monte Carlo, walk-forward, leakage guard
tests/
  unit/             — 59+ tests (market mode, bias exit, participation matrix)
  integration/      — Storage and pipeline integration tests
```

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

**API keys needed:**

| Provider | Purpose | Free tier available |
|---|---|---|
| [Alpaca](https://alpaca.markets) | US stock/ETF OHLCV + live prices | Yes (paper trading) |
| [Binance](https://www.binance.com) | Crypto spot + futures OHLCV + live prices | Yes (no key needed for public data) |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/vimalbosamia/alpha-signal-engine.git
cd alpha-signal-engine

# With uv (recommended)
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
AGENT_MODE=paper

# Alpaca (stocks + ETFs)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Binance (crypto) — no key needed for public candle data
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
BINANCE_ENV=live

# Watchlists
STOCK_WATCHLIST=AAPL,MSFT,NVDA,AMD,AMZN,GOOGL,META,TSLA,AVGO,SMCI,MU,ARM,PLTR,TSM,JPM,GS,BAC,MS,NFLX,SHOP,SNOW,CRM,UBER,SPY,QQQ,IWM,DIA,SMH,XLF,XLK,TLT,GLD
CRYPTO_WATCHLIST=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,NEARUSDT,SUIUSDT,ATOMUSDT,UNIUSDT,AAVEUSDT,INJUSDT,ARBUSDT,OPUSDT,IMXUSDT,PEPEUSDT,SHIBUSDT,LTCUSDT,APTUSDT,STXUSDT,FETUSDT,RENDERUSDT,TONUSDT,BCHUSDT,POLUSDT
FUTURES_WATCHLIST=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT,DOTUSDT,AVAXUSDT,POLUSDT,LTCUSDT,NEARUSDT,SUIUSDT,APTUSDT,ARBUSDT,INJUSDT,FETUSDT,RENDERUSDT,TONUSDT
ENABLE_FUTURES=true
```

### 3. Run tests

```bash
uv run pytest tests/unit/ -q
```

---

## Signal Output

Each signal is a structured object — never a market order:

```python
SignalOutput(
    symbol="BTCUSDT",
    asset_class=AssetClass.CRYPTO,
    action=SignalAction.BUY,
    trading_mode=TradingMode.FUTURES,    # SPOT | FUTURES
    confidence=0.78,
    entry_zone_low=107100.0,
    entry_zone_high=107500.0,
    stop_loss=106200.0,
    take_profit_1=109000.0,
    strategy_name="breakout_momentum",
    market_regime=MarketRegime.TRENDING_UP,
    patterns_detected=["hammer", "breakout_candle"],
)
```

---

## Pattern Engine

40 candlestick detectors organized into 4 groups:

| Group | Count | Examples |
|---|---|---|
| Single-candle | 10 | Hammer, Shooting Star, Doji, Marubozu, Pin Bar |
| Two-candle | 8 | Engulfing, Harami, Piercing Line, Dark Cloud Cover, Tweezer |
| Multi-candle | 10 | Morning/Evening Star, Three White Soldiers, Abandoned Baby |
| Context-aware | 12 | Breakout, Exhaustion, Trap, Failed Breakout, Wide/Narrow Range |

**Patterns alone never create signals** — they only contribute to confluence scoring.

---

## 27 Strategies

| Category | Strategies |
|---|---|
| **Trend** | EMA Crossover, Trend Following, MTF Alignment |
| **Breakout** | Resistance Breakout, Break and Retest, Range Breakout |
| **Momentum** | MACD Crossover, RSI Divergence, Momentum Burst |
| **Reversal** | Shooting Star Reversal, Hammer Reversal, Double Top/Bottom |
| **Mean Reversion** | Bollinger Band Reversion, Z-Score Reversion |
| **Structure** | Order Block, Fair Value Gap, Wyckoff Accumulation |
| **Continuation** | Flag/Pennant, Inside Bar Breakout |
| **Scalp** | Quick Scalp, Micro Breakout |

---

## Development

```bash
# Lint
uv run ruff check libs/ apps/ tests/

# Type check
uv run mypy libs/ apps/

# Tests with coverage
uv run pytest tests/unit/ --cov=libs --cov-report=term-missing

# Run specific test suite
uv run pytest tests/unit/test_market_mode_validator.py -v
uv run pytest tests/unit/test_bias_exit.py -v
uv run pytest tests/unit/test_participation_matrix.py -v
```

---

## License

MIT
