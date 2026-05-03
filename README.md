# AI Trading Signal Agent

A production-grade, **signal-only** AI trading agent for US stocks and crypto. It analyzes market data, detects candlestick patterns, scores confluence across multiple technical signals, and emits structured BUY/SELL/NO_TRADE recommendations.

**It never places, modifies, or cancels orders.** All output is signals and analytics only.

---

## What It Does

| Capability | Detail |
|---|---|
| **Data ingestion** | Alpaca (stocks), Binance (crypto), real-time OHLCV |
| **Data quality** | 14-check candle validator — blocks signals on critical data issues |
| **Pattern detection** | 40 candlestick pattern detectors (reversal, continuation, indecision) |
| **Technical analysis** | Indicators, market structure, volume profile, regime detection |
| **Confluence scoring** | Weighted combination of patterns + indicators → signal strength 0–1 |
| **Risk guardrails** | Max signals per symbol, correlated-signal limits, portfolio guard |
| **Outcome tracking** | Records WIN/LOSS/EXPIRED per signal, per-strategy win rates |
| **ML classifier** | Online learner — improves signal filtering from outcome history |
| **Backtesting** | Walk-forward validation against historical OHLCV |
| **Dashboard** | FastAPI + web UI for live signal monitoring |
| **Observability** | Structured logs (structlog), Prometheus metrics, audit trail |

---

## Architecture

```
apps/
  signal_agent/   — main agent loop, pipeline orchestrator
  cli/            — backtest CLI
  dashboard/      — FastAPI monitoring UI
libs/
  analysis/       — patterns (40 detectors), indicators, regime, structure
  core/           — domain models, config, logging
  data/           — providers (Alpaca/Binance), storage (SQLite), quality
  ml/             — feature extraction, online signal classifier
  monitoring/     — outcome tracker, portfolio guard, metrics, alerts
  risk/           — risk engine, position sizing
  signals/        — confluence scorer, signal output
  strategies/     — breakout, momentum, reversal, trend, continuation
tests/
  unit/           — 534 tests across all modules
  integration/    — storage and pipeline integration tests
```

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

**API keys needed:**

| Provider | Purpose | Free tier available |
|---|---|---|
| [Alpaca](https://alpaca.markets) | US stock OHLCV + live prices | Yes (paper trading) |
| [Binance](https://www.binance.com) | Crypto OHLCV + live prices | Yes (testnet) |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/ai-trading-signal-agent.git
cd ai-trading-signal-agent

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
# Agent mode: research | backtest | paper | live
AGENT_MODE=paper

# Alpaca (stocks)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Binance (crypto)
BINANCE_API_KEY=your_key
BINANCE_SECRET_KEY=your_secret
BINANCE_ENV=testnet

# Watchlists
STOCK_WATCHLIST=AAPL,MSFT,NVDA,TSLA
CRYPTO_WATCHLIST=BTCUSDT,ETHUSDT,SOLUSDT

# Signal quality thresholds
MIN_CONFLUENCE_SCORE=0.65
MIN_REWARD_RISK=2.0
```

See `.env.example` for the full reference with all options.

### 3. Run tests

```bash
uv run pytest tests/unit/ -q
# Should show: 534 passed
```

---

## Running the Agent

### Continuous scan loop

```bash
uv run python -m apps.signal_agent.main run
uv run python -m apps.signal_agent.main run --timeframe 15m --interval 300
```

### Single symbol scan

```bash
uv run python -m apps.signal_agent.main once --symbol AAPL --asset-class stock
uv run python -m apps.signal_agent.main once --symbol BTCUSDT --asset-class crypto
```

### Backtest a strategy

```bash
uv run python -m apps.cli.backtest \
  --symbol BTCUSDT --asset-class crypto --timeframe 15m \
  --strategy macd_crossover --days 90

uv run python -m apps.cli.backtest \
  --symbol AAPL --asset-class stock --timeframe 1h \
  --strategy ema_crossover --days 180
```

### Dashboard

```bash
uv run python -m apps.dashboard.server
# Open http://localhost:8000
```

---

## Signal Output

Each signal is a structured object — never a market order:

```python
SignalOutput(
    signal_id=UUID(...),
    symbol="BTCUSDT",
    asset_class=AssetClass.CRYPTO,
    timeframe=Timeframe.FIFTEEN_MIN,
    action=SignalAction.BUY,           # BUY | SELL | NO_TRADE
    confluence_score=0.78,             # 0.0–1.0
    entry_zone_low=42100.0,
    entry_zone_high=42500.0,
    stop_loss=41200.0,
    take_profit_1=44000.0,
    take_profit_2=46000.0,
    strategy_name="breakout_momentum",
    patterns_detected=["hammer", "breakout_candle", "morning_star"],
    explanation="...",
)
```

NO_TRADE is emitted when data quality is critical, confluence is below threshold, risk limits are reached, or the strategy is muted due to poor win rate.

---

## Pattern Engine

40 candlestick detectors organized into 4 groups:

| Group | Count | Examples |
|---|---|---|
| Single-candle | 10 | Hammer, Shooting Star, Doji, Marubozu, Pin Bar |
| Two-candle | 8 | Engulfing, Harami, Piercing Line, Dark Cloud Cover, Tweezer |
| Multi-candle | 10 | Morning/Evening Star, Three White Soldiers, Abandoned Baby |
| Context-aware | 12 | Breakout, Exhaustion, Trap, Failed Breakout, Wide/Narrow Range |

Each detected pattern returns: `name`, `bias` (bullish/bearish/neutral), `category` (reversal/continuation/indecision), `strength` (0–100), `reliability` (0–100), `explanation`, `candle_index`, `timestamp`.

**Patterns alone never create signals** — they only contribute to confluence scoring.

---

## Data Quality

The `CandleValidator` runs 14 checks before any analysis:

**Critical (blocks signal):** empty data, negative prices, OHLC violations (high < open/close, low > open/close), duplicate timestamps, future timestamps, naive/non-UTC timestamps, ≥50% zero-volume candles.

**Warning (logs, does not block by default):** stale data, minority zero-volume bars, statistical price outliers, crypto continuity gaps, stock intraday session gaps.

Set `DATA_QUALITY_STRICT=true` in `.env` to block signals on warnings too.

---

## Risk Guardrails

Configured in `.env`:

```env
MAX_RISK_PER_SIGNAL_PCT=2.0      # max portfolio % at risk per signal
MAX_SIGNALS_PER_SYMBOL=2         # max concurrent signals per ticker
MAX_CORRELATED_SIGNALS=4         # max signals in the same sector/direction
MAX_ACTIVE_SIGNALS=20            # total active signals cap
```

Strategies with a win rate below threshold over recent signals are automatically muted until performance recovers.

---

## Development

```bash
# Lint
uv run ruff check libs/ apps/ tests/

# Type check
uv run mypy libs/ apps/

# Tests with coverage
uv run pytest tests/unit/ --cov=libs --cov-report=term-missing
```

### Branch strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, tested |
| `stage3-institutional-upgrades` | Pattern engine, data quality, ML classifier |

---

## Claude Code Plugins Used

This project was built using [Claude Code](https://claude.ai/code) with the following plugins and skills. If you want to contribute or extend this project using the same AI-assisted workflow:

### Required plugins

| Plugin | Install | Purpose |
|---|---|---|
| **superpowers** | `claude plugin install superpowers` | Core workflow skills: brainstorming, writing plans, subagent-driven development, TDD, code review |
| **ruflo** (Rufalo) | `claude plugin install ruflo-core` | Specialized subagents: coder, reviewer, researcher, tester, security auditor |

### Skills used

| Skill | What it did |
|---|---|
| `superpowers:brainstorming` | Designed each phase before implementation |
| `superpowers:writing-plans` | Produced step-by-step implementation plans |
| `superpowers:subagent-driven-development` | Dispatched fresh implementer + 2-stage review (spec compliance then code quality) per task |
| `superpowers:test-driven-development` | Enforced RED → GREEN → refactor discipline |
| `ruflo-core:coder` | Wrote focused implementations per task |
| `ruflo-core:reviewer` | Code quality gate after each task |
| `superpowers:code-reviewer` | Spec compliance gate after each task |

### Session hooks

| Hook | Purpose |
|---|---|
| **caveman mode** | Terse, filler-free responses — drop: "Sure!", "Certainly", "I'd be happy to". Enable: automatic on session start |
| **graphify** | Trigger `/graphify` to push any input to a knowledge graph for cross-session memory |

### Workflow used to build this project

```
/brainstorm → design doc → /plan → subagent-driven-development
  └── per task: implementer → spec review → code quality review → commit
```

To reproduce the development workflow on a new feature:

```
# In Claude Code terminal
/brainstorm     # design the feature
/plan           # create implementation plan
# Claude then auto-invokes subagent-driven-development
```

---

## What Was Built (Phase Summary)

| Phase | What | Tests added |
|---|---|---|
| Phase 1 | Domain models, strategy base, confluence scorer, signal output | — |
| Phase 2 | `CandleValidator` — 14-check data quality gate | +65 |
| Phase 3 — Task 1 | `PatternResult` extended: strength, reliability_score, explanation, category | +9 |
| Phase 3 — Task 2 | 28 existing detectors enriched with category/reliability/explanation | — |
| Phase 3 — Task 3 | 12 new context-aware detectors (breakout, exhaustion, trap, etc.) | — |
| Phase 3 — Task 4 | `CandlePatternEngine` — runs all 40 detectors, confidence-sorted output | — |
| Phase 3 — Task 5 | Comprehensive test suite — all 40 detectors verified | +80 |
| **Total** | | **534 passing** |

---

## License

MIT
