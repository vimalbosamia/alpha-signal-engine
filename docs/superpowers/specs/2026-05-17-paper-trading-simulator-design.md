# Paper Trading Simulator — Multi-Agent Hedge Fund

## Overview

Six competing hedge fund bots trade on $10,000 virtual capital using existing signal pipeline infrastructure. Each bot specializes in a different strategy style, sizes positions via Kelly criterion, and tracks P&L in real-time. Weekly Darwinian rebalance rewards winners with more capital.

**Core rule: No real money. No real orders. Virtual only.**

## Architecture

```
SignalPipeline (existing)
    │
    ▼ EventBus.SIGNAL_GENERATED
    │
PaperTradingEngine (new)
    │
    ├── CapitalAllocator
    │     $10K split across 5 bots
    │     Weekly rebalance by Sharpe ratio
    │
    ├── BotAgent (base class)
    │     Kelly criterion sizing
    │     Virtual position management
    │     Per-bot P&L tracking
    │
    ├── MomentumBot    → trends, breakouts, EMA crossovers
    ├── ReversalBot    → reversal patterns, key-level bounces
    ├── MeanReversionBot → pullbacks, range-bound, Bollinger
    ├── ScalperBot     → 1m/5m, tight stops, TP1-only exits
    ├── SwingBot       → 1h/4h, wide stops, holds for TP2
    ├── AdaptiveBot    → learns from losses, evolves filters
    │
    ├── PaperPortfolio
    │     Virtual balance per bot
    │     Trade history
    │     Equity curve snapshots
    │
    └── Dashboard endpoint: /paper
          Single page with giant H1 P&L counter
```

## Data Flow

1. Pipeline generates signal (existing, unchanged)
2. EventBus dispatches to PaperTradingEngine
3. Engine fans signal to all 6 bots
4. Each bot decides: take or skip (based on strategy match)
5. If take → Kelly sizes position → opens virtual trade in PaperPortfolio
6. Live prices from existing Binance WebSocket + Alpaca polling
7. When price hits TP1/TP2/SL → close trade, update bot P&L
8. Dashboard polls /api/paper/* endpoints, H1 updates every second via JS

## Bot Specifications

### MomentumBot
- **Strategies:** EMA crossover, ResistanceBreakoutStrategy
- **Regime filter:** Only trades when regime = TRENDING
- **Signals:** BUY in uptrends, SELL in downtrends
- **Timeframes:** 15m, 30m, 1h
- **Exit:** TP1 default, hold for TP2 if momentum strong

### ReversalBot
- **Strategies:** HammerReversalStrategy, ShootingStarReversalStrategy, SupportBreakdownStrategy
- **Pattern filter:** Only takes signals with reversal-type patterns from CandlePatternEngine
- **Regime filter:** Trades all regimes except CHOPPY
- **Timeframes:** 15m, 1h
- **Exit:** TP1 default

### MeanReversionBot
- **Strategies:** PullbackContinuationStrategy, PullbackBearContinuationStrategy
- **Regime filter:** Only trades when regime = RANGING or SIDEWAYS
- **Timeframes:** 15m, 30m
- **Exit:** TP1 default

### ScalperBot
- **Strategies:** All strategies on short timeframes
- **Timeframes:** 1m, 5m only
- **R:R filter:** Only takes R:R >= 1.5
- **Exit:** Always TP1 (never holds for TP2)
- **Max hold:** 1 hour, then force-close at market price

### SwingBot
- **Strategies:** All strategies on higher timeframes
- **Timeframes:** 1h, 4h only
- **R:R filter:** Only takes R:R >= 2.5
- **Exit:** Holds through TP1 → targets TP2
- **Max hold:** 48 hours

### AdaptiveBot (Self-Learning)
- **Strategies:** All strategies — no preset filter
- **Core behavior:** Starts by taking all signals above 60% confidence. After every loss, analyzes WHY it lost and tightens filters.
- **Loss analysis tracks:**
  - Which strategy lost
  - Which regime it lost in
  - Which symbol/timeframe
  - Which patterns were present
  - What the R:R was
  - Whether HTF conflicted
- **Adaptation rules:**
  - After 3 losses on same strategy → raise confidence threshold for that strategy by 5%
  - After 3 losses in same regime → avoid that regime for 10 trades
  - After 3 losses on same symbol → avoid that symbol for 10 trades
  - After loss with HTF conflict → require HTF alignment going forward
  - After loss with R:R < 2 → raise minimum R:R by 0.2
- **Recovery rules:**
  - After 5 consecutive wins → relax one filter slightly (lowest penalty first)
  - Never relax below original baseline
  - All adaptations stored in `adaptive_filters` JSON field
- **Safety:**
  - Min 10 samples before any adaptation
  - Rolling 30-trade window for stats
  - Logs every filter change with reason
  - Existing ML classifier win-prob used as additional gate
- **Timeframes:** All
- **Exit:** TP1 default, TP2 if confidence > 80%

## Kelly Criterion Position Sizing

```
kelly_fraction = win_rate - (loss_rate / avg_win_loss_ratio)
position_size = kelly_fraction * bot_capital * 0.5  # Half-Kelly for safety
```

### Three Phases Per Bot

| Phase | Trades | Sizing | Purpose |
|-------|--------|--------|---------|
| Cold Start | 1-10 | Fixed 1% of bot capital | Gather stats |
| Learning | 11-30 | Quarter-Kelly | Build confidence |
| Full | 31+ | Half-Kelly | Statistically meaningful |

### Guardrails
- Max single trade: 5% of bot capital
- Min trade size: $10 (skip if Kelly says less)
- Kelly capped at 25%
- Negative Kelly = bot paused (no edge)
- Win rate on rolling 50-trade window

## Capital Allocation

### Initial
- ~$1,667 per bot (equal split of $10,000 across 6 bots)

### Weekly Rebalance (Sunday 00:00 UTC)
- Rank bots by Sharpe ratio
- #1: 25% of total capital
- #2: 20%
- #3: 18%
- #4: 15%
- #5: 12%
- #6: 10%
- Negative Kelly bot → paused, capital redistributed
- Never force-close open positions during rebalance

## P&L Tracking

### Per Trade
- Entry price, exit price, position size, direction
- Simulated fees: 0.1% per side (0.2% round trip)
- Realized P&L (dollar + percentage)
- Hold duration
- Bot name, strategy name

### Per Bot (real-time)
- Balance (allocated + unrealized)
- Total P&L ($, %)
- Win rate, trade count
- Sharpe ratio, max drawdown, profit factor
- Current Kelly fraction
- Open positions count + unrealized P&L
- Phase (cold_start / learning / full / paused)

### Portfolio-Wide (real-time)
- Total balance = sum of all bot balances
- Combined P&L ($, %)
- Best/worst bot
- Total open positions
- Equity curve (snapshot every 5 minutes)

## Storage

### New Tables (SQLite, same DB)

**paper_trades:**
- id, bot_name, symbol, asset_class, action (BUY/SELL)
- entry_price, exit_price, position_size_usd, fees
- realized_pnl, pnl_pct, hold_duration_seconds
- strategy_name, signal_id
- opened_at, closed_at, status (OPEN/CLOSED/FORCE_CLOSED)

**paper_bot_stats:**
- bot_name, allocated_capital, current_balance
- total_pnl, win_count, loss_count, trade_count
- sharpe_ratio, max_drawdown, profit_factor
- kelly_fraction, phase, is_paused
- last_rebalance_at, updated_at

**paper_equity_curve:**
- timestamp, total_balance, bot_balances_json

## Dashboard — Single Page at /paper

### Layout (top to bottom)

**1. Giant H1 Money Counter**
- Shows total portfolio value: `$10,247.83 (+$247.83 / +2.47%)`
- Green text + pulse animation when profit
- Red text when loss
- Updates every second via JS (unrealized P&L from live prices)
- Subtitle: `Started: $10,000 · Running 3d 14h`

**2. Bot Leaderboard Table**
- Rank, bot name, balance, P&L%, win rate, trades, phase, status
- Color-coded bars for performance
- Pause/resume toggle per bot

**3. Equity Curve Chart**
- Line chart showing total balance over time
- 5-minute resolution
- Color lines per bot (optional toggle)

**4. Open Positions Table**
- Bot, symbol, side, entry, live price, unrealized P&L%, stop, TP
- Flash animation on price changes
- Progress bar to TP1

**5. Trade History Table**
- All closed trades, newest first
- Filterable by bot name
- Columns: time, bot, symbol, side, entry, exit, P&L, R:R, hold time

**6. Stats Footer**
- Capital allocation breakdown
- Combined metrics (Sharpe, drawdown, profit factor)
- Next rebalance countdown

### API Endpoints (new)

- `GET /api/paper/summary` — total P&L, bot stats, open count
- `GET /api/paper/positions` — all open virtual positions
- `GET /api/paper/trades` — closed trade history (paginated)
- `GET /api/paper/equity` — equity curve data points
- `POST /api/paper/bot/{name}/pause` — pause a bot
- `POST /api/paper/bot/{name}/resume` — resume a bot
- `POST /api/paper/reset` — reset everything to $10K fresh start

## Existing Code Reuse

| Existing Module | How Paper Trading Uses It |
|----------------|--------------------------|
| SignalPipeline | Generates signals — unchanged |
| EventBus | Dispatches signals to PaperTradingEngine |
| CandlePatternEngine | All 40 detectors wired in (fixes unused detectors) |
| PositionWatcher | Live price tracking for open paper positions |
| SignalOutcomeTracker | Outcome data feeds bot Kelly calculations |
| PortfolioGuard | Per-bot exposure limits |
| BacktestEngine | Historical Kelly parameter bootstrap |
| ML Classifier | Bot can use ML win-prob as additional filter |
| Dashboard (server.py) | Add /paper page + API endpoints |
| Binance WebSocket | Real-time prices for P&L calculation |

## New Files

```
libs/paper_trading/
    __init__.py
    engine.py           # PaperTradingEngine orchestrator
    bot_agent.py        # BotAgent base class
    bots/
        __init__.py
        momentum.py     # MomentumBot
        reversal.py     # ReversalBot
        mean_reversion.py # MeanReversionBot
        scalper.py      # ScalperBot
        swing.py        # SwingBot
        adaptive.py     # AdaptiveBot (self-learning)
    portfolio.py        # PaperPortfolio (balance, trades, metrics)
    allocator.py        # CapitalAllocator (Kelly + rebalance)
    models.py           # SQLAlchemy models for paper_trades etc.

apps/dashboard/
    paper_page.py       # /paper page HTML + API endpoints

tests/unit/test_paper_trading/
    test_engine.py
    test_kelly.py
    test_bots.py
    test_allocator.py
    test_portfolio.py
```

## Integration Point

In `apps/signal_agent/runner.py`, after existing `PositionWatcher` and `SignalOutcomeTracker` setup:

```python
# Start paper trading engine
from libs.paper_trading.engine import PaperTradingEngine
paper_engine = PaperTradingEngine()
await paper_engine.start()
# Engine subscribes to EventBus.SIGNAL_GENERATED internally
```

## Success Criteria

1. All 6 bots receive and filter signals independently
2. Kelly sizing works through all 3 phases
3. Live P&L updates every second on /paper page
4. H1 shows accurate total including unrealized
5. Trade history persists across restarts
6. Weekly rebalance adjusts capital allocation
7. 538+ existing tests still pass
8. New tests cover all paper trading logic
9. AdaptiveBot logs every filter change with reason
10. AdaptiveBot tightens filters after losses, relaxes after wins
