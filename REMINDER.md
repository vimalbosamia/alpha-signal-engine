# REMINDER — Build Execution Layer

**Date set:** May 20, 2026
**Check back:** June 3, 2026 (2 weeks)

## What to do when you return:

1. Check paper trading results at http://localhost:8000/paper
2. Review: win rate, total P&L, Sharpe ratio, max drawdown
3. If consistently profitable → build the execution layer (real money trading)
4. If not profitable → tune strategies, fix issues first

## Execution layer scope (2-3 days):
- Order management (place/cancel/modify real orders on Binance/Alpaca)
- Exchange API with WRITE permissions
- Slippage handling, partial fills, error recovery
- Kill switch (emergency stop all orders)
- Position reconciliation

## Current system:
- 102 commits, 939 tests, 35K lines
- 27 strategies, 40 patterns, 32 engines, 6 bots
- Paper trading live at http://localhost:8000/paper

## To restart server:
```bash
cd "/Users/chandnibosamiya/Documents/AI Project/ai-trading-signal-agent"
lsof -ti :8000 | xargs kill -9; sleep 2; uv run python -m apps.signal_agent.main serve
```

## WARNING
Real money = real risk. Only proceed if paper results prove the system works.
