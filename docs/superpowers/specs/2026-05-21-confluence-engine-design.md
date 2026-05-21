# Institutional Confluence Engine — Design Spec

## Problem

Current system: 1 strategy fires = 1 trade. No multi-strategy confirmation.
Target: institutional-grade multi-layer probabilistic confluence where 1 trade = multiple strategy confirmations.

## Architecture: 5-Layer Hierarchical Confluence

### Layer 1 — Market Regime (0-15 points)
- Uses existing `RegimeEngine` (18 regimes)
- Hard blocks: `climactic`, `panic_selloff`, `liquidation_event`
- High scores: `breakout` (14), `trending_up/down` (13)

### Layer 2 — Directional Bias (0-20 points)
- Uses existing `BullBearBiasEngine` + structure + regime
- Outputs `probability_long`, `probability_short`
- Counter-bias penalized, aligned bias boosted

### Layer 3 — Entry Trigger (15-30 points)
- MANDATORY — no trade without trigger
- Max 1 trigger per strategy category (dedup correlated)
- Primary trigger = highest confidence across categories

### Layer 4 — Confirmation (5-10 each, stacked)
- Strategy confirmations: `mtf_alignment`, `trend_following`, `candle_momentum`, etc.
- Indicator confirmations: volume spike, RSI aligned, MACD aligned, VWAP support, EMA stack, ADX strong
- Max 1 per category, diversity bonus: +5 per additional unique category

### Layer 5 — Risk Suppression (-5 to -15 each)
- Exhaustion, divergence, weak volume, overextension, volatility collapse
- 3+ suppressions = hard block

## Scoring

```
final = regime + bias + trigger + confirmation + suppression + diversity_bonus
capped at 0-100
```

Quality grades: A (80+), B (60+), C (40+), D (<40)

## Graduated Thresholds

| Phase | Trades | Min Score | Min Layers |
|-------|--------|-----------|------------|
| Cold Start | < 50 | 35 | Trigger + 1 |
| Learning | 50-200 | 50 | Trigger + 2 |
| Mature | 200+ | 65 | Trigger + 3 |

## Trade Structure

Each signal carries full provenance: primary strategy, supporting strategies,
confirmation signals, suppression signals, all layer scores, diversity bonus,
final confluence score, quality grade.

## Integration

Pipeline changed from single-pass to two-pass:
1. Collect ALL strategy candidates
2. Group by direction, run through 5-layer engine
3. Emit enriched signals with V2 metadata

Existing `ConfluenceEngine` (V1) preserved for backward compatibility.
V2 runs alongside, enriches signals with multi-layer scoring.

## Module Structure

```
libs/confluence/
  engine.py              — ConfluenceV2Engine (orchestrator)
  scorer.py              — scoring + thresholds
  layer_regime.py        — Layer 1
  layer_bias.py          — Layer 2
  layer_trigger.py       — Layer 3
  layer_confirmation.py  — Layer 4
  layer_suppression.py   — Layer 5
  trade_structure.py     — ConfluenceTradeResult dataclass
  strategy_registry.py   — strategy → layer + category mapping
```

## Anti-Overtrading (existing, preserved)

- Portfolio guard: exposure limits
- ML classifier: win probability filter
- Direction lock: bias engine gates
- Shared loss memory: pattern avoidance
