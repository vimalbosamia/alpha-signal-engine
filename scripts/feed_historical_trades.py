"""Feed closed trade history into the self-training coordinator.

Run: .venv/bin/python scripts/feed_historical_trades.py
"""
from libs.learning.coordinator import get_coordinator
from libs.learning.pattern_scorer import get_pattern_store
from libs.learning.strategy_tuner import get_strategy_tuner
from libs.learning.reward_engine import get_reward_engine
from libs.paper_trading.state_persistence import _save_learning_systems, _ensure_dir

TRADES = [
    {"symbol": "LINKUSDT", "side": "SELL", "pnl": -0.69, "pnl_pct": -1.00, "strategy": "shooting_star_reversal"},
    {"symbol": "XRPUSDT", "side": "SELL", "pnl": -0.39, "pnl_pct": -0.54, "strategy": "shooting_star_reversal"},
    {"symbol": "BNBUSDT", "side": "SELL", "pnl": -0.29, "pnl_pct": -0.59, "strategy": "shooting_star_reversal"},
    {"symbol": "XRPUSDT", "side": "SELL", "pnl": -0.53, "pnl_pct": -0.96, "strategy": "support_breakdown"},
    {"symbol": "SOLUSDT", "side": "SELL", "pnl": -0.48, "pnl_pct": -0.94, "strategy": "shooting_star_reversal"},
    {"symbol": "SOLUSDT", "side": "SELL", "pnl": -0.31, "pnl_pct": -0.94, "strategy": "candle_momentum"},
    {"symbol": "SOLUSDT", "side": "SELL", "pnl": -0.62, "pnl_pct": -0.94, "strategy": "shooting_star_reversal"},
    {"symbol": "BNBUSDT", "side": "SELL", "pnl": -0.29, "pnl_pct": -0.43, "strategy": "shooting_star_reversal"},
    {"symbol": "XRPUSDT", "side": "SELL", "pnl": -0.61, "pnl_pct": -1.00, "strategy": "mtf_alignment"},
    {"symbol": "APTUSDT", "side": "SELL", "pnl": -0.42, "pnl_pct": -0.98, "strategy": "shooting_star_reversal"},
    {"symbol": "APTUSDT", "side": "SELL", "pnl": -0.67, "pnl_pct": -0.98, "strategy": "shooting_star_reversal"},
    {"symbol": "XRPUSDT", "side": "SELL", "pnl": -0.53, "pnl_pct": -0.96, "strategy": "support_breakdown"},
    {"symbol": "ARBUSDT", "side": "SELL", "pnl": -0.84, "pnl_pct": -1.12, "strategy": "candle_momentum"},
    {"symbol": "ARBUSDT", "side": "SELL", "pnl": -0.67, "pnl_pct": -1.12, "strategy": "break_and_retest"},
    {"symbol": "FETUSDT", "side": "SELL", "pnl": -0.38, "pnl_pct": -0.81, "strategy": "shooting_star_reversal"},
    {"symbol": "FETUSDT", "side": "SELL", "pnl": -0.46, "pnl_pct": -0.81, "strategy": "shooting_star_reversal"},
    {"symbol": "LTCUSDT", "side": "SELL", "pnl": -0.35, "pnl_pct": -0.76, "strategy": "shooting_star_reversal"},
    {"symbol": "LTCUSDT", "side": "SELL", "pnl": -0.47, "pnl_pct": -0.76, "strategy": "shooting_star_reversal"},
    {"symbol": "LINKUSDT", "side": "SELL", "pnl": -0.64, "pnl_pct": -1.00, "strategy": "mtf_alignment"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": 0.03, "pnl_pct": 0.05, "strategy": "hammer_reversal"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": 0.15, "pnl_pct": 0.26, "strategy": "hammer_reversal"},
    {"symbol": "ATOMUSDT", "side": "BUY", "pnl": -0.20, "pnl_pct": -0.44, "strategy": "resistance_breakout"},
    {"symbol": "INJUSDT", "side": "SELL", "pnl": -0.71, "pnl_pct": -1.23, "strategy": "support_breakdown"},
    {"symbol": "INJUSDT", "side": "SELL", "pnl": -0.51, "pnl_pct": -1.11, "strategy": "support_breakdown"},
    {"symbol": "ATOMUSDT", "side": "BUY", "pnl": -0.33, "pnl_pct": -0.49, "strategy": "trend_following"},
    {"symbol": "ATOMUSDT", "side": "BUY", "pnl": -0.26, "pnl_pct": -0.44, "strategy": "resistance_breakout"},
    {"symbol": "ATOMUSDT", "side": "BUY", "pnl": -0.37, "pnl_pct": -0.44, "strategy": "resistance_breakout"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": -0.01, "pnl_pct": -0.01, "strategy": "hammer_reversal"},
    {"symbol": "TONUSDT", "side": "SELL", "pnl": -1.61, "pnl_pct": -2.69, "strategy": "break_and_retest"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": 0.32, "pnl_pct": 0.38, "strategy": "bollinger_mean_reversion"},
    {"symbol": "ARBUSDT", "side": "SELL", "pnl": -0.62, "pnl_pct": -1.12, "strategy": "ema_crossover"},
    {"symbol": "RENDERUSDT", "side": "BUY", "pnl": 0.09, "pnl_pct": 0.12, "strategy": "mtf_alignment"},
    {"symbol": "RENDERUSDT", "side": "BUY", "pnl": 0.10, "pnl_pct": 0.12, "strategy": "mtf_alignment"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": -0.14, "pnl_pct": -0.17, "strategy": "bollinger_mean_reversion"},
    {"symbol": "TONUSDT", "side": "SELL", "pnl": -0.42, "pnl_pct": -1.20, "strategy": "candle_momentum"},
    {"symbol": "DOTUSDT", "side": "BUY", "pnl": -0.44, "pnl_pct": -0.70, "strategy": "resistance_breakout"},
    {"symbol": "LINKUSDT", "side": "SELL", "pnl": 1.34, "pnl_pct": 1.60, "strategy": "candle_momentum"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.13, "pnl_pct": -0.37, "strategy": "candlestick_reversal"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.23, "pnl_pct": -0.42, "strategy": "resistance_breakout"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.19, "pnl_pct": -0.37, "strategy": "mtf_alignment"},
    {"symbol": "RENDERUSDT", "side": "BUY", "pnl": -0.56, "pnl_pct": -0.98, "strategy": "resistance_breakout"},
    {"symbol": "RENDERUSDT", "side": "BUY", "pnl": -0.48, "pnl_pct": -0.97, "strategy": "mtf_alignment"},
    {"symbol": "AVAXUSDT", "side": "BUY", "pnl": -0.31, "pnl_pct": -0.60, "strategy": "volume_breakout"},
    {"symbol": "DOTUSDT", "side": "BUY", "pnl": -0.41, "pnl_pct": -0.81, "strategy": "trend_following"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.32, "pnl_pct": -0.68, "strategy": "resistance_breakout"},
    {"symbol": "BNBUSDT", "side": "SELL", "pnl": 0.24, "pnl_pct": 0.73, "strategy": "shooting_star_reversal"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": -0.11, "pnl_pct": -0.14, "strategy": "bollinger_mean_reversion"},
    {"symbol": "UNIUSDT", "side": "BUY", "pnl": -0.45, "pnl_pct": -0.59, "strategy": "hammer_reversal"},
    {"symbol": "NEARUSDT", "side": "SELL", "pnl": 0.80, "pnl_pct": 1.35, "strategy": "shooting_star_reversal"},
    {"symbol": "FETUSDT", "side": "BUY", "pnl": -0.24, "pnl_pct": -0.50, "strategy": "mtf_alignment"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": -0.24, "pnl_pct": -0.28, "strategy": "bollinger_mean_reversion"},
    {"symbol": "NEARUSDT", "side": "BUY", "pnl": -0.53, "pnl_pct": -1.13, "strategy": "resistance_breakout"},
    {"symbol": "INJUSDT", "side": "BUY", "pnl": -0.38, "pnl_pct": -0.71, "strategy": "sma_crossover"},
    {"symbol": "INJUSDT", "side": "BUY", "pnl": -0.22, "pnl_pct": -0.45, "strategy": "resistance_breakout"},
    {"symbol": "INJUSDT", "side": "BUY", "pnl": -0.34, "pnl_pct": -0.45, "strategy": "resistance_breakout"},
    {"symbol": "UNIUSDT", "side": "BUY", "pnl": -0.35, "pnl_pct": -0.70, "strategy": "mtf_alignment"},
    {"symbol": "DOGEUSDT", "side": "BUY", "pnl": -0.71, "pnl_pct": -2.25, "strategy": "mtf_alignment"},
    {"symbol": "UNIUSDT", "side": "BUY", "pnl": -0.25, "pnl_pct": -0.40, "strategy": "mtf_alignment"},
    {"symbol": "UNIUSDT", "side": "BUY", "pnl": -0.30, "pnl_pct": -0.40, "strategy": "mtf_alignment"},
    {"symbol": "INJUSDT", "side": "BUY", "pnl": -0.47, "pnl_pct": -0.91, "strategy": "resistance_breakout"},
    {"symbol": "SUIUSDT", "side": "BUY", "pnl": -1.05, "pnl_pct": -1.47, "strategy": "mtf_alignment"},
    {"symbol": "SUIUSDT", "side": "BUY", "pnl": -0.73, "pnl_pct": -1.47, "strategy": "mtf_alignment"},
    {"symbol": "DOGEUSDT", "side": "BUY", "pnl": -0.58, "pnl_pct": -0.95, "strategy": "trend_following"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": 0.03, "pnl_pct": 0.03, "strategy": "bollinger_mean_reversion"},
    {"symbol": "INJUSDT", "side": "BUY", "pnl": -0.75, "pnl_pct": -0.95, "strategy": "hammer_reversal"},
    {"symbol": "NEARUSDT", "side": "BUY", "pnl": -0.76, "pnl_pct": -2.42, "strategy": "resistance_breakout"},
    {"symbol": "NEARUSDT", "side": "BUY", "pnl": -1.48, "pnl_pct": -2.42, "strategy": "resistance_breakout"},
    {"symbol": "NEARUSDT", "side": "BUY", "pnl": -1.32, "pnl_pct": -2.27, "strategy": "mtf_alignment"},
    {"symbol": "DOGEUSDT", "side": "BUY", "pnl": -0.70, "pnl_pct": -0.98, "strategy": "mtf_alignment"},
    {"symbol": "POLUSDT", "side": "SELL", "pnl": -0.73, "pnl_pct": -0.96, "strategy": "candle_momentum"},
    {"symbol": "LTCUSDT", "side": "SELL", "pnl": -0.27, "pnl_pct": -0.87, "strategy": "candle_momentum"},
    {"symbol": "FETUSDT", "side": "SELL", "pnl": -0.94, "pnl_pct": -1.12, "strategy": "fibonacci_bounce"},
    {"symbol": "FETUSDT", "side": "SELL", "pnl": -0.43, "pnl_pct": -1.12, "strategy": "fibonacci_bounce"},
    {"symbol": "FETUSDT", "side": "SELL", "pnl": -0.76, "pnl_pct": -1.12, "strategy": "fibonacci_bounce"},
    {"symbol": "APTUSDT", "side": "SELL", "pnl": -0.96, "pnl_pct": -1.42, "strategy": "candle_momentum"},
    {"symbol": "ARBUSDT", "side": "SELL", "pnl": -0.80, "pnl_pct": -1.12, "strategy": "candle_momentum"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": 0.06, "pnl_pct": 0.14, "strategy": "candlestick_reversal"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": 0.05, "pnl_pct": 0.14, "strategy": "candlestick_reversal"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": 0.22, "pnl_pct": 0.28, "strategy": "bollinger_mean_reversion"},
    {"symbol": "DOTUSDT", "side": "SELL", "pnl": -0.29, "pnl_pct": -0.98, "strategy": "break_and_retest"},
    {"symbol": "DOTUSDT", "side": "SELL", "pnl": -0.53, "pnl_pct": -0.98, "strategy": "break_and_retest"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": 0.16, "pnl_pct": 0.19, "strategy": "bollinger_mean_reversion"},
    {"symbol": "AVAXUSDT", "side": "SELL", "pnl": -0.36, "pnl_pct": -0.68, "strategy": "candlestick_reversal"},
    {"symbol": "AVAXUSDT", "side": "SELL", "pnl": -0.49, "pnl_pct": -0.71, "strategy": "shooting_star_reversal"},
    {"symbol": "DOGEUSDT", "side": "SELL", "pnl": 0.25, "pnl_pct": 0.41, "strategy": "support_breakdown"},
    {"symbol": "UNIUSDT", "side": "BUY", "pnl": -0.24, "pnl_pct": -0.40, "strategy": "break_and_retest"},
    {"symbol": "RENDERUSDT", "side": "BUY", "pnl": -0.61, "pnl_pct": -0.97, "strategy": "break_and_retest"},
    {"symbol": "INJUSDT", "side": "BUY", "pnl": -0.43, "pnl_pct": -0.78, "strategy": "trend_following"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": -0.05, "pnl_pct": -0.09, "strategy": "mtf_alignment"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": -0.03, "pnl_pct": -0.09, "strategy": "mtf_alignment"},
    {"symbol": "BTCUSDT", "side": "BUY", "pnl": 0.14, "pnl_pct": 0.17, "strategy": "bollinger_mean_reversion"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": -0.16, "pnl_pct": -0.19, "strategy": "bollinger_mean_reversion"},
    {"symbol": "NEARUSDT", "side": "BUY", "pnl": -1.11, "pnl_pct": -1.91, "strategy": "resistance_breakout"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.07, "pnl_pct": -0.15, "strategy": "resistance_breakout"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.11, "pnl_pct": -0.15, "strategy": "resistance_breakout"},
    {"symbol": "PEPEUSDT", "side": "BUY", "pnl": -0.06, "pnl_pct": -0.10, "strategy": "mtf_alignment"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": 0.01, "pnl_pct": 0.01, "strategy": "hammer_reversal"},
    {"symbol": "AVAXUSDT", "side": "SELL", "pnl": -0.51, "pnl_pct": -1.20, "strategy": "candle_momentum"},
    {"symbol": "DOTUSDT", "side": "SELL", "pnl": -0.36, "pnl_pct": -0.99, "strategy": "candle_momentum"},
    {"symbol": "ETHUSDT", "side": "BUY", "pnl": 0.23, "pnl_pct": 0.28, "strategy": "bollinger_mean_reversion"},
    {"symbol": "LTCUSDT", "side": "SELL", "pnl": -0.34, "pnl_pct": -0.43, "strategy": "shooting_star_reversal"},
    {"symbol": "LTCUSDT", "side": "SELL", "pnl": -0.13, "pnl_pct": -0.32, "strategy": "shooting_star_reversal"},
    {"symbol": "LTCUSDT", "side": "SELL", "pnl": -0.26, "pnl_pct": -0.32, "strategy": "shooting_star_reversal"},
    {"symbol": "DOGEUSDT", "side": "SELL", "pnl": -0.65, "pnl_pct": -1.24, "strategy": "support_breakdown"},
    {"symbol": "DOGEUSDT", "side": "SELL", "pnl": -0.98, "pnl_pct": -1.24, "strategy": "support_breakdown"},
    {"symbol": "TONUSDT", "side": "BUY", "pnl": -0.17, "pnl_pct": -0.49, "strategy": "mtf_alignment"},
]


def main():
    coord = get_coordinator()
    tuner = get_strategy_tuner()

    print(f"Before: phase={coord.current_phase.name}, trades={coord.total_trades}")

    for t in TRADES:
        won = t["pnl"] > 0
        rr = abs(t["pnl_pct"]) / 100.0

        patterns = []
        strat = t["strategy"]
        if "shooting_star" in strat: patterns = ["shooting_star"]
        elif "hammer" in strat: patterns = ["hammer"]
        elif "engulfing" in strat: patterns = ["engulfing"]
        elif "candlestick" in strat: patterns = ["candlestick_pattern"]
        elif "bollinger" in strat: patterns = ["bollinger_bounce"]

        regime = "unknown"
        if t["side"] == "SELL" and not won: regime = "trending_up"
        elif t["side"] == "BUY" and not won: regime = "trending_down"
        elif t["side"] == "BUY" and won: regime = "trending_up"
        elif t["side"] == "SELL" and won: regime = "trending_down"

        coord.on_trade_close(
            strategy=strat, won=won, pnl=t["pnl"], rr=rr,
            confidence=0.6, patterns=patterns, regime=regime,
            disciplined_exit=True, regime_aligned=won,
            force_all=True,  # bypass phase gating for retroactive training
        )

    tuner.tune_all()

    progress = coord.get_training_progress()
    print(f"After: phase={coord.current_phase.name}, trades={coord.total_trades}")
    print(f"Win rate: {progress['win_rate']:.1%}")
    print(f"Next phase: {progress['next_phase']} at {progress['next_phase_at_trades']} trades")
    print(f"Active: {', '.join(progress['active_subsystems'])}")
    print()

    # Pattern scores
    ps = get_pattern_store().get_all_stats()
    total_p = ps.get("total_patterns", 0)
    print(f"=== PATTERNS LEARNED ({total_p} combos) ===")
    for regime, patterns_list in ps.get("by_regime", {}).items():
        for p in patterns_list[:8]:
            tag = "WIN" if p["win_rate"] > 0.5 else "AVOID"
            print(f"  {p['pattern']:20s} + {regime:15s} | WR={p['win_rate']:.0%} n={p['total_trades']} [{tag}]")

    print()
    print("=== STRATEGY THRESHOLDS (auto-tuned) ===")
    ts = tuner.get_all_stats()
    for name, s in ts.get("strategies", {}).items():
        arrow = "^" if s["min_confidence"] > 0.60 else "v" if s["min_confidence"] < 0.60 else "="
        print(f"  {name:30s} | conf>={s['min_confidence']:.2f}{arrow} rr>={s['min_rr']:.2f} tuned={s['tune_count']}x")

    print()
    print("=== STRATEGY PROBABILITIES (RL) ===")
    rs = get_reward_engine().get_stats()
    for name, s in sorted(rs.get("strategy_probabilities", {}).items(), key=lambda x: x[1]["probability"], reverse=True):
        bar = "#" * int(s["probability"] * 40)
        print(f"  {name:30s} | {s['probability']:.1%} {bar}")

    # Save
    _ensure_dir()
    _save_learning_systems()
    print("\nState saved to data/paper_state/learning/")


if __name__ == "__main__":
    main()
