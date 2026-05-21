"""
ConfluenceV2Engine — Institutional-grade 5-layer hierarchical confluence engine.

Replaces single-strategy-per-trade with multi-strategy probabilistic scoring.

Flow:
  1. Receive ALL strategy candidates for one symbol
  2. Group by direction (BUY / SELL)
  3. For each direction group:
     a. Layer 1: Score regime
     b. Layer 2: Score directional bias
     c. Layer 3: Select best triggers (max 1 per category)
     d. Layer 4: Stack confirmations (strategy + indicator based)
     e. Layer 5: Check suppressions
  4. Compute final score with diversity bonus
  5. Return ConfluenceTradeResult with full provenance

Integration: called from SignalPipeline AFTER all strategies have run.
Does NOT replace the existing ConfluenceEngine — works alongside it.
"""
from __future__ import annotations

from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.confluence.layer_bias import score_bias
from libs.confluence.layer_confirmation import score_confirmations
from libs.confluence.layer_regime import get_regime_name, score_regime
from libs.confluence.layer_suppression import score_suppressions
from libs.confluence.layer_trigger import score_triggers
from libs.confluence.scorer import compute_final_score, quality_grade
from libs.confluence.strategy_registry import get_category
from libs.confluence.trade_structure import ConfluenceTradeResult
from libs.core.models.domain import SignalCandidate


class ConfluenceV2Engine:
    """5-layer hierarchical confluence engine.

    Stateless — safe for concurrent use. All context passed per call.
    """

    def evaluate(
        self,
        symbol: str,
        candidates: list[SignalCandidate],
        regime: RegimeAnalysis | None = None,
        structure: MarketStructure | None = None,
        bias_data: dict | None = None,
        indicators: dict | None = None,
    ) -> list[ConfluenceTradeResult]:
        """Evaluate all candidates for a symbol through the 5-layer hierarchy.

        Args:
            symbol: Trading symbol.
            candidates: ALL strategy candidates (triggers + confirmations).
            regime: Regime engine output.
            structure: Market structure analysis.
            bias_data: Bull/bear bias engine output dict.
            indicators: Computed indicator values dict.

        Returns:
            List of ConfluenceTradeResult (one per viable direction).
            May be empty if no triggers fire or all directions blocked.
        """
        results: list[ConfluenceTradeResult] = []

        # Group candidates by direction
        buy_candidates = [
            c for c in candidates if c.proposed_action.value == "BUY"
        ]
        sell_candidates = [
            c for c in candidates if c.proposed_action.value == "SELL"
        ]

        for direction, group in [("BUY", buy_candidates), ("SELL", sell_candidates)]:
            if not group:
                continue

            result = self._evaluate_direction(
                symbol=symbol,
                direction=direction,
                candidates=group,
                all_candidates=candidates,
                regime=regime,
                structure=structure,
                bias_data=bias_data,
                indicators=indicators,
            )
            if result is not None:
                results.append(result)

        return results

    def _evaluate_direction(
        self,
        symbol: str,
        direction: str,
        candidates: list[SignalCandidate],
        all_candidates: list[SignalCandidate],
        regime: RegimeAnalysis | None,
        structure: MarketStructure | None,
        bias_data: dict | None,
        indicators: dict | None,
    ) -> ConfluenceTradeResult | None:
        """Evaluate one direction through all 5 layers."""
        notes: dict[str, str] = {}

        # ── Layer 1: Regime ──
        regime_score, regime_note, regime_blocked = score_regime(regime)
        notes["regime"] = regime_note
        if regime_blocked:
            return None

        # ── Layer 2: Bias ──
        bias_score, prob_long, prob_short, bias_note = score_bias(
            bias_data, structure, regime, direction,
        )
        notes["bias"] = bias_note

        # ── Layer 3: Trigger ──
        trigger_score, trigger_note, primary_trigger, trigger_categories = score_triggers(
            candidates, direction,
        )
        notes["trigger"] = trigger_note
        if primary_trigger is None:
            return None  # no trigger = no trade

        # ── Layer 4: Confirmation ──
        conf_score, supporting, confirmations, diversity_bonus = score_confirmations(
            all_candidates, direction, indicators,
        )
        notes["confirmation"] = (
            f"supporting={supporting}, indicators={confirmations}, "
            f"diversity_bonus={diversity_bonus}"
        )

        # ── Layer 5: Suppression ──
        supp_penalty, suppressions, supp_blocked = score_suppressions(
            indicators, regime, structure, direction, primary_trigger.strategy_name,
        )
        notes["suppression"] = f"signals={suppressions}, penalty={supp_penalty}"
        if supp_blocked:
            notes["suppression"] += " — HARD BLOCKED (3+ suppressions)"
            return None

        # ── Final scoring ──
        final_score = compute_final_score(
            regime_score=regime_score,
            bias_score=bias_score,
            trigger_score=trigger_score,
            confirmation_score=conf_score,
            suppression_penalty=supp_penalty,
            diversity_bonus=diversity_bonus,
        )

        # Count contributing layers
        layer_count = 1  # trigger always counts
        if regime_score > 5:
            layer_count += 1
        if bias_score > 8:
            layer_count += 1
        if conf_score > 0:
            layer_count += 1
        if suppressions:
            layer_count += 1  # suppression layer is "active" even if penalty

        # Collect all unique categories
        all_categories = set(trigger_categories)
        for s in supporting:
            all_categories.add(get_category(s).value)

        # Extract entry/exit from primary trigger
        entry = (primary_trigger.entry_zone_low + primary_trigger.entry_zone_high) / 2.0
        sl = primary_trigger.stop_loss
        tp1 = primary_trigger.take_profit_1
        tp2 = primary_trigger.take_profit_2 if hasattr(primary_trigger, "take_profit_2") else None

        rr = 0.0
        risk = abs(entry - sl) if sl else 0.0
        reward = abs(tp1 - entry) if tp1 else 0.0
        if risk > 0:
            rr = reward / risk

        return ConfluenceTradeResult(
            symbol=symbol,
            direction=direction,
            market_regime=get_regime_name(regime),
            regime_score=regime_score,
            bias_direction="bullish" if prob_long > 0.55 else "bearish" if prob_short > 0.55 else "neutral",
            probability_long=prob_long,
            probability_short=prob_short,
            bias_score=bias_score,
            primary_strategy=primary_trigger.strategy_name,
            trigger_confidence=trigger_score / 30.0,  # normalized from trigger layer
            trigger_score=trigger_score,
            supporting_strategies=tuple(supporting),
            confirmation_signals=tuple(confirmations),
            confirmation_score=conf_score,
            suppression_signals=tuple(suppressions),
            suppression_penalty=supp_penalty,
            diversity_bonus=diversity_bonus,
            confluence_score=final_score,
            layer_count=layer_count,
            categories_represented=tuple(sorted(all_categories)),
            trade_quality=quality_grade(final_score),
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward=rr,
            factor_notes=notes,
        )
