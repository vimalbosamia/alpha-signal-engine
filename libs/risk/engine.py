"""
Risk engine: evaluates a SignalCandidate and returns a RiskAssessment.

This module is purely analytical — no trade execution ever occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass

from libs.core.config.settings import RiskSettings
from libs.core.models.domain import AssetClass, MarketRegime, SignalAction, SignalCandidate


# ── Frozen dataclasses ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostModel:
    """Per-asset transaction cost estimates (all values in %)."""

    slippage_pct: float
    spread_pct: float
    commission_pct: float

    @property
    def total_friction_pct(self) -> float:
        return self.slippage_pct + self.spread_pct + self.commission_pct


@dataclass(frozen=True)
class RiskAssessment:
    """Outcome of running RiskEngine.assess() on a SignalCandidate."""

    passed: bool
    reward_risk: float
    stop_distance_pct: float
    risk_per_trade_pct: float
    blocked_reasons: list[str]
    warnings: list[str]
    score: float


# ── Engine ────────────────────────────────────────────────────────────────────

class RiskEngine:
    """
    Stateless risk assessor.  Instantiate once; call assess() per candidate.

    All constants are class-level so callers can inspect and override them
    in tests without monkey-patching the module.
    """

    # Stop-distance thresholds (%)
    MIN_STOP_DISTANCE_PCT: float = 0.10
    MAX_STOP_DISTANCE_PCT: float = 15.0
    MIN_CRYPTO_STOP_PCT: float = 0.10
    MAX_CRYPTO_STOP_PCT: float = 25.0

    # R:R minimum (mirrors SignalSettings default; kept here so RiskEngine
    # is self-contained and does not depend on SignalSettings)
    DEFAULT_MIN_REWARD_RISK: float = 2.0

    # Score penalties
    HIGH_VOL_REGIME_PENALTY: float = 0.15

    # Stock cost model (%)
    COST_STOCK_SLIPPAGE: float = 0.05
    COST_STOCK_SPREAD: float = 0.02
    COST_STOCK_COMMISSION: float = 0.00   # modern zero-commission brokers

    # Crypto cost model (%)
    COST_CRYPTO_SLIPPAGE: float = 0.10
    COST_CRYPTO_SPREAD: float = 0.05
    COST_CRYPTO_COMMISSION: float = 0.10  # taker fee ~0.1 %

    def __init__(self, settings: RiskSettings | None = None) -> None:
        self._settings = settings or RiskSettings()

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(self, candidate: SignalCandidate) -> RiskAssessment:
        """Return a RiskAssessment for *candidate*. Never raises."""
        blocked: list[str] = []
        warnings: list[str] = []

        entry = self._entry_mid(candidate)
        rr = self._calc_reward_risk(candidate, entry)
        stop_pct = self._calc_stop_distance_pct(candidate, entry)

        # NO_TRADE is always blocked — there is no valid setup to assess.
        if candidate.proposed_action is SignalAction.NO_TRADE:
            blocked.append("proposed_action is NO_TRADE — no valid setup")

        # Stop-distance checks
        blocked.extend(
            self._check_stop_distance(stop_pct, candidate.asset_class)
        )

        # R:R check (skip for NO_TRADE — already blocked above)
        if candidate.proposed_action is not SignalAction.NO_TRADE:
            blocked.extend(
                self._check_reward_risk(rr, self.DEFAULT_MIN_REWARD_RISK)
            )

        # Edge-after-costs (warning only)
        cost = self._cost_model(candidate.asset_class)
        if not self._edge_after_costs(rr, cost, stop_pct):
            warnings.append(
                "Effective R:R after transaction costs is below 1.5 — "
                "edge may be insufficient."
            )

        score = self._calc_score(
            rr=rr,
            stop_pct=stop_pct,
            asset_class=candidate.asset_class,
            regime=candidate.regime,
            blocked=bool(blocked),
        )

        return RiskAssessment(
            passed=len(blocked) == 0,
            reward_risk=rr,
            stop_distance_pct=stop_pct,
            risk_per_trade_pct=self._settings.max_risk_per_signal_pct,
            blocked_reasons=blocked,
            warnings=warnings,
            score=score,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _entry_mid(self, candidate: SignalCandidate) -> float:
        return (candidate.entry_zone_low + candidate.entry_zone_high) / 2.0

    def _calc_reward_risk(self, candidate: SignalCandidate, entry: float) -> float:
        action = candidate.proposed_action
        tp1 = candidate.take_profit_1
        sl = candidate.stop_loss

        if action is SignalAction.BUY:
            return (tp1 - entry) / max(entry - sl, 1e-9)
        if action is SignalAction.SELL:
            return (entry - tp1) / max(sl - entry, 1e-9)
        # NO_TRADE
        return 0.0

    def _calc_stop_distance_pct(
        self, candidate: SignalCandidate, entry: float
    ) -> float:
        if entry == 0.0:
            return 0.0
        return abs(entry - candidate.stop_loss) / abs(entry) * 100.0

    def _check_stop_distance(
        self, stop_pct: float, asset_class: AssetClass
    ) -> list[str]:
        reasons: list[str] = []
        if asset_class is AssetClass.CRYPTO:
            lo, hi = self.MIN_CRYPTO_STOP_PCT, self.MAX_CRYPTO_STOP_PCT
        else:
            lo, hi = self.MIN_STOP_DISTANCE_PCT, self.MAX_STOP_DISTANCE_PCT

        if stop_pct < lo:
            reasons.append(
                f"Stop distance {stop_pct:.4f}% is below minimum "
                f"{lo}% for {asset_class.value}."
            )
        elif stop_pct > hi:
            reasons.append(
                f"Stop distance {stop_pct:.4f}% exceeds maximum "
                f"{hi}% for {asset_class.value}."
            )
        return reasons

    def _check_reward_risk(self, rr: float, min_rr: float) -> list[str]:
        if rr < min_rr:
            return [
                f"Reward:Risk ratio {rr:.2f} is below minimum {min_rr:.2f}."
            ]
        return []

    def _cost_model(self, asset_class: AssetClass) -> CostModel:
        if asset_class is AssetClass.CRYPTO:
            return CostModel(
                slippage_pct=self.COST_CRYPTO_SLIPPAGE,
                spread_pct=self.COST_CRYPTO_SPREAD,
                commission_pct=self.COST_CRYPTO_COMMISSION,
            )
        return CostModel(
            slippage_pct=self.COST_STOCK_SLIPPAGE,
            spread_pct=self.COST_STOCK_SPREAD,
            commission_pct=self.COST_STOCK_COMMISSION,
        )

    def _edge_after_costs(
        self, rr: float, cost: CostModel, stop_pct: float
    ) -> bool:
        """
        Approximate the effective R:R after subtracting friction from both
        the profit leg and the risk leg.

        friction_risk_units = total_friction_pct / stop_pct
        effective_rr        = rr - 2 * friction_risk_units
        (costs eat into reward AND inflate risk — hence multiply by 2)
        """
        if stop_pct <= 0:
            return False
        friction_in_r = cost.total_friction_pct / stop_pct
        effective_rr = rr - 2.0 * friction_in_r
        return effective_rr >= 1.5

    def _calc_score(
        self,
        rr: float,
        stop_pct: float,
        asset_class: AssetClass,
        regime: MarketRegime,
        blocked: bool,
    ) -> float:
        if blocked:
            return 0.0

        score = 0.6

        # R:R bonus
        if rr >= 3.0:
            score += 0.20
        elif rr >= 2.0:
            score += 0.10

        # Tight stop bonus
        if stop_pct <= 1.0:
            score += 0.10

        # High-volatility / climactic regime penalty
        if regime in (MarketRegime.RANGING_HIGH_VOL, MarketRegime.CLIMACTIC):
            score -= self.HIGH_VOL_REGIME_PENALTY

        # Crypto friction penalty
        if asset_class is AssetClass.CRYPTO:
            score -= 0.10

        return max(0.0, min(1.0, score))
