"""
Unit tests for FeatureCorrelationReducer.

Validates audit counts, weight logic, and inflation risk classification.
No external I/O.
"""
from __future__ import annotations

import pytest

from libs.analysis.filters.correlation_reducer import (
    CorrelationAudit,
    CorrelationGroup,
    FeatureCorrelationReducer,
)

REDUCER = FeatureCorrelationReducer()


class TestAuditCounts:
    def test_audit_counts(self):
        """Audit totals must match the GROUPS definition."""
        audit = REDUCER.audit()

        # Known counts from GROUPS definition:
        # momentum: 5 indicators, 3 redundant
        # trend:    6 indicators, 2 redundant
        # volatility: 3 indicators, 0 redundant
        # volume:   2 indicators, 0 redundant
        assert audit.total_indicators == 16
        assert audit.removed_indicators == 5
        assert audit.kept_indicators == 11
        assert audit.kept_indicators == audit.total_indicators - audit.removed_indicators

    def test_audit_returns_all_groups(self):
        """Audit must include all 4 groups."""
        audit = REDUCER.audit()
        group_names = {g.name for g in audit.groups}
        assert group_names == {"momentum", "trend", "volatility", "volume"}

    def test_audit_explanation_mentions_redundant_count(self):
        """Explanation string should reference the redundant indicator count."""
        audit = REDUCER.audit()
        assert str(audit.removed_indicators) in audit.explanation

    def test_audit_is_immutable(self):
        """CorrelationAudit is a frozen dataclass."""
        audit = REDUCER.audit()
        with pytest.raises((AttributeError, TypeError)):
            audit.inflation_risk = "low"  # type: ignore[misc]


class TestWeightLogic:
    def test_representative_full_weight(self):
        """Representative indicators receive weight 1.0."""
        representatives = ["rsi", "ema_20", "atr", "relative_volume"]
        for ind in representatives:
            weight = REDUCER.should_discount(ind)
            assert weight == 1.0, f"{ind!r} representative should have weight 1.0, got {weight}"

    def test_redundant_discounted(self):
        """Redundant indicators receive weight 0.3."""
        redundant = ["stoch_rsi_k", "stoch_rsi_d", "roc", "sma_20", "sma_50"]
        for ind in redundant:
            weight = REDUCER.should_discount(ind)
            assert weight == 0.3, f"{ind!r} redundant should have weight 0.3, got {weight}"

    def test_unknown_full_weight(self):
        """Indicators not in any group default to full weight 1.0."""
        unknowns = ["custom_oscillator", "proprietary_signal", "obv"]
        for ind in unknowns:
            weight = REDUCER.should_discount(ind)
            assert weight == 1.0, f"Unknown {ind!r} should default to 1.0, got {weight}"


class TestInflationRisk:
    def test_inflation_risk_high(self):
        """5 redundant indicators → inflation_risk == 'high' (> 3 threshold)."""
        audit = REDUCER.audit()
        assert audit.removed_indicators == 5
        assert audit.inflation_risk == "high"

    def test_inflation_risk_moderate(self):
        """2 redundant indicators → 'moderate'."""
        reducer = FeatureCorrelationReducer()
        # Temporarily patch GROUPS to have exactly 2 redundant
        from libs.analysis.filters.correlation_reducer import CorrelationGroup as CG
        reducer.GROUPS = [
            CG("test", ["a", "b", "c"], "a", ["b", "c"]),
        ]
        audit = reducer.audit()
        assert audit.removed_indicators == 2
        assert audit.inflation_risk == "moderate"

    def test_inflation_risk_low(self):
        """1 redundant indicator → 'low'."""
        reducer = FeatureCorrelationReducer()
        from libs.analysis.filters.correlation_reducer import CorrelationGroup as CG
        reducer.GROUPS = [
            CG("test", ["a", "b"], "a", ["b"]),
        ]
        audit = reducer.audit()
        assert audit.removed_indicators == 1
        assert audit.inflation_risk == "low"
