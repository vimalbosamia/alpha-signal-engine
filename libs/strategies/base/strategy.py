"""
Abstract base class for all signal strategies.

Design rules:
  - No execution logic ever lives here or in any subclass.
  - generate_candidate must never raise — return None on any data issue.
  - is_eligible is a pre-flight guard; call it first in generate_candidate.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from libs.core.models.domain import (
    AssetClass,
    DataQualityReport,
    DataQualityStatus,
    SessionState,
    SignalCandidate,
)
from libs.analysis.indicators.engine import IndicatorSnapshot
from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext


class BaseStrategy(ABC):
    """All strategies inherit from this.  No execution logic ever lives here."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_asset_classes(self) -> list[AssetClass]: ...

    @property
    @abstractmethod
    def min_bars_required(self) -> int: ...

    @abstractmethod
    def generate_candidate(
        self,
        symbol: str,
        asset_class: AssetClass,
        df: pd.DataFrame,              # primary timeframe OHLCV (enriched)
        df_htf: pd.DataFrame | None,   # higher timeframe OHLCV (may be None)
        session: SessionState | None,
        quality: DataQualityReport | None,
        structure: MarketStructure | None,
        levels: list[KeyLevel] | None,
        volume: VolumeContext | None,
        regime: RegimeAnalysis | None,
        indicators: IndicatorSnapshot | None = None,
    ) -> SignalCandidate | None:
        """
        Analyse context and return a SignalCandidate or None (no setup found).
        Must never raise — return None on any data issue.
        Must never execute trades.
        """
        ...

    def is_eligible(
        self,
        asset_class: AssetClass,
        session: SessionState | None,
        quality: DataQualityReport | None,
    ) -> bool:
        """Pre-flight: check asset class + session + data quality."""
        if asset_class not in self.supported_asset_classes:
            return False
        if quality is not None and quality.status == DataQualityStatus.BLOCKED:
            return False
        if session is not None and not session.is_tradable:
            return False
        return True

    def _atr_stop(self, df: pd.DataFrame, multiplier: float = 2.0) -> float:
        """Rough ATR-based stop distance in price units (uses last 14 bars)."""
        if df.empty or len(df) < 2:
            return 0.0
        tr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        return float(tr) * multiplier if not pd.isna(tr) else 0.0
