"""
Core domain models shared across all libs.

These are the canonical data types for the entire system.
Every component works with these types — no ad-hoc dicts.

Design rules:
  - All models are immutable (frozen=True) unless mutation is needed
  - All models use Pydantic v2 for validation and serialisation
  - Enums for every categorical field
  - No Optional[str] when "" suffices

Phase 1 additions (institutional upgrade):
  - TradingMode, SignalType, Direction, RiskLevel, SignalQuality,
    SetupGrade, TradeDecision, ValidationStatus enums
  - BiasScores, EntryZone, StrategyResult, IndicatorResult,
    RiskResult, FuturesRiskResult models
  - Candle.source, PatternResult.category/reliability/candle_index
  - SignalOutput extended with new optional fields (backward compat)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"


# ── Phase 1: New enumerations ─────────────────────────────────────────────────

class TradingMode(str, Enum):
    """Whether this signal is for a spot or leveraged-futures position."""
    SPOT = "spot"
    FUTURES = "futures"


class SignalType(str, Enum):
    """
    Granular signal classification replacing the legacy BUY/SELL.

    SPOT signals:
      SPOT_BUY       — enter or add to a spot long position
      SPOT_SELL_EXIT — exit or reduce an existing spot position
      SPOT_NO_TRADE  — no actionable spot setup

    FUTURES signals:
      FUTURES_LONG   — enter a leveraged long (perpetual futures)
      FUTURES_SHORT  — enter a leveraged short (perpetual futures)
      FUTURES_NO_TRADE — no actionable futures setup
    """
    SPOT_BUY = "SPOT_BUY"
    SPOT_SELL_EXIT = "SPOT_SELL_EXIT"
    SPOT_NO_TRADE = "SPOT_NO_TRADE"
    FUTURES_LONG = "FUTURES_LONG"
    FUTURES_SHORT = "FUTURES_SHORT"
    FUTURES_NO_TRADE = "FUTURES_NO_TRADE"


class Direction(str, Enum):
    """Normalised direction across spot and futures signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    NO_TRADE = "NO_TRADE"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class SignalQuality(str, Enum):
    """Operator-facing quality tier, distinct from numeric confidence."""
    INSTITUTIONAL = "institutional"   # A+ setup, all confirmations
    HIGH = "high"                     # A  setup
    MODERATE = "moderate"             # B  setup
    LOW = "low"                       # C  setup
    REJECT = "reject"                 # Avoid — do not act


class SetupGrade(str, Enum):
    """
    Graded quality of the trade setup.

    A+ — all confirmations, regime aligned, HTF aligned, strong volume
    A  — most confirmations, minor conflict only
    B  — tradable but requires manual review
    C  — weak, usually pass
    Avoid — should never be traded
    """
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    AVOID = "Avoid"


class TradeDecision(str, Enum):
    """
    Recommended operator action for this signal.

    TAKE  — execute if account risk allows (A+/A grade only)
    WAIT  — monitor; conditions not fully confirmed yet
    SKIP  — setup exists but too risky / low quality to act
    NO_TRADE — no setup; discard
    """
    TAKE = "TAKE"
    WAIT = "WAIT"
    SKIP = "SKIP"
    NO_TRADE = "NO_TRADE"


class ValidationStatus(str, Enum):
    """Whether this strategy has a proven walk-forward edge."""
    VALIDATED = "validated"          # passed OOS walk-forward
    UNVALIDATED = "unvalidated"      # no walk-forward test run yet
    INSUFFICIENT_DATA = "insufficient_data"  # < minimum sample threshold


# ── End Phase 1 new enumerations ──────────────────────────────────────────────

class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class Timeframe(str, Enum):
    ONE_MIN = "1m"
    THREE_MIN = "3m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class TrendDirection(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING_LOW_VOL = "ranging_low_vol"
    RANGING_HIGH_VOL = "ranging_high_vol"
    BREAKOUT = "breakout"
    CLIMACTIC = "climactic"
    UNKNOWN = "unknown"


class SessionType(str, Enum):
    # Stocks
    REGULAR = "regular"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    CLOSED = "closed"
    HOLIDAY = "holiday"
    HALF_DAY = "half_day"
    # Crypto
    CONTINUOUS = "continuous"         # crypto is always open
    EXCHANGE_MAINTENANCE = "exchange_maintenance"


class DataQualityStatus(str, Enum):
    CLEAN = "clean"
    WARNING = "warning"
    BLOCKED = "blocked"              # signals must not run on BLOCKED data


class PatternBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ── Core candle model ─────────────────────────────────────────────────────────

class Candle(BaseModel):
    """
    A single OHLCV candle.  Immutable after creation.
    timestamp is the bar's OPEN time in UTC.
    """
    model_config = {"frozen": True}

    symbol: str
    asset_class: AssetClass
    timeframe: Timeframe
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None
    is_confirmed: bool = True        # False = candle still building
    source: str = ""                 # provider identifier, e.g. "binance", "alpaca"

    @field_validator("high")
    @classmethod
    def high_gte_open_close(cls, v: float, info: Any) -> float:
        data = info.data
        if "open" in data and v < data["open"]:
            raise ValueError(f"high {v} < open {data['open']}")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_open_close(cls, v: float, info: Any) -> float:
        data = info.data
        if "open" in data and v > data["open"]:
            raise ValueError(f"low {v} > open {data['open']}")
        if "high" in data and v > data["high"]:
            raise ValueError(f"low {v} > high {data['high']}")
        return v

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def total_range(self) -> float:
        return self.high - self.low

    @property
    def body_pct(self) -> float:
        return self.body_size / self.total_range if self.total_range > 0 else 0.0

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        return self.body_pct < 0.10


# ── Symbol metadata ───────────────────────────────────────────────────────────

class SymbolMetadata(BaseModel):
    """Asset-specific metadata used by session logic and cost model."""
    model_config = {"frozen": True}

    symbol: str
    asset_class: AssetClass
    base_currency: str = ""
    quote_currency: str = "USD"
    tick_size: float = 0.01          # minimum price increment
    lot_size: float = 1.0            # minimum order size
    is_tradable: bool = True
    exchange: str = ""
    sector: str = ""                 # stocks only
    market_cap_tier: str = ""        # large/mid/small cap for stocks


# ── Session state ─────────────────────────────────────────────────────────────

class SessionState(BaseModel):
    """Current session status for a symbol at a given timestamp."""
    model_config = {"frozen": True}

    symbol: str
    asset_class: AssetClass
    session_type: SessionType
    is_tradable: bool
    quality_score: float = Field(ge=0.0, le=1.0)  # 0=off, 1=prime
    minutes_to_close: float | None = None
    notes: str = ""


# ── Data quality report ───────────────────────────────────────────────────────

class DataQualityReport(BaseModel):
    """Result of running the data quality validator on a candle series."""

    symbol: str
    asset_class: AssetClass
    status: DataQualityStatus
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rows_checked: int = 0
    rows_with_issues: int = 0
    checked_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_safe(self) -> bool:
        return self.status != DataQualityStatus.BLOCKED


# ── Pattern result ────────────────────────────────────────────────────────────

class PatternResult(BaseModel):
    """Output of a single candle pattern detector."""
    model_config = {"frozen": True}

    pattern_name: str
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    bias: PatternBias = PatternBias.NEUTRAL
    candle_span: int = 1
    details: dict[str, Any] = Field(default_factory=dict)

    # Phase 1 additions
    category: str = ""               # "single" | "two_candle" | "multi_candle"
    reliability: float = Field(ge=0.0, le=1.0, default=0.0)   # historical hit rate (0 = unknown)
    candle_index: int = -1           # index of last candle in pattern (-1 = not set)
    source_timestamp: datetime | None = None  # timestamp of the pattern's last candle

    @property
    def is_actionable(self) -> bool:
        return self.detected and self.confidence >= 0.60


# ── Signal candidate (pre-scoring) ────────────────────────────────────────────

class SignalCandidate(BaseModel):
    """
    Intermediate object produced by a strategy before final scoring.
    Contains everything the confluence engine needs.
    """
    candidate_id: UUID = Field(default_factory=uuid4)
    symbol: str
    asset_class: AssetClass
    strategy_name: str
    proposed_action: SignalAction
    timeframe: Timeframe
    higher_tf_bias: TrendDirection
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    stop_limit_price: float | None = None   # for stop-limit orders: trigger price ≠ limit price
    take_profit_1: float
    take_profit_2: float | None = None
    pattern_results: list[PatternResult] = Field(default_factory=list)
    regime: MarketRegime = MarketRegime.UNKNOWN
    session: SessionState | None = None
    quality: DataQualityReport | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    raw_features: dict[str, Any] = Field(default_factory=dict)


# ── Confluence breakdown ──────────────────────────────────────────────────────

class ConfluenceBreakdown(BaseModel):
    """Transparent scoring breakdown — one field per factor."""
    model_config = {"frozen": True}

    pattern_score: float = Field(ge=0.0, le=1.0)
    structure_score: float = Field(ge=0.0, le=1.0)
    level_score: float = Field(ge=0.0, le=1.0)
    volume_score: float = Field(ge=0.0, le=1.0)
    regime_score: float = Field(ge=0.0, le=1.0)
    session_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    weighted_total: float = Field(ge=0.0, le=1.0)
    weights: dict[str, float] = Field(default_factory=dict)

    # Explanation tree
    factor_notes: dict[str, str] = Field(default_factory=dict)
    blocked_reasons: list[str] = Field(default_factory=list)
    warning_tags: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        from libs.core.config import get_settings
        return (
            self.weighted_total >= get_settings().signal.min_confluence_score
            and not self.blocked_reasons
        )


# ── Phase 1: New institutional models ────────────────────────────────────────

class BiasScores(BaseModel):
    """
    Composite directional scores produced by BullBearBiasEngine.
    All four scores sum to approximately 1.0 but are not strictly normalised —
    conflictScore can push the total above 1.0 when signals disagree.
    """
    model_config = {"frozen": True}

    bullish_score: float = Field(ge=0.0, le=1.0, default=0.0)
    bearish_score: float = Field(ge=0.0, le=1.0, default=0.0)
    neutral_score: float = Field(ge=0.0, le=1.0, default=0.0)
    conflict_score: float = Field(ge=0.0, le=1.0, default=0.0)   # high = mixed signals
    upside_probability: float = Field(ge=0.0, le=1.0, default=0.0)
    downside_probability: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def dominant_bias(self) -> str:
        """Return 'bullish', 'bearish', 'neutral', or 'conflict'."""
        scores = {
            "bullish": self.bullish_score,
            "bearish": self.bearish_score,
            "neutral": self.neutral_score,
            "conflict": self.conflict_score,
        }
        return max(scores, key=lambda k: scores[k])


class EntryZone(BaseModel):
    """Price zone where entry is considered valid."""
    model_config = {"frozen": True}

    low: float
    high: float

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def width_pct(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 0.0
        return abs(self.high - self.low) / mid * 100.0


class StrategyResult(BaseModel):
    """
    Per-strategy analysis output attached to a signal for transparency.
    Used by the BullBearBiasEngine and signal grading layer.
    """
    model_config = {"frozen": True}

    strategy_name: str
    produced_candidate: bool         # True if strategy found a valid setup
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    bullish_score: float = Field(ge=0.0, le=1.0, default=0.0)
    bearish_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasons: list[str] = Field(default_factory=list)
    failed_confirmations: list[str] = Field(default_factory=list)
    required_confirmations: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED


class IndicatorResult(BaseModel):
    """Snapshot of a single indicator's current state and bias."""
    model_config = {"frozen": True}

    name: str
    value: float
    bias: PatternBias = PatternBias.NEUTRAL
    strength: float = Field(ge=0.0, le=1.0, default=0.0)
    explanation: str = ""


class RiskResult(BaseModel):
    """
    Full risk profile for a signal setup.
    Superset of what RiskEngine currently returns — adds invalidation
    level and operator-facing cost warnings.
    """
    model_config = {"frozen": True}

    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None = None
    risk_reward_ratio: float = 0.0
    invalidation_level: float | None = None   # price that definitively invalidates setup
    max_loss_percent: float = 0.0             # (entry - stop_loss) / entry * 100
    volatility_warning: str = ""
    liquidity_warning: str = ""
    spread_warning: str = ""
    slippage_warning: str = ""

    @property
    def has_warnings(self) -> bool:
        return bool(
            self.volatility_warning or self.liquidity_warning
            or self.spread_warning or self.slippage_warning
        )


class FuturesRiskResult(BaseModel):
    """
    Futures-specific risk fields.
    Required on every FUTURES_LONG or FUTURES_SHORT signal.
    """
    model_config = {"frozen": True}

    leverage: float = Field(ge=1.0, default=1.0)
    margin_type: str = "isolated"                   # "isolated" | "cross"
    estimated_liquidation_price: float = 0.0
    liquidation_buffer_percent: float = 0.0        # (liq - entry) / entry * 100
    liquidation_risk: RiskLevel = RiskLevel.LOW
    funding_fee_risk: RiskLevel = RiskLevel.LOW
    max_loss_before_stop: float = 0.0              # $ loss at stop level after fees


# ── End Phase 1 new institutional models ─────────────────────────────────────

# ── Final signal output ───────────────────────────────────────────────────────

class SignalOutput(BaseModel):
    """
    The final signal output of the agent.

    This is what gets written to the audit log, sent to dashboards,
    and consumed by the human operator.

    The agent NEVER places trades — this is purely informational.
    """
    signal_id: UUID = Field(default_factory=uuid4)

    # Core identification
    symbol: str
    asset_class: AssetClass
    strategy_name: str
    timeframe: Timeframe

    # The decision
    action: SignalAction
    confidence: float = Field(ge=0.0, le=1.0)

    # Trade setup details
    higher_tf_bias: TrendDirection
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    stop_limit_price: float | None = None   # stop-limit order: limit leg price
    take_profit_1: float
    take_profit_2: float | None = None
    estimated_risk_reward: float

    # Context snapshot
    market_regime: MarketRegime
    session_status: SessionType
    data_quality_status: DataQualityStatus

    # Transparency
    confluence: ConfluenceBreakdown
    patterns_detected: list[str] = Field(default_factory=list)
    explanation: str = ""
    warnings: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    agent_mode: str = ""
    data_provider: str = ""

    # ── Phase 1: institutional output contract (all optional, backward compat) ─
    trading_mode: TradingMode = TradingMode.SPOT
    signal_type: SignalType | None = None        # None = legacy BUY/SELL not yet migrated
    direction: Direction = Direction.NO_TRADE
    bias_scores: BiasScores | None = None
    risk_level: RiskLevel = RiskLevel.MODERATE
    signal_quality: SignalQuality = SignalQuality.MODERATE
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    setup_grade: SetupGrade | None = None
    trade_decision: TradeDecision = TradeDecision.NO_TRADE
    invalidation_level: float | None = None
    failed_confirmations: list[str] = Field(default_factory=list)
    risk_result: RiskResult | None = None
    futures_risk: FuturesRiskResult | None = None
    strategy_results: list[StrategyResult] = Field(default_factory=list)
    indicator_results: list[IndicatorResult] = Field(default_factory=list)
    # ── End Phase 1 additions ──────────────────────────────────────────────────

    def to_display(self) -> str:
        """Human-readable one-liner for terminal output."""
        emoji = {"BUY": "▲", "SELL": "▼", "NO_TRADE": "─"}[self.action.value]
        return (
            f"{emoji} {self.action.value:8s} {self.symbol:10s} "
            f"[{self.asset_class.value:6s}] "
            f"| Score: {self.confidence:.2f} "
            f"| R:R {self.estimated_risk_reward:.1f}x "
            f"| Pattern: {', '.join(self.patterns_detected[:2]) or 'none'} "
            f"| Regime: {self.market_regime.value}"
        )
