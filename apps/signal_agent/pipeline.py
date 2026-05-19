"""
SignalPipeline: full end-to-end signal generation for ONE symbol.

Steps:
  1. Fetch historical candles from provider
  2. Validate data quality
  3. Check session eligibility
  4. Build/enrich candles
  5. Run all analysis engines
  6. Run all eligible strategies
  7. Score each candidate through confluence engine
  8. Emit final SignalOutput for each candidate
  9. Record to audit log + storage
 10. Publish to event bus

This class NEVER places trades. It only produces SignalOutput objects.
"""
from __future__ import annotations
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from libs.core.models.domain import (
    AssetClass, SignalOutput, SignalAction, Timeframe
)
from libs.core.config.settings import get_settings
from libs.core.logging.logger import get_logger
from libs.data.providers.base import BaseDataProvider
from libs.data.candles.builder import CandleBuilder
from libs.data.quality.validator import DataQualityValidator
from libs.data.session.session_manager import get_session_manager
from libs.analysis.indicators.engine import IndicatorsEngine
from libs.analysis.structure.engine import MarketStructureEngine
from libs.analysis.levels.engine import KeyLevelsEngine
from libs.analysis.volume.engine import VolumeEngine
from libs.analysis.regime.engine import RegimeEngine
from libs.analysis.patterns.single_candle import (
    HammerDetector, InvertedHammerDetector, ShootingStarDetector,
    HangingManDetector, DojiDetector, DragonflyDojiDetector,
    GravestoneDojiDetector, SpinningTopDetector,
    BullishMarubozuDetector, BearishMarubozuDetector,
)
from libs.analysis.patterns.two_candle import (
    BullishEngulfingDetector, BearishEngulfingDetector,
    BullishHaramiDetector, BearishHaramiDetector,
    PiercingLineDetector, DarkCloudCoverDetector,
    TweezerTopDetector, TweezerBottomDetector,
)
from libs.analysis.patterns.multi_candle import (
    MorningStarDetector, EveningStarDetector,
    MorningDojiStarDetector, EveningDojiStarDetector,
    ThreeWhiteSoldiersDetector, ThreeBlackCrowsDetector,
    BullishAbandonedBabyDetector, BearishAbandonedBabyDetector,
    RisingThreeMethodsDetector, FallingThreeMethodsDetector,
)
from libs.analysis.patterns.context_candle import (
    InsideBarDetector, OutsideBarDetector, PinBarDetector,
    RejectionCandleDetector, BreakoutCandleDetector,
    ExhaustionCandleDetector, MomentumCandleDetector,
    LongWickCandleDetector, NarrowRangeCandleDetector,
    WideRangeCandleDetector, TrapCandleDetector,
    FailedBreakoutCandleDetector,
)
from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
from libs.analysis.structure.market_structure import MarketStructureAnalyzer
from libs.analysis.levels.supply_demand import SupplyDemandDetector
from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
from libs.signals.confluence.engine import ConfluenceEngine
from libs.signals.output.emitter import SignalEmitter
from libs.risk.engine import RiskEngine
from libs.strategies.base.strategy import BaseStrategy
from libs.monitoring.outcome_tracker import is_strategy_muted
from libs.monitoring.portfolio_guard import PortfolioGuard, GuardConfig, get_portfolio_guard
from libs.audit.audit_log import AuditLog, AuditEvent
from libs.monitoring.metrics import MetricsCollector
from libs.core.events.bus import EventBus
from libs.data.storage.repository import SignalRepository
from libs.data.storage.db import get_session_factory

log = get_logger(__name__)

# All 40 pattern detectors (single + two + multi + context candle)
DEFAULT_DETECTORS = [
    # Single candle (10)
    HammerDetector(), InvertedHammerDetector(), ShootingStarDetector(),
    HangingManDetector(), DojiDetector(), DragonflyDojiDetector(),
    GravestoneDojiDetector(), SpinningTopDetector(),
    BullishMarubozuDetector(), BearishMarubozuDetector(),
    # Two candle (8)
    BullishEngulfingDetector(), BearishEngulfingDetector(),
    BullishHaramiDetector(), BearishHaramiDetector(),
    PiercingLineDetector(), DarkCloudCoverDetector(),
    TweezerTopDetector(), TweezerBottomDetector(),
    # Multi candle (10)
    MorningStarDetector(), EveningStarDetector(),
    MorningDojiStarDetector(), EveningDojiStarDetector(),
    ThreeWhiteSoldiersDetector(), ThreeBlackCrowsDetector(),
    BullishAbandonedBabyDetector(), BearishAbandonedBabyDetector(),
    RisingThreeMethodsDetector(), FallingThreeMethodsDetector(),
    # Context candle (12)
    InsideBarDetector(), OutsideBarDetector(), PinBarDetector(),
    RejectionCandleDetector(), BreakoutCandleDetector(),
    ExhaustionCandleDetector(), MomentumCandleDetector(),
    LongWickCandleDetector(), NarrowRangeCandleDetector(),
    WideRangeCandleDetector(), TrapCandleDetector(),
    FailedBreakoutCandleDetector(),
]


class SignalPipeline:
    """
    Orchestrates the signal generation pipeline for a single symbol+timeframe.
    Instantiate once; call run_once() each bar.
    """

    def __init__(
        self,
        provider: BaseDataProvider,
        strategies: list[BaseStrategy],
        audit_log: AuditLog | None = None,
        metrics: MetricsCollector | None = None,
        event_bus: EventBus | None = None,
        portfolio_guard: PortfolioGuard | None = None,
        candle_lookback: int = 200,
        htf_multiplier: int = 4,   # e.g. 5m → 20m for HTF
    ) -> None:
        self._provider = provider
        self._strategies = strategies
        self._audit = audit_log or AuditLog(enabled=False)
        self._metrics = metrics or MetricsCollector()
        self._bus = event_bus or EventBus()
        self._guard = portfolio_guard or get_portfolio_guard()

        self._builder = CandleBuilder()
        self._validator = DataQualityValidator()
        self._structure = MarketStructureEngine()
        self._levels = KeyLevelsEngine()
        self._volume = VolumeEngine()
        self._regime = RegimeEngine()
        self._indicators = IndicatorsEngine()
        self._confluence = ConfluenceEngine()
        self._risk = RiskEngine()
        self._emitter = SignalEmitter(
            agent_mode=get_settings().agent_mode.value,
            data_provider=provider.name,
        )
        self._lookback = candle_lookback
        self._htf_multiplier = htf_multiplier

    async def run_once(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe: Timeframe,
    ) -> list[SignalOutput]:
        """
        Run the full pipeline for one symbol. Returns all signal outputs
        (may be empty if no setups found or data is unsafe).
        Never raises — all exceptions are caught and logged.
        """
        outputs: list[SignalOutput] = []
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=5)   # fetch ~5 days of history

        try:
            # ── 1. Fetch data ─────────────────────────────────────────────
            df = await self._provider.get_candles(symbol, timeframe, start, now)
            df = self._builder.enrich(df)

            # ── 1b. Fetch HTF data (best-effort; silently fall back to None) ──
            df_htf: pd.DataFrame | None = None
            htf_tf = self._htf_timeframe(timeframe)
            if htf_tf is not None:
                try:
                    df_htf = await self._provider.get_candles(symbol, htf_tf, start, now)
                    df_htf = self._builder.enrich(df_htf)
                except Exception:
                    df_htf = None  # HTF failure is non-fatal

            # ── 2. Session (needed before quality to gate staleness check) ──
            session_mgr = get_session_manager(asset_class)
            session = session_mgr.get_state(symbol, now)

            # ── 3. Data quality ───────────────────────────────────────────
            tf_seconds = self._timeframe_seconds(timeframe)
            # For stocks: staleness check only makes sense during market hours.
            # Outside regular trading hours the latest bar will naturally be
            # hours old — that is not a data error.
            is_live_check = session.is_tradable if asset_class.value == "stock" else True
            quality = self._validator.validate(
                df, symbol, asset_class,
                timeframe_seconds=tf_seconds,
                is_live=is_live_check,
            )

            if not quality.is_safe:
                self._metrics.record_data_quality_block(symbol, asset_class.value)
                await self._bus.publish(EventBus.DATA_QUALITY_BLOCKED, {
                    "symbol": symbol, "errors": quality.errors,
                })
                log.warning("pipeline_data_blocked", symbol=symbol, errors=quality.errors)
                return []

            # ── 4. Context engines ────────────────────────────────────────
            structure = self._structure.analyze(df)
            htf_structure = (
                self._structure.analyze(df_htf)
                if df_htf is not None and len(df_htf) >= 11
                else None
            )
            levels = self._levels.analyze(df, asset_class)
            regime = self._regime.analyze(df)
            indicators = self._indicators.compute(df)

            # ── 4b. Deep analysis engines ────────────────────────────────
            # Indicator bias (bullish/bearish/neutral per indicator)
            try:
                last_close = float(df["close"].iloc[-1])
                rel_vol = float(df["relative_volume"].iloc[-1]) if "relative_volume" in df else 1.0
                is_bull_candle = bool(df["is_bullish"].iloc[-1]) if "is_bullish" in df else True
                indicator_bias = IndicatorBiasAnalyzer().analyze_all(
                    rsi=indicators.rsi or 50, rsi_prev=indicators.rsi_prev or 50,
                    macd_line=indicators.macd_line or 0, macd_signal=indicators.macd_signal or 0,
                    histogram=indicators.macd_histogram or 0, histogram_prev=indicators.macd_histogram_prev or 0,
                    close=last_close, bb_upper=indicators.bb_upper or last_close + 1,
                    bb_lower=indicators.bb_lower or last_close - 1, bb_pct_b=indicators.bb_pct_b or 0.5,
                    ema_9=indicators.ema_9 or last_close, ema_20=indicators.ema_20 or last_close,
                    ema_50=indicators.ema_50 or last_close,
                    adx=indicators.adx or 0,
                    relative_volume=rel_vol, is_bullish_candle=is_bull_candle,
                )
            except Exception:
                indicator_bias = None

            # Market structure (HH/HL/LH/LL, BOS, CHoCH)
            try:
                deep_structure = MarketStructureAnalyzer().analyze(df)
            except Exception:
                deep_structure = None

            # Supply/demand zones
            try:
                sd_zones = SupplyDemandDetector().detect(df)
            except Exception:
                sd_zones = []

            # HTF bias string for bias engine
            htf_bias_str = "neutral"
            if htf_structure is not None:
                trend_val = getattr(htf_structure, "trend", None)
                if trend_val is not None:
                    tv = trend_val.value if hasattr(trend_val, "value") else str(trend_val)
                    if "up" in tv.lower():
                        htf_bias_str = "bullish"
                    elif "down" in tv.lower():
                        htf_bias_str = "bearish"

            # ── 5. Strategies ─────────────────────────────────────────────
            for strategy in self._strategies:
                if not strategy.is_eligible(asset_class, session, quality):
                    continue
                if len(df) < strategy.min_bars_required:
                    continue
                # Skip strategies auto-muted by poor win rate
                if is_strategy_muted(strategy.name):
                    log.debug("strategy_muted_skipped", strategy=strategy.name, symbol=symbol)
                    continue

                # Run pattern detectors
                pattern_results = [det.detect(df) for det in DEFAULT_DETECTORS]

                # Volume context per strategy (uses proposed_action from candidate)
                try:
                    candidate = strategy.generate_candidate(
                        symbol=symbol,
                        asset_class=asset_class,
                        df=df,
                        df_htf=df_htf,
                        session=session,
                        quality=quality,
                        structure=structure,
                        levels=levels,
                        volume=None,   # volume assessed after candidate exists
                        regime=regime,
                        indicators=indicators,
                    )
                except Exception as exc:
                    log.warning(
                        "strategy_error",
                        strategy=strategy.name,
                        symbol=symbol,
                        error=str(exc),
                    )
                    continue

                if candidate is None:
                    continue

                # Attach patterns from detectors to candidate
                candidate = candidate.model_copy(
                    update={"pattern_results": pattern_results}
                )

                # Override higher_tf_bias with real HTF structure if available
                if htf_structure is not None:
                    candidate = candidate.model_copy(
                        update={"higher_tf_bias": htf_structure.trend}
                    )

                # Volume context now that we have proposed_action
                volume = self._volume.analyze(df, candidate.proposed_action)

                # Risk assessment
                risk = self._risk.assess(candidate)

                # Confluence scoring
                breakdown = self._confluence.score(
                    candidate, structure, levels, volume, regime, risk
                )

                # ── 6. Bull/Bear bias gate ───────────────────────────────
                # Block signals that conflict with composite directional bias
                bias_result = None
                try:
                    candle_bull = sum(1 for p in pattern_results if getattr(p, "bias", "") == "bullish")
                    candle_bear = sum(1 for p in pattern_results if getattr(p, "bias", "") == "bearish")
                    vol_ctx = self._volume.analyze(df, candidate.proposed_action)
                    vol_confirms = getattr(vol_ctx, "is_confirming", False) if vol_ctx else False

                    # Check if regime supports the proposed direction
                    trending_regimes = {"trending_up", "trending_down", "breakout"}
                    regime_val = regime.regime.value if hasattr(regime.regime, "value") else str(regime.regime)
                    regime_supports = regime_val.lower() in trending_regimes

                    bias_result = BullBearBiasEngine().score(BiasInput(
                        indicator_bullish=indicator_bias.bullish_score if indicator_bias else 0.5,
                        indicator_bearish=indicator_bias.bearish_score if indicator_bias else 0.5,
                        structure_bias=deep_structure.trend_bias if deep_structure else "neutral",
                        structure_strength=deep_structure.strength if deep_structure else 0.0,
                        candle_bullish_count=candle_bull,
                        candle_bearish_count=candle_bear,
                        candle_total=len(pattern_results),
                        regime_supports_direction=regime_supports,
                        volume_confirms=vol_confirms,
                        htf_bias=htf_bias_str,
                    ))

                    # Block if bias strongly conflicts with proposed action
                    proposed = candidate.proposed_action
                    if proposed == SignalAction.BUY and bias_result.net_bias == "bearish" and bias_result.bearish_score > 0.6:
                        log.info(
                            "bias_gate_blocked",
                            symbol=symbol, strategy=strategy.name,
                            proposed="BUY", bias="bearish",
                            bearish_score=bias_result.bearish_score,
                            conflict=bias_result.conflict_score,
                        )
                        continue  # Skip this signal — bias says bearish
                    if proposed == SignalAction.SELL and bias_result.net_bias == "bullish" and bias_result.bullish_score > 0.6:
                        log.info(
                            "bias_gate_blocked",
                            symbol=symbol, strategy=strategy.name,
                            proposed="SELL", bias="bullish",
                            bullish_score=bias_result.bullish_score,
                            conflict=bias_result.conflict_score,
                        )
                        continue  # Skip this signal — bias says bullish

                    # High conflict → reduce confidence
                    if bias_result.conflict_score > 0.7:
                        log.debug("bias_high_conflict", symbol=symbol, conflict=bias_result.conflict_score)
                except Exception as exc:
                    log.debug("bias_engine_error", error=str(exc))

                # ── 7. Confidence calibration ────────────────────────────
                try:
                    from libs.signals.calibration.engine import ConfidenceCalibrator
                    cal_result = ConfidenceCalibrator().calibrate(
                        raw_confidence=breakdown.weighted_total,
                        strategy_name=strategy.name,
                        strategy_win_rate=None,  # TODO: wire from outcome DB
                        strategy_trade_count=15,  # Assume some history
                        symbol_win_rate=None,
                        regime_win_rate=None,
                        conflict_score=bias_result.conflict_score if bias_result else 0.0,
                    )
                    if hasattr(breakdown, 'model_copy'):
                        breakdown = breakdown.model_copy(update={
                            "weighted_total": cal_result.calibrated_confidence,
                        })
                except Exception:
                    cal_result = None

                # ── 8. Trade grading ─────────────────────────────────────
                grading_result = None
                try:
                    from libs.signals.grading.engine import TradeDecisionEngine, GradingInput
                    vol_ctx = self._volume.analyze(df, candidate.proposed_action)
                    grading_result = TradeDecisionEngine().grade(GradingInput(
                        confidence=breakdown.weighted_total,
                        risk_reward=self._emitter._calc_rr(candidate) if candidate.proposed_action != SignalAction.NO_TRADE else 0,
                        bias_net=bias_result.net_bias if bias_result else "neutral",
                        bias_conflict=bias_result.conflict_score if bias_result else 0.5,
                        bias_bullish=bias_result.bullish_score if bias_result else 0.5,
                        bias_bearish=bias_result.bearish_score if bias_result else 0.5,
                        htf_aligned=htf_bias_str != "neutral",
                        regime_supports=regime.vol_score >= 0.4 if hasattr(regime, 'vol_score') else True,
                        structure_strength=deep_structure.strength if deep_structure else 0.0,
                        volume_confirms=getattr(vol_ctx, 'is_confirming', False) if vol_ctx else False,
                        data_quality_clean=quality.is_safe,
                        action=candidate.proposed_action.value if candidate.proposed_action != SignalAction.NO_TRADE else "BUY",
                        is_late_entry=False,
                        is_overextended=False,
                    ))
                except Exception:
                    pass

                # ── 9. Futures risk ──────────────────────────────────────
                try:
                    from libs.core.models.domain import TradingMode
                    if getattr(candidate, 'trading_mode', None) == TradingMode.FUTURES:
                        from libs.risk.futures import FuturesRiskEngine
                        futures_result = FuturesRiskEngine().assess(
                            entry_price=(candidate.entry_zone_low + candidate.entry_zone_high) / 2,
                            leverage=getattr(candidate, 'leverage', 3.0),
                            margin_type="isolated",
                            direction="LONG" if candidate.proposed_action == SignalAction.BUY else "SHORT",
                            stop_loss=candidate.stop_loss,
                        )
                        if futures_result.should_reject:
                            log.info("futures_risk_rejected", symbol=symbol, reason=futures_result.rejection_reason)
                            continue
                except Exception:
                    pass

                # Emit signal
                output = self._emitter.emit(candidate, breakdown)

                # Attach grading to output
                if grading_result is not None:
                    try:
                        output = output.model_copy(update={
                            "setup_grade": grading_result.setup_grade,
                            "trade_decision": grading_result.trade_decision,
                        })
                    except Exception:
                        pass

                # ── Paper trading: dispatch with original proposed action ────
                # Paper bots get the signal with the strategy's proposed action,
                # bypassing confluence/ML/guard NO_TRADE overrides.
                paper_signal = output
                if output.action == SignalAction.NO_TRADE and candidate.proposed_action != SignalAction.NO_TRADE:
                    paper_signal = output.model_copy(update={
                        "action": candidate.proposed_action,
                    })
                if paper_signal.action != SignalAction.NO_TRADE:
                    await self._bus.publish("paper.signal.raw", {
                        "signal": paper_signal,
                    })

                # ── ML filter: block signals predicted as low-win-probability ──
                try:
                    from libs.ml.signal_classifier import get_classifier
                    clf = get_classifier()
                    if clf.is_trained and output.action != SignalAction.NO_TRADE:
                        win_prob = clf.predict_win_prob(output)
                        if clf.should_block(output):
                            log.info(
                                "ml_filter_blocked",
                                symbol=symbol,
                                strategy=output.strategy_name,
                                win_prob=round(win_prob, 3),
                            )
                            # Re-emit as NO_TRADE with ML block note
                            from libs.signals.output.emitter import SignalEmitter
                            output = output.model_copy(update={
                                "action": SignalAction.NO_TRADE,
                                "warnings": list(output.warnings or []) + [
                                    f"ML_BLOCKED: predicted win prob {win_prob:.0%}"
                                ],
                            })
                        else:
                            # Attach ML win probability to warnings for visibility
                            output = output.model_copy(update={
                                "warnings": list(output.warnings or []) + [
                                    f"ML_WIN_PROB: {win_prob:.0%}"
                                ],
                            })
                except Exception:
                    pass  # ML unavailable — proceed without filtering

                # ── Portfolio guard: check exposure limits ─────────────────
                if output.action != SignalAction.NO_TRADE:
                    decision = self._guard.is_allowed(output)
                    if decision.blocked:
                        log.info(
                            "portfolio_guard_blocked",
                            symbol=symbol,
                            strategy=output.strategy_name,
                            reason=decision.reason,
                        )
                        output = output.model_copy(update={
                            "action": SignalAction.NO_TRADE,
                            "warnings": list(output.warnings or []) + [
                                f"GUARD_BLOCKED: {decision.reason}"
                            ],
                        })
                    else:
                        self._guard.register(output)

                outputs.append(output)

                # Persist to DB + audit log + metrics + event bus
                await self._audit.record_signal(output)
                self._metrics.record_signal(output)
                try:
                    async with get_session_factory()() as db_session:
                        repo = SignalRepository(db_session)
                        await repo.save_signal(output)
                except Exception as db_exc:
                    log.warning("db_save_failed", symbol=symbol, error=str(db_exc))
                await self._bus.publish(EventBus.SIGNAL_GENERATED, {
                    "symbol": symbol,
                    "action": output.action.value,
                    "confidence": output.confidence,
                    "strategy": output.strategy_name,
                    "signal": output,
                })

                log.info("signal_emitted", signal=output.to_display())

        except Exception as exc:
            log.error(
                "pipeline_error", symbol=symbol, timeframe=timeframe.value, error=str(exc)
            )

        return outputs

    _HTF_MAP: dict[Timeframe, Timeframe] = {
        Timeframe.ONE_MIN:     Timeframe.FIVE_MIN,
        Timeframe.THREE_MIN:   Timeframe.FIFTEEN_MIN,
        Timeframe.FIVE_MIN:    Timeframe.FIFTEEN_MIN,
        Timeframe.FIFTEEN_MIN: Timeframe.ONE_HOUR,
        Timeframe.THIRTY_MIN:  Timeframe.FOUR_HOUR,
        Timeframe.ONE_HOUR:    Timeframe.FOUR_HOUR,
        Timeframe.FOUR_HOUR:   Timeframe.ONE_DAY,
        Timeframe.ONE_DAY:     Timeframe.ONE_WEEK,
    }

    @classmethod
    def _htf_timeframe(cls, tf: Timeframe) -> Timeframe | None:
        return cls._HTF_MAP.get(tf)

    @staticmethod
    def _timeframe_seconds(tf: Timeframe) -> int:
        _MAP = {
            Timeframe.ONE_MIN: 60, Timeframe.THREE_MIN: 180,
            Timeframe.FIVE_MIN: 300, Timeframe.FIFTEEN_MIN: 900,
            Timeframe.THIRTY_MIN: 1800, Timeframe.ONE_HOUR: 3600,
            Timeframe.FOUR_HOUR: 14400, Timeframe.ONE_DAY: 86400,
            Timeframe.ONE_WEEK: 604800,
        }
        return _MAP.get(tf, 300)
