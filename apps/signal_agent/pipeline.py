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
    ThreeWhiteSoldiersDetector, ThreeBlackCrowsDetector,
)
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

# Default pattern detectors (all instantiated with BALANCED mode)
DEFAULT_DETECTORS = [
    HammerDetector(), InvertedHammerDetector(), ShootingStarDetector(),
    HangingManDetector(), DojiDetector(), DragonflyDojiDetector(),
    GravestoneDojiDetector(), SpinningTopDetector(),
    BullishMarubozuDetector(), BearishMarubozuDetector(),
    BullishEngulfingDetector(), BearishEngulfingDetector(),
    BullishHaramiDetector(), BearishHaramiDetector(),
    PiercingLineDetector(), DarkCloudCoverDetector(),
    TweezerTopDetector(), TweezerBottomDetector(),
    MorningStarDetector(), EveningStarDetector(),
    ThreeWhiteSoldiersDetector(), ThreeBlackCrowsDetector(),
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

                # Emit signal
                output = self._emitter.emit(candidate, breakdown)

                # ── Paper trading: dispatch BEFORE ML/guard filters ──────────
                # Paper bots make their own decisions; they need unfiltered signals
                if output.action != SignalAction.NO_TRADE:
                    await self._bus.publish("paper.signal.raw", {
                        "signal": output,
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
