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
    AssetClass, SignalCandidate, SignalOutput, SignalAction, Timeframe
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
from libs.confluence.engine import ConfluenceV2Engine
from libs.confluence.scorer import meets_threshold
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
        self._confluence_v2 = ConfluenceV2Engine()
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
            levels = self._levels.analyze(df, asset_class)
            regime = self._regime.analyze(df)
            indicators = self._indicators.compute(df)

            # ── 5. Pass 1: Collect ALL strategy candidates ─────────────────
            all_candidates: list[SignalCandidate] = []
            pattern_results = [det.detect(df) for det in DEFAULT_DETECTORS]

            for strategy in self._strategies:
                if not strategy.is_eligible(asset_class, session, quality):
                    continue
                if len(df) < strategy.min_bars_required:
                    continue
                if is_strategy_muted(strategy.name):
                    log.debug("strategy_muted_skipped", strategy=strategy.name, symbol=symbol)
                    continue

                try:
                    candidate = strategy.generate_candidate(
                        symbol=symbol,
                        asset_class=asset_class,
                        df=df,
                        df_htf=None,
                        session=session,
                        quality=quality,
                        structure=structure,
                        levels=levels,
                        volume=None,
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

                if candidate is not None:
                    candidate = candidate.model_copy(
                        update={"pattern_results": pattern_results}
                    )
                    all_candidates.append(candidate)

            if not all_candidates:
                return []

            # ── 6. Pass 2: Confluence V2 — multi-layer scoring ───────────
            # Build indicator dict for confirmation/suppression layers
            ind_dict = None
            if indicators:
                last_row = df.iloc[-1] if len(df) > 0 else None
                ind_dict = dict(indicators)
                if last_row is not None:
                    ind_dict.setdefault("close", float(last_row.get("close", 0)))
                    ind_dict.setdefault("volume_relative", indicators.get("volume_relative"))
                # Add prev_rsi for divergence detection
                if len(df) > 1:
                    prev_row = df.iloc[-2]
                    prev_indicators = self._indicators.compute(df.iloc[:-1])
                    ind_dict["prev_rsi"] = prev_indicators.get("rsi")

            # Build bias data from indicators
            bias_data = None
            if ind_dict:
                bias_data = {
                    "bullish": ind_dict.get("bullish_score", 0.0),
                    "bearish": ind_dict.get("bearish_score", 0.0),
                    "net": ind_dict.get("bias_net", "neutral"),
                }

            confluence_results = self._confluence_v2.evaluate(
                symbol=symbol,
                candidates=all_candidates,
                regime=regime,
                structure=structure,
                bias_data=bias_data,
                indicators=ind_dict,
            )

            log.debug(
                "confluence_v2_results",
                symbol=symbol,
                candidates=len(all_candidates),
                results=len(confluence_results),
                scores=[
                    f"{r.direction}:{r.primary_strategy}={r.confluence_score:.0f}"
                    for r in confluence_results
                ],
            )

            # ── 7. Emit signals — one per confluence result ──────────────
            for cr in confluence_results:
                # Find the primary trigger candidate
                primary = next(
                    (c for c in all_candidates
                     if c.strategy_name == cr.primary_strategy
                     and c.proposed_action.value == cr.direction),
                    None,
                )
                if primary is None:
                    continue

                volume = self._volume.analyze(df, primary.proposed_action)
                risk = self._risk.assess(primary)
                breakdown = self._confluence.score(
                    primary, structure, levels, volume, regime, risk
                )

                output = self._emitter.emit(primary, breakdown)

                # Enrich with V2 confluence metadata
                v2_warnings = list(output.warnings or [])
                v2_warnings.append(
                    f"CONFLUENCE_V2: score={cr.confluence_score:.0f} "
                    f"quality={cr.trade_quality} layers={cr.layer_count} "
                    f"primary={cr.primary_strategy} "
                    f"supporting=[{','.join(cr.supporting_strategies)}] "
                    f"confirmations=[{','.join(cr.confirmation_signals)}] "
                    f"suppressions=[{','.join(cr.suppression_signals)}]"
                )

                # Override confidence with V2 score (normalized 0-1)
                v2_confidence = min(1.0, cr.confluence_score / 100.0)
                output = output.model_copy(update={
                    "confidence": v2_confidence,
                    "warnings": v2_warnings,
                    "explanation": (
                        f"Multi-layer confluence: {cr.primary_strategy} "
                        f"+ {len(cr.supporting_strategies)} supporting "
                        f"+ {len(cr.confirmation_signals)} confirmations. "
                        f"Score: {cr.confluence_score:.0f}/100 ({cr.trade_quality})"
                    ),
                })

                # ── ML filter ──
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
                            output = output.model_copy(update={
                                "action": SignalAction.NO_TRADE,
                                "warnings": list(output.warnings or []) + [
                                    f"ML_BLOCKED: predicted win prob {win_prob:.0%}"
                                ],
                            })
                        else:
                            output = output.model_copy(update={
                                "warnings": list(output.warnings or []) + [
                                    f"ML_WIN_PROB: {win_prob:.0%}"
                                ],
                            })
                except Exception:
                    pass

                # ── Portfolio guard ──
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

                # Persist + publish
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
                    "confluence_score": cr.confluence_score,
                    "trade_quality": cr.trade_quality,
                    "supporting": list(cr.supporting_strategies),
                    "confirmations": list(cr.confirmation_signals),
                })

                log.info(
                    "signal_emitted",
                    signal=output.to_display(),
                    confluence_score=cr.confluence_score,
                    quality=cr.trade_quality,
                    layers=cr.layer_count,
                )

        except Exception as exc:
            log.error(
                "pipeline_error", symbol=symbol, timeframe=timeframe.value, error=str(exc)
            )

        return outputs

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
