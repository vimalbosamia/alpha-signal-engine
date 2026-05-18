from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import orjson
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.core.models.domain import SignalOutput
from libs.data.storage.models import (
    AuditRecord, ProviderHealthRecord, SignalRecord,
    SignalOutcomeRecord, StrategyPerformanceRecord,
)


class SignalRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_signal(self, signal: SignalOutput) -> None:
        """Persist a SignalOutput to the signals table."""
        record = SignalRecord(
            id=str(signal.signal_id),
            symbol=signal.symbol,
            asset_class=signal.asset_class.value,
            action=signal.action.value,
            confidence=signal.confidence,
            strategy_name=signal.strategy_name,
            timeframe=signal.timeframe.value,
            regime=signal.market_regime.value,
            estimated_rr=signal.estimated_risk_reward,
            generated_at=signal.generated_at,
            payload=signal.model_dump_json(),
        )
        self._session.add(record)
        await self._session.commit()

    async def get_latest_signals(
        self, symbol: str | None = None, limit: int = 50
    ) -> list[dict]:
        """Return latest signals as dicts (deserialized payload)."""
        stmt = (
            select(SignalRecord)
            .order_by(desc(SignalRecord.generated_at))
            .limit(limit)
        )
        if symbol:
            stmt = stmt.where(SignalRecord.symbol == symbol)
        result = await self._session.execute(stmt)
        return [orjson.loads(r.payload) for r in result.scalars()]

    async def get_by_id(self, signal_id: str) -> dict | None:
        result = await self._session.get(SignalRecord, signal_id)
        if result is None:
            return None
        return orjson.loads(result.payload)


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_event(
        self,
        event_type: str,
        symbol: str,
        asset_class: str,
        payload: dict,
        signal_id: str | None = None,
    ) -> None:
        record = AuditRecord(
            id=str(uuid4()),
            signal_id=signal_id,
            event_type=event_type,
            symbol=symbol,
            asset_class=asset_class,
            timestamp=datetime.now(timezone.utc),
            payload=orjson.dumps(payload).decode(),
        )
        self._session.add(record)
        await self._session.commit()

    async def get_signal_trail(self, signal_id: str) -> list[dict]:
        stmt = (
            select(AuditRecord)
            .where(AuditRecord.signal_id == signal_id)
            .order_by(AuditRecord.timestamp)
        )
        result = await self._session.execute(stmt)
        return [
            {
                "event_type": r.event_type,
                "timestamp": r.timestamp.isoformat(),
                "payload": orjson.loads(r.payload),
            }
            for r in result.scalars()
        ]


class OutcomeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_outcome(self, outcome: SignalOutcomeRecord) -> None:
        existing = await self._session.get(SignalOutcomeRecord, outcome.id)
        if existing is None:
            self._session.add(outcome)
        else:
            existing.outcome = outcome.outcome
            existing.check_price = outcome.check_price
            existing.direction_correct = outcome.direction_correct
            existing.checked_at = outcome.checked_at
            if outcome.ml_features:
                existing.ml_features = outcome.ml_features
        await self._session.commit()

    async def get_resolved_with_features(self) -> list[dict]:
        """Return all WIN/LOSS outcomes that have ML features stored."""
        stmt = select(SignalOutcomeRecord).where(
            SignalOutcomeRecord.outcome.in_(["WIN", "LOSS"]),
            SignalOutcomeRecord.ml_features.is_not(None),
        )
        result = await self._session.execute(stmt)
        rows = []
        for r in result.scalars():
            try:
                features = orjson.loads(r.ml_features)
                rows.append({
                    "features": features,
                    "win": r.direction_correct,
                    "strategy": r.strategy_name,
                    "symbol": r.symbol,
                })
            except Exception:
                continue
        return rows

    async def get_pending(self) -> list[SignalOutcomeRecord]:
        stmt = select(SignalOutcomeRecord).where(
            SignalOutcomeRecord.outcome == "PENDING"
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def get_strategy_stats(self) -> list[dict]:
        stmt = select(StrategyPerformanceRecord).order_by(
            StrategyPerformanceRecord.win_rate.desc()
        )
        result = await self._session.execute(stmt)
        return [
            {
                "strategy": r.strategy_name,
                "total": r.total_signals,
                "wins": r.wins,
                "losses": r.losses,
                "win_rate": round(r.win_rate * 100, 1),
                "muted": r.is_muted,
            }
            for r in result.scalars()
        ]

    async def upsert_strategy_performance(
        self,
        strategy_name: str,
        won: bool,
    ) -> StrategyPerformanceRecord:
        from datetime import datetime
        rec = await self._session.get(StrategyPerformanceRecord, strategy_name)
        if rec is None:
            rec = StrategyPerformanceRecord(
                strategy_name=strategy_name,
                total_signals=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                is_muted=False,
                last_updated=datetime.now(timezone.utc),
            )
            self._session.add(rec)

        rec.total_signals += 1
        if won:
            rec.wins += 1
        else:
            rec.losses += 1

        total = rec.wins + rec.losses
        rec.win_rate = rec.wins / total if total > 0 else 0.0
        rec.is_muted = total >= 5 and rec.win_rate < 0.40
        rec.last_updated = datetime.now(timezone.utc)

        await self._session.commit()
        return rec
