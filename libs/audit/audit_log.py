from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel, Field

from libs.core.logging.logger import get_logger
from libs.core.models.domain import AssetClass, SignalOutput

log = get_logger(__name__)


class AuditEvent(BaseModel):
    """Single step in the signal pipeline trail."""

    event_id: UUID = Field(default_factory=uuid4)
    signal_id: UUID | None = None
    event_type: str  # e.g. "data_fetched", "quality_checked", "signal_generated"
    symbol: str
    asset_class: AssetClass
    payload: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditLog:
    """
    Writes AuditEvents to rotating JSONL files and optionally a DB.
    Thread-safe via asyncio lock.
    """

    def __init__(self, log_dir: str = "data/audit", enabled: bool = True) -> None:
        self._dir = Path(log_dir)
        self._enabled = enabled
        self._lock = asyncio.Lock()
        if enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    async def record(self, event: AuditEvent) -> None:
        """Append event to today's JSONL file."""
        if not self._enabled:
            return
        line = (
            orjson.dumps(event.model_dump(mode="json"), default=str).decode() + "\n"
        )
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._dir / f"audit-{date_str}.jsonl"
        async with self._lock:
            await asyncio.to_thread(self._append, path, line)

    def _append(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    async def get_signal_trail(self, signal_id: UUID) -> list[AuditEvent]:
        """Load all events for a signal_id from today's JSONL."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._dir / f"audit-{date_str}.jsonl"
        if not path.exists():
            return []
        events: list[AuditEvent] = []

        def _read() -> list[str]:
            return path.read_text(encoding="utf-8").splitlines()

        lines = await asyncio.to_thread(_read)
        for line in lines:
            try:
                data = orjson.loads(line)
                if str(data.get("signal_id")) == str(signal_id):
                    events.append(AuditEvent(**data))
            except Exception:
                pass
        return events

    async def record_signal(self, signal: SignalOutput) -> None:
        """Convenience: record a generated signal as an audit event."""
        event = AuditEvent(
            signal_id=signal.signal_id,
            event_type="signal_generated",
            symbol=signal.symbol,
            asset_class=signal.asset_class,
            payload={
                "action": signal.action.value,
                "confidence": signal.confidence,
                "strategy": signal.strategy_name,
                "regime": signal.market_regime.value,
                "rr": signal.estimated_risk_reward,
            },
        )
        await self.record(event)
