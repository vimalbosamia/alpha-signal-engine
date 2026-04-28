from __future__ import annotations

import httpx

from libs.core.config.settings import get_settings
from libs.core.logging.logger import get_logger

log = get_logger(__name__)


class AlertManager:
    """Emit structured log alerts + optional Slack webhook."""

    def __init__(self, webhook_url: str = "") -> None:
        self._webhook = webhook_url or get_settings().observability.slack_webhook_url

    async def stale_data(self, symbol: str, age_seconds: float) -> None:
        log.warning("alert_stale_data", symbol=symbol, age_seconds=age_seconds)
        await self._send(
            f":warning: Stale data: `{symbol}` last bar {age_seconds:.0f}s ago"
        )

    async def missing_bars(self, symbol: str, count: int) -> None:
        log.warning("alert_missing_bars", symbol=symbol, count=count)
        await self._send(
            f":warning: Missing bars: `{symbol}` missing {count} bars"
        )

    async def provider_disconnect(self, provider: str) -> None:
        log.error("alert_provider_disconnect", provider=provider)
        await self._send(f":red_circle: Provider disconnect: `{provider}`")

    async def abnormal_signal_rate(self, signals_per_minute: float) -> None:
        log.warning(
            "alert_signal_rate", signals_per_minute=signals_per_minute
        )
        await self._send(
            f":warning: Abnormal signal rate: {signals_per_minute:.1f}/min"
        )

    async def _send(self, text: str) -> None:
        if not self._webhook:
            return
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._webhook, json={"text": text})
        except Exception as exc:
            log.debug("slack_alert_failed", error=str(exc))
