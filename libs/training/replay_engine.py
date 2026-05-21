"""
libs.training.replay_engine — Simulation Replay for Extreme Market Events.

Replays historical extreme events (COVID crash, FTX collapse, flash crashes,
meme squeezes, CPI spikes) to train survival behavior and adaptive risk
management.

Each event is a named time window on specific symbols. The engine downloads
the event-period data and trains the agent specifically on extreme conditions.

Design rules:
  - Events are immutable (frozen dataclass)
  - All events have start_date / end_date / affected symbols
  - Trains with emphasis on: survival, drawdown control, defensive mode activation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── Event definitions ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarketEvent:
    """Definition of a historical extreme market event."""
    name: str
    description: str
    start_date: str          # YYYY-MM-DD
    end_date: str            # YYYY-MM-DD
    symbols: list[str]       # affected symbols to train on
    event_type: str          # crash, rally, squeeze, macro, flash_crash
    severity: str            # extreme, high, moderate
    expected_regime: str     # what regime the agent should detect
    lessons: list[str]       # what the agent should learn from this event


# Curated extreme events for training
EXTREME_EVENTS: list[MarketEvent] = [
    MarketEvent(
        name="covid_crash_2020",
        description="COVID-19 pandemic crash — BTC fell 50%+ in 24 hours",
        start_date="2020-03-09",
        end_date="2020-03-20",
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"],
        event_type="crash",
        severity="extreme",
        expected_regime="panic_selloff",
        lessons=["Stop trading during extreme vol", "Cut leverage immediately",
                 "Don't catch falling knives", "Cash is a position"],
    ),
    MarketEvent(
        name="may_2021_crash",
        description="China mining ban + Elon tweet crash — BTC -55% in weeks",
        start_date="2021-05-12",
        end_date="2021-05-25",
        symbols=["BTCUSDT", "ETHUSDT", "DOGEUSDT", "BNBUSDT", "ADAUSDT"],
        event_type="crash",
        severity="extreme",
        expected_regime="panic_selloff",
        lessons=["News events can override technical analysis",
                 "Correlated assets crash together", "Reduce exposure fast"],
    ),
    MarketEvent(
        name="luna_collapse_2022",
        description="LUNA/UST depeg and death spiral",
        start_date="2022-05-07",
        end_date="2022-05-15",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT"],
        event_type="crash",
        severity="extreme",
        expected_regime="liquidation_event",
        lessons=["Contagion risk across crypto", "Stablecoin risk is systemic",
                 "Liquidation cascades accelerate selloffs"],
    ),
    MarketEvent(
        name="ftx_collapse_2022",
        description="FTX exchange collapse — trust crisis across crypto",
        start_date="2022-11-06",
        end_date="2022-11-14",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"],
        event_type="crash",
        severity="extreme",
        expected_regime="liquidation_event",
        lessons=["Exchange risk is real", "Cascading liquidations",
                 "Exit positions before full information available"],
    ),
    MarketEvent(
        name="btc_etf_rally_2024",
        description="BTC ETF approval rally — sustained bullish momentum",
        start_date="2024-01-08",
        end_date="2024-01-20",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "AVAXUSDT"],
        event_type="rally",
        severity="high",
        expected_regime="trending_up",
        lessons=["Institutional flows create sustained trends",
                 "Don't short strong momentum", "Scale into trends"],
    ),
    MarketEvent(
        name="meme_squeeze_doge_2021",
        description="DOGE meme squeeze — 10x in weeks",
        start_date="2021-04-12",
        end_date="2021-04-20",
        symbols=["DOGEUSDT", "SHIBUSDT"],
        event_type="squeeze",
        severity="high",
        expected_regime="climactic",
        lessons=["Meme rallies are unsustainable", "Take profits aggressively",
                 "Don't chase parabolic moves"],
    ),
    MarketEvent(
        name="flash_crash_may_2021",
        description="BTC flash crash from 43k to 30k in minutes",
        start_date="2021-05-19",
        end_date="2021-05-20",
        symbols=["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT"],
        event_type="flash_crash",
        severity="extreme",
        expected_regime="panic_selloff",
        lessons=["Flash crashes recover partially", "Wide stops get hunted",
                 "Liquidity evaporates in panic"],
    ),
    MarketEvent(
        name="cpi_volatility_2022",
        description="CPI surprise — hot inflation print spiked vol across markets",
        start_date="2022-09-13",
        end_date="2022-09-15",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        event_type="macro",
        severity="moderate",
        expected_regime="news_driven",
        lessons=["Don't hold through CPI releases", "Reduce size before macro events",
                 "Fade the initial spike after data settles"],
    ),
    MarketEvent(
        name="bear_market_2022",
        description="Extended crypto bear market — BTC 69k to 15k",
        start_date="2022-04-01",
        end_date="2022-06-30",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "DOTUSDT"],
        event_type="crash",
        severity="high",
        expected_regime="trending_down",
        lessons=["Bear markets last longer than expected",
                 "Mean reversion fails in strong trends",
                 "Capital preservation > profit seeking"],
    ),
    MarketEvent(
        name="low_liquidity_holiday_2023",
        description="Christmas/New Year low liquidity period — thin books, fakeouts",
        start_date="2023-12-23",
        end_date="2024-01-02",
        symbols=["BTCUSDT", "ETHUSDT"],
        event_type="low_liquidity",
        severity="moderate",
        expected_regime="low_liquidity",
        lessons=["Reduce position size in low liquidity",
                 "Breakouts in thin markets are unreliable",
                 "Spreads widen significantly"],
    ),
]


class ReplayEngine:
    """Replays extreme market events for agent training.

    Downloads event-period data and runs it through the training pipeline
    with emphasis on survival behavior.
    """

    def __init__(self, data_dir: str = "data/historical") -> None:
        self._data_dir = data_dir

    def list_events(self) -> list[dict[str, Any]]:
        """List all available replay events."""
        return [
            {
                "name": e.name,
                "description": e.description,
                "type": e.event_type,
                "severity": e.severity,
                "date_range": f"{e.start_date} → {e.end_date}",
                "symbols": len(e.symbols),
                "lessons": len(e.lessons),
            }
            for e in EXTREME_EVENTS
        ]

    async def download_event(
        self,
        event_name: str,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Download data for a specific event period.

        Returns dict of results.
        """
        event = self._find_event(event_name)
        if event is None:
            return {"error": f"Unknown event: {event_name}"}

        timeframes = timeframes or ["5m", "15m", "1h"]

        try:
            from libs.data.providers.historical.downloader import BinanceHistoricalDownloader

            dl = BinanceHistoricalDownloader(data_dir=self._data_dir)
            results: dict[str, Any] = {"event": event_name, "downloads": {}}

            for tf in timeframes:
                tf_results = await dl.download_all(
                    symbols=list(event.symbols),
                    timeframe=tf,
                    months=1,  # events are short — 1 month covers them
                    max_concurrent=3,
                )
                results["downloads"][tf] = len(tf_results)

            return results
        except Exception as exc:
            return {"error": str(exc)}

    async def replay_event(
        self,
        event_name: str,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Replay a specific event through the training pipeline.

        Returns training results for the event period.
        """
        event = self._find_event(event_name)
        if event is None:
            return {"error": f"Unknown event: {event_name}"}

        timeframes = timeframes or ["15m", "1h"]

        try:
            from libs.training.historical_trainer import HistoricalTrainer

            trainer = HistoricalTrainer(data_dir=self._data_dir)
            summary = await trainer.train(
                symbols=list(event.symbols),
                timeframes=timeframes,
            )

            return {
                "event": event_name,
                "description": event.description,
                "severity": event.severity,
                "expected_regime": event.expected_regime,
                "total_trades": summary.total_trades,
                "win_rate": round(summary.win_rate, 4),
                "phase_before": summary.phase_before,
                "phase_after": summary.phase_after,
                "duration_seconds": round(summary.duration_seconds, 1),
                "lessons": event.lessons,
            }
        except Exception as exc:
            return {"error": str(exc)}

    async def replay_all(
        self,
        severity_filter: str | None = None,
        timeframes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Replay all events (optionally filtered by severity).

        Returns list of per-event results.
        """
        events = EXTREME_EVENTS
        if severity_filter:
            events = [e for e in events if e.severity == severity_filter]

        results: list[dict[str, Any]] = []
        for event in events:
            _log.info("replay_engine.replaying", event=event.name, severity=event.severity)
            result = await self.replay_event(event.name, timeframes)
            results.append(result)

        return results

    def _find_event(self, name: str) -> MarketEvent | None:
        """Find an event by name."""
        for event in EXTREME_EVENTS:
            if event.name == name:
                return event
        return None
