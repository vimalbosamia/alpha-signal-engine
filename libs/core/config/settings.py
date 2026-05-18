"""
Central settings management.

All config is typed with Pydantic v2. Secrets are exclusively
sourced from environment variables — never from committed files.

Layered override order (highest wins):
  1. Environment variables / .env
  2. Explicit constructor arguments (for tests)
  3. Pydantic defaults
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentMode(str, Enum):
    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


# ── Provider sub-settings ─────────────────────────────────────────────────────

class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPACA_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""))
    secret_key: SecretStr = Field(default=SecretStr(""))
    base_url: str = "https://paper-api.alpaca.markets"
    data_url: str = "https://data.alpaca.markets"
    data_feed: str = "iex"       # iex (free) | sip (paid)
    timeout_seconds: int = 30


class BinanceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""))
    secret_key: SecretStr = Field(default=SecretStr(""))
    env: str = "testnet"          # live | testnet
    testnet_url: str = "https://testnet.binance.vision"
    ws_reconnect_attempts: int = 5
    ws_ping_interval: int = 20

    @property
    def base_url(self) -> str:
        if self.env == "testnet":
            return self.testnet_url
        return "https://api.binance.com"


class CoinbaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COINBASE_", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""))
    secret_key: SecretStr = Field(default=SecretStr(""))
    enabled: bool = False


# ── Signal settings ───────────────────────────────────────────────────────────

class SignalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    min_confluence_score: float = Field(default=0.40, ge=0.0, le=1.0)
    min_reward_risk: float = Field(default=1.0, ge=0.5)
    max_active_signals: int = Field(default=100, ge=1)
    max_signals_per_symbol: int = Field(default=5, ge=1)
    stock_watchlist: str = "AAPL,MSFT,NVDA,TSLA,SPY,QQQ"
    crypto_watchlist: str = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,AVAXUSDT,TRXUSDT,LINKUSDT,SUIUSDT,NEARUSDT,PEPEUSDT,LTCUSDT,UNIUSDT,AAVEUSDT,DASHUSDT,ZECUSDT"
    futures_watchlist: str = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT"
    enable_futures: bool = False

    @property
    def stock_symbols(self) -> list[str]:
        return [s.strip() for s in self.stock_watchlist.split(",") if s.strip()]

    @property
    def crypto_symbols(self) -> list[str]:
        return [s.strip() for s in self.crypto_watchlist.split(",") if s.strip()]

    @property
    def futures_symbols(self) -> list[str]:
        return [s.strip() for s in self.futures_watchlist.split(",") if s.strip()]


# ── Risk settings ─────────────────────────────────────────────────────────────

class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    max_risk_per_signal_pct: float = Field(default=2.0, ge=0.1, le=10.0)
    max_correlated_signals: int = Field(default=4, ge=1)
    data_quality_strict: bool = False


# ── Storage settings ──────────────────────────────────────────────────────────

class StorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./data/signals.db"
    enable_audit_log: bool = True
    enable_replay: bool = True
    audit_log_dir: str = "data/audit"
    max_candle_cache_bars: int = 1000


# ── Observability settings ────────────────────────────────────────────────────

class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    prometheus_port: int = 9090
    enable_metrics: bool = True
    slack_webhook_url: str = ""
    alert_email: str = ""


# ── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    agent_mode: AgentMode = AgentMode.PAPER
    enable_stocks: bool = True
    enable_crypto: bool = True

    # Nested settings (read from same env vars)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    coinbase: CoinbaseSettings = Field(default_factory=CoinbaseSettings)
    signal: SignalSettings = Field(default_factory=SignalSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("agent_mode", mode="before")
    @classmethod
    def normalise_mode(cls, v: object) -> object:
        if isinstance(v, AgentMode):
            return v
        if isinstance(v, str):
            return v.lower()
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached validated settings. Call once at startup."""
    return Settings()


def is_live() -> bool:
    return get_settings().agent_mode == AgentMode.LIVE


def is_paper() -> bool:
    return get_settings().agent_mode == AgentMode.PAPER


def is_backtest() -> bool:
    return get_settings().agent_mode == AgentMode.BACKTEST
