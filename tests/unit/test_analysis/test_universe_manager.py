"""Tests for UniverseManager."""
from libs.analysis.filters.universe_manager import UniverseManager, get_group


def _sym(symbol, volume=50_000_000, spread=0.05, atr=2.0, momentum=0.5, regime="trending_up"):
    return {"symbol": symbol, "asset_class": "crypto", "volume_24h": volume,
            "spread_pct": spread, "atr_pct": atr, "momentum": momentum, "regime": regime}


def test_rank_top_candidates():
    mgr = UniverseManager(max_candidates=3)
    data = [_sym("BTCUSDT", volume=100e6), _sym("ETHUSDT", volume=80e6),
            _sym("SOLUSDT", volume=60e6), _sym("DOGEUSDT", volume=40e6)]
    result = mgr.rank(data)
    assert result.eligible_count >= 3
    assert len(result.top_candidates) <= 3


def test_reject_low_volume():
    mgr = UniverseManager(min_volume_usd=50_000_000)
    data = [_sym("JUNKUSDT", volume=1_000_000)]
    result = mgr.rank(data)
    assert result.eligible_count == 0
    assert result.rejection_reasons.get("low_volume", 0) >= 1


def test_reject_wide_spread():
    mgr = UniverseManager()
    data = [_sym("BADUSDT", spread=1.0)]
    result = mgr.rank(data)
    assert result.eligible_count == 0


def test_reject_climactic_regime():
    mgr = UniverseManager()
    data = [_sym("BTCUSDT", regime="climactic")]
    result = mgr.rank(data)
    assert result.eligible_count == 0


def test_group_assignment():
    assert get_group("BTCUSDT") == "major"
    assert get_group("SOLUSDT") == "layer1"
    assert get_group("DOGEUSDT") == "meme"
    assert get_group("AAPL") == "tech_stock"
    assert get_group("RANDOMXYZ") == "other"


def test_news_sensitive():
    mgr = UniverseManager()
    assert mgr.is_news_sensitive("BTCUSDT") is True
    assert mgr.is_news_sensitive("RANDOMUSDT") is False
