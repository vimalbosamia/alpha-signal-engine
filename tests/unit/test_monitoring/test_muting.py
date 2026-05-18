"""Tests for strategy muting persistence."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import libs.monitoring.outcome_tracker as ot_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(muted_names: list[str], unmuted_names: list[str] | None = None) -> list[dict]:
    """Build a fake get_strategy_stats return value."""
    stats = [{"strategy": name, "muted": True} for name in muted_names]
    for name in (unmuted_names or []):
        stats.append({"strategy": name, "muted": False})
    return stats


def _reset_muted(names: frozenset[str] | None = None) -> None:
    """Reset the module-level _MUTED_STRATEGIES frozenset."""
    ot_mod._MUTED_STRATEGIES = names if names is not None else frozenset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_muted_from_db_populates_set():
    """DB returns 2 muted strategies → is_strategy_muted reflects them."""
    fake_stats = _make_stats(["bad_strategy", "awful_strategy"], ["ok_strategy"])

    mock_repo = AsyncMock()
    mock_repo.get_strategy_stats.return_value = fake_stats

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock(return_value=mock_session)

    with patch(
        "libs.monitoring.outcome_tracker.get_session_factory",
        return_value=mock_factory,
    ), patch(
        "libs.monitoring.outcome_tracker.OutcomeRepository",
        return_value=mock_repo,
    ):
        _reset_muted()

        count = await ot_mod.load_muted_strategies_from_db()

    assert count == 2
    assert ot_mod.is_strategy_muted("bad_strategy") is True
    assert ot_mod.is_strategy_muted("awful_strategy") is True
    assert ot_mod.is_strategy_muted("ok_strategy") is False


@pytest.mark.asyncio
async def test_load_muted_clears_stale_mutes():
    """Stale name in set before load is removed when DB returns different names."""
    fake_stats = _make_stats(["new_bad_strategy"])

    mock_repo = AsyncMock()
    mock_repo.get_strategy_stats.return_value = fake_stats

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock(return_value=mock_session)

    with patch(
        "libs.monitoring.outcome_tracker.get_session_factory",
        return_value=mock_factory,
    ), patch(
        "libs.monitoring.outcome_tracker.OutcomeRepository",
        return_value=mock_repo,
    ):
        # Pre-populate with a stale entry
        _reset_muted(frozenset({"stale_strategy"}))

        await ot_mod.load_muted_strategies_from_db()

    assert ot_mod.is_strategy_muted("stale_strategy") is False
    assert ot_mod.is_strategy_muted("new_bad_strategy") is True


@pytest.mark.asyncio
async def test_load_muted_handles_empty_db():
    """Empty stats list → 0 muted strategies, no exception raised."""
    mock_repo = AsyncMock()
    mock_repo.get_strategy_stats.return_value = []

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock(return_value=mock_session)

    with patch(
        "libs.monitoring.outcome_tracker.get_session_factory",
        return_value=mock_factory,
    ), patch(
        "libs.monitoring.outcome_tracker.OutcomeRepository",
        return_value=mock_repo,
    ):
        _reset_muted()

        count = await ot_mod.load_muted_strategies_from_db()

    assert count == 0


@pytest.mark.asyncio
async def test_load_muted_handles_db_error():
    """DB raises an exception → returns 0, no propagation."""
    mock_factory = MagicMock(side_effect=RuntimeError("db down"))

    with patch(
        "libs.monitoring.outcome_tracker.get_session_factory",
        return_value=mock_factory,
    ):
        count = await ot_mod.load_muted_strategies_from_db()

    assert count == 0


@pytest.mark.asyncio
async def test_is_strategy_muted_returns_false_for_unknown():
    """Unknown strategy name returns False without any DB call."""
    _reset_muted()

    assert ot_mod.is_strategy_muted("completely_unknown_strategy") is False
