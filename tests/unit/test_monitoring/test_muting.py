"""Tests for strategy muting persistence."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(muted_names: list[str], unmuted_names: list[str] | None = None) -> list[dict]:
    """Build a fake get_strategy_stats return value."""
    stats = [{"strategy": name, "muted": True} for name in muted_names]
    for name in (unmuted_names or []):
        stats.append({"strategy": name, "muted": False})
    return stats


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
        from libs.monitoring.outcome_tracker import (
            load_muted_strategies_from_db,
            is_strategy_muted,
            _MUTED_STRATEGIES,
        )
        _MUTED_STRATEGIES.clear()

        count = await load_muted_strategies_from_db()

    assert count == 2
    assert is_strategy_muted("bad_strategy") is True
    assert is_strategy_muted("awful_strategy") is True
    assert is_strategy_muted("ok_strategy") is False


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
        from libs.monitoring.outcome_tracker import (
            load_muted_strategies_from_db,
            is_strategy_muted,
            _MUTED_STRATEGIES,
        )
        # Pre-populate with a stale entry
        _MUTED_STRATEGIES.clear()
        _MUTED_STRATEGIES.add("stale_strategy")

        await load_muted_strategies_from_db()

    assert is_strategy_muted("stale_strategy") is False
    assert is_strategy_muted("new_bad_strategy") is True


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
        from libs.monitoring.outcome_tracker import (
            load_muted_strategies_from_db,
            _MUTED_STRATEGIES,
        )
        _MUTED_STRATEGIES.clear()

        count = await load_muted_strategies_from_db()

    assert count == 0


@pytest.mark.asyncio
async def test_load_muted_handles_db_error():
    """DB raises an exception → returns 0, no propagation."""
    mock_factory = MagicMock(side_effect=RuntimeError("db down"))

    with patch(
        "libs.monitoring.outcome_tracker.get_session_factory",
        return_value=mock_factory,
    ):
        from libs.monitoring.outcome_tracker import load_muted_strategies_from_db

        count = await load_muted_strategies_from_db()

    assert count == 0


@pytest.mark.asyncio
async def test_is_strategy_muted_returns_false_for_unknown():
    """Unknown strategy name returns False without any DB call."""
    from libs.monitoring.outcome_tracker import is_strategy_muted, _MUTED_STRATEGIES

    _MUTED_STRATEGIES.clear()

    assert is_strategy_muted("completely_unknown_strategy") is False
