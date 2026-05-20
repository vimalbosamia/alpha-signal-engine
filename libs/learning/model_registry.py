"""
libs.learning.model_registry — Model versioning registry for strategy weights.

Tracks registered model versions per strategy, supports activation, comparison,
and rollback without mutating existing version records.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ModelVersion:
    version_id: str
    strategy_name: str
    weights: dict[str, float]
    win_rate: float
    sharpe_ratio: float
    trade_count: int
    created_at: str
    is_active: bool = False


_MIN_TRADES_TO_ACTIVATE = 20
_WIN_RATE_REGRESSION_TOLERANCE = 0.05
_SHARPE_REGRESSION_TOLERANCE = 0.20


class ModelVersionRegistry:
    """
    Registry that stores model versions per strategy and manages activation.

    Rules enforced by compare():
    - New model must have >= 20 trades.
    - New win_rate must be > active win_rate - 0.05 (allow small regression).
    - New sharpe must be > active sharpe - 0.20.
    - Both conditions must hold for should_switch=True.
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[ModelVersion]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(
        self,
        strategy_name: str,
        weights: dict[str, float],
        win_rate: float,
        sharpe_ratio: float,
        trade_count: int,
    ) -> ModelVersion:
        """Register a new model version. Does NOT activate it."""
        version = ModelVersion(
            version_id=str(uuid.uuid4()),
            strategy_name=strategy_name,
            weights=dict(weights),
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            trade_count=trade_count,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            is_active=False,
        )
        self._versions.setdefault(strategy_name, []).append(version)
        return version

    def compare(self, strategy_name: str) -> dict:
        """
        Compare the latest registered version against the currently active one.

        Returns a dict with keys:
            should_switch (bool), reason (str),
            latest (ModelVersion | None), active (ModelVersion | None)
        """
        versions = self._versions.get(strategy_name, [])
        active = self._find_active(versions)
        latest = versions[-1] if versions else None

        if latest is None:
            return {
                "should_switch": False,
                "reason": "No versions registered for strategy.",
                "latest": None,
                "active": active,
            }

        if active is None:
            return {
                "should_switch": False,
                "reason": "No active version to compare against.",
                "latest": latest,
                "active": None,
            }

        if latest.version_id == active.version_id:
            return {
                "should_switch": False,
                "reason": "Latest version is already the active version.",
                "latest": latest,
                "active": active,
            }

        if latest.trade_count < _MIN_TRADES_TO_ACTIVATE:
            return {
                "should_switch": False,
                "reason": (
                    f"Latest version has only {latest.trade_count} trades "
                    f"(minimum {_MIN_TRADES_TO_ACTIVATE} required)."
                ),
                "latest": latest,
                "active": active,
            }

        win_rate_ok = latest.win_rate > active.win_rate - _WIN_RATE_REGRESSION_TOLERANCE
        sharpe_ok = latest.sharpe_ratio > active.sharpe_ratio - _SHARPE_REGRESSION_TOLERANCE

        if win_rate_ok and sharpe_ok:
            return {
                "should_switch": True,
                "reason": (
                    f"Latest model meets quality thresholds "
                    f"(win_rate={latest.win_rate:.3f} vs active={active.win_rate:.3f}, "
                    f"sharpe={latest.sharpe_ratio:.3f} vs active={active.sharpe_ratio:.3f})."
                ),
                "latest": latest,
                "active": active,
            }

        reasons: list[str] = []
        if not win_rate_ok:
            reasons.append(
                f"win_rate regression too large "
                f"({latest.win_rate:.3f} vs active={active.win_rate:.3f}, "
                f"tolerance={_WIN_RATE_REGRESSION_TOLERANCE})"
            )
        if not sharpe_ok:
            reasons.append(
                f"sharpe regression too large "
                f"({latest.sharpe_ratio:.3f} vs active={active.sharpe_ratio:.3f}, "
                f"tolerance={_SHARPE_REGRESSION_TOLERANCE})"
            )

        return {
            "should_switch": False,
            "reason": "; ".join(reasons),
            "latest": latest,
            "active": active,
        }

    def activate(self, strategy_name: str, version_id: str) -> bool:
        """
        Activate a specific version by version_id.

        Deactivates the currently active version first.
        Returns True if the version was found and activated, False otherwise.
        """
        versions = self._versions.get(strategy_name, [])
        target_idx = next(
            (i for i, v in enumerate(versions) if v.version_id == version_id), None
        )
        if target_idx is None:
            return False

        # Deactivate all (immutable-style replacement), then activate target.
        for i, v in enumerate(versions):
            if v.is_active:
                versions[i] = _with_active(v, is_active=False)

        versions[target_idx] = _with_active(versions[target_idx], is_active=True)
        return True

    def get_active(self, strategy_name: str) -> ModelVersion | None:
        """Return the currently active version, or None if none is active."""
        versions = self._versions.get(strategy_name, [])
        return self._find_active(versions)

    def rollback(self, strategy_name: str) -> bool:
        """
        Rollback to the previous active version.

        Scans versions in reverse order for the first inactive version that
        preceded the current active one.  Returns True on success.
        """
        versions = self._versions.get(strategy_name, [])
        active_idx = next(
            (i for i, v in enumerate(versions) if v.is_active), None
        )
        if active_idx is None:
            return False

        # Find the most recent version before active_idx that was previously
        # considered (any version before it in registration order).
        previous_idx = active_idx - 1
        if previous_idx < 0:
            return False

        # Deactivate current active.
        versions[active_idx] = _with_active(versions[active_idx], is_active=False)
        # Activate previous.
        versions[previous_idx] = _with_active(versions[previous_idx], is_active=True)
        return True

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_active(versions: list[ModelVersion]) -> ModelVersion | None:
        return next((v for v in versions if v.is_active), None)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _with_active(version: ModelVersion, *, is_active: bool) -> ModelVersion:
    """Return a new ModelVersion identical to *version* but with is_active set."""
    return ModelVersion(
        version_id=version.version_id,
        strategy_name=version.strategy_name,
        weights=version.weights,
        win_rate=version.win_rate,
        sharpe_ratio=version.sharpe_ratio,
        trade_count=version.trade_count,
        created_at=version.created_at,
        is_active=is_active,
    )
