"""
ML signal classifier — learns which signals are actually correct.

Lifecycle:
  1. Outcome tracker records WIN/LOSS for each signal after 30 min.
  2. Once MIN_SAMPLES outcomes exist, train() is called automatically.
  3. predict(signal) returns a win-probability (0.0–1.0).
  4. Pipeline blocks signals where probability < BLOCK_THRESHOLD.
  5. Retrains every RETRAIN_EVERY new outcomes.

Model: GradientBoostingClassifier (handles small datasets, non-linear,
       no scaling needed, interpretable via feature_importances_).

Persistence: data/ml_model.pkl (joblib).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from libs.core.logging.logger import get_logger

if TYPE_CHECKING:
    from libs.core.models.domain import SignalOutput

log = get_logger(__name__)

MIN_SAMPLES      = 20      # need this many outcomes before model activates
RETRAIN_EVERY    = 10      # retrain after this many new outcomes
BLOCK_THRESHOLD  = 0.85    # block signal if model win-prob < this
MODEL_PATH       = Path("data/ml_model.pkl")


@dataclass
class ModelStats:
    trained:          bool  = False
    samples:          int   = 0
    accuracy:         float = 0.0
    precision:        float = 0.0
    win_rate:         float = 0.0        # raw win rate in training data
    last_trained_at:  str   = ""
    outcomes_since_retrain: int = 0
    feature_importances: dict[str, float] = field(default_factory=dict)
    blocked_last_hour:   int = 0


class SignalClassifier:
    """
    Gradient-boosted classifier trained on signal outcomes.
    Thread-safe: prediction uses a read lock; training uses a write lock.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock  = threading.RLock()
        self._stats = ModelStats()
        self._outcomes_since_retrain = 0
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_if_exists()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_win_prob(self, signal: "SignalOutput") -> float:
        """Return estimated win probability for this signal (0.0–1.0).
        Returns 1.0 (pass-through) when model not yet trained."""
        from libs.ml.features import extract
        with self._lock:
            if self._model is None:
                return 1.0
            X = np.array([extract(signal)], dtype=np.float32)
            prob = self._model.predict_proba(X)[0][1]  # class 1 = WIN
            return float(prob)

    def should_block(self, signal: "SignalOutput") -> bool:
        """True if model is trained and predicts low win probability."""
        if self._model is None:
            return False
        prob = self.predict_win_prob(signal)
        if prob < BLOCK_THRESHOLD:
            with self._lock:
                self._stats.blocked_last_hour += 1
            log.debug(
                "ml_signal_blocked",
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                win_prob=round(prob, 3),
                threshold=BLOCK_THRESHOLD,
            )
            return True
        return False

    def record_outcome(self, features: list[float], win: bool) -> None:
        """Register a new labeled outcome. Triggers retrain when threshold met."""
        with self._lock:
            self._outcomes_since_retrain += 1
            self._stats.outcomes_since_retrain = self._outcomes_since_retrain

        # Async retrain check — don't block caller
        if self._outcomes_since_retrain >= RETRAIN_EVERY or (
            self._model is None and self._get_total_outcomes() >= MIN_SAMPLES
        ):
            threading.Thread(target=self._retrain_async, daemon=True).start()

    def train(self, X: list[list[float]], y: list[int]) -> None:
        """Train (or retrain) the classifier on all accumulated outcomes."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import precision_score
        from libs.ml.features import FEATURE_NAMES

        if len(X) < MIN_SAMPLES:
            log.info("ml_train_skipped_insufficient_data", samples=len(X), needed=MIN_SAMPLES)
            return

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

        # Cross-val accuracy (3-fold, min)
        n_splits = min(3, len(X) // 5) if len(X) >= 10 else 2
        try:
            cv_scores = cross_val_score(model, X_arr, y_arr, cv=n_splits, scoring="accuracy")
            cv_acc = float(cv_scores.mean())
        except Exception:
            cv_acc = 0.0

        model.fit(X_arr, y_arr)

        # Precision on full training set (optimistic but useful for monitoring)
        y_pred = model.predict(X_arr)
        try:
            prec = float(precision_score(y_arr, y_pred, zero_division=0))
        except Exception:
            prec = 0.0

        importances = {
            FEATURE_NAMES[i]: round(float(v), 4)
            for i, v in enumerate(model.feature_importances_)
        }

        win_rate = float(y_arr.mean())

        with self._lock:
            self._model = model
            self._outcomes_since_retrain = 0
            self._stats = ModelStats(
                trained=True,
                samples=len(X),
                accuracy=round(cv_acc, 4),
                precision=round(prec, 4),
                win_rate=round(win_rate, 4),
                last_trained_at=datetime.now(timezone.utc).isoformat(),
                outcomes_since_retrain=0,
                feature_importances=importances,
                blocked_last_hour=self._stats.blocked_last_hour,
            )
            self._save()

        log.info(
            "ml_model_trained",
            samples=len(X),
            cv_accuracy=round(cv_acc, 3),
            precision=round(prec, 3),
            win_rate=round(win_rate, 3),
            top_feature=max(importances, key=importances.get),
        )

    @property
    def stats(self) -> ModelStats:
        with self._lock:
            return self._stats

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _retrain_async(self) -> None:
        """Load all outcomes from DB and retrain.

        Runs in a daemon thread — create a fresh event loop since
        asyncio.run() fails when a loop is already running.
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._load_and_retrain())
            finally:
                loop.close()
        except Exception as exc:
            log.warning("ml_retrain_failed", error=str(exc))

    async def _load_and_retrain(self) -> None:
        from libs.data.storage.db import get_session_factory
        from libs.data.storage.repository import OutcomeRepository

        async with get_session_factory()() as session:
            repo = OutcomeRepository(session)
            rows = await repo.get_resolved_with_features()

        if not rows:
            return

        X = [r["features"] for r in rows]
        y = [1 if r["win"] else 0 for r in rows]
        self.train(X, y)

    def _get_total_outcomes(self) -> int:
        """Quick sync count of resolved outcomes in DB."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._count_resolved())
            finally:
                loop.close()
        except Exception:
            return 0

    async def _count_resolved(self) -> int:
        from libs.data.storage.db import get_session_factory
        from libs.data.storage.repository import OutcomeRepository

        async with get_session_factory()() as session:
            repo = OutcomeRepository(session)
            rows = await repo.get_resolved_with_features()
        return len(rows)

    def _save(self) -> None:
        try:
            import joblib
            joblib.dump(self._model, MODEL_PATH)
            log.debug("ml_model_saved", path=str(MODEL_PATH))
        except Exception as exc:
            log.warning("ml_model_save_failed", error=str(exc))

    def _load_if_exists(self) -> None:
        if not MODEL_PATH.exists():
            return
        try:
            import joblib
            model = joblib.load(MODEL_PATH)
            # Verify loaded object is actually a classifier
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"Loaded object is not a classifier: {type(model)}")
            self._model = model
            self._stats.trained = True
            log.info("ml_model_loaded", path=str(MODEL_PATH))
        except Exception as exc:
            log.warning("ml_model_load_failed", error=str(exc))
            self._model = None


# ── Singleton ─────────────────────────────────────────────────────────────────

_classifier: SignalClassifier | None = None


def get_classifier() -> SignalClassifier:
    global _classifier
    if _classifier is None:
        _classifier = SignalClassifier()
    return _classifier
