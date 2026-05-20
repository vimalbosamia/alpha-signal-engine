"""
Trailing stop manager for open positions.

Tracks the highest/lowest price seen since entry and updates the stop-loss
level as price moves favorably.  Stop only ever moves in the direction that
reduces risk — it never widens.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrailingStopState:
    entry_price: float
    original_stop: float
    current_stop: float
    highest_price: float      # for BUY trades — tracks peak seen
    lowest_price: float       # for SELL trades — tracks trough seen
    trail_distance_pct: float
    activated: bool           # True once price moved far enough from entry


class TrailingStopManager:
    """
    Manages trailing stops for BUY and SELL positions.

    Args:
        trail_pct:      Distance behind the peak/trough at which the stop
                        is placed (as a percentage of the peak/trough price).
        activation_pct: Minimum favourable move (% from original stop toward
                        entry) required before the trail activates.
    """

    def __init__(self, trail_pct: float = 1.0, activation_pct: float = 0.5) -> None:
        self._trail_pct = trail_pct
        self._activation_pct = activation_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        entry_price: float,
        original_stop: float,
        action: str,
    ) -> TrailingStopState:
        """
        Create the initial TrailingStopState for a new position.

        Args:
            entry_price:    Price at which the trade was entered.
            original_stop:  Initial hard stop-loss price.
            action:         'BUY' or 'SELL' (case-insensitive).

        Returns:
            A fresh TrailingStopState (not yet activated).
        """
        return TrailingStopState(
            entry_price=entry_price,
            original_stop=original_stop,
            current_stop=original_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            trail_distance_pct=self._trail_pct,
            activated=False,
        )

    def update(
        self,
        state: TrailingStopState,
        current_price: float,
        action: str,
    ) -> TrailingStopState:
        """
        Update trailing stop given the latest price tick.

        For BUY positions:
            - Track the highest price seen.
            - Once profit exceeds activation_pct (measured from original stop
              to the new high), set stop = highest * (1 - trail_pct / 100).
            - The stop only moves up — never down.

        For SELL positions:
            - Track the lowest price seen.
            - Once profit exceeds activation_pct (measured from original stop
              down to the new low), set stop = lowest * (1 + trail_pct / 100).
            - The stop only moves down — never up.

        Args:
            state:         Current TrailingStopState.
            current_price: Latest market price.
            action:        'BUY' or 'SELL' (case-insensitive).

        Returns:
            New TrailingStopState (original state is never mutated).
        """
        action_upper = action.upper()

        if action_upper == "BUY":
            return self._update_buy(state, current_price)
        return self._update_sell(state, current_price)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_buy(
        self, state: TrailingStopState, current_price: float
    ) -> TrailingStopState:
        new_highest = max(state.highest_price, current_price)

        # Check activation: price must have moved activation_pct% above
        # the original stop before the trail kicks in.
        # Activation: price must move activation_pct% above entry before trail kicks in
        profit_pct = (new_highest - state.entry_price) / state.entry_price * 100.0 if state.entry_price > 0 else 0.0
        activated = state.activated or profit_pct >= self._activation_pct

        if activated:
            candidate_stop = new_highest * (1.0 - self._trail_pct / 100.0)
            # Stop can only move up — never widen
            new_stop = max(state.current_stop, candidate_stop)
        else:
            new_stop = state.current_stop

        return TrailingStopState(
            entry_price=state.entry_price,
            original_stop=state.original_stop,
            current_stop=new_stop,
            highest_price=new_highest,
            lowest_price=state.lowest_price,
            trail_distance_pct=state.trail_distance_pct,
            activated=activated,
        )

    def _update_sell(
        self, state: TrailingStopState, current_price: float
    ) -> TrailingStopState:
        new_lowest = min(state.lowest_price, current_price)

        # Check activation: price must have moved activation_pct% below
        # the original stop before the trail kicks in.
        profit_from_stop_pct = (
            (state.original_stop - new_lowest) / state.original_stop * 100.0
        )
        activated = state.activated or profit_from_stop_pct >= self._activation_pct

        if activated:
            candidate_stop = new_lowest * (1.0 + self._trail_pct / 100.0)
            # Stop can only move down — never widen
            new_stop = min(state.current_stop, candidate_stop)
        else:
            new_stop = state.current_stop

        return TrailingStopState(
            entry_price=state.entry_price,
            original_stop=state.original_stop,
            current_stop=new_stop,
            highest_price=state.highest_price,
            lowest_price=new_lowest,
            trail_distance_pct=state.trail_distance_pct,
            activated=activated,
        )
