# Bias Flip Auto-Exit Logic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the live bias flips against an open trade for 2 consecutive reanalysis checks, auto-close the position — with separate rules for SPOT and FUTURES.

**Architecture:** Add `bias_adverse_count` to VirtualTrade to track consecutive adverse bias checks. Update the existing `_trade_reanalysis_loop` in runner.py to (a) update `current_bias` + `bias_status` on each trade every 5 min, (b) increment/reset `bias_adverse_count`, (c) close only after 2 consecutive adverse checks. Use `direction` field (LONG/SHORT) instead of `action` (BUY/SELL) for proper spot/futures logic.

**Tech Stack:** Python 3.12+, dataclasses, existing BullBearBiasEngine

---

### Task 1: Add `bias_adverse_count` field to VirtualTrade

**Files:**
- Modify: `libs/paper_trading/portfolio.py:48` (VirtualTrade dataclass)
- Test: `tests/unit/test_market_mode_validator.py` (import check only)

- [ ] **Step 1: Add field to VirtualTrade**

In `libs/paper_trading/portfolio.py`, add after `risk_rejection_reason` field (line ~60):

```python
    bias_adverse_count: int = 0        # consecutive reanalysis checks where bias opposes direction
```

- [ ] **Step 2: Verify file compiles**

Run: `uv run python -c "from libs.paper_trading.portfolio import VirtualTrade; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add libs/paper_trading/portfolio.py
git commit -m "feat: add bias_adverse_count to VirtualTrade for 2-check exit rule"
```

---

### Task 2: Create bias exit logic module

**Files:**
- Create: `libs/paper_trading/bias_exit.py`
- Test: `tests/unit/test_bias_exit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_bias_exit.py`:

```python
"""Tests for bias flip auto-exit logic — document (4) section 9."""
import pytest
from libs.paper_trading.bias_exit import (
    compute_bias_status,
    should_exit_on_bias_flip,
    BIAS_ADVERSE_THRESHOLD,
)


class TestComputeBiasStatus:
    def test_long_bullish_is_aligned(self):
        assert compute_bias_status("LONG", "bullish") == "ALIGNED"

    def test_long_bearish_is_conflict(self):
        assert compute_bias_status("LONG", "bearish") == "CONFLICT"

    def test_long_neutral_is_warning(self):
        assert compute_bias_status("LONG", "neutral") == "WARNING"

    def test_short_bearish_is_aligned(self):
        assert compute_bias_status("SHORT", "bearish") == "ALIGNED"

    def test_short_bullish_is_conflict(self):
        assert compute_bias_status("SHORT", "bullish") == "CONFLICT"

    def test_short_neutral_is_warning(self):
        assert compute_bias_status("SHORT", "neutral") == "WARNING"

    def test_flat_anything_is_no_position(self):
        assert compute_bias_status("FLAT", "bullish") == "NO_POSITION"


class TestShouldExitOnBiasFlip:
    def test_spot_long_bearish_first_check_no_exit(self):
        """First adverse check: increment count, don't exit."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="bearish", bias_adverse_count=0,
        )
        assert exit_now is False
        assert new_count == 1

    def test_spot_long_bearish_second_check_exits(self):
        """Second consecutive adverse check: exit."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="bearish", bias_adverse_count=1,
        )
        assert exit_now is True
        assert new_count == 2

    def test_spot_long_bullish_resets_count(self):
        """Bias re-aligns: reset counter."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="bullish", bias_adverse_count=1,
        )
        assert exit_now is False
        assert new_count == 0

    def test_spot_long_neutral_increments(self):
        """SPOT neutral = no new position, but don't force exit until bearish."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="SPOT",
            current_bias="neutral", bias_adverse_count=0,
        )
        assert exit_now is False
        assert new_count == 0  # neutral doesn't increment for SPOT longs

    def test_futures_long_bearish_second_check_exits(self):
        """FUTURES LONG: bearish for 2 checks → CLOSE_LONG."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="LONG", market_mode="FUTURES",
            current_bias="bearish", bias_adverse_count=1,
        )
        assert exit_now is True
        assert new_count == 2

    def test_futures_short_bullish_second_check_exits(self):
        """FUTURES SHORT: bullish for 2 checks → CLOSE_SHORT."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="SHORT", market_mode="FUTURES",
            current_bias="bullish", bias_adverse_count=1,
        )
        assert exit_now is True
        assert new_count == 2

    def test_futures_short_bearish_resets(self):
        """FUTURES SHORT: bias aligns (bearish) → reset."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="SHORT", market_mode="FUTURES",
            current_bias="bearish", bias_adverse_count=1,
        )
        assert exit_now is False
        assert new_count == 0

    def test_futures_short_neutral_no_exit(self):
        """FUTURES neutral = manage only, no new position, no force exit."""
        exit_now, new_count = should_exit_on_bias_flip(
            direction="SHORT", market_mode="FUTURES",
            current_bias="neutral", bias_adverse_count=0,
        )
        assert exit_now is False
        assert new_count == 0

    def test_threshold_is_2(self):
        assert BIAS_ADVERSE_THRESHOLD == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_bias_exit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'libs.paper_trading.bias_exit'`

- [ ] **Step 3: Implement bias_exit.py**

Create `libs/paper_trading/bias_exit.py`:

```python
"""
Bias flip auto-exit logic — document (4) section 9.

Rules:
  SPOT LONG + bearish for 2 checks → CLOSE_LONG
  FUTURES LONG + bearish for 2 checks → CLOSE_LONG
  FUTURES SHORT + bullish for 2 checks → CLOSE_SHORT
  Neutral does not increment — only hard flips count.
  Bias re-alignment resets counter to 0.
"""
from __future__ import annotations

BIAS_ADVERSE_THRESHOLD: int = 2


def compute_bias_status(direction: str, current_bias: str) -> str:
    """
    Compute bias status for a trade.

    Returns: ALIGNED | WARNING | CONFLICT | NO_POSITION
    """
    direction = direction.upper()
    current_bias = current_bias.lower()

    if direction == "FLAT":
        return "NO_POSITION"

    if direction == "LONG":
        if current_bias == "bullish":
            return "ALIGNED"
        if current_bias == "neutral":
            return "WARNING"
        return "CONFLICT"

    if direction == "SHORT":
        if current_bias == "bearish":
            return "ALIGNED"
        if current_bias == "neutral":
            return "WARNING"
        return "CONFLICT"

    return "NO_POSITION"


def _is_adverse(direction: str, current_bias: str) -> bool:
    """Check if bias is directly opposed to direction (not neutral)."""
    direction = direction.upper()
    bias = current_bias.lower()

    if direction == "LONG" and bias == "bearish":
        return True
    if direction == "SHORT" and bias == "bullish":
        return True
    return False


def _is_aligned(direction: str, current_bias: str) -> bool:
    """Check if bias supports the direction."""
    direction = direction.upper()
    bias = current_bias.lower()

    if direction == "LONG" and bias == "bullish":
        return True
    if direction == "SHORT" and bias == "bearish":
        return True
    return False


def should_exit_on_bias_flip(
    direction: str,
    market_mode: str,
    current_bias: str,
    bias_adverse_count: int,
) -> tuple[bool, int]:
    """
    Determine if a trade should be closed due to bias flip.

    Args:
        direction: LONG | SHORT | FLAT
        market_mode: SPOT | FUTURES
        current_bias: bullish | bearish | neutral
        bias_adverse_count: how many consecutive adverse checks so far

    Returns:
        (should_exit, new_bias_adverse_count)

    Rules per document (4) section 9:
      - Only hard flips (bearish for long, bullish for short) increment
      - Neutral does NOT increment — manage only, no forced exit
      - Aligned bias resets counter to 0
      - Exit after BIAS_ADVERSE_THRESHOLD consecutive adverse checks
    """
    if direction.upper() == "FLAT":
        return False, 0

    if _is_adverse(direction, current_bias):
        new_count = bias_adverse_count + 1
        if new_count >= BIAS_ADVERSE_THRESHOLD:
            return True, new_count
        return False, new_count

    if _is_aligned(direction, current_bias):
        return False, 0

    # Neutral — don't increment, don't reset
    return False, bias_adverse_count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_bias_exit.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add libs/paper_trading/bias_exit.py tests/unit/test_bias_exit.py
git commit -m "feat: bias flip exit logic with 2-check threshold per doc (4) section 9"
```

---

### Task 3: Wire bias exit into the reanalysis loop

**Files:**
- Modify: `apps/signal_agent/runner.py:358-382` (_trade_reanalysis_loop)

- [ ] **Step 1: Update the reanalysis loop**

Replace the block in `apps/signal_agent/runner.py` from line 358 (`# Check if bias flipped...`) through line 382 (`log.info("trade_reanalysis_closed"...`) with:

```python
                            # Update bias status + check for 2-check exit on all positions
                            from libs.paper_trading.bias_exit import (
                                compute_bias_status,
                                should_exit_on_bias_flip,
                            )

                            for bot in self._paper_engine.bots:
                                for trade in list(bot.portfolio.open_trades):
                                    if trade.symbol != sym:
                                        continue

                                    # Update live bias fields on trade
                                    trade.current_bias = new_bias.net_bias
                                    trade.bias_status = compute_bias_status(
                                        trade.direction, new_bias.net_bias,
                                    )

                                    # 2-check exit rule
                                    should_close, new_count = should_exit_on_bias_flip(
                                        direction=trade.direction,
                                        market_mode=trade.market_mode,
                                        current_bias=new_bias.net_bias,
                                        bias_adverse_count=trade.bias_adverse_count,
                                    )
                                    trade.bias_adverse_count = new_count

                                    if should_close:
                                        price = await provider.get_latest_price(sym)
                                        if price:
                                            reason = (
                                                f"Bias flipped to {new_bias.net_bias.upper()} "
                                                f"for {new_count} checks — closing {trade.direction}"
                                            )
                                            trade.exit_reason = reason
                                            result = bot.portfolio.close_trade(
                                                trade.id, price, f"BIAS_FLIP: {reason}",
                                            )
                                            log.info("trade_bias_flip_closed",
                                                     bot=bot.name, symbol=sym,
                                                     direction=trade.direction,
                                                     market_mode=trade.market_mode,
                                                     adverse_checks=new_count,
                                                     new_bias=new_bias.net_bias,
                                                     pnl=result.get("realized_pnl", 0))
                                    elif trade.bias_status == "CONFLICT":
                                        log.info("trade_bias_conflict_warning",
                                                 bot=bot.name, symbol=sym,
                                                 direction=trade.direction,
                                                 adverse_count=new_count,
                                                 bias=new_bias.net_bias)
```

- [ ] **Step 2: Verify file compiles**

Run: `uv run python -c "import py_compile; py_compile.compile('apps/signal_agent/runner.py', doraise=True); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add apps/signal_agent/runner.py
git commit -m "feat: wire 2-check bias flip exit into reanalysis loop"
```

---

### Task 4: Expose bias_adverse_count in API + dashboard

**Files:**
- Modify: `libs/paper_trading/engine.py:170-180` (get_all_open_positions)
- Modify: `apps/dashboard/paper_page.py` (positions table tooltip)

- [ ] **Step 1: Add field to position serialization**

In `libs/paper_trading/engine.py`, in the `get_all_open_positions` method, add after the `"current_bias"` line:

```python
                    "bias_adverse_count": getattr(trade, 'bias_adverse_count', 0),
```

- [ ] **Step 2: Update dashboard bias status column to show adverse count**

In `apps/dashboard/paper_page.py`, in the `renderPositions` function, update the bias status cell to include the count:

Replace the bias_status `<td>`:
```javascript
      <td style="font-size:0.68rem;font-weight:bold;color:${bsClr}">${biasStatus.toUpperCase()}</td>
```

With:
```javascript
      <td style="font-size:0.68rem;font-weight:bold;color:${bsClr}" title="Adverse checks: ${p.bias_adverse_count || 0}/2">${biasStatus.toUpperCase()}${(p.bias_adverse_count || 0) > 0 ? ' (' + p.bias_adverse_count + '/2)' : ''}</td>
```

- [ ] **Step 3: Verify both files compile**

Run: `uv run python -c "import py_compile; py_compile.compile('libs/paper_trading/engine.py', doraise=True); py_compile.compile('apps/dashboard/paper_page.py', doraise=True); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add libs/paper_trading/engine.py apps/dashboard/paper_page.py
git commit -m "feat: expose bias_adverse_count in API + dashboard with visual indicator"
```

---

### Task 5: Run full test suite and push

- [ ] **Step 1: Run all bias-related tests**

Run: `uv run python -m pytest tests/unit/test_bias_exit.py tests/unit/test_market_mode_validator.py -v`
Expected: All pass (14 + 27 = 41 tests)

- [ ] **Step 2: Push**

```bash
git push origin stage3-institutional-upgrades
```
