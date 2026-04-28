from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from libs.audit.audit_log import AuditEvent, AuditLog
from libs.core.models.domain import AssetClass


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(signal_id=None, event_type: str = "data_fetched") -> AuditEvent:
    return AuditEvent(
        signal_id=signal_id,
        event_type=event_type,
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        payload={"key": "value"},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_record_writes_jsonl_file(tmp_path):
    """record() writes a JSONL file when enabled."""
    audit = AuditLog(log_dir=str(tmp_path), enabled=True)
    event = make_event()
    await audit.record(event)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_file = tmp_path / f"audit-{date_str}.jsonl"
    assert jsonl_file.exists()
    content = jsonl_file.read_text(encoding="utf-8")
    assert "data_fetched" in content


async def test_get_signal_trail_returns_matching_events(tmp_path):
    """get_signal_trail returns only events matching signal_id."""
    audit = AuditLog(log_dir=str(tmp_path), enabled=True)
    sid = uuid4()
    other_sid = uuid4()

    await audit.record(make_event(signal_id=sid, event_type="target_event"))
    await audit.record(make_event(signal_id=other_sid, event_type="other_event"))

    trail = await audit.get_signal_trail(sid)
    assert len(trail) == 1
    assert trail[0].event_type == "target_event"


async def test_disabled_creates_no_file(tmp_path):
    """enabled=False → no file created."""
    audit = AuditLog(log_dir=str(tmp_path), enabled=False)
    event = make_event()
    await audit.record(event)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_file = tmp_path / f"audit-{date_str}.jsonl"
    assert not jsonl_file.exists()


async def test_multiple_events_for_same_signal_id(tmp_path):
    """Multiple events for same signal_id are returned in write order."""
    audit = AuditLog(log_dir=str(tmp_path), enabled=True)
    sid = uuid4()

    await audit.record(make_event(signal_id=sid, event_type="step_one"))
    await audit.record(make_event(signal_id=sid, event_type="step_two"))
    await audit.record(make_event(signal_id=sid, event_type="step_three"))

    trail = await audit.get_signal_trail(sid)
    assert len(trail) == 3
    event_types = [e.event_type for e in trail]
    assert event_types == ["step_one", "step_two", "step_three"]


async def test_corrupted_line_skipped_no_exception(tmp_path):
    """Corrupted line in JSONL → skipped, no exception raised."""
    audit = AuditLog(log_dir=str(tmp_path), enabled=True)
    sid = uuid4()

    # Write a valid event first
    await audit.record(make_event(signal_id=sid, event_type="valid_event"))

    # Manually corrupt the file by prepending a bad line
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    jsonl_file = tmp_path / f"audit-{date_str}.jsonl"
    good_content = jsonl_file.read_text(encoding="utf-8")
    jsonl_file.write_text("{{CORRUPTED_JSON}}\n" + good_content, encoding="utf-8")

    # Should not raise
    trail = await audit.get_signal_trail(sid)
    # Valid event should still be returned
    assert len(trail) == 1
    assert trail[0].event_type == "valid_event"


async def test_get_signal_trail_no_file_returns_empty(tmp_path):
    """get_signal_trail returns [] when JSONL file does not exist."""
    audit = AuditLog(log_dir=str(tmp_path), enabled=True)
    trail = await audit.get_signal_trail(uuid4())
    assert trail == []
