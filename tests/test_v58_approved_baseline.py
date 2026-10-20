"""Offline tests of explicit baseline approval."""

import json
from datetime import datetime, timezone

import pytest

from cfb_quant.live.approved_baseline_v58 import approve, verify


def schedule(tmp_path):
    snapshot = tmp_path / "games.json"
    game = {
        "id": 401000001,
        "season": 2026,
        "startDate": datetime(
            2026, 10, 24, 17, tzinfo=timezone.utc
        ).isoformat(),
        "homeId": 1,
        "awayId": 2,
        "homeTeam": "Kansas",
        "awayTeam": "Baylor",
        "homeClassification": "fbs",
        "awayClassification": "fbs",
        "startTimeTBD": False,
    }
    snapshot.write_text(json.dumps([game]), encoding="utf-8")
    return snapshot


def test_approval_verifies(tmp_path):
    source = schedule(tmp_path)
    registry = tmp_path / "approved.json"
    approve(source, registry, 2026, "Synthetic test approval")
    verified, record = verify(registry, 2026)
    assert verified == source.resolve()
    assert record["auto_rotation_allowed"] is False


def test_reapproval_refused(tmp_path):
    source = schedule(tmp_path)
    registry = tmp_path / "approved.json"
    approve(source, registry, 2026, "First")
    with pytest.raises(FileExistsError):
        approve(source, registry, 2026, "Second")


def test_tampered_schedule_rejected(tmp_path):
    source = schedule(tmp_path)
    registry = tmp_path / "approved.json"
    approve(source, registry, 2026, "Test")
    source.write_text(source.read_text() + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify(registry, 2026)


def test_wrong_season_rejected(tmp_path):
    source = schedule(tmp_path)
    registry = tmp_path / "approved.json"
    approve(source, registry, 2026, "Test")
    with pytest.raises(ValueError, match="season mismatch"):
        verify(registry, 2025)


def test_blank_reason_rejected(tmp_path):
    source = schedule(tmp_path)
    with pytest.raises(ValueError, match="reason"):
        approve(source, tmp_path / "approved.json", 2026, "   ")
