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


def portable_fixture(tmp_path):
    from cfb_quant.live.approved_baseline_v58 import fingerprint

    source = schedule(tmp_path)
    directory = tmp_path / "portable"
    directory.mkdir()
    target = directory / "schedule.json"
    target.write_bytes(source.read_bytes())

    record = {
        "schema": "portable_baseline_v58_v1",
        "research_only": True,
        "season": 2026,
        "schedule_file": "schedule.json",
        "snapshot_sha256": fingerprint(target),
        "season_games": 1,
        "derived_from_original_approval": True,
        "auto_rotation_allowed": False,
    }

    registry = directory / "approval.json"
    registry.write_text(json.dumps(record), encoding="utf-8")
    return target, registry


def test_portable_approval_verifies(tmp_path):
    from cfb_quant.live.approved_baseline_v58 import verify_any

    target, registry = portable_fixture(tmp_path)
    verified, record = verify_any(registry, 2026)

    assert verified == target.resolve()
    assert record["season_games"] == 1


def test_portable_path_traversal_rejected(tmp_path):
    from cfb_quant.live.approved_baseline_v58 import verify_any

    target, registry = portable_fixture(tmp_path)
    record = json.loads(registry.read_text(encoding="utf-8"))
    record["schedule_file"] = "../outside.json"
    registry.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe portable schedule path"):
        verify_any(registry, 2026)


def test_portable_hash_tampering_rejected(tmp_path):
    from cfb_quant.live.approved_baseline_v58 import verify_any

    # The fixture creates schedule.json before calculating its SHA256.
    target, registry = portable_fixture(tmp_path)
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_any(registry, 2026)


def test_portable_wrong_season_rejected(tmp_path):
    from cfb_quant.live.approved_baseline_v58 import verify_any

    _, registry = portable_fixture(tmp_path)

    with pytest.raises(ValueError, match="season mismatch"):
        verify_any(registry, 2025)


def test_legacy_dispatch_still_works(tmp_path):
    from cfb_quant.live.approved_baseline_v58 import verify_any

    source = schedule(tmp_path)
    registry = tmp_path / "legacy.json"
    approve(source, registry, 2026, "Compatibility test")

    verified, _ = verify_any(registry, 2026)
    assert verified == source.resolve()
