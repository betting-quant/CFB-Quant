"""Offline readiness decision tests for CFB-Quant v5.8."""

import json

import pytest

from cfb_quant.live import daily_readiness_v58 as daily


def execute(tmp_path, monkeypatch, *, changes=None,
            added=None, removed=None, unresolved=None,
            statuses=None):

    baseline = tmp_path / "baseline.json"
    baseline.write_text("[]", encoding="utf-8")

    archive = tmp_path / "forecast.csv"
    archive.write_text("synthetic archive", encoding="utf-8")

    sync_audit = {
        "added_game_ids": added or [],
        "removed_game_ids": removed or [],
        "unresolved_kickoffs": unresolved or [],
        "changed_games": changes or [],
    }

    def fake_sync(season, start, end, baseline_path, output):
        destination = output / "synthetic_sync"
        destination.mkdir(parents=True)
        (destination / "cfbd_games.json").write_text(
            "[]", encoding="utf-8"
        )
        return sync_audit

    def fake_coverage(snapshot, archives, start, end, output):
        return {
            "statuses": statuses or {"ARCHIVED": 2},
            "total_games": 2,
            "all_fbs_fbs_forecasted": (
                (statuses or {"ARCHIVED": 2}).get("ARCHIVED", 0)
                == 2
            ),
        }

    monkeypatch.setattr(
        daily, "verify_approved",
        lambda _registry, _season: (
            baseline,
            {"snapshot_sha256": "synthetic-test-hash"},
        ),
    )
    monkeypatch.setattr(daily, "sync_schedule", fake_sync)
    monkeypatch.setattr(daily, "audit_coverage", fake_coverage)

    result = daily.run(
        2026,
        "2026-10-20",
        "2026-10-24",
        baseline,
        [archive],
        tmp_path / "output"
    )

    reports = list((tmp_path / "output").glob("*/readiness.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["ready"] == result["ready"]

    return result


def change(field):
    return [{
        "game_id": "401000001",
        "changes": {
            field: {"old": "BEFORE", "new": "AFTER"}
        },
    }]


def test_unchanged_schedule_is_ready(tmp_path, monkeypatch):
    result = execute(tmp_path, monkeypatch)
    assert result["ready"] is True
    assert result["schedule_alerts"] == 0


def test_changed_kickoff_blocks_readiness(tmp_path, monkeypatch):
    result = execute(
        tmp_path, monkeypatch,
        changes=change("startDate"),
    )
    assert result["ready"] is False
    assert result["schedule_alerts"] == 1


def test_changed_opponent_blocks_readiness(tmp_path, monkeypatch):
    result = execute(
        tmp_path, monkeypatch,
        changes=change("awayId"),
    )
    assert result["ready"] is False
    assert result["schedule_alerts"] == 1


def test_added_game_blocks_readiness(tmp_path, monkeypatch):
    result = execute(
        tmp_path, monkeypatch,
        added=["401000002"],
    )
    assert result["ready"] is False
    assert result["schedule_alerts"] == 1


def test_completion_update_not_schedule_alert(tmp_path, monkeypatch):
    result = execute(
        tmp_path, monkeypatch,
        changes=change("completed"),
    )
    assert result["ready"] is True
    assert result["schedule_alerts"] == 0


def test_missing_forecast_blocks_readiness(tmp_path, monkeypatch):
    result = execute(
        tmp_path, monkeypatch,
        statuses={
            "ARCHIVED": 1,
            "MISSING_PREGAME_FORECAST": 1,
        },
    )
    assert result["ready"] is False
    assert result["missing_or_unresolved_forecasts"] == 1


def test_unresolved_kickoff_blocks_readiness(tmp_path, monkeypatch):
    result = execute(
        tmp_path, monkeypatch,
        unresolved=[{"game_id": "401000001"}],
    )
    assert result["ready"] is False
    assert result["schedule_alerts"] == 1


def test_missing_approval_blocks_api(tmp_path, monkeypatch):
    def forbidden_sync(*args, **kwargs):
        raise AssertionError("API called before approval verification")

    monkeypatch.setattr(daily, "sync_schedule", forbidden_sync)

    with pytest.raises(FileNotFoundError):
        daily.run(
            2026, "2026-10-20", "2026-10-24",
            tmp_path / "missing.json", [], tmp_path / "output",
        )

    assert not (tmp_path / "output").exists()


def test_wrong_season_approval_blocks_api(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from cfb_quant.live.approved_baseline_v58 import approve

    snapshot = tmp_path / "schedule.json"
    snapshot.write_text(json.dumps([{
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
    }]), encoding="utf-8")

    registry = tmp_path / "approval.json"
    approve(snapshot, registry, 2026, "Test-only approval")

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("API called with wrong-season approval")

    monkeypatch.setattr(daily, "sync_schedule", forbidden_sync)

    with pytest.raises(ValueError, match="season mismatch"):
        daily.run(
            2025, "2026-10-20", "2026-10-24",
            registry, [], tmp_path / "output",
        )

    assert not (tmp_path / "output").exists()


def test_tampered_approval_snapshot_blocks_api(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from cfb_quant.live.approved_baseline_v58 import approve

    snapshot = tmp_path / "schedule.json"
    snapshot.write_text(json.dumps([{
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
    }]), encoding="utf-8")

    registry = tmp_path / "approval.json"
    approve(snapshot, registry, 2026, "Test-only approval")
    snapshot.write_text(
        snapshot.read_text(encoding="utf-8") + " ",
        encoding="utf-8"
    )

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("API called with tampered schedule")

    monkeypatch.setattr(daily, "sync_schedule", forbidden_sync)

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        daily.run(
            2026, "2026-10-20", "2026-10-24",
            registry, [], tmp_path / "output",
        )

    assert not (tmp_path / "output").exists()
