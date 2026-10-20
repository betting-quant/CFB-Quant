"""Contract tests for daily readiness JSON reports."""

import json

from cfb_quant.live import daily_readiness_v58 as daily


def generate(tmp_path, monkeypatch, statuses, alerts=None):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("[]", encoding="utf-8")

    archive = tmp_path / "forecast.csv"
    archive.write_text("synthetic", encoding="utf-8")

    monkeypatch.setattr(
        daily,
        "verify_approved",
        lambda _registry, _season: (
            baseline,
            {"snapshot_sha256": "test-hash"},
        ),
    )

    def fake_sync(season, start, end, baseline_path, output):
        folder = output / "synthetic_sync"
        folder.mkdir(parents=True)
        (folder / "cfbd_games.json").write_text(
            "[]", encoding="utf-8"
        )
        return {
            "added_game_ids": [],
            "removed_game_ids": [],
            "unresolved_kickoffs": [],
            "changed_games": alerts or [],
        }

    def fake_coverage(snapshot, archives, start, end, output):
        return {
            "statuses": statuses,
            "total_games": 2,
            "all_fbs_fbs_forecasted": statuses.get(
                "ARCHIVED", 0
            ) == 2,
        }

    monkeypatch.setattr(daily, "sync_schedule", fake_sync)
    monkeypatch.setattr(daily, "audit_coverage", fake_coverage)

    result = daily.run(
        2026,
        "2026-10-20",
        "2026-10-24",
        baseline,
        [archive],
        tmp_path / "output",
    )

    reports = list((tmp_path / "output").glob("*/readiness.json"))
    assert len(reports) == 1

    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved == result
    return saved


def test_successful_report_schema(tmp_path, monkeypatch):
    report = generate(
        tmp_path, monkeypatch, {"ARCHIVED": 2}
    )

    required = {
        "version",
        "created_utc",
        "research_only",
        "approval_registry",
        "approved_baseline_sha256",
        "window_eastern",
        "schedule_snapshot",
        "scheduled_games",
        "statuses",
        "schedule_alerts",
        "missing_or_unresolved_forecasts",
        "unsupported_fbs_opponent_games",
        "all_fbs_fbs_forecasted",
        "ready",
        "model_executed",
        "production_modified",
        "historical_pit_certified",
        "external_publication_certified",
    }

    assert required <= report.keys()
    assert report["version"] == "daily_readiness_v58_v1"
    assert report["statuses"]["ARCHIVED"] == 2
    assert report["scheduled_games"] == 2
    assert report["schedule_alerts"] == 0
    assert report["ready"] is True
    assert report["model_executed"] is False
    assert report["production_modified"] is False
    assert report["historical_pit_certified"] is False
    assert report["external_publication_certified"] is False

    # Prevent consumers from assuming these nonexistent fields.
    assert "archived_forecasts" not in report
    assert "unsupported_fbs_opponents" not in report


def test_missing_forecast_fails_report(tmp_path, monkeypatch):
    report = generate(
        tmp_path,
        monkeypatch,
        {
            "ARCHIVED": 1,
            "MISSING_PREGAME_FORECAST": 1,
        },
    )

    assert report["ready"] is False
    assert report["statuses"]["ARCHIVED"] == 1
    assert report["missing_or_unresolved_forecasts"] == 1


def test_schedule_change_fails_report(tmp_path, monkeypatch):
    report = generate(
        tmp_path,
        monkeypatch,
        {"ARCHIVED": 2},
        alerts=[{
            "game_id": "401000001",
            "changes": {
                "startDate": {
                    "old": "BEFORE",
                    "new": "AFTER",
                }
            },
        }],
    )

    assert report["ready"] is False
    assert report["schedule_alerts"] == 1
    assert report["statuses"]["ARCHIVED"] == 2
