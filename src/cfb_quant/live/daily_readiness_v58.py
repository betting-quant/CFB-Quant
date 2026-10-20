"""Research-only daily schedule and forecast readiness pipeline."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from cfb_quant.live.daily_schedule_sync_v58 import run as sync_schedule
from cfb_quant.live.forecast_coverage_v58 import run as audit_coverage
from cfb_quant.live.approved_baseline_v58 import verify as verify_approved


def run(season, start, end, approval_registry, archives, output):
    # Reject invalid approvals before contacting CFBD.
    baseline, approval = verify_approved(approval_registry, season)
    if not archives:
        raise ValueError("At least one forecast archive is required")

    for path in [*archives]:
        if not path.is_file():
            raise FileNotFoundError(path)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    root = output / stamp

    if root.exists():
        raise FileExistsError(root)

    schedule_dir = root / "schedule"

    print("=" * 78)
    print("CFB-QUANT v5.8 — DAILY READINESS PREFLIGHT")
    print("=" * 78)

    sync_audit = sync_schedule(
        season, start, end, baseline, schedule_dir
    )

    snapshots = list(schedule_dir.glob("*/cfbd_games.json"))
    if len(snapshots) != 1:
        raise RuntimeError("Expected exactly one schedule snapshot")

    snapshot = snapshots[0]

    # Only actionable schedule changes affect readiness.
    # A completed-status update is not itself a schedule change.
    critical_fields = {
        "startDate",
        "startTimeTBD",
        "homeId",
        "awayId",
        "homeTeam",
        "awayTeam",
        "homeClassification",
        "awayClassification",
    }

    critical_changes = [
        item for item in sync_audit["changed_games"]
        if critical_fields.intersection(item["changes"])
    ]

    schedule_alerts = (
        len(sync_audit["added_game_ids"])
        + len(sync_audit["removed_game_ids"])
        + len(sync_audit["unresolved_kickoffs"])
        + len(critical_changes)
    )

    # Preserve schedule findings even if forecast validation fails.
    # The coverage registry itself fails closed for invalid evidence.
    coverage = audit_coverage(
        snapshot, archives, start, end, root / "coverage"
    )

    counts = coverage["statuses"]

    missing = sum(
        counts.get(status, 0)
        for status in (
            "MISSING_PREGAME_FORECAST",
            "KICKOFF_UNRESOLVED",
            "COMPLETED_WITHOUT_FORECAST",
            "STARTED_WITHOUT_FORECAST",
        )
    )

    unsupported = counts.get("FBS_OPPONENT_DATA_ONLY", 0)

    ready = (
        schedule_alerts == 0
        and missing == 0
        and unsupported == 0
        and coverage["all_fbs_fbs_forecasted"]
    )

    summary = {
        "version": "daily_readiness_v58_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "approval_registry": str(Path(approval_registry).resolve()),
        "approved_baseline_sha256": approval["snapshot_sha256"],
        "window_eastern": [start, end],
        "schedule_snapshot": snapshot.as_posix(),
        "scheduled_games": coverage["total_games"],
        "statuses": counts,
        "schedule_alerts": schedule_alerts,
        "missing_or_unresolved_forecasts": missing,
        "unsupported_fbs_opponent_games": unsupported,
        "all_fbs_fbs_forecasted": (
            coverage["all_fbs_fbs_forecasted"]
        ),
        "ready": bool(ready),
        "model_executed": False,
        "production_modified": False,
        "historical_pit_certified": False,
        "external_publication_certified": False,
    }

    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "readiness.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("DAILY READINESS SUMMARY")
    print("=" * 78)
    print("Scheduled games:", summary["scheduled_games"])
    print("Archived forecasts:", counts.get("ARCHIVED", 0))
    print("Schedule alerts:", schedule_alerts)
    print("Missing/unresolved:", missing)
    print("Unsupported FBS opponents:", unsupported)
    print("Research readiness:", "PASS" if ready else "ATTENTION")
    print("Summary:", summary_path)
    print("Production unchanged.")
    print("=" * 78)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path,
        action="append", required=True
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/v58_development/daily_readiness_v58")
    )
    args = parser.parse_args()

    for value in (args.start, args.end):
        datetime.strptime(value, "%Y-%m-%d")

    result = run(
        args.season, args.start, args.end,
        args.approval, args.archive, args.output
    )

    if not result["ready"]:
        raise SystemExit(
            "Readiness ATTENTION: inspect archived audit before forecasting."
        )


if __name__ == "__main__":
    main()
