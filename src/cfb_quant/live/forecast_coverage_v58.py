"""Research-only daily FBS forecast coverage registry."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cfb_quant.live.daily_schedule_sync_v58 import (
    index_games, matchup_type, kickoff, select_window,
)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_archives(paths):
    found = {}
    sources = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path, dtype={"game_id": str})

        required = {
            "game_id", "kickoff_utc", "away", "home",
            "fair_home_spread", "fair_total",
            "blend_home_spread",
        }
        if not required <= set(frame):
            raise ValueError(f"Missing forecast columns: {path}")

        if frame.game_id.duplicated().any():
            raise ValueError(f"Duplicate forecast IDs: {path}")

        numbers = frame[[
            "fair_home_spread",
            "fair_total",
            "blend_home_spread"
        ]].apply(pd.to_numeric, errors="raise")

        import numpy as np
        if not np.isfinite(numbers.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite forecasts: {path}")

        sources.append({
            "path": path.as_posix(),
            "sha256": sha(path),
            "games": len(frame),
        })

        for row in frame.itertuples(index=False):
            gid = str(row.game_id)

            if gid in found:
                raise ValueError(
                    f"Game {gid} exists in multiple forecast archives"
                )

            saved_kickoff = pd.to_datetime(
                row.kickoff_utc, utc=True, errors="raise"
            )

            found[gid] = {
                "archive": path.as_posix(),
                "forecast_kickoff_utc": saved_kickoff,
                "away": str(row.away),
                "home": str(row.home),
            }

    return found, sources

def run(snapshot, archives, start, end, output):
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)
    if end < start:
        raise ValueError("End date is before start date")

    schedule_hash = sha(snapshot)
    games = index_games(
        json.loads(snapshot.read_text(encoding="utf-8"))
    )
    selected = select_window(games, start, end)
    forecasts, sources = load_archives(archives)

    # Fail closed: archive presence alone is not sufficient.
    # Reuse the audited v5.8 authenticity validator.
    from cfb_quant.live.forecast_authenticity_v58 import verify_all

    evidence = verify_all(Path.cwd().resolve(), archives)
    results = evidence["results"]

    if len(results) != len(archives):
        raise RuntimeError("Authenticity result count mismatch")

    for archive_path, item in zip(archives, results):
        if item["status"] != "GIT_PREKICKOFF_SUPPORTED":
            raise RuntimeError(
                "Forecast authenticity rejected: "
                f"{archive_path} | {item['errors']}"
            )

    sources_by_path = {
        Path(item["path"]).resolve(): item
        for item in sources
    }

    for archive_path, item in zip(archives, results):
        source = sources_by_path[Path(archive_path).resolve()]

        if item["csv_sha256"] != source["sha256"]:
            raise RuntimeError(
                f"Authenticity hash mismatch: {archive_path}"
            )

    records = []

    for gid, game in sorted(
        selected.items(),
        key=lambda item: (kickoff(item[1]), item[0])
    ):
        category = matchup_type(game)
        tbd = game.get("startTimeTBD") is True
        archived = forecasts.get(gid)

        scheduled_kickoff = kickoff(game)
        kickoff_difference = False

        if archived:
            kickoff_difference = (
                archived["forecast_kickoff_utc"].to_pydatetime()
                != scheduled_kickoff
            )

        if archived:
            # A matching game ID alone is never sufficient.
            # Verify both team identities against the schedule.
            expected_home = str(game["homeTeam"])
            expected_away = str(game["awayTeam"])
            actual_home = archived["home"]
            actual_away = archived["away"]

            if (
                actual_home != expected_home
                or actual_away != expected_away
            ):
                raise RuntimeError(
                    f"Archived team identity mismatch for game {gid}: "
                    f"expected {expected_away} @ {expected_home}; "
                    f"received {actual_away} @ {actual_home}"
                )

        if archived:
            status = "ARCHIVED"
        elif category != "FBS_FBS":
            status = "FBS_OPPONENT_DATA_ONLY"
        elif tbd:
            status = "KICKOFF_UNRESOLVED"
        elif game.get("completed") is True:
            status = "COMPLETED_WITHOUT_FORECAST"
        elif scheduled_kickoff <= datetime.now(timezone.utc):
            status = "STARTED_WITHOUT_FORECAST"
        else:
            status = "MISSING_PREGAME_FORECAST"

        records.append({
            "game_id": gid,
            "date_eastern": scheduled_kickoff.astimezone(
                __import__("zoneinfo").ZoneInfo("America/New_York")
            ).date().isoformat(),
            "kickoff_utc": scheduled_kickoff.isoformat(),
            "away": game["awayTeam"],
            "home": game["homeTeam"],
            "matchup_type": category,
            "start_time_tbd": tbd,
            "status": status,
            "archived_kickoff_mismatch": kickoff_difference,
            "forecast_archive": (
                archived["archive"] if archived else ""
            ),
        })

    registry = pd.DataFrame(records)

    if registry.empty:
        raise RuntimeError("No FBS-involving games in selected window")

    duplicates = registry.game_id.duplicated()
    if duplicates.any():
        raise RuntimeError("Duplicate games in coverage registry")

    selected_ids = set(registry.game_id)
    archived_ids = set(forecasts)

    # For this evaluation window, all supplied archives must belong
    # to the selected schedule. No silent extras are permitted.
    extras = archived_ids - selected_ids
    if extras:
        raise RuntimeError(
            f"Archived forecasts outside schedule/window: {sorted(extras)}"
        )

    if registry.archived_kickoff_mismatch.any():
        problematic = registry.loc[
            registry.archived_kickoff_mismatch,
            ["game_id", "away", "home"]
        ]
        raise RuntimeError(
            "Archived/scheduled kickoff mismatch:\n"
            + problematic.to_string(index=False)
        )

    if sha(snapshot) != schedule_hash:
        raise RuntimeError("Schedule snapshot changed during audit")

    destination = output / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    if destination.exists():
        raise FileExistsError(destination)

    destination.mkdir(parents=True)
    csv_path = destination / "coverage_registry.csv"
    registry.to_csv(csv_path, index=False)

    counts = Counter(registry.status)
    types = Counter(registry.matchup_type)

    audit = {
        "version": "forecast_coverage_v58_v1",
        "research_only": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "window_eastern": [start, end],
        "schedule_snapshot": snapshot.as_posix(),
        "schedule_sha256": schedule_hash,
        "forecast_sources": sources,
        "coverage_registry_sha256": sha(csv_path),
        "total_games": len(registry),
        "matchup_types": dict(types),
        "statuses": dict(counts),
        "all_fbs_games_accounted_for": True,
        "all_fbs_fbs_forecasted": (
            all(
                row["status"] == "ARCHIVED"
                for row in records
                if row["matchup_type"] == "FBS_FBS"
            )
        ),
        "model_run": False,
        "production_modified": False,
        "independent_schedule_completeness_certified": False,
    }

    (destination / "audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8"
    )

    print("=" * 76)
    print("STEP 146 — FBS FORECAST COVERAGE REGISTRY")
    print("=" * 76)
    print("Window:", start, "through", end)
    print("Total FBS-involving games:", len(registry))
    print("FBS-FBS:", types["FBS_FBS"])
    print("FBS-FCS:", types["FBS_FCS"])
    print("FBS-other:", types["FBS_OTHER"])
    print("Archived forecasts:", counts["ARCHIVED"])
    print("Missing pregame forecasts:",
          counts["MISSING_PREGAME_FORECAST"])
    print("Unresolved kickoffs:", counts["KICKOFF_UNRESOLVED"])
    print("FBS data-only games:", counts["FBS_OPPONENT_DATA_ONLY"])
    print("All FBS-FBS forecasted:",
          audit["all_fbs_fbs_forecasted"])
    print("Archive:", destination)

    incomplete = registry[
        registry.status != "ARCHIVED"
    ]
    if not incomplete.empty:
        print("\nGAMES REQUIRING ATTENTION:")
        print(incomplete[[
            "date_eastern", "game_id",
            "away", "home", "matchup_type", "status"
        ]].to_string(index=False))

    print("\nNo predictions or production files changed.")
    print("=" * 76)

    return audit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--archive", type=Path,
        action="append", required=True
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "reports/v58_development/forecast_coverage_v58"
        )
    )
    args = parser.parse_args()

    for value in (args.start, args.end):
        datetime.strptime(value, "%Y-%m-%d")

    run(
        args.snapshot, args.archive,
        args.start, args.end, args.output
    )

if __name__ == "__main__":
    main()
