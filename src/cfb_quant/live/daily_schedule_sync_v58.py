"""Research-only CFBD schedule synchronization and coverage audit."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from cfb_quant.data.download_history import cfbd_get

EASTERN = ZoneInfo("America/New_York")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_games(games):
    if not isinstance(games, list):
        raise ValueError("Schedule must be a JSON list")
    indexed = {}
    for game in games:
        key = str(int(game["id"]))
        if key in indexed:
            raise ValueError(f"Duplicate game ID: {key}")
        indexed[key] = game
    return indexed


def matchup_type(game):
    h = str(game.get("homeClassification") or "").lower()
    a = str(game.get("awayClassification") or "").lower()
    if h == "fbs" and a == "fbs":
        return "FBS_FBS"
    if {h, a} == {"fbs", "fcs"}:
        return "FBS_FCS"
    if "fbs" in (h, a):
        return "FBS_OTHER"
    return "NON_FBS"


def kickoff(game):
    value = str(game["startDate"])
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Naive kickoff for game {game['id']}")
    return result.astimezone(timezone.utc)


def select_window(indexed, start, end):
    selected = {}
    for gid, game in indexed.items():
        if matchup_type(game) == "NON_FBS":
            continue
        day = kickoff(game).astimezone(EASTERN).date().isoformat()
        if start <= day <= end:
            selected[gid] = game
    return selected


def validate(games, season, baseline_count):
    indexed = index_games(games)
    minimum = max(1, int(baseline_count * 0.90))
    if len(indexed) < minimum:
        raise ValueError(
            f"Schedule unexpectedly shrank: {len(indexed)} < {minimum}"
        )
    for gid, game in indexed.items():
        if int(game.get("season") or -1) != season:
            raise ValueError(f"Incorrect season: {gid}")
        for field in (
            "startDate", "homeId", "awayId", "homeTeam",
            "awayTeam", "homeClassification", "awayClassification"
        ):
            if game.get(field) is None:
                raise ValueError(f"{gid}: missing {field}")
        kickoff(game)
    return indexed


def run(season, start, end, baseline, output, *, capture_http=False):
    if end < start:
        raise ValueError("End date precedes start date")
    if not baseline.is_file():
        raise FileNotFoundError(baseline)

    baseline_hash = fingerprint(baseline)
    old = json.loads(baseline.read_text(encoding="utf-8"))
    old_index = index_games(old)

    started = datetime.now(timezone.utc)
    request_params = {
        "year": season,
        "seasonType": "both",
        "classification": "fbs",
    }
    acquisition = {}

    if capture_http:
        from cfb_quant.live.acquisition_receipt_v58 import (
            capture_response,
            verify_receipt,
        )

        def save_acquisition(**kwargs):
            archive_dir = capture_response(
                output / "_http_acquisitions",
                **kwargs,
            )
            receipt = verify_receipt(archive_dir)
            acquisition["directory"] = str(archive_dir)
            acquisition["receipt"] = receipt

        fresh = cfbd_get(
            "/games",
            params=request_params,
            response_capture=save_acquisition,
        )
        if not acquisition:
            raise RuntimeError(
                "HTTP capture requested but no receipt was created"
            )
    else:
        fresh = cfbd_get("/games", params=request_params)

    fetched = datetime.now(timezone.utc)

    new_index = validate(fresh, season, len(old_index))
    old_window = select_window(old_index, start, end)
    new_window = select_window(new_index, start, end)

    added = sorted(set(new_window) - set(old_window))
    removed = sorted(set(old_window) - set(new_window))
    changed = []

    fields = (
        "startDate", "startTimeTBD", "completed",
        "homeId", "awayId", "homeClassification",
        "awayClassification", "homeTeam", "awayTeam"
    )

    for gid in sorted(set(old_window) & set(new_window)):
        before, after = old_window[gid], new_window[gid]
        differences = {
            key: {"old": before.get(key), "new": after.get(key)}
            for key in fields
            if before.get(key) != after.get(key)
        }
        if differences:
            changed.append({"game_id": gid, "changes": differences})

    unresolved = [
        {
            "game_id": gid,
            "away": game["awayTeam"],
            "home": game["homeTeam"],
            "reason": "startTimeTBD",
        }
        for gid, game in new_window.items()
        if game.get("startTimeTBD") is True
    ]

    counts = Counter(map(matchup_type, new_window.values()))
    stamp = fetched.strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / stamp

    if destination.exists():
        raise FileExistsError(destination)

    # Validate everything before creating the immutable archive.
    if fingerprint(baseline) != baseline_hash:
        raise RuntimeError("Baseline changed while synchronizing")

    destination.mkdir(parents=True)
    snapshot = destination / "cfbd_games.json"
    snapshot.write_text(json.dumps(fresh, indent=2), encoding="utf-8")

    audit = {
        "version": "daily_schedule_sync_v58_v1",
        "research_only": True,
        "season": season,
        "window_eastern": [start, end],
        "request_started_utc": started.isoformat(),
        "request_completed_utc": fetched.isoformat(),
        "source": "CFBD /games",
        "parameters": {
            "year": season,
            "seasonType": "both",
            "classification": "fbs",
        },
        "baseline_sha256": baseline_hash,
        "snapshot_sha256": fingerprint(snapshot),
        "baseline_games": len(old_index),
        "fresh_games": len(new_index),
        "eligible_games": len(new_window),
        "matchup_counts": dict(counts),
        "added_game_ids": added,
        "removed_game_ids": removed,
        "changed_games": changed,
        "unresolved_kickoffs": unresolved,
        "schedule_completeness_independently_certified": False,
        "production_modified": False,
        "models_executed": False,
    }

    if capture_http:
        audit["http_acquisition"] = {
            "receipt_directory": acquisition["directory"],
            "receipt_relative_to_output": str(
                Path(acquisition["directory"]).resolve().relative_to(
                    output.resolve()
                )
            ).replace("\\", "/"),
            "response_sha256": acquisition["receipt"]["response_sha256"],
            "request_started_utc": acquisition["receipt"]["request_started_utc"],
            "request_completed_utc": acquisition["receipt"]["request_completed_utc"],
            "source_bytes_archived": True,
            "historical_point_in_time_certified": False,
            "production_forecast_authorized": False,
        }

    audit_path = destination / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    if fingerprint(baseline) != baseline_hash:
        raise RuntimeError("Baseline changed during archive creation")

    print("=" * 72)
    print("DAILY CFBD SCHEDULE SYNC — RESEARCH")
    print("=" * 72)
    print("Season:", season)
    print("Eastern window:", start, "through", end)
    print("Fresh season games:", len(new_index))
    print("Eligible window games:", len(new_window))
    print("FBS-FBS:", counts["FBS_FBS"])
    print("FBS-FCS:", counts["FBS_FCS"])
    print("FBS-other:", counts["FBS_OTHER"])
    print("Added:", len(added))
    print("Removed:", len(removed))
    print("Changed:", len(changed))
    print("Unresolved kickoff times:", len(unresolved))
    print("Archive:", destination)
    print("Snapshot SHA256:", fingerprint(snapshot))
    print("Production untouched.")
    print("=" * 72)

    if added or removed or unresolved:
        print("ATTENTION: Schedule requires reconciliation.")
        for gid in added:
            game = new_window[gid]
            print("ADDED:", gid, game["awayTeam"], "@", game["homeTeam"])
        for gid in removed:
            game = old_window[gid]
            print("REMOVED:", gid, game["awayTeam"], "@", game["homeTeam"])
        for item in unresolved:
            print("UNRESOLVED:", item["game_id"],
                  item["away"], "@", item["home"])

    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "reports/v58_development/schedule_refresh_v1/"
            "games_2026_candidate.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v58_development/daily_schedule_sync_v58"),
    )
    args = parser.parse_args()
    for day in (args.start, args.end):
        datetime.strptime(day, "%Y-%m-%d")
    run(args.season, args.start, args.end,
        args.baseline, args.output)


if __name__ == "__main__":
    main()
