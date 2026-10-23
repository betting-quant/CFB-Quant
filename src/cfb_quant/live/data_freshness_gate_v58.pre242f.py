"""CFB-Quant v5.8 research data freshness gate."""

from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd


def normalize_id(value):
    return str(value).strip().removesuffix(".0")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source(root, relative, cutoff, max_age_hours):
    path = root / relative
    metadata_path = path.with_suffix(".meta.json")

    if not path.is_file():
        return {
            "status": "INCOMPLETE",
            "reason": "Source file missing",
        }

    if not metadata_path.is_file():
        return {
            "status": "UNVERIFIED",
            "reason": "Retrieval metadata missing",
        }

    try:
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8-sig")
        )
        timestamp = pd.to_datetime(
            metadata["downloaded_at_utc"],
            utc=True,
            errors="raise",
        )
        expected_hash = metadata["sha256"]
    except (KeyError, ValueError, TypeError):
        return {
            "status": "UNVERIFIED",
            "reason": "Invalid source metadata",
        }

    if pd.isna(timestamp):
        return {
            "status": "UNVERIFIED",
            "reason": "Invalid retrieval timestamp",
        }

    if sha256(path) != expected_hash:
        return {
            "status": "INCOMPLETE",
            "reason": "SHA256 fingerprint mismatch",
        }

    age_hours = (cutoff - timestamp).total_seconds() / 3600

    if age_hours < 0:
        status = "UNVERIFIED"
        reason = "Source retrieved after forecast cutoff"
    elif age_hours > max_age_hours:
        status = "STALE"
        reason = f"Source age {age_hours:.2f} hours"
    else:
        status = "READY"
        reason = f"Source age {age_hours:.2f} hours"

    return {
        "status": status,
        "reason": reason,
        "retrieved_at_utc": timestamp.isoformat(),
    }


def audit(root, cutoff, max_age_hours=24):
    sources = {
        "schedule": Path(
            "data/raw/cfbd/games/games_2026.json"
        ),
        "history": Path(
            "data/processed/team_games_2026.parquet"
        ),
        "advanced": Path(
            "data/raw/cfbd/advanced_game_v4/advanced_2026.json"
        ),
    }

    checks = {
        name: check_source(root, path, cutoff, max_age_hours)
        for name, path in sources.items()
    }

    issues = []

    if all((root / path).is_file() for path in sources.values()):
        schedule = json.loads(
            (root / sources["schedule"]).read_text(
                encoding="utf-8-sig"
            )
        )
        history = pd.read_parquet(root / sources["history"])
        advanced = json.loads(
            (root / sources["advanced"]).read_text(
                encoding="utf-8-sig"
            )
        )

        if not isinstance(schedule, list) or not isinstance(advanced, list):
            issues.append("Unexpected source JSON structure")
        else:
            schedule_map = {}

            for game in schedule:
                key = normalize_id(game["id"])
                if key in schedule_map:
                    issues.append(f"Duplicate schedule ID: {key}")
                schedule_map[key] = game

            advanced_counts = {}

            for record in advanced:
                key = normalize_id(record["gameId"])
                advanced_counts[key] = (
                    advanced_counts.get(key, 0) + 1
                )

            history = history.copy()
            history["game_id"] = history.game_id.map(
                normalize_id
            )

            for game_id, group in history.groupby("game_id"):
                if (
                    len(group) != 2
                    or set(group.home_away) != {"home", "away"}
                ):
                    issues.append(
                        f"Invalid historical game pair: {game_id}"
                    )

                game = schedule_map.get(game_id)

                if game is None:
                    issues.append(
                        f"Historical game missing schedule: {game_id}"
                    )
                    continue

                kickoff = pd.to_datetime(
                    game.get("startDate"),
                    utc=True,
                    errors="coerce",
                )

                if (
                    game.get("completed") is not True
                    or pd.isna(kickoff)
                    or kickoff >= cutoff
                ):
                    issues.append(
                        f"Unsafe historical outcome: {game_id}"
                    )

                if advanced_counts.get(game_id, 0) != 2:
                    issues.append(
                        f"Incomplete advanced statistics: {game_id}"
                    )

    statuses = [value["status"] for value in checks.values()]

    if issues or "INCOMPLETE" in statuses:
        overall = "INCOMPLETE"
    elif "STALE" in statuses:
        overall = "STALE"
    elif "UNVERIFIED" in statuses:
        overall = "UNVERIFIED"
    else:
        overall = "READY"

    return {
        "status": overall,
        "cutoff_utc": cutoff.isoformat(),
        "max_age_hours": max_age_hours,
        "sources": checks,
        "issue_count": len(issues),
        "issues": issues[:25],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--cutoff-utc", required=True)
    parser.add_argument(
        "--max-age-hours", type=float, default=24
    )
    args = parser.parse_args()

    if args.max_age_hours <= 0:
        parser.error("--max-age-hours must be positive")

    cutoff = pd.to_datetime(
        args.cutoff_utc, utc=True, errors="raise"
    )

    result = audit(
        Path(args.root).resolve(),
        cutoff,
        args.max_age_hours,
    )

    print(json.dumps(result, indent=2))

    if result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
