"""Automatic, approval-enforced CFB v5.8 readiness preflight."""

import argparse
import hashlib
from pathlib import Path

from cfb_quant.live.forecast_archive_discovery_v58 import discover
from cfb_quant.live.daily_readiness_v58 import run as readiness_run

DEFAULT_ROOTS = (
    "reports/v58_development/midweek_forecasts",
    "reports/v58_development/prospective_forecasts",
    "reports/v58_development/supplementary_forecasts",
)


def verify_discovered(discovery):
    if discovery["overlap_count"] != 0:
        raise ValueError("Conflicting forecast archives")
    if discovery["archive_count"] != len(discovery["archives"]):
        raise ValueError("Archive count mismatch")

    paths = []
    unique = set()

    for entry in discovery["archives"]:
        path = Path(entry["path"]).resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"Forecast archive changed after discovery: {path}"
            )

        game_ids = entry["game_ids"]
        if entry["rows"] != len(game_ids):
            raise ValueError("Forecast row count mismatch")

        for gid in game_ids:
            if gid in unique:
                raise ValueError(f"Duplicate forecast game ID: {gid}")
            unique.add(gid)

        paths.append(path)

    if len(unique) != discovery["unique_game_count"]:
        raise ValueError("Unique game count mismatch")

    if not paths:
        raise ValueError("No archives discovered")

    return paths


def run(season, start, end, approval, roots, output):
    # Validate the approval before discovery or external API use.
    from cfb_quant.live.approved_baseline_v58 import verify_any

    verify_any(approval, season)

    discovery = discover(roots, start, end)
    archives = verify_discovered(discovery)

    result = readiness_run(
        season, start, end, approval, archives, output
    )

    # Recheck that none of the input files changed during processing.
    verify_discovered(discovery)

    if not result.get("ready", False):
        raise RuntimeError(
            "Research readiness ATTENTION. "
            "Inspect the archived readiness report."
        )

    if result.get("statuses", {}).get("ARCHIVED", 0) != (
        discovery["unique_game_count"]
    ):
        raise RuntimeError(
            "Discovered and authenticated game counts disagree"
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--approval", type=Path, required=True
    )
    parser.add_argument(
        "--root", action="append", type=Path
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/v58_development/daily_readiness_v58")
    )
    args = parser.parse_args()

    roots = args.root or [Path(p) for p in DEFAULT_ROOTS]

    result = run(
        args.season, args.start, args.end,
        args.approval, roots, args.output
    )

    print("=" * 72)
    print("AUTOMATIC CFB v5.8 RESEARCH READINESS: PASS")
    print("Scheduled games:", result["scheduled_games"])
    print("Archived games:", result["statuses"].get("ARCHIVED", 0))
    print("Schedule alerts:", result["schedule_alerts"])
    print("Model executed: NO")
    print("Production modified: NO")
    print("=" * 72)


if __name__ == "__main__":
    main()
