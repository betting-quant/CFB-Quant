"""Read-only, fail-closed discovery of v5.8 forecast archives."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
REQUIRED = {
    "game_id", "kickoff_utc", "away", "home",
    "fair_margin_home", "fair_home_spread", "fair_total",
    "boosting_margin_home", "blend_margin_home",
    "blend_home_spread", "blend_total",
}
NUMERIC = REQUIRED - {
    "game_id", "kickoff_utc", "away", "home"
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kickoff(value):
    value = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Kickoff must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def discover(roots, start, end):
    first = date.fromisoformat(str(start))
    last = date.fromisoformat(str(end))
    if first > last:
        raise ValueError("Start date is after end date")

    candidates = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Missing archive root: {root}")
        candidates.update(
            p.resolve() for p in root.rglob("paired_forecasts.csv")
        )

    selected = []
    by_game = defaultdict(list)

    for path in sorted(candidates):
        with path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            if len(columns) != len(set(columns)):
                raise ValueError(f"Duplicate CSV columns: {path}")
            if not REQUIRED.issubset(columns):
                raise ValueError(f"Missing required columns: {path}")
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty forecast archive: {path}")

        in_window = []
        out_window = []
        seen = set()

        for row in rows:
            gid = (row.get("game_id") or "").strip()
            if not gid or gid in seen:
                raise ValueError(f"Blank/duplicate game ID: {path}")
            seen.add(gid)

            if not (row.get("away") or "").strip():
                raise ValueError(f"Missing away team: {path}")
            if not (row.get("home") or "").strip():
                raise ValueError(f"Missing home team: {path}")

            dt = kickoff(row.get("kickoff_utc") or "")
            eastern_day = dt.astimezone(EASTERN).date()

            for column in NUMERIC:
                try:
                    value = float(row[column])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Invalid {column} for game {gid}: {path}"
                    ) from None
                if not math.isfinite(value):
                    raise ValueError(
                        f"Nonfinite {column} for game {gid}: {path}"
                    )

            if first <= eastern_day <= last:
                in_window.append(gid)
            else:
                out_window.append(gid)

        if not in_window:
            continue

        # Downstream coverage currently receives whole CSVs.
        # Refuse a mixed-window file rather than quietly passing
        # games outside the requested window.
        if out_window:
            raise ValueError(
                f"Mixed-window archive requires explicit review: {path}"
            )

        item = {
            "path": str(path),
            "sha256": sha256(path),
            "game_ids": sorted(in_window),
            "rows": len(in_window),
        }
        selected.append(item)

        for gid in in_window:
            by_game[gid].append(str(path))

    overlaps = {
        gid: sources
        for gid, sources in by_game.items()
        if len(sources) > 1
    }
    if overlaps:
        sample = sorted(overlaps)[:10]
        raise ValueError(
            f"Overlapping forecast game IDs: {sample}. "
            "Manual reconciliation required."
        )

    if not selected:
        raise ValueError("No forecast archives found in date window")

    return {
        "schema": "forecast_archive_discovery_v58_v1",
        "research_only": True,
        "window_eastern": [first.isoformat(), last.isoformat()],
        "archives": selected,
        "archive_count": len(selected),
        "unique_game_count": len(by_game),
        "overlap_count": 0,
        "authenticity_verified": False,
        "coverage_verified": False,
        "model_executed": False,
        "production_modified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    result = discover(args.root, args.start, args.end)

    print("ARCHIVE DISCOVERY — STRUCTURAL PASS")
    print("Window:", result["window_eastern"])
    print("Archive files:", result["archive_count"])
    print("Unique game IDs:", result["unique_game_count"])
    print("Overlaps:", result["overlap_count"])
    print("Git authenticity verified: NO (separate check)")
    print("Schedule coverage verified: NO (separate check)")

    for archive in result["archives"]:
        print(f"  {archive['rows']:>3} games  {archive['path']}")

    if args.manifest is not None:
        target = args.manifest.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
        print("Manifest:", target)


if __name__ == "__main__":
    main()
