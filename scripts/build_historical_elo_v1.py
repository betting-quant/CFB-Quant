"""Retrospective chronological Elo features for CFB research.

Result availability is ASSUMED to be kickoff + 24 hours.
This is not historical point-in-time certification.
No market inputs. No production model modifications.
"""

from datetime import timedelta, datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import subprocess

import pandas as pd

from cfb_quant.spread_total.elo_engine_v58 import EloConfig
from cfb_quant.spread_total.elo_replay_v58 import (
    ReplayGame,
    replay_games,
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / (
    "data/processed/cfb_matchup_features_opponent_v45.parquet"
)

OUT = ROOT / "reports/research/historical_elo_v1"
CSV = OUT / "pregame_elo_features.csv"
MANIFEST = OUT / "manifest.json"

DELAY_HOURS = 24
CONFIG = EloConfig(
    k_factor=24.0,
    home_advantage=0.0,
    scale=400.0,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    require(
        SOURCE.exists(),
        f"Historical dataset missing: {SOURCE}"
    )
    require(
        not CSV.exists() and not MANIFEST.exists(),
        "Existing research artifact detected; refusing overwrite"
    )

    columns = [
        "game_id",
        "season",
        "week",
        "team",
        "home_away",
        "points",
        "opponent_points",
        "kickoff_utc",
    ]

    df = pd.read_parquet(SOURCE, columns=columns)

    require(
        df["game_id"].notna().all(),
        "Missing historical game IDs"
    )
    require(
        not df.duplicated(["game_id", "home_away"]).any(),
        "Duplicate historical game/side"
    )
    require(
        len(df) == 2 * df.game_id.nunique(),
        "Historical dataset does not contain complete game pairs"
    )

    df["kickoff_utc"] = pd.to_datetime(
        df["kickoff_utc"], utc=True, errors="raise"
    )
    require(
        df["kickoff_utc"].notna().all(),
        "Missing historical kickoff"
    )

    df["points"] = pd.to_numeric(
        df["points"], errors="raise"
    )
    df["opponent_points"] = pd.to_numeric(
        df["opponent_points"], errors="raise"
    )

    require(
        df[["points", "opponent_points"]].notna().all().all(),
        "Missing historical final scores"
    )

    games = []
    metadata = {}

    for game_id, group in df.groupby("game_id", sort=False):
        gid = str(game_id)

        require(
            len(group) == 2
            and set(group["home_away"]) == {"home", "away"},
            f"Invalid home/away pair: {gid}"
        )

        home = group.loc[
            group["home_away"] == "home"
        ].iloc[0]

        away = group.loc[
            group["home_away"] == "away"
        ].iloc[0]

        require(
            home["team"] != away["team"],
            f"Same-team matchup: {gid}"
        )
        require(
            home["kickoff_utc"] == away["kickoff_utc"],
            f"Kickoff mismatch: {gid}"
        )
        require(
            home["season"] == away["season"]
            and home["week"] == away["week"],
            f"Season/week mismatch: {gid}"
        )

        scores = [
            float(home["points"]),
            float(away["points"]),
            float(home["opponent_points"]),
            float(away["opponent_points"]),
        ]

        require(
            all(math.isfinite(x) and x >= 0 for x in scores),
            f"Invalid score: {gid}"
        )
        require(
            scores[0] == scores[3]
            and scores[1] == scores[2],
            f"Home/away final scores disagree: {gid}"
        )

        kickoff = home["kickoff_utc"].to_pydatetime()

        games.append(
            ReplayGame(
                game_id=gid,
                kickoff_utc=kickoff,
                home=str(home["team"]),
                away=str(away["team"]),
                home_points=scores[0],
                away_points=scores[1],
                result_available_utc=(
                    kickoff + timedelta(hours=DELAY_HOURS)
                ),
            )
        )

        metadata[gid] = {
            "season": int(home["season"]),
            "week": int(home["week"]),
            "actual_home_points": scores[0],
            "actual_away_points": scores[1],
            "actual_home_margin": scores[0] - scores[1],
        }

    print("\n=== HISTORICAL ELO REPLAY ===")
    print("Historical games:", len(games))
    print("Availability assumption:", DELAY_HOURS, "hours")

    replay = replay_games(
        games,
        config=CONFIG,
        initial_rating=1500.0,
    )

    require(
        len(replay.snapshots) == len(games),
        "Replay snapshot coverage mismatch"
    )
    require(
        replay.updates_applied == len(games),
        "Historical rating updates incomplete"
    )

    rows = []

    for snap in replay.snapshots:
        info = metadata[snap.game_id]

        rows.append({
            "game_id": snap.game_id,
            "season": info["season"],
            "week": info["week"],
            "kickoff_utc": snap.kickoff_utc.isoformat(),
            "home_team": snap.home,
            "away_team": snap.away,
            "home_elo_pregame": snap.home_elo,
            "away_elo_pregame": snap.away_elo,
            "elo_home_advantage": CONFIG.home_advantage,
            "elo_rating_difference": (
                snap.home_elo - snap.away_elo
            ),
            "elo_home_win_probability": (
                snap.expected_home_win
            ),
            "actual_home_points": info["actual_home_points"],
            "actual_away_points": info["actual_away_points"],
            "actual_home_margin": info["actual_home_margin"],
        })

    result = pd.DataFrame(rows).sort_values(
        ["kickoff_utc", "game_id"]
    ).reset_index(drop=True)

    require(
        result.game_id.is_unique,
        "Duplicate output game IDs"
    )

    require(
        result["season"].between(2014, 2025).all(),
        "Unexpected historical season"
    )

    # Research outputs are separate from live/production artifacts.
    OUT.mkdir(parents=True, exist_ok=True)

    result.to_csv(CSV, index=False)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    manifest = {
        "schema_version": "historical_elo_research_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "output_sha256": sha256(CSV),
        "git_revision": commit,
        "games": len(games),
        "seasons": sorted(
            result["season"].unique().tolist()
        ),
        "rating_updates_applied": replay.updates_applied,
        "teams_represented": len(replay.final_ratings),
        "initial_rating": 1500.0,
        "k_factor": CONFIG.k_factor,
        "elo_scale": CONFIG.scale,
        "home_advantage": CONFIG.home_advantage,
        "result_delay_assumed_hours": DELAY_HOURS,
        "result_timing_policy": (
            "Retrospective assumption: each completed game "
            "becomes available 24 hours after kickoff"
        ),
        "historical_point_in_time_certified": False,
        "production_approved": False,
        "research_only": True,
    }

    with MANIFEST.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("\n=== HISTORICAL ELO RESEARCH COMPLETE ===")
    print("Games:", len(result))
    print("Rating updates:", replay.updates_applied)
    print("Teams:", len(replay.final_ratings))
    print("Seasons:", manifest["seasons"])
    print("Unique game IDs:", result.game_id.is_unique)
    print("Historical PIT certified: NO")
    print("Production modified: NO")
    print("Output:", CSV.relative_to(ROOT))
    print("Source SHA256:", manifest["source_sha256"])
    print("Output SHA256:", manifest["output_sha256"])


if __name__ == "__main__":
    main()
