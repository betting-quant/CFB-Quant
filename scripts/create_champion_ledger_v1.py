from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "reports/locked_forecasts/20261029_preweek9_baseline"
GAME_PATH = SOURCE / "matchup_games_2026_week_9.csv"
TEAM_PATH = SOURCE / "matchup_predictions_2026_week_9.csv"
SUBSET_PATH = SOURCE / "cfb_oct29_baseline_2026.csv"

OUT = ROOT / "reports/research/champion_ledger_v1_20261030"

OUTPUT_CSV = OUT / "champion_forecasts.csv"
OUTPUT_JSON = OUT / "manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"game_id": str})


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(
        not OUTPUT_CSV.exists() and not OUTPUT_JSON.exists(),
        "Ledger already exists. Refusing to overwrite."
    )

    games = read_source(GAME_PATH)
    teams = read_source(TEAM_PATH)
    subset = read_source(SUBSET_PATH)

    required_games = {
        "game_id", "start_date", "home_team", "away_team",
        "home_points", "away_points", "projected_total",
        "projected_home_margin"
    }

    required_teams = {
        "game_id", "start_date", "team", "opponent",
        "home_away", "proj_points"
    }

    require(
        required_games.issubset(games.columns),
        "Missing required game forecast columns"
    )
    require(
        required_teams.issubset(teams.columns),
        "Missing required team forecast columns"
    )

    require(len(games) > 0, "Empty game predictions")
    require(
        games["game_id"].notna().all()
        and not games["game_id"].duplicated().any(),
        "Invalid or duplicate game IDs"
    )

    require(
        not teams.duplicated(["game_id", "home_away"]).any(),
        "Duplicate team/side predictions"
    )

    require(
        set(games.game_id) == set(teams.game_id),
        "Game and team forecast coverage differs"
    )

    require(
        len(teams) == 2 * len(games),
        "Expected two team predictions per game"
    )

    for game in games.itertuples(index=False):
        pair = teams.loc[teams.game_id == game.game_id]

        require(
            len(pair) == 2
            and set(pair.home_away) == {"home", "away"},
            f"Invalid team pair: {game.game_id}"
        )

        home = pair.loc[pair.home_away == "home"].iloc[0]
        away = pair.loc[pair.home_away == "away"].iloc[0]

        require(
            home.team == game.home_team
            and away.team == game.away_team,
            f"Team identity mismatch: {game.game_id}"
        )

        require(
            home.opponent == away.team
            and away.opponent == home.team,
            f"Opponent mismatch: {game.game_id}"
        )

        require(
            pd.to_datetime(pair.start_date, utc=True).eq(
                pd.to_datetime(game.start_date, utc=True)
            ).all(),
            f"Kickoff mismatch: {game.game_id}"
        )

        require(
            abs(float(home.proj_points) - float(game.home_points))
            <= 0.002
            and abs(float(away.proj_points) - float(game.away_points))
            <= 0.002,
            f"Team points mismatch: {game.game_id}"
        )

    numeric = [
        "home_points", "away_points",
        "projected_total", "projected_home_margin"
    ]

    for col in numeric:
        games[col] = pd.to_numeric(games[col], errors="raise")

    require(
        np.isfinite(games[numeric].to_numpy()).all(),
        "Nonfinite predictions found"
    )

    require(
        (games[["home_points", "away_points"]] >= 0).all().all(),
        "Negative projected score"
    )

    require(
        np.allclose(
            games.home_points + games.away_points,
            games.projected_total,
            atol=0.002,
            rtol=0
        ),
        "Projected totals disagree with team scores"
    )

    require(
        np.allclose(
            games.home_points - games.away_points,
            games.projected_home_margin,
            atol=0.002,
            rtol=0
        ),
        "Projected margins disagree with team scores"
    )

    require(
        set(subset.game_id).issubset(set(games.game_id)),
        "Baseline subset contains unknown games"
    )

    check_cols = [
        "home_team", "away_team",
        "home_points", "away_points",
        "projected_total", "projected_home_margin"
    ]

    joined = subset.merge(
        games,
        on="game_id",
        how="left",
        validate="one_to_one",
        suffixes=("_subset", "_full")
    )

    for col in check_cols:
        a = joined[f"{col}_subset"]
        b = joined[f"{col}_full"]

        if col.endswith("_team"):
            require(
                a.equals(b),
                f"Subset identity mismatch: {col}"
            )
        else:
            require(
                np.allclose(
                    pd.to_numeric(a),
                    pd.to_numeric(b),
                    atol=0.002,
                    rtol=0
                ),
                f"Subset prediction mismatch: {col}"
            )

    kickoff = pd.to_datetime(
        games.start_date, utc=True, errors="raise"
    )

    ledger = pd.DataFrame({
        "game_id": games.game_id,
        "kickoff_utc": kickoff.dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "home_team": games.home_team,
        "away_team": games.away_team,
        "champion_home_points": games.home_points,
        "champion_away_points": games.away_points,
        "champion_total": games.projected_total,
        "champion_home_margin": games.projected_home_margin,
        "champion_model": "archived_matchup_baseline_20261029",
        "provenance_status": "archived_timestamp_not_certified",
    })

    ledger = ledger.sort_values(
        ["kickoff_utc", "game_id"]
    ).reset_index(drop=True)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    manifest = {
        "schema_version": "champion_ledger_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "research_only": True,
        "production_modified": False,
        "historical_pregame_certified": False,
        "provenance_status": "archived_timestamp_not_certified",
        "margin_convention": "home_points_minus_away_points",
        "games": len(ledger),
        "team_rows": len(teams),
        "subset_games": len(subset),
        "git_commit_at_export": commit,
        "sources": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [GAME_PATH, TEAM_PATH, SUBSET_PATH]
        }
    }

    OUT.mkdir(parents=True, exist_ok=True)

    ledger.to_csv(OUTPUT_CSV, index=False)
    manifest["output_sha256"] = sha256(OUTPUT_CSV)

    with OUTPUT_JSON.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print("\n=== CHAMPION LEDGER CREATED ===")
    print("Games:", len(ledger))
    print("Team predictions:", len(teams))
    print("Subset games:", len(subset))
    print("Game IDs unique:", ledger.game_id.is_unique)
    print("Score arithmetic: PASS")
    print("Team reconciliation: PASS")
    print("Subset reconciliation: PASS")
    print("Historical PIT certified: NO")
    print("Production modified: NO")
    print("Output:", OUTPUT_CSV.relative_to(ROOT))
    print("SHA256:", manifest["output_sha256"])


if __name__ == "__main__":
    main()
