from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cfb_quant.features.matchup_engine import (
    build_features,
    load_team_history,
)
from cfb_quant.live.predict import (
    _load_schedule,
    _select_games,
)
from cfb_quant.models.matchup_tournament import (
    TARGETS,
    _to_numeric_frame,
)


ARTIFACT_DIR = Path("models/artifacts")
REPORT_DIR = Path("reports")


def _prepare_kickoff_aware_history(
    history: pd.DataFrame,
    games: list[dict],
    *,
    safety_hours: float = 6.0,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Prepare conservative, timestamp-ordered pregame history."""
    import json

    if not games:
        raise ValueError("No target games provided.")

    if safety_hours < 0:
        raise ValueError("safety_hours must be nonnegative.")

    kicks = pd.to_datetime(
        [g.get("startDate") for g in games],
        utc=True,
        errors="coerce",
    )

    if kicks.isna().any():
        raise ValueError("Target kickoff timestamp missing.")

    if len(set(kicks)) != 1:
        raise ValueError(
            "Kickoff-aware mode requires one target kickoff time. "
            "Mixed-time weekly slates are not yet supported."
        )

    target_kickoff = pd.Timestamp(kicks[0])
    cutoff = target_kickoff - pd.Timedelta(hours=safety_hours)

    schedules = {}

    for schedule_path in sorted(
        Path("data/raw/cfbd/games").glob("games_*.json")
    ):
        games_data = json.loads(
            schedule_path.read_text(encoding="utf-8-sig")
        )

        for game in games_data:
            gid = int(game["id"])

            if gid in schedules:
                raise ValueError(
                    f"Duplicate schedule game ID: {gid}"
                )

            schedules[gid] = game

    prepared = history.copy()

    prepared["game_id"] = pd.to_numeric(
        prepared["game_id"],
        errors="raise",
    ).astype("int64")

    missing = set(prepared["game_id"]) - set(schedules)

    if missing:
        raise ValueError(
            f"Missing historical schedule entries: "
            f"{len(missing)}"
        )

    prepared["kickoff"] = pd.to_datetime(
        prepared["game_id"].map(
            lambda gid: schedules[int(gid)].get("startDate")
        ),
        utc=True,
        errors="coerce",
    )

    if prepared["kickoff"].isna().any():
        raise ValueError(
            "Historical kickoff timestamps missing."
        )

    completed = prepared["game_id"].map(
        lambda gid: schedules[int(gid)].get("completed") is True
    )

    if not completed.all():
        raise ValueError(
            "Historical dataset contains uncompleted games."
        )

    eligible = prepared.loc[
        prepared["kickoff"] < cutoff
    ].copy()

    print(
        "Kickoff-aware cutoff (UTC):",
        cutoff.isoformat(),
    )
    print(
        "Historical team rows excluded:",
        len(prepared) - len(eligible),
    )

    return eligible, target_kickoff


def _make_team_rows(
    history: pd.DataFrame,
    games: list[dict],
    *,
    season: int,
    week: int,
) -> pd.DataFrame:

    rows = []

    for game in games:
        game_id = int(game["id"])

        sides = [
            {
                "team": str(game["homeTeam"]),
                "team_id": game.get("homeId"),
                "opponent": str(game["awayTeam"]),
                "home_away": "home",
                "conference": game.get(
                    "homeConference"
                ),
            },
            {
                "team": str(game["awayTeam"]),
                "team_id": game.get("awayId"),
                "opponent": str(game["homeTeam"]),
                "home_away": "away",
                "conference": game.get(
                    "awayConference"
                ),
            },
        ]

        for side in sides:
            row = {
                column: pd.NA
                for column in history.columns
            }

            row.update(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "season_type": str(
                        game.get(
                            "seasonType",
                            "regular",
                        )
                    ).lower(),
                    "team_id": side["team_id"],
                    "team": side["team"],
                    "conference":
                        side["conference"],
                    "opponent":
                        side["opponent"],
                    "home_away":
                        side["home_away"],
                }
            )

            rows.append(row)

    return pd.DataFrame(
        rows,
        columns=history.columns,
    )


def _load_production_artifacts() -> dict[str, dict]:

    artifacts = {}

    for target in TARGETS:
        path = (
            ARTIFACT_DIR
            / (
                f"cfb_matchup_{target}_"
                "production_v0_1.joblib"
            )
        )

        if not path.exists():
            raise RuntimeError(
                f"Production artifact missing: {path}"
            )

        artifact = joblib.load(path)

        if (
            artifact.get("artifact_type")
            != "cfb_matchup_production_model"
        ):
            raise RuntimeError(
                f"Wrong artifact type: {path}"
            )

        if artifact.get("target") != target:
            raise RuntimeError(
                f"Target mismatch: {path}"
            )

        artifacts[target] = artifact

    return artifacts


def _predict(
    live_rows: pd.DataFrame,
    artifacts: dict[str, dict],
) -> pd.DataFrame:

    output = live_rows[
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent",
            "home_away",
            "team_games_before",
            "team_season_games_before",
        ]
    ].copy()

    for target, artifact in artifacts.items():

        feature_columns = list(
            artifact["feature_columns"]
        )

        missing = [
            column
            for column in feature_columns
            if column not in live_rows.columns
        ]

        if missing:
            raise RuntimeError(
                f"{target}: missing live features: "
                f"{missing[:20]}"
            )

        X = _to_numeric_frame(
            live_rows,
            feature_columns,
        )

        prediction = np.asarray(
            artifact["model"].predict(X),
            dtype=float,
        )

        lo, hi = artifact[
            "prediction_bounds"
        ]

        prediction = np.clip(
            prediction,
            lo,
            hi,
        )

        output[
            f"proj_{target}"
        ] = prediction

    return output


def _add_team_derived(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    out = frame.copy()

    out[
        "proj_pass_plus_rush"
    ] = (
        out["proj_pass_attempts"]
        + out["proj_rush_attempts"]
    )

    out[
        "proj_pass_rate"
    ] = (
        out["proj_pass_attempts"]
        / out[
            "proj_pass_plus_rush"
        ].replace(0, np.nan)
    )

    out[
        "proj_plays_model_gap"
    ] = (
        out["proj_total_plays"]
        - out["proj_pass_plus_rush"]
    )

    out[
        "proj_yards"
    ] = (
        out["proj_passing_yards"]
        + out["proj_rushing_yards"]
    )

    possession_sum = (
        out.groupby(
            "game_id"
        )[
            "proj_possession_seconds"
        ]
        .transform("sum")
        .replace(0, np.nan)
    )

    out[
        "proj_possession_share"
    ] = (
        out[
            "proj_possession_seconds"
        ]
        / possession_sum
    )

    out[
        "proj_possession_seconds_normalized"
    ] = (
        out[
            "proj_possession_share"
        ]
        * 3600.0
    )

    return out


def _build_game_summary(
    teams: pd.DataFrame,
    games: list[dict],
) -> pd.DataFrame:

    rows = []

    lookup = {
        int(game["id"]): game
        for game in games
    }

    for game_id, group in teams.groupby(
        "game_id",
        sort=False,
    ):

        home = group.loc[
            group["home_away"].eq("home")
        ]

        away = group.loc[
            group["home_away"].eq("away")
        ]

        if len(home) != 1 or len(away) != 1:
            raise RuntimeError(
                f"Expected one home and one away "
                f"row for game {game_id}"
            )

        home = home.iloc[0]
        away = away.iloc[0]

        game = lookup[int(game_id)]

        rows.append(
            {
                "game_id": int(game_id),
                "start_date":
                    game.get("startDate"),
                "away_team":
                    away["team"],
                "home_team":
                    home["team"],

                "away_points":
                    away["proj_points"],
                "home_points":
                    home["proj_points"],

                "projected_total":
                    (
                        away["proj_points"]
                        + home["proj_points"]
                    ),

                "projected_home_margin":
                    (
                        home["proj_points"]
                        - away["proj_points"]
                    ),

                "away_total_plays":
                    away["proj_total_plays"],
                "home_total_plays":
                    home["proj_total_plays"],

                "away_pass_attempts":
                    away["proj_pass_attempts"],
                "home_pass_attempts":
                    home["proj_pass_attempts"],

                "away_rush_attempts":
                    away["proj_rush_attempts"],
                "home_rush_attempts":
                    home["proj_rush_attempts"],

                "away_pass_rate":
                    away["proj_pass_rate"],
                "home_pass_rate":
                    home["proj_pass_rate"],

                "away_passing_yards":
                    away["proj_passing_yards"],
                "home_passing_yards":
                    home["proj_passing_yards"],

                "away_rushing_yards":
                    away["proj_rushing_yards"],
                "home_rushing_yards":
                    home["proj_rushing_yards"],

                "away_possession_seconds":
                    away[
                        "proj_possession_seconds_normalized"
                    ],
                "home_possession_seconds":
                    home[
                        "proj_possession_seconds_normalized"
                    ],
            }
        )

    return pd.DataFrame(rows)


def run(
    *,
    season: int,
    week: int,
    kickoff_aware: bool = False,
) -> None:

    print("=" * 90)
    print(
        f"CFB LIVE MATCHUP PROJECTIONS — "
        f"{season} WEEK {week}"
    )
    print("=" * 90)

    schedule = _load_schedule(
        season
    )

    games = _select_games(
        schedule,
        season=season,
        week=week,
    )

    if not games:
        raise RuntimeError(
            f"No uncompleted games found for "
            f"{season} Week {week}"
        )

    print(
        f"Upcoming games selected: "
        f"{len(games)}"
    )

    for game in games:
        print(
            f"  {game['awayTeam']} @ "
            f"{game['homeTeam']} "
            f"[{game['id']}]"
        )

    print()
    print(
        "Loading completed team history..."
    )

    history = load_team_history()

    print(
        f"Historical team-game rows: "
        f"{len(history):,}"
    )

    if kickoff_aware:
        history, target_kickoff = (
            _prepare_kickoff_aware_history(
                history,
                games,
            )
        )
        print(
            "Kickoff-aware eligible history rows:",
            len(history),
        )

    # Safety: live history must not already
    # contain target-week completed rows.
    current_or_future = history.loc[
        (
            pd.to_numeric(
                history["season"],
                errors="coerce",
            )
            == season
        )
        & (
            pd.to_numeric(
                history["week"],
                errors="coerce",
            )
            >= week
        )
    ]

    if not kickoff_aware and not current_or_future.empty:
        raise RuntimeError(
            "History contains rows from the "
            f"target week or later: "
            f"{season} Week {week}+"
        )

    synthetic = _make_team_rows(
        history,
        games,
        season=season,
        week=week,
    )

    if kickoff_aware:
        synthetic["kickoff_utc"] = target_kickoff

    print(
        f"Synthetic live team rows: "
        f"{len(synthetic):,}"
    )

    combined = pd.concat(
        [
            history,
            synthetic,
        ],
        ignore_index=True,
    )

    print(
        "Building leakage-safe matchup "
        "features..."
    )

    featured, feature_columns = (
        build_features(
            combined
        )
    )

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    game_ids = {
        int(game["id"])
        for game in games
    }

    live_rows = featured.loc[
        (
            pd.to_numeric(
                featured["game_id"],
                errors="coerce",
            )
            .isin(game_ids)
        )
        & (
            pd.to_numeric(
                featured["season"],
                errors="coerce",
            )
            .eq(season)
        )
        & (
            pd.to_numeric(
                featured["week"],
                errors="coerce",
            )
            .eq(week)
        )
    ].copy()

    expected_rows = (
        len(games) * 2
    )

    if len(live_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} live "
            f"team rows; found {len(live_rows)}"
        )

    if live_rows[
        ["game_id", "team"]
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate live team-game rows."
        )

    print(
        f"Live rows ready: "
        f"{len(live_rows)}"
    )

    artifacts = (
        _load_production_artifacts()
    )

    print(
        "Running production matchup models..."
    )

    teams = _predict(
        live_rows,
        artifacts,
    )

    teams = _add_team_derived(
        teams
    )

    schedule_lookup = {
        int(game["id"]): game
        for game in games
    }

    teams["start_date"] = (
        teams["game_id"].map(
            lambda game_id:
            schedule_lookup[
                int(game_id)
            ].get("startDate")
        )
    )

    teams = teams[
        [
            "game_id",
            "start_date",
            "season",
            "week",
            "team",
            "opponent",
            "home_away",
            "team_games_before",
            "team_season_games_before",
            "proj_points",
            "proj_pass_attempts",
            "proj_rush_attempts",
            "proj_total_plays",
            "proj_pass_plus_rush",
            "proj_plays_model_gap",
            "proj_pass_rate",
            "proj_passing_yards",
            "proj_rushing_yards",
            "proj_yards",
            "proj_possession_seconds",
            "proj_possession_share",
            "proj_possession_seconds_normalized",
        ]
    ].copy()

    numeric_columns = [
        column
        for column in teams.columns
        if column.startswith("proj_")
    ]

    teams[
        numeric_columns
    ] = teams[
        numeric_columns
    ].round(3)

    games_summary = (
        _build_game_summary(
            teams,
            games,
        )
    )

    game_numeric = games_summary.select_dtypes(
        include="number"
    ).columns.tolist()

    games_summary[
        game_numeric
    ] = games_summary[
        game_numeric
    ].round(3)

    games_summary = (
        games_summary.sort_values(
            [
                "start_date",
                "game_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    teams = (
        teams.sort_values(
            [
                "start_date",
                "game_id",
                "home_away",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    team_path = (
        REPORT_DIR
        / (
            f"matchup_predictions_"
            f"{season}_week_{week}.csv"
        )
    )

    game_path = (
        REPORT_DIR
        / (
            f"matchup_games_"
            f"{season}_week_{week}.csv"
        )
    )

    teams.to_csv(
        team_path,
        index=False,
    )

    games_summary.to_csv(
        game_path,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "CFB LIVE MATCHUP BUILD COMPLETE"
    )
    print("=" * 90)

    print(
        f"Games: {len(games_summary):,}"
    )
    print(
        f"Team projections: "
        f"{len(teams):,}"
    )

    print()
    print(
        games_summary[
            [
                "away_team",
                "home_team",
                "away_points",
                "home_points",
                "projected_total",
                "projected_home_margin",
                "away_pass_attempts",
                "home_pass_attempts",
                "away_rush_attempts",
                "home_rush_attempts",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Team report: {team_path}"
    )
    print(
        f"Game report: {game_path}"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--kickoff-aware",
        action="store_true",
        help=(
            "Enable timestamp-ordered historical features "
            "with a conservative six-hour cutoff."
        ),
    )

    args = parser.parse_args()

    run(
        season=args.season,
        week=args.week,
        kickoff_aware=args.kickoff_aware,
    )


if __name__ == "__main__":
    main()
