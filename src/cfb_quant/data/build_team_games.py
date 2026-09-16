from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_ROOT = Path(
    "data/raw/cfbd/team_stats"
)

PROCESSED_ROOT = Path(
    "data/processed"
)

REPORTS_ROOT = Path(
    "reports"
)


FILE_PATTERN = re.compile(
    r"team_stats_"
    r"(?P<season>\d{4})_"
    r"(?P<season_type>[A-Za-z]+)_"
    r"week_(?P<week>\d+)\.json$"
)


# ============================================================
# PARSING HELPERS
# ============================================================


def parse_number(
    value: Any,
) -> float:
    if value is None:
        return np.nan

    text = str(
        value
    ).strip()

    if text in {
        "",
        "-",
        "--",
        "NA",
        "N/A",
    }:
        return np.nan

    text = text.replace(
        ",",
        "",
    )

    try:
        return float(
            text
        )

    except ValueError:
        return np.nan


def parse_completion_attempts(
    value: Any,
) -> tuple[
    float,
    float,
]:
    if value is None:
        return (
            np.nan,
            np.nan,
        )

    text = str(
        value
    ).strip()

    separator = None

    for candidate in [
        "-",
        "/",
    ]:
        if candidate in text:
            separator = candidate
            break

    if separator is None:
        return (
            np.nan,
            np.nan,
        )

    left, right = text.split(
        separator,
        1,
    )

    return (
        parse_number(
            left
        ),
        parse_number(
            right
        ),
    )


def parse_penalties(
    value: Any,
) -> tuple[
    float,
    float,
]:
    if value is None:
        return (
            np.nan,
            np.nan,
        )

    text = str(
        value
    ).strip()

    if "-" not in text:
        return (
            np.nan,
            np.nan,
        )

    left, right = text.split(
        "-",
        1,
    )

    return (
        parse_number(
            left
        ),
        parse_number(
            right
        ),
    )


def parse_possession_seconds(
    value: Any,
) -> float:
    if value is None:
        return np.nan

    text = str(
        value
    ).strip()

    if ":" not in text:
        return np.nan

    minutes, seconds = text.split(
        ":",
        1,
    )

    minute_value = parse_number(
        minutes
    )

    second_value = parse_number(
        seconds
    )

    if (
        np.isnan(
            minute_value
        )
        or np.isnan(
            second_value
        )
    ):
        return np.nan

    return (
        minute_value
        * 60.0
        + second_value
    )


# ============================================================
# TEAM ROW
# ============================================================


def parse_team(
    *,
    game_id: int,
    season: int,
    week: int,
    season_type: str,
    team_block: dict[str, Any],
    opponent_block: dict[str, Any],
) -> dict[str, Any]:
    stats = {
        str(
            item.get(
                "category",
                "",
            )
        ): item.get(
            "stat"
        )
        for item in team_block.get(
            "stats",
            []
        )
    }

    completions, pass_attempts = (
        parse_completion_attempts(
            stats.get(
                "completionAttempts"
            )
        )
    )

    penalties, penalty_yards = (
        parse_penalties(
            stats.get(
                "totalPenaltiesYards"
            )
        )
    )

    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": season_type,

        "team_id": (
            team_block.get(
                "teamId"
            )
        ),
        "team": (
            team_block.get(
                "team"
            )
        ),
        "conference": (
            team_block.get(
                "conference"
            )
        ),
        "opponent": (
            opponent_block.get(
                "team"
            )
        ),
        "home_away": (
            team_block.get(
                "homeAway"
            )
        ),

        "points": parse_number(
            team_block.get(
                "points"
            )
        ),
        "opponent_points": parse_number(
            opponent_block.get(
                "points"
            )
        ),

        "completions": completions,
        "pass_attempts": pass_attempts,
        "passing_yards": parse_number(
            stats.get(
                "netPassingYards"
            )
        ),
        "passing_touchdowns": parse_number(
            stats.get(
                "passingTDs"
            )
        ),

        "rush_attempts": parse_number(
            stats.get(
                "rushingAttempts"
            )
        ),
        "rushing_yards": parse_number(
            stats.get(
                "rushingYards"
            )
        ),
        "rushing_touchdowns": parse_number(
            stats.get(
                "rushingTDs"
            )
        ),

        "interceptions": parse_number(
            stats.get(
                "interceptions"
            )
        ),
        "turnovers": parse_number(
            stats.get(
                "turnovers"
            )
        ),
        "fumbles_lost": parse_number(
            stats.get(
                "fumblesLost"
            )
        ),

        "sacks": parse_number(
            stats.get(
                "sacks"
            )
        ),
        "tackles_for_loss": parse_number(
            stats.get(
                "tacklesForLoss"
            )
        ),
        "qb_hurries": parse_number(
            stats.get(
                "qbHurries"
            )
        ),

        "possession_seconds": (
            parse_possession_seconds(
                stats.get(
                    "possessionTime"
                )
            )
        ),

        "penalties": penalties,
        "penalty_yards": penalty_yards,

        "yards_per_rush_attempt": (
            parse_number(
                stats.get(
                    "yardsPerRushAttempt"
                )
            )
        ),
        "yards_per_pass": (
            parse_number(
                stats.get(
                    "yardsPerPass"
                )
            )
        ),
    }


# ============================================================
# RAW TEAM FILES
# ============================================================


def parse_file(
    path: Path,
) -> list[
    dict[str, Any]
]:
    match = FILE_PATTERN.match(
        path.name
    )

    if not match:
        raise RuntimeError(
            "Unexpected team-stat filename: "
            f"{path.name}"
        )

    season = int(
        match.group(
            "season"
        )
    )

    week = int(
        match.group(
            "week"
        )
    )

    season_type = match.group(
        "season_type"
    )

    games = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for game in games:
        game_id = int(
            game[
                "id"
            ]
        )

        teams = game.get(
            "teams",
            []
        )

        if len(
            teams
        ) != 2:
            raise RuntimeError(
                f"Game {game_id} has "
                f"{len(teams)} team blocks."
            )

        for index in [
            0,
            1,
        ]:
            rows.append(
                parse_team(
                    game_id=game_id,
                    season=season,
                    week=week,
                    season_type=season_type,
                    team_block=teams[
                        index
                    ],
                    opponent_block=teams[
                        1 - index
                    ],
                )
            )

    return rows


def build_team_games(
    season: int,
) -> pd.DataFrame:
    root = (
        RAW_ROOT
        / str(
            season
        )
    )

    paths = sorted(
        root.glob(
            f"team_stats_{season}_*.json"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No team-stat files found "
            f"for {season}."
        )

    rows: list[
        dict[str, Any]
    ] = []

    for path in paths:
        file_rows = parse_file(
            path
        )

        rows.extend(
            file_rows
        )

        print(
            f"  {path.name}: "
            f"{len(file_rows):,} team-games"
        )

    frame = pd.DataFrame(
        rows
    )

    duplicates = frame.duplicated(
        subset=[
            "game_id",
            "team",
        ],
        keep=False,
    )

    if duplicates.any():
        raise RuntimeError(
            f"Found {int(duplicates.sum())} "
            "duplicate team-game rows."
        )

    frame = frame.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
        ]
    ).reset_index(
        drop=True
    )

    return frame


# ============================================================
# PLAYER / TEAM RECONCILIATION
# ============================================================


def reconcile(
    *,
    season: int,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    player_path = (
        PROCESSED_ROOT
        / f"player_games_{season}.parquet"
    )

    if not player_path.exists():
        raise RuntimeError(
            "Player-game table is missing: "
            f"{player_path}"
        )

    players = pd.read_parquet(
        player_path
    )

    players[
        "is_team_row"
    ] = (
        players[
            "player_name"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(
            "team"
        )
    )

    print()
    print(
        "Pseudo-player Team rows:",
        f"{int(players['is_team_row'].sum()):,}",
    )

    # --------------------------------------------------------
    # ALL source rows.
    #
    # Keep Team pseudo-rows here because they can contain
    # legitimate team rushing adjustments.
    # --------------------------------------------------------

    all_grouped = (
        players.groupby(
            [
                "game_id",
                "team",
            ],
            as_index=False,
        )
        .agg(
            player_completions=(
                "completions",
                "sum",
            ),
            player_pass_attempts=(
                "pass_attempts",
                "sum",
            ),
            player_passing_yards=(
                "passing_yards",
                "sum",
            ),
            player_rush_attempts=(
                "rush_attempts",
                "sum",
            ),
            player_rushing_yards=(
                "rushing_yards",
                "sum",
            ),
            player_receptions=(
                "receptions",
                "sum",
            ),
            player_receiving_yards=(
                "receiving_yards",
                "sum",
            ),
        )
    )

    # --------------------------------------------------------
    # INDIVIDUAL PLAYERS ONLY.
    # --------------------------------------------------------

    individual = players[
        ~players[
            "is_team_row"
        ]
    ].copy()

    individual_grouped = (
        individual.groupby(
            [
                "game_id",
                "team",
            ],
            as_index=False,
        )
        .agg(
            individual_rush_attempts=(
                "rush_attempts",
                "sum",
            ),
            individual_rushing_yards=(
                "rushing_yards",
                "sum",
            ),
        )
    )

    audit = team_games.merge(
        all_grouped,
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    audit = audit.merge(
        individual_grouped,
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="one_to_one",
    )

    player_total_columns = [
        "player_completions",
        "player_pass_attempts",
        "player_passing_yards",
        "player_rush_attempts",
        "player_rushing_yards",
        "player_receptions",
        "player_receiving_yards",
        "individual_rush_attempts",
        "individual_rushing_yards",
    ]

    for column in player_total_columns:
        audit[
            column
        ] = audit[
            column
        ].fillna(
            0.0
        )

    audit[
        "completion_delta"
    ] = (
        audit[
            "player_completions"
        ]
        - audit[
            "completions"
        ]
    )

    audit[
        "pass_attempt_delta"
    ] = (
        audit[
            "player_pass_attempts"
        ]
        - audit[
            "pass_attempts"
        ]
    )

    audit[
        "passing_yards_delta"
    ] = (
        audit[
            "player_passing_yards"
        ]
        - audit[
            "passing_yards"
        ]
    )

    audit[
        "rush_attempt_delta"
    ] = (
        audit[
            "player_rush_attempts"
        ]
        - audit[
            "rush_attempts"
        ]
    )

    audit[
        "rushing_yards_delta"
    ] = (
        audit[
            "player_rushing_yards"
        ]
        - audit[
            "rushing_yards"
        ]
    )

    # Useful diagnostic:
    # how much the pseudo-Team row contributes.
    audit[
        "individual_rush_attempt_delta"
    ] = (
        audit[
            "individual_rush_attempts"
        ]
        - audit[
            "rush_attempts"
        ]
    )

    audit[
        "individual_rushing_yards_delta"
    ] = (
        audit[
            "individual_rushing_yards"
        ]
        - audit[
            "rushing_yards"
        ]
    )

    # A completed forward pass normally produces one
    # reception, making this a useful source integrity check.
    audit[
        "receptions_vs_completions_delta"
    ] = (
        audit[
            "player_receptions"
        ]
        - audit[
            "completions"
        ]
    )

    audit[
        "receiving_vs_passing_yards_delta"
    ] = (
        audit[
            "player_receiving_yards"
        ]
        - audit[
            "passing_yards"
        ]
    )

    # Save clean prop-player table.
    props = individual.copy()

    props_path = (
        PROCESSED_ROOT
        / (
            f"player_games_props_"
            f"{season}.parquet"
        )
    )

    props.to_parquet(
        props_path,
        index=False,
    )

    return audit


# ============================================================
# AUDIT SUMMARY
# ============================================================


def print_metric(
    frame: pd.DataFrame,
    *,
    column: str,
    label: str,
) -> None:
    values = pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )

    valid = values.notna()

    if not valid.any():
        print(
            f"{label}: no valid rows"
        )

        return

    values = values[
        valid
    ]

    exact = (
        values.abs()
        < 1e-9
    )

    within_one = (
        values.abs()
        <= 1.0
    )

    print()
    print(label)

    print(
        "  rows:",
        f"{len(values):,}",
    )

    print(
        "  exact:",
        f"{exact.mean() * 100:.2f}%",
    )

    print(
        "  within 1:",
        f"{within_one.mean() * 100:.2f}%",
    )

    print(
        "  mean abs delta:",
        f"{values.abs().mean():.4f}",
    )

    print(
        "  max abs delta:",
        f"{values.abs().max():.1f}",
    )


def display_summary(
    *,
    season: int,
    team_games: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print(
        f"CFB SOURCE RECONCILIATION — {season}"
    )
    print("=" * 80)

    print()
    print(
        "Team-game rows:",
        f"{len(team_games):,}",
    )

    print(
        "Games:",
        f"{team_games['game_id'].nunique():,}",
    )

    print_metric(
        audit,
        column="completion_delta",
        label="COMPLETIONS",
    )

    print_metric(
        audit,
        column="pass_attempt_delta",
        label="PASS ATTEMPTS",
    )

    print_metric(
        audit,
        column="passing_yards_delta",
        label="PASSING YARDS",
    )

    print_metric(
        audit,
        column="rush_attempt_delta",
        label=(
            "RUSH ATTEMPTS "
            "(including Team row)"
        ),
    )

    print_metric(
        audit,
        column="rushing_yards_delta",
        label=(
            "RUSHING YARDS "
            "(including Team row)"
        ),
    )

    print_metric(
        audit,
        column=(
            "individual_rush_attempt_delta"
        ),
        label=(
            "RUSH ATTEMPTS "
            "(individual players only)"
        ),
    )

    print_metric(
        audit,
        column=(
            "individual_rushing_yards_delta"
        ),
        label=(
            "RUSHING YARDS "
            "(individual players only)"
        ),
    )

    print_metric(
        audit,
        column=(
            "receptions_vs_completions_delta"
        ),
        label=(
            "PLAYER RECEPTIONS "
            "VS TEAM COMPLETIONS"
        ),
    )

    print_metric(
        audit,
        column=(
            "receiving_vs_passing_yards_delta"
        ),
        label=(
            "PLAYER RECEIVING YARDS "
            "VS TEAM PASSING YARDS"
        ),
    )


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    season: int,
    team_games: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[
    Path,
    Path,
]:
    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    team_path = (
        PROCESSED_ROOT
        / f"team_games_{season}.parquet"
    )

    audit_path = (
        REPORTS_ROOT
        / (
            f"source_reconciliation_"
            f"{season}.csv"
        )
    )

    team_games.to_parquet(
        team_path,
        index=False,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    return (
        team_path,
        audit_path,
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized CFB team-game "
            "records and reconcile them against "
            "player box scores."
        )
    )

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print(
        f"BUILD TEAM GAMES — {args.season}"
    )
    print("=" * 80)

    team_games = build_team_games(
        args.season
    )

    audit = reconcile(
        season=args.season,
        team_games=team_games,
    )

    team_path, audit_path = (
        save_outputs(
            season=args.season,
            team_games=team_games,
            audit=audit,
        )
    )

    display_summary(
        season=args.season,
        team_games=team_games,
        audit=audit,
    )

    print()
    print(
        f"Team games: {team_path}"
    )

    print(
        "Prop-safe player games: "
        f"data/processed/"
        f"player_games_props_"
        f"{args.season}.parquet"
    )

    print(
        f"Audit: {audit_path}"
    )


if __name__ == "__main__":
    main()
