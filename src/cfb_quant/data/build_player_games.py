from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_ROOT = Path(
    "data/raw/cfbd/player_stats"
)

OUTPUT_ROOT = Path(
    "data/processed"
)


# ============================================================
# HELPERS
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


def parse_comp_att(
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

    if "/" not in text:
        return (
            np.nan,
            np.nan,
        )

    left, right = text.split(
        "/",
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


def get_team_name(
    team_block: dict[
        str,
        Any
    ],
) -> str:
    value = team_block.get(
        "team"
    )

    if isinstance(
        value,
        dict,
    ):
        for key in [
            "school",
            "name",
            "displayName",
            "abbreviation",
        ]:
            candidate = value.get(
                key
            )

            if candidate:
                return str(
                    candidate
                )

    if value is not None:
        return str(
            value
        )

    return ""


def normalize_stat_name(
    value: Any,
) -> str:
    return (
        str(
            value
        )
        .strip()
        .upper()
        .replace(
            " ",
            "",
        )
    )


def player_key(
    athlete: dict[
        str,
        Any
    ],
) -> str:
    player_id = athlete.get(
        "id"
    )

    if player_id not in {
        None,
        "",
    }:
        return (
            f"id:{player_id}"
        )

    return (
        "name:"
        + str(
            athlete.get(
                "name",
                "",
            )
        )
    )


# ============================================================
# PLAYER RECORD
# ============================================================


def new_player_record(
    *,
    athlete: dict[str, Any],
    game_id: int,
    season: int,
    week: int,
    season_type: str,
    team: str,
    opponent: str,
    home_away: str,
    team_points: Any,
    opponent_points: Any,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "season_type": season_type,
        "player_id": athlete.get(
            "id"
        ),
        "player_name": athlete.get(
            "name"
        ),
        "team": team,
        "opponent": opponent,
        "home_away": home_away,
        "team_points": parse_number(
            team_points
        ),
        "opponent_points": parse_number(
            opponent_points
        ),

        # Passing
        "completions": 0.0,
        "pass_attempts": 0.0,
        "passing_yards": 0.0,
        "passing_touchdowns": 0.0,
        "interceptions": 0.0,
        "qbr": np.nan,

        # Rushing
        "rush_attempts": 0.0,
        "rushing_yards": 0.0,
        "rushing_touchdowns": 0.0,

        # Receiving
        "receptions": 0.0,
        "receiving_yards": 0.0,
        "receiving_touchdowns": 0.0,
        "targets": np.nan,

        # Audit flags
        "has_passing_stats": False,
        "has_rushing_stats": False,
        "has_receiving_stats": False,
    }


# ============================================================
# CATEGORY PARSING
# ============================================================


def apply_passing_stat(
    record: dict[str, Any],
    *,
    stat_name: str,
    stat_value: Any,
) -> None:
    record[
        "has_passing_stats"
    ] = True

    if stat_name in {
        "C/ATT",
        "COMP/ATT",
    }:
        completions, attempts = (
            parse_comp_att(
                stat_value
            )
        )

        if not np.isnan(
            completions
        ):
            record[
                "completions"
            ] = completions

        if not np.isnan(
            attempts
        ):
            record[
                "pass_attempts"
            ] = attempts

    elif stat_name == "YDS":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "passing_yards"
            ] = value

    elif stat_name == "TD":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "passing_touchdowns"
            ] = value

    elif stat_name == "INT":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "interceptions"
            ] = value

    elif stat_name == "QBR":
        record[
            "qbr"
        ] = parse_number(
            stat_value
        )


def apply_rushing_stat(
    record: dict[str, Any],
    *,
    stat_name: str,
    stat_value: Any,
) -> None:
    record[
        "has_rushing_stats"
    ] = True

    if stat_name in {
        "CAR",
        "ATT",
    }:
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "rush_attempts"
            ] = value

    elif stat_name == "YDS":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "rushing_yards"
            ] = value

    elif stat_name == "TD":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "rushing_touchdowns"
            ] = value


def apply_receiving_stat(
    record: dict[str, Any],
    *,
    stat_name: str,
    stat_value: Any,
) -> None:
    record[
        "has_receiving_stats"
    ] = True

    if stat_name in {
        "REC",
        "RECEPTIONS",
    }:
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "receptions"
            ] = value

    elif stat_name == "YDS":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "receiving_yards"
            ] = value

    elif stat_name == "TD":
        value = parse_number(
            stat_value
        )

        if not np.isnan(
            value
        ):
            record[
                "receiving_touchdowns"
            ] = value

    elif stat_name in {
        "TGT",
        "TGTS",
        "TARGET",
        "TARGETS",
    }:
        record[
            "targets"
        ] = parse_number(
            stat_value
        )


# ============================================================
# GAME PARSER
# ============================================================


def parse_game(
    *,
    game: dict[str, Any],
    season: int,
    week: int,
    season_type: str,
) -> list[
    dict[str, Any]
]:
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

    team_names = [
        get_team_name(
            team
        )
        for team in teams
    ]

    records: list[
        dict[str, Any]
    ] = []

    for team_index, team_block in enumerate(
        teams
    ):
        opponent_index = (
            1 - team_index
        )

        team_name = (
            team_names[
                team_index
            ]
        )

        opponent_name = (
            team_names[
                opponent_index
            ]
        )

        opponent_block = teams[
            opponent_index
        ]

        players: dict[
            str,
            dict[str, Any],
        ] = {}

        categories = team_block.get(
            "categories",
            []
        )

        for category in categories:
            category_name = (
                str(
                    category.get(
                        "name",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            if category_name not in {
                "passing",
                "rushing",
                "receiving",
            }:
                continue

            types = category.get(
                "types",
                []
            )

            for stat_type in types:
                stat_name = (
                    normalize_stat_name(
                        stat_type.get(
                            "name",
                            "",
                        )
                    )
                )

                athletes = stat_type.get(
                    "athletes",
                    []
                )

                for athlete in athletes:
                    key = player_key(
                        athlete
                    )

                    if key not in players:
                        players[
                            key
                        ] = new_player_record(
                            athlete=athlete,
                            game_id=game_id,
                            season=season,
                            week=week,
                            season_type=season_type,
                            team=team_name,
                            opponent=opponent_name,
                            home_away=str(
                                team_block.get(
                                    "homeAway",
                                    "",
                                )
                            ),
                            team_points=(
                                team_block.get(
                                    "points"
                                )
                            ),
                            opponent_points=(
                                opponent_block.get(
                                    "points"
                                )
                            ),
                        )

                    record = players[
                        key
                    ]

                    stat_value = (
                        athlete.get(
                            "stat"
                        )
                    )

                    if (
                        category_name
                        == "passing"
                    ):
                        apply_passing_stat(
                            record,
                            stat_name=stat_name,
                            stat_value=stat_value,
                        )

                    elif (
                        category_name
                        == "rushing"
                    ):
                        apply_rushing_stat(
                            record,
                            stat_name=stat_name,
                            stat_value=stat_value,
                        )

                    elif (
                        category_name
                        == "receiving"
                    ):
                        apply_receiving_stat(
                            record,
                            stat_name=stat_name,
                            stat_value=stat_value,
                        )

        records.extend(
            players.values()
        )

    return records


# ============================================================
# FILE PARSER
# ============================================================


FILE_PATTERN = re.compile(
    r"player_stats_"
    r"(?P<season>\d{4})_"
    r"(?P<season_type>[A-Za-z]+)_"
    r"week_(?P<week>\d+)\.json$"
)


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
            "Unexpected player stats filename: "
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
        rows.extend(
            parse_game(
                game=game,
                season=season,
                week=week,
                season_type=season_type,
            )
        )

    return rows


# ============================================================
# SEASON BUILD
# ============================================================


def build_player_games(
    season: int,
) -> pd.DataFrame:
    season_root = (
        RAW_ROOT
        / str(
            season
        )
    )

    if not season_root.exists():
        raise RuntimeError(
            "Player stats directory "
            f"does not exist: {season_root}"
        )

    paths = sorted(
        season_root.glob(
            f"player_stats_{season}_*.json"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No player-stat files found "
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
            f"{len(file_rows):,} player-games"
        )

    frame = pd.DataFrame(
        rows
    )

    if frame.empty:
        raise RuntimeError(
            f"No player-game rows built "
            f"for {season}."
        )

    # --------------------------------------------------------
    # Basic normalization
    # --------------------------------------------------------

    numeric_columns = [
        "game_id",
        "season",
        "week",
        "team_points",
        "opponent_points",
        "completions",
        "pass_attempts",
        "passing_yards",
        "passing_touchdowns",
        "interceptions",
        "qbr",
        "rush_attempts",
        "rushing_yards",
        "rushing_touchdowns",
        "receptions",
        "receiving_yards",
        "receiving_touchdowns",
        "targets",
    ]

    for column in numeric_columns:
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Duplicate protection
    # --------------------------------------------------------

    duplicates = frame.duplicated(
        subset=[
            "game_id",
            "team",
            "player_id",
            "player_name",
        ],
        keep=False,
    )

    if duplicates.any():
        bad = frame.loc[
            duplicates,
            [
                "game_id",
                "team",
                "player_id",
                "player_name",
            ],
        ]

        print()
        print(
            "DUPLICATE PLAYER-GAME ROWS"
        )

        print(
            bad.head(
                20
            ).to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"Found {int(duplicates.sum())} "
            "duplicate player-game rows."
        )

    frame = frame.sort_values(
        [
            "season",
            "week",
            "game_id",
            "team",
            "player_name",
        ]
    ).reset_index(
        drop=True
    )

    return frame


def save_player_games(
    *,
    season: int,
    frame: pd.DataFrame,
) -> Path:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        OUTPUT_ROOT
        / f"player_games_{season}.parquet"
    )

    temporary = (
        output.with_suffix(
            ".parquet.tmp"
        )
    )

    frame.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(
        output
    )

    return output


# ============================================================
# VALIDATION SUMMARY
# ============================================================


def display_summary(
    frame: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print("PLAYER GAME TABLE SUMMARY")
    print("=" * 80)

    print(
        "Rows:",
        f"{len(frame):,}",
    )

    print(
        "Games:",
        f"{frame['game_id'].nunique():,}",
    )

    print(
        "Players:",
        f"{frame['player_id'].nunique(dropna=True):,}",
    )

    print()

    print(
        "QB passing rows:",
        f"{int(frame['pass_attempts'].gt(0).sum()):,}",
    )

    print(
        "Rushing rows:",
        f"{int(frame['rush_attempts'].gt(0).sum()):,}",
    )

    print(
        "Receiving rows:",
        f"{int(frame['receptions'].gt(0).sum()):,}",
    )

    print()

    print(
        "Total pass attempts:",
        f"{frame['pass_attempts'].sum():,.0f}",
    )

    print(
        "Total completions:",
        f"{frame['completions'].sum():,.0f}",
    )

    print(
        "Total rushing attempts:",
        f"{frame['rush_attempts'].sum():,.0f}",
    )

    print(
        "Total receptions:",
        f"{frame['receptions'].sum():,.0f}",
    )

    print()

    sample_columns = [
        "season",
        "week",
        "game_id",
        "player_name",
        "team",
        "opponent",
        "home_away",
        "completions",
        "pass_attempts",
        "passing_yards",
        "rush_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
    ]

    print(
        frame[
            sample_columns
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize CFBD player box scores "
            "into one row per player-game."
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
        f"BUILD PLAYER GAMES — {args.season}"
    )
    print("=" * 80)

    frame = build_player_games(
        args.season
    )

    output = save_player_games(
        season=args.season,
        frame=frame,
    )

    display_summary(
        frame
    )

    print()
    print(
        f"Output: {output}"
    )


if __name__ == "__main__":
    main()
