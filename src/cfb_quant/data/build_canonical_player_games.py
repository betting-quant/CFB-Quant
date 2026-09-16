from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_quant.data.zero_catch_receivers import append_safe_zero_catch_receivers


PROCESSED_ROOT = Path(
    "data/processed"
)

REPORTS_ROOT = Path(
    "reports"
)


# ============================================================
# HELPERS
# ============================================================


def normalize_id(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .round()
        .astype(
            "Int64"
        )
    )


def safe_pct(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


# ============================================================
# LOAD OFFICIAL PLAYER OUTCOMES
# ============================================================


def load_official_players(
    season: int,
) -> pd.DataFrame:
    path = (
        PROCESSED_ROOT
        / f"player_games_props_{season}.parquet"
    )

    if not path.exists():
        raise RuntimeError(
            "Official player-game table "
            f"not found: {path}"
        )

    frame = pd.read_parquet(
        path
    ).copy()

    frame["game_id"] = normalize_id(
        frame["game_id"]
    )

    frame["player_id"] = normalize_id(
        frame["player_id"]
    )

    # The prop-safe table should already exclude
    # pseudo-player "Team" rows, but protect against
    # accidental leakage anyway.
    frame = frame[
        ~(
            frame["player_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("team")
        )
    ].copy()

    duplicate = frame.duplicated(
        subset=[
            "game_id",
            "player_id",
        ],
        keep=False,
    )

    if duplicate.any():
        bad = frame.loc[
            duplicate,
            [
                "game_id",
                "player_id",
                "player_name",
                "team",
            ],
        ]

        print()
        print(
            "DUPLICATE OFFICIAL PLAYER-GAME IDS"
        )

        print(
            bad.head(20).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Official player table is not "
            "unique on game_id + player_id."
        )

    return frame


# ============================================================
# ADD CANONICAL TEAM IDS
# ============================================================


def add_team_ids(
    *,
    season: int,
    players: pd.DataFrame,
) -> pd.DataFrame:
    path = (
        PROCESSED_ROOT
        / f"team_games_{season}.parquet"
    )

    if not path.exists():
        raise RuntimeError(
            f"Team-game table not found: {path}"
        )

    teams = pd.read_parquet(
        path
    ).copy()

    teams["game_id"] = normalize_id(
        teams["game_id"]
    )

    teams["team_id"] = normalize_id(
        teams["team_id"]
    )

    team_lookup = (
        teams[
            [
                "game_id",
                "team",
                "team_id",
            ]
        ]
        .drop_duplicates()
    )

    opponent_lookup = (
        team_lookup.rename(
            columns={
                "team": "opponent",
                "team_id": (
                    "opponent_team_id"
                ),
            }
        )
    )

    result = players.merge(
        team_lookup,
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        opponent_lookup,
        on=[
            "game_id",
            "opponent",
        ],
        how="left",
        validate="many_to_one",
    )

    missing = result[
        "team_id"
    ].isna()

    if missing.any():
        print()
        print(
            "PLAYER ROWS MISSING TEAM ID"
        )

        print(
            result.loc[
                missing,
                [
                    "game_id",
                    "player_name",
                    "team",
                    "opponent",
                ],
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{int(missing.sum())} player rows "
            "could not be assigned a team ID."
        )

    return result


# ============================================================
# PBP QUALITY
# ============================================================


def load_quality(
    season: int,
) -> pd.DataFrame:
    path = (
        REPORTS_ROOT
        / f"pbp_quality_team_{season}.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"PBP quality table not found: {path}"
        )

    frame = pd.read_csv(
        path
    ).copy()

    frame["game_id"] = normalize_id(
        frame["game_id"]
    )

    frame["team_id"] = normalize_id(
        frame["team_id"]
    )

    frame[
        "pbp_pass_strict"
    ] = (
        frame[
            "pbp_present"
        ].astype(bool)
        & pd.to_numeric(
            frame[
                "pass_attempt_delta_valid"
            ],
            errors="coerce",
        ).eq(0)
        & pd.to_numeric(
            frame[
                "completion_delta_valid"
            ],
            errors="coerce",
        ).eq(0)
    )

    frame[
        "pbp_pass_tolerant"
    ] = (
        frame[
            "pbp_present"
        ].astype(bool)
        & pd.to_numeric(
            frame[
                "pass_attempt_delta_valid"
            ],
            errors="coerce",
        ).abs().le(1)
        & pd.to_numeric(
            frame[
                "completion_delta_valid"
            ],
            errors="coerce",
        ).abs().le(1)
    )

    frame[
        "pbp_rush_attempt_proxy"
    ] = (
        pd.to_numeric(
            frame[
                "pbp_rushes_valid"
            ],
            errors="coerce",
        ).fillna(0)
        + pd.to_numeric(
            frame[
                "pbp_sacks_valid"
            ],
            errors="coerce",
        ).fillna(0)
    )

    frame[
        "pbp_rush_attempt_delta"
    ] = (
        frame[
            "pbp_rush_attempt_proxy"
        ]
        - pd.to_numeric(
            frame[
                "rush_attempts"
            ],
            errors="coerce",
        )
    )

    frame[
        "pbp_rush_strict"
    ] = (
        frame[
            "pbp_present"
        ].astype(bool)
        & frame[
            "pbp_rush_attempt_delta"
        ].eq(0)
    )

    frame[
        "pbp_rush_tolerant"
    ] = (
        frame[
            "pbp_present"
        ].astype(bool)
        & frame[
            "pbp_rush_attempt_delta"
        ].abs().le(1)
    )

    frame[
        "pbp_receiving_tolerant"
    ] = frame[
        "pbp_pass_tolerant"
    ]

    frame[
        "pbp_full_tolerant"
    ] = (
        frame[
            "pbp_pass_tolerant"
        ]
        & frame[
            "pbp_rush_tolerant"
        ]
    )

    keep = [
        "game_id",
        "team_id",
        "pbp_present",
        "pbp_issue",
        "pbp_pass_strict",
        "pbp_pass_tolerant",
        "pbp_rush_attempt_proxy",
        "pbp_rush_attempt_delta",
        "pbp_rush_strict",
        "pbp_rush_tolerant",
        "pbp_receiving_tolerant",
        "pbp_full_tolerant",
    ]

    return frame[
        keep
    ].copy()


# ============================================================
# PBP PLAYER FEATURES
# ============================================================


def load_pbp_players(
    season: int,
) -> pd.DataFrame:
    path = (
        PROCESSED_ROOT
        / f"pbp_player_games_{season}.parquet"
    )

    if not path.exists():
        raise RuntimeError(
            f"PBP player table not found: {path}"
        )

    frame = pd.read_parquet(
        path
    ).copy()

    frame["game_id"] = normalize_id(
        frame["game_id"]
    )

    frame["player_id"] = normalize_id(
        frame["player_id"]
    )

    duplicate = frame.duplicated(
        subset=[
            "game_id",
            "player_id",
        ],
        keep=False,
    )

    if duplicate.any():
        raise RuntimeError(
            "PBP player table is not unique "
            "on game_id + player_id."
        )

    # Rename descriptive columns so the official CFBD
    # fields remain canonical.
    rename = {}

    for column in [
        "season",
        "week",
        "player_name",
        "team",
        "opponent",
    ]:
        if column in frame.columns:
            rename[
                column
            ] = (
                f"{column}_pbp"
            )

    return frame.rename(
        columns=rename
    )


# ============================================================
# QUALITY MASKS
# ============================================================


PASS_FEATURES = [
    "dropbacks",
    "pass_attempts_pbp",
    "completions_pbp",
    "sacks_taken",
    "pass_touchdowns_pbp",
    "pass_epa",
    "pass_successes",
    "total_air_yards",
    "red_zone_dropbacks",
    "early_down_dropbacks",
    "late_down_dropbacks",
    "completion_rate_pbp",
    "pass_success_rate",
    "epa_per_dropback",
    "air_yards_per_attempt",
]


RUSH_FEATURES = [
    "rush_plays_pbp",
    "non_kneel_carries",
    "kneels",
    "rushing_yards_pbp",
    "rushing_touchdowns_pbp",
    "rush_epa",
    "rush_successes",
    "red_zone_carries",
    "short_yardage_carries",
    "power_carries",
    "explosive_rushes",
    "rush_success_rate",
    "rush_epa_per_play",
]


RECEIVING_FEATURES = [
    "targets_pbp",
    "receptions_pbp",
    "receiving_touchdowns_pbp",
    "receiving_epa",
    "receiving_successes",
    "total_target_air_yards",
    "total_yac",
    "red_zone_targets",
    "explosive_pass_targets",
    "catch_rate_pbp",
    "receiving_success_rate",
    "receiving_epa_per_target",
    "air_yards_per_target",
]


def mask_untrusted_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    for column in PASS_FEATURES:
        if column in result.columns:
            result.loc[
                ~result[
                    "pbp_pass_tolerant"
                ].fillna(False),
                column,
            ] = np.nan

    for column in RECEIVING_FEATURES:
        if column in result.columns:
            result.loc[
                ~result[
                    "pbp_receiving_tolerant"
                ].fillna(False),
                column,
            ] = np.nan

    for column in RUSH_FEATURES:
        if column in result.columns:
            result.loc[
                ~result[
                    "pbp_rush_tolerant"
                ].fillna(False),
                column,
            ] = np.nan

    return result


# ============================================================
# BUILD CANONICAL TABLE
# ============================================================


def build_canonical(
    season: int,
) -> pd.DataFrame:
    official = load_official_players(
        season
    )

    official = add_team_ids(
        season=season,
        players=official,
    )

    quality = load_quality(
        season
    )

    official = official.merge(
        quality,
        on=[
            "game_id",
            "team_id",
        ],
        how="left",
        validate="many_to_one",
    )

    for column in [
        "pbp_present",
        "pbp_pass_strict",
        "pbp_pass_tolerant",
        "pbp_rush_strict",
        "pbp_rush_tolerant",
        "pbp_receiving_tolerant",
        "pbp_full_tolerant",
    ]:
        official[
            column
        ] = (
            official[
                column
            ]
            .fillna(False)
            .astype(bool)
        )

    pbp = load_pbp_players(
        season
    )

    result = official.merge(
        pbp,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        validate="one_to_one",
    )

    role_columns = [
        column
        for column in [
            "dropbacks",
            "rush_plays_pbp",
            "targets_pbp",
        ]
        if column in result.columns
    ]

    if role_columns:
        result[
            "pbp_player_matched"
        ] = (
            result[
                role_columns
            ]
            .notna()
            .any(
                axis=1
            )
        )
    else:
        result[
            "pbp_player_matched"
        ] = False

    # --------------------------------------------------------
    # Player-level feature usability
    # --------------------------------------------------------

    result[
        "pbp_pass_features_usable"
    ] = (
        result[
            "pbp_pass_tolerant"
        ]
        & result[
            "pbp_player_matched"
        ]
    )

    result[
        "pbp_rush_features_usable"
    ] = (
        result[
            "pbp_rush_tolerant"
        ]
        & result[
            "pbp_player_matched"
        ]
    )

    result[
        "pbp_receiving_features_usable"
    ] = (
        result[
            "pbp_receiving_tolerant"
        ]
        & result[
            "pbp_player_matched"
        ]
    )

    # --------------------------------------------------------
    # CRITICAL:
    #
    # Do not allow corrupted PBP values to survive into
    # downstream feature engineering.
    #
    # Official CFBD outcomes remain untouched.
    # --------------------------------------------------------

    result = mask_untrusted_features(
        result
    )

    # --------------------------------------------------------
    # Useful role flags based ONLY on official box scores.
    # --------------------------------------------------------

    result[
        "official_passer"
    ] = (
        pd.to_numeric(
            result[
                "pass_attempts"
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
    )

    result[
        "official_rusher"
    ] = (
        pd.to_numeric(
            result[
                "rush_attempts"
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
    )

    result[
        "official_receiver"
    ] = (
        pd.to_numeric(
            result[
                "receptions"
            ],
            errors="coerce",
        )
        .fillna(0)
        .gt(0)
    )

    result = append_safe_zero_catch_receivers(
        season=season,
        frame=result,
    )

    # Keep official columns near the front.
    front = [
        "game_id",
        "season",
        "week",
        "season_type",
        "player_id",
        "player_name",
        "team_id",
        "team",
        "opponent_team_id",
        "opponent",
        "home_away",
        "team_points",
        "opponent_points",

        "completions",
        "pass_attempts",
        "passing_yards",
        "passing_touchdowns",
        "interceptions",

        "rush_attempts",
        "rushing_yards",
        "rushing_touchdowns",

        "receptions",
        "receiving_yards",
        "receiving_touchdowns",

        "official_passer",
        "official_rusher",
        "official_receiver",

        "pbp_present",
        "pbp_player_matched",
        "pbp_issue",

        "pbp_pass_strict",
        "pbp_pass_tolerant",
        "pbp_rush_strict",
        "pbp_rush_tolerant",
        "pbp_receiving_tolerant",
        "pbp_full_tolerant",

        "pbp_pass_features_usable",
        "pbp_rush_features_usable",
        "pbp_receiving_features_usable",
    ]

    front = [
        column
        for column in front
        if column in result.columns
    ]

    remaining = [
        column
        for column in result.columns
        if column not in front
    ]

    result = result[
        front
        + remaining
    ]

    result = result.sort_values(
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

    return result


# ============================================================
# VALIDATION
# ============================================================


def display_summary(
    frame: pd.DataFrame,
    season: int,
) -> None:
    total = len(
        frame
    )

    matched = int(
        frame[
            "pbp_player_matched"
        ].sum()
    )

    print()
    print("=" * 80)
    print(
        f"CANONICAL PLAYER GAMES — {season}"
    )
    print("=" * 80)

    print()
    print(
        "Player-game rows:",
        f"{total:,}",
    )

    print(
        "Games:",
        f"{frame['game_id'].nunique():,}",
    )

    print(
        "Players:",
        f"{frame['player_id'].nunique():,}",
    )

    print(
        "PBP player matches:",
        f"{matched:,} "
        f"({safe_pct(matched, total):.2f}%)",
    )

    roles = [
        (
            "official_passer",
            "pbp_pass_features_usable",
            "PASSERS",
        ),
        (
            "official_rusher",
            "pbp_rush_features_usable",
            "RUSHERS",
        ),
        (
            "official_receiver",
            "pbp_receiving_features_usable",
            "RECEIVERS",
        ),
    ]

    for role_column, quality_column, label in roles:
        subset = frame[
            frame[
                role_column
            ]
        ]

        count = len(
            subset
        )

        usable = int(
            subset[
                quality_column
            ].sum()
        )

        print()
        print(label)

        print(
            "  official player-games:",
            f"{count:,}",
        )

        print(
            "  PBP features usable:",
            f"{usable:,} "
            f"({safe_pct(usable, count):.2f}%)",
        )

    print()
    print(
        "Pass-tolerant team context:",
        f"{frame['pbp_pass_tolerant'].mean() * 100:.2f}%"
    )

    print(
        "Rush-tolerant team context:",
        f"{frame['pbp_rush_tolerant'].mean() * 100:.2f}%"
    )

    print(
        "Full-tolerant team context:",
        f"{frame['pbp_full_tolerant'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Ensure official outcomes survived untouched.
    # --------------------------------------------------------

    required_outcomes = [
        "pass_attempts",
        "completions",
        "passing_yards",
        "rush_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
    ]

    print()
    print("OFFICIAL OUTCOME NULL COUNTS")

    for column in required_outcomes:
        print(
            f"  {column}: "
            f"{int(frame[column].isna().sum()):,}"
        )


# ============================================================
# SAVE
# ============================================================


def save(
    *,
    season: int,
    frame: pd.DataFrame,
) -> Path:
    PROCESSED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        PROCESSED_ROOT
        / (
            f"canonical_player_games_"
            f"{season}.parquet"
        )
    )

    temporary = (
        path.with_suffix(
            ".parquet.tmp"
        )
    )

    frame.to_parquet(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )

    return path


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    frame = build_canonical(
        args.season
    )

    output = save(
        season=args.season,
        frame=frame,
    )

    display_summary(
        frame,
        args.season,
    )

    print()
    print(
        f"Output: {output}"
    )


if __name__ == "__main__":
    main()
