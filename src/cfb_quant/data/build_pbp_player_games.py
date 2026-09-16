from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PBP_ROOT = Path(
    "data/raw/sportsdataverse/pbp"
)

PROCESSED_ROOT = Path(
    "data/processed"
)

REPORTS_ROOT = Path(
    "reports"
)


# ============================================================
# HELPERS
# ============================================================


def bool_col(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
        )

    series = frame[
        column
    ]

    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.fillna(
            False
        )

    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


def numeric_col(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )


def normalize_player_id(
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


def safe_rate(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    result = (
        numerator
        / denominator.replace(
            0,
            np.nan,
        )
    )

    return result


# ============================================================
# PREPARE PLAY DATA
# ============================================================


def prepare_pbp(
    season: int,
) -> pd.DataFrame:
    path = (
        PBP_ROOT
        / f"cfb_pbp_{season}.parquet"
    )

    if not path.exists():
        raise RuntimeError(
            f"PBP file not found: {path}"
        )

    frame = pd.read_parquet(
        path
    ).copy()

    required = {
        "game_id",
        "season",
        "week",
        "pos_team",
        "def_pos_team",
        "pass",
        "rush",
        "completion",
        "pass_attempt",
        "target",
        "sack",
        "kneel_down",
        "passer_player_id",
        "rusher_player_id",
        "receiver_player_id",
        "EPA",
        "EPA_success",
        "start.yardsToEndzone",
    }

    missing = sorted(
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "PBP file is missing required "
            f"columns: {missing}"
        )

    frame[
        "game_id"
    ] = pd.to_numeric(
        frame[
            "game_id"
        ],
        errors="coerce",
    ).astype(
        "Int64"
    )

    frame[
        "passer_player_id"
    ] = normalize_player_id(
        frame[
            "passer_player_id"
        ]
    )

    frame[
        "rusher_player_id"
    ] = normalize_player_id(
        frame[
            "rusher_player_id"
        ]
    )

    frame[
        "receiver_player_id"
    ] = normalize_player_id(
        frame[
            "receiver_player_id"
        ]
    )

    frame[
        "_pass"
    ] = bool_col(
        frame,
        "pass",
    )

    frame[
        "_rush"
    ] = bool_col(
        frame,
        "rush",
    )

    frame[
        "_pass_attempt"
    ] = bool_col(
        frame,
        "pass_attempt",
    )

    frame[
        "_completion"
    ] = bool_col(
        frame,
        "completion",
    )

    frame[
        "_target"
    ] = bool_col(
        frame,
        "target",
    )

    frame[
        "_sack"
    ] = bool_col(
        frame,
        "sack",
    )

    frame[
        "_pass_td"
    ] = bool_col(
        frame,
        "pass_td",
    )

    frame[
        "_rush_td"
    ] = bool_col(
        frame,
        "rush_td",
    )

    frame[
        "_kneel"
    ] = bool_col(
        frame,
        "kneel_down",
    )

    frame[
        "_success"
    ] = bool_col(
        frame,
        "EPA_success",
    )

    frame[
        "_explosive_pass"
    ] = bool_col(
        frame,
        "EPA_explosive_pass",
    )

    frame[
        "_explosive_rush"
    ] = bool_col(
        frame,
        "EPA_explosive_rush",
    )

    frame[
        "_short_rush_attempt"
    ] = bool_col(
        frame,
        "short_rush_attempt",
    )

    frame[
        "_power_rush_attempt"
    ] = bool_col(
        frame,
        "power_rush_attempt",
    )

    frame[
        "_early_down"
    ] = bool_col(
        frame,
        "early_down",
    )

    frame[
        "_late_down"
    ] = bool_col(
        frame,
        "late_down",
    )

    frame[
        "_epa"
    ] = numeric_col(
        frame,
        "EPA",
    )

    frame[
        "_air_yards"
    ] = numeric_col(
        frame,
        "air_yards",
    )

    frame[
        "_yac"
    ] = numeric_col(
        frame,
        "yards_after_catch",
    )

    frame[
        "_rush_yards"
    ] = numeric_col(
        frame,
        "yds_rushed",
    )

    frame[
        "_yards_to_endzone"
    ] = numeric_col(
        frame,
        "start.yardsToEndzone",
    )

    frame[
        "_red_zone"
    ] = (
        frame[
            "_yards_to_endzone"
        ].le(
            20
        )
    )

    return frame


# ============================================================
# PASSER FEATURES
# ============================================================


def build_passer_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = frame[
        frame[
            "passer_player_id"
        ].notna()
        & (
            frame[
                "_pass_attempt"
            ]
            | frame[
                "_sack"
            ]
        )
    ].copy()

    if rows.empty:
        return pd.DataFrame()

    rows[
        "_dropback"
    ] = True

    grouped = (
        rows.groupby(
            [
                "game_id",
                "passer_player_id",
            ],
            as_index=False,
        )
        .agg(
            season=(
                "season",
                "first",
            ),
            week=(
                "week",
                "first",
            ),
            team=(
                "pos_team",
                "first",
            ),
            opponent=(
                "def_pos_team",
                "first",
            ),
            player_name=(
                "passer_player_name",
                "first",
            ),
            dropbacks=(
                "_dropback",
                "sum",
            ),
            pass_attempts_pbp=(
                "_pass_attempt",
                "sum",
            ),
            completions_pbp=(
                "_completion",
                "sum",
            ),
            sacks_taken=(
                "_sack",
                "sum",
            ),
            pass_touchdowns_pbp=(
                "_pass_td",
                "sum",
            ),
            pass_epa=(
                "_epa",
                "sum",
            ),
            pass_successes=(
                "_success",
                "sum",
            ),
            total_air_yards=(
                "_air_yards",
                "sum",
            ),
            red_zone_dropbacks=(
                "_red_zone",
                "sum",
            ),
            early_down_dropbacks=(
                "_early_down",
                "sum",
            ),
            late_down_dropbacks=(
                "_late_down",
                "sum",
            ),
        )
    )

    grouped = grouped.rename(
        columns={
            "passer_player_id": (
                "player_id"
            )
        }
    )

    grouped[
        "completion_rate_pbp"
    ] = safe_rate(
        grouped[
            "completions_pbp"
        ],
        grouped[
            "pass_attempts_pbp"
        ],
    )

    grouped[
        "pass_success_rate"
    ] = safe_rate(
        grouped[
            "pass_successes"
        ],
        grouped[
            "dropbacks"
        ],
    )

    grouped[
        "epa_per_dropback"
    ] = safe_rate(
        grouped[
            "pass_epa"
        ],
        grouped[
            "dropbacks"
        ],
    )

    grouped[
        "air_yards_per_attempt"
    ] = safe_rate(
        grouped[
            "total_air_yards"
        ],
        grouped[
            "pass_attempts_pbp"
        ],
    )

    return grouped


# ============================================================
# RUSHER FEATURES
# ============================================================


def build_rusher_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = frame[
        frame[
            "rusher_player_id"
        ].notna()
        & frame[
            "_rush"
        ]
    ].copy()

    if rows.empty:
        return pd.DataFrame()

    rows[
        "_rush_play"
    ] = True

    rows[
        "_non_kneel_carry"
    ] = (
        ~rows[
            "_kneel"
        ]
    )

    grouped = (
        rows.groupby(
            [
                "game_id",
                "rusher_player_id",
            ],
            as_index=False,
        )
        .agg(
            season=(
                "season",
                "first",
            ),
            week=(
                "week",
                "first",
            ),
            team=(
                "pos_team",
                "first",
            ),
            opponent=(
                "def_pos_team",
                "first",
            ),
            player_name=(
                "rusher_player_name",
                "first",
            ),
            rush_plays_pbp=(
                "_rush_play",
                "sum",
            ),
            non_kneel_carries=(
                "_non_kneel_carry",
                "sum",
            ),
            kneels=(
                "_kneel",
                "sum",
            ),
            rushing_yards_pbp=(
                "_rush_yards",
                "sum",
            ),
            rushing_touchdowns_pbp=(
                "_rush_td",
                "sum",
            ),
            rush_epa=(
                "_epa",
                "sum",
            ),
            rush_successes=(
                "_success",
                "sum",
            ),
            red_zone_carries=(
                "_red_zone",
                "sum",
            ),
            short_yardage_carries=(
                "_short_rush_attempt",
                "sum",
            ),
            power_carries=(
                "_power_rush_attempt",
                "sum",
            ),
            explosive_rushes=(
                "_explosive_rush",
                "sum",
            ),
        )
    )

    grouped = grouped.rename(
        columns={
            "rusher_player_id": (
                "player_id"
            )
        }
    )

    grouped[
        "rush_success_rate"
    ] = safe_rate(
        grouped[
            "rush_successes"
        ],
        grouped[
            "rush_plays_pbp"
        ],
    )

    grouped[
        "rush_epa_per_play"
    ] = safe_rate(
        grouped[
            "rush_epa"
        ],
        grouped[
            "rush_plays_pbp"
        ],
    )

    return grouped


# ============================================================
# RECEIVER FEATURES
# ============================================================


def build_receiver_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = frame[
        frame[
            "receiver_player_id"
        ].notna()
        & frame[
            "_target"
        ]
    ].copy()

    if rows.empty:
        return pd.DataFrame()

    rows[
        "_target_play"
    ] = True

    grouped = (
        rows.groupby(
            [
                "game_id",
                "receiver_player_id",
            ],
            as_index=False,
        )
        .agg(
            season=(
                "season",
                "first",
            ),
            week=(
                "week",
                "first",
            ),
            team=(
                "pos_team",
                "first",
            ),
            opponent=(
                "def_pos_team",
                "first",
            ),
            player_name=(
                "receiver_player_name",
                "first",
            ),
            targets_pbp=(
                "_target_play",
                "sum",
            ),
            receptions_pbp=(
                "_completion",
                "sum",
            ),
            receiving_touchdowns_pbp=(
                "_pass_td",
                "sum",
            ),
            receiving_epa=(
                "_epa",
                "sum",
            ),
            receiving_successes=(
                "_success",
                "sum",
            ),
            total_target_air_yards=(
                "_air_yards",
                "sum",
            ),
            total_yac=(
                "_yac",
                "sum",
            ),
            red_zone_targets=(
                "_red_zone",
                "sum",
            ),
            explosive_pass_targets=(
                "_explosive_pass",
                "sum",
            ),
        )
    )

    grouped = grouped.rename(
        columns={
            "receiver_player_id": (
                "player_id"
            )
        }
    )

    grouped[
        "catch_rate_pbp"
    ] = safe_rate(
        grouped[
            "receptions_pbp"
        ],
        grouped[
            "targets_pbp"
        ],
    )

    grouped[
        "receiving_success_rate"
    ] = safe_rate(
        grouped[
            "receiving_successes"
        ],
        grouped[
            "targets_pbp"
        ],
    )

    grouped[
        "receiving_epa_per_target"
    ] = safe_rate(
        grouped[
            "receiving_epa"
        ],
        grouped[
            "targets_pbp"
        ],
    )

    grouped[
        "air_yards_per_target"
    ] = safe_rate(
        grouped[
            "total_target_air_yards"
        ],
        grouped[
            "targets_pbp"
        ],
    )

    return grouped


# ============================================================
# COMBINE PLAYER ROLES
# ============================================================


def merge_role_tables(
    passer: pd.DataFrame,
    rusher: pd.DataFrame,
    receiver: pd.DataFrame,
) -> pd.DataFrame:
    metadata = [
        "game_id",
        "player_id",
    ]

    frames = [
        frame
        for frame in [
            passer,
            rusher,
            receiver,
        ]
        if not frame.empty
    ]

    if not frames:
        raise RuntimeError(
            "No PBP player rows were produced."
        )

    combined = frames[
        0
    ].copy()

    for other in frames[
        1:
    ]:
        descriptive = [
            column
            for column in [
                "season",
                "week",
                "team",
                "opponent",
                "player_name",
            ]
            if column in other.columns
        ]

        other_copy = other.drop(
            columns=descriptive,
            errors="ignore",
        )

        combined = combined.merge(
            other_copy,
            on=metadata,
            how="outer",
            validate="one_to_one",
        )

    # --------------------------------------------------------
    # Restore descriptive information from the raw role
    # tables in priority order.
    # --------------------------------------------------------

    descriptors = pd.concat(
        [
            frame[
                [
                    "game_id",
                    "player_id",
                    "season",
                    "week",
                    "team",
                    "opponent",
                    "player_name",
                ]
            ]
            for frame in frames
        ],
        ignore_index=True,
    )

    descriptors = (
        descriptors.drop_duplicates(
            subset=[
                "game_id",
                "player_id",
            ],
            keep="first",
        )
    )

    combined = combined.drop(
        columns=[
            "season",
            "week",
            "team",
            "opponent",
            "player_name",
        ],
        errors="ignore",
    )

    combined = combined.merge(
        descriptors,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        validate="one_to_one",
    )

    front = [
        "game_id",
        "season",
        "week",
        "player_id",
        "player_name",
        "team",
        "opponent",
    ]

    remaining = [
        column
        for column in combined.columns
        if column not in front
    ]

    combined = combined[
        front
        + remaining
    ]

    return combined.sort_values(
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


# ============================================================
# CFBD RECONCILIATION
# ============================================================


def reconcile_with_box_scores(
    *,
    season: int,
    pbp_players: pd.DataFrame,
) -> pd.DataFrame:
    path = (
        PROCESSED_ROOT
        / (
            f"player_games_props_"
            f"{season}.parquet"
        )
    )

    if not path.exists():
        raise RuntimeError(
            "Validated CFBD player table "
            f"not found: {path}"
        )

    box = pd.read_parquet(
        path
    ).copy()

    box[
        "player_id"
    ] = normalize_player_id(
        box[
            "player_id"
        ]
    )

    box = box[
        box[
            "player_id"
        ].notna()
    ].copy()

    audit = box.merge(
        pbp_players,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        suffixes=(
            "_box",
            "_pbp",
        ),
        validate="one_to_one",
    )

    comparisons = {
        "pass_attempt_delta": (
            "pass_attempts",
            "pass_attempts_pbp",
        ),
        "completion_delta": (
            "completions",
            "completions_pbp",
        ),
        "rush_attempt_delta": (
            "rush_attempts",
            "rush_plays_pbp",
        ),
        "rushing_yards_delta": (
            "rushing_yards",
            "rushing_yards_pbp",
        ),
        "reception_delta": (
            "receptions",
            "receptions_pbp",
        ),
    }

    for output, (
        box_column,
        pbp_column,
    ) in comparisons.items():
        if pbp_column not in audit.columns:
            audit[
                pbp_column
            ] = 0.0

        audit[
            output
        ] = (
            pd.to_numeric(
                audit[
                    pbp_column
                ],
                errors="coerce",
            ).fillna(
                0.0
            )
            - pd.to_numeric(
                audit[
                    box_column
                ],
                errors="coerce",
            ).fillna(
                0.0
            )
        )

    return audit


# ============================================================
# SUMMARY
# ============================================================


def print_audit_metric(
    audit: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    values = pd.to_numeric(
        audit[
            column
        ],
        errors="coerce",
    ).dropna()

    print()
    print(label)

    print(
        "  rows:",
        f"{len(values):,}",
    )

    print(
        "  exact:",
        f"{values.eq(0).mean() * 100:.2f}%",
    )

    print(
        "  within 1:",
        f"{values.abs().le(1).mean() * 100:.2f}%",
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
    players: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    print()
    print("=" * 80)
    print(
        f"PBP PLAYER OPPORTUNITY SUMMARY — {season}"
    )
    print("=" * 80)

    print()
    print(
        "Player-game rows:",
        f"{len(players):,}",
    )

    print(
        "Games:",
        f"{players['game_id'].nunique():,}",
    )

    print(
        "Players:",
        f"{players['player_id'].nunique():,}",
    )

    if "dropbacks" in players.columns:
        print(
            "QB player-games:",
            f"{players['dropbacks'].notna().sum():,}",
        )

    if "rush_plays_pbp" in players.columns:
        print(
            "Rusher player-games:",
            f"{players['rush_plays_pbp'].notna().sum():,}",
        )

    if "targets_pbp" in players.columns:
        print(
            "Targeted receiver player-games:",
            f"{players['targets_pbp'].notna().sum():,}",
        )

    print()
    print("=" * 80)
    print("PBP VS VALIDATED CFBD BOX SCORE")
    print("=" * 80)

    print_audit_metric(
        audit,
        "pass_attempt_delta",
        "PASS ATTEMPTS",
    )

    print_audit_metric(
        audit,
        "completion_delta",
        "COMPLETIONS",
    )

    print_audit_metric(
        audit,
        "rush_attempt_delta",
        "RUSH ATTEMPTS",
    )

    print_audit_metric(
        audit,
        "rushing_yards_delta",
        "RUSHING YARDS",
    )

    print_audit_metric(
        audit,
        "reception_delta",
        "RECEPTIONS",
    )


# ============================================================
# MAIN BUILD
# ============================================================


def build(
    season: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pbp = prepare_pbp(
        season
    )

    print(
        "Raw PBP rows:",
        f"{len(pbp):,}",
    )

    passer = build_passer_rows(
        pbp
    )

    rusher = build_rusher_rows(
        pbp
    )

    receiver = build_receiver_rows(
        pbp
    )

    print(
        "Passer rows:",
        f"{len(passer):,}",
    )

    print(
        "Rusher rows:",
        f"{len(rusher):,}",
    )

    print(
        "Receiver rows:",
        f"{len(receiver):,}",
    )

    players = merge_role_tables(
        passer,
        rusher,
        receiver,
    )

    audit = reconcile_with_box_scores(
        season=season,
        pbp_players=players,
    )

    return (
        players,
        audit,
    )


def save_outputs(
    *,
    season: int,
    players: pd.DataFrame,
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

    player_path = (
        PROCESSED_ROOT
        / (
            f"pbp_player_games_"
            f"{season}.parquet"
        )
    )

    audit_path = (
        REPORTS_ROOT
        / (
            f"pbp_player_reconciliation_"
            f"{season}.csv"
        )
    )

    players.to_parquet(
        player_path,
        index=False,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    return (
        player_path,
        audit_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build player-game opportunity "
            "features from CFB play-by-play."
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
        f"BUILD PBP PLAYER GAMES — "
        f"{args.season}"
    )
    print("=" * 80)
    print()

    players, audit = build(
        args.season
    )

    player_path, audit_path = (
        save_outputs(
            season=args.season,
            players=players,
            audit=audit,
        )
    )

    display_summary(
        season=args.season,
        players=players,
        audit=audit,
    )

    print()
    print(
        f"PBP player games: {player_path}"
    )

    print(
        f"Audit: {audit_path}"
    )


if __name__ == "__main__":
    main()
