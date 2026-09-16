from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_ROOT = Path("data/processed")
PBP_ROOT = Path("data/raw/sportsdataverse/pbp")


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


OFFICIAL_ZERO_COLUMNS = [
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
]


def normalize_id(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .round()
        .astype("Int64")
    )


def normalize_name(
    value,
) -> str:
    if pd.isna(value):
        return ""

    text = str(value).casefold()

    text = re.sub(
        r"\b(jr|sr|ii|iii|iv)\b",
        "",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )

    return text


def bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

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


def append_safe_zero_catch_receivers(
    *,
    season: int,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["game_id"] = normalize_id(
        result["game_id"]
    )

    result["player_id"] = normalize_id(
        result["player_id"]
    )

    result["team_id"] = normalize_id(
        result["team_id"]
    )

    # --------------------------------------------------------
    # Provenance columns.
    # --------------------------------------------------------

    if "zero_catch_augmented" not in result.columns:
        result[
            "zero_catch_augmented"
        ] = False

    if "receiving_outcome_source" not in result.columns:
        result[
            "receiving_outcome_source"
        ] = "cfbd_box"

    # --------------------------------------------------------
    # Load player-game PBP.
    # --------------------------------------------------------

    pbp_path = (
        PROCESSED_ROOT
        / f"pbp_player_games_{season}.parquet"
    )

    raw_path = (
        PBP_ROOT
        / f"cfb_pbp_{season}.parquet"
    )

    pbp = pd.read_parquet(
        pbp_path
    ).copy()

    raw = pd.read_parquet(
        raw_path
    ).copy()

    pbp["game_id"] = normalize_id(
        pbp["game_id"]
    )

    pbp["player_id"] = normalize_id(
        pbp["player_id"]
    )

    raw["game_id"] = normalize_id(
        raw["game_id"]
    )

    raw["pos_team_id"] = normalize_id(
        raw["pos_team_id"]
    )

    raw["receiver_player_id"] = normalize_id(
        raw["receiver_player_id"]
    )

    game_ids = set(
        result["game_id"]
        .dropna()
        .unique()
    )

    pbp = pbp[
        pbp["game_id"].isin(
            game_ids
        )
    ].copy()

    # --------------------------------------------------------
    # PBP-only player-games.
    # --------------------------------------------------------

    existing_keys = (
        result[
            [
                "game_id",
                "player_id",
            ]
        ]
        .drop_duplicates()
        .assign(
            in_canonical=True
        )
    )

    candidates = pbp.merge(
        existing_keys,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
    )

    candidates = candidates[
        candidates[
            "in_canonical"
        ].isna()
    ].copy()

    for column in [
        "targets_pbp",
        "receptions_pbp",
        "rush_plays_pbp",
        "dropbacks",
    ]:
        if column not in candidates.columns:
            candidates[column] = 0

        candidates[column] = (
            pd.to_numeric(
                candidates[column],
                errors="coerce",
            )
            .fillna(0)
        )

    # Pure receiving observation:
    # targeted, zero catches, no passing/rushing role.
    candidates = candidates[
        candidates["targets_pbp"].gt(0)
        & candidates["receptions_pbp"].eq(0)
        & candidates["rush_plays_pbp"].eq(0)
        & candidates["dropbacks"].eq(0)
    ].copy()

    initial_candidates = len(
        candidates
    )

    # --------------------------------------------------------
    # Authoritative offense team assignment from receiver
    # participant ID in the raw PBP.
    # --------------------------------------------------------

    receiver_map = raw[
        [
            "game_id",
            "pos_team_id",
            "receiver_player_id",
        ]
    ].copy()

    receiver_map = receiver_map[
        receiver_map[
            "receiver_player_id"
        ].notna()
        & receiver_map[
            "pos_team_id"
        ].notna()
    ]

    receiver_map = receiver_map.rename(
        columns={
            "receiver_player_id":
                "player_id",
        }
    )[
        [
            "game_id",
            "player_id",
            "pos_team_id",
        ]
    ].drop_duplicates()

    team_counts = (
        receiver_map.groupby(
            [
                "game_id",
                "player_id",
            ]
        )[
            "pos_team_id"
        ]
        .nunique()
        .reset_index(
            name="team_count"
        )
    )

    receiver_map = receiver_map.merge(
        team_counts,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
    )

    receiver_map = (
        receiver_map[
            receiver_map[
                "team_count"
            ].eq(1)
        ]
        .drop(
            columns="team_count"
        )
        .drop_duplicates(
            [
                "game_id",
                "player_id",
            ]
        )
    )

    candidates = candidates.merge(
        receiver_map,
        on=[
            "game_id",
            "player_id",
        ],
        how="left",
        validate="one_to_one",
    )

    candidates = candidates[
        candidates[
            "pos_team_id"
        ].notna()
    ].copy()

    # --------------------------------------------------------
    # Team/game metadata and quality flags from canonical
    # official rows.
    # --------------------------------------------------------

    context_columns = [
        column
        for column in [
            "season",
            "week",
            "season_type",
            "team",
            "opponent_team_id",
            "opponent",
            "home_away",
            "team_points",
            "opponent_points",
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
        if column in result.columns
    ]

    context = (
        result[
            [
                "game_id",
                "team_id",
            ]
            + context_columns
        ]
        .drop_duplicates(
            [
                "game_id",
                "team_id",
            ]
        )
        .rename(
            columns={
                "team_id":
                    "pos_team_id",
                **{
                    column:
                        f"ctx_{column}"
                    for column
                    in context_columns
                },
            }
        )
    )

    candidates = candidates.merge(
        context,
        on=[
            "game_id",
            "pos_team_id",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Official receiving-box closure.
    # --------------------------------------------------------

    closure_source = result.copy()

    for column in [
        "completions",
        "passing_yards",
        "receptions",
        "receiving_yards",
    ]:
        closure_source[column] = (
            pd.to_numeric(
                closure_source[column],
                errors="coerce",
            )
            .fillna(0)
        )

    closure = (
        closure_source.groupby(
            [
                "game_id",
                "team_id",
            ],
            as_index=False,
        )
        .agg(
            official_completions=(
                "completions",
                "sum",
            ),
            official_pass_yards=(
                "passing_yards",
                "sum",
            ),
            official_receptions=(
                "receptions",
                "sum",
            ),
            official_rec_yards=(
                "receiving_yards",
                "sum",
            ),
        )
    )

    closure[
        "receiving_box_closed"
    ] = (
        closure[
            "official_completions"
        ].eq(
            closure[
                "official_receptions"
            ]
        )
        & closure[
            "official_pass_yards"
        ].eq(
            closure[
                "official_rec_yards"
            ]
        )
    )

    closure = closure.rename(
        columns={
            "team_id":
                "pos_team_id",
        }
    )

    candidates = candidates.merge(
        closure[
            [
                "game_id",
                "pos_team_id",
                "receiving_box_closed",
            ]
        ],
        on=[
            "game_id",
            "pos_team_id",
        ],
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Prevent cross-provider ID aliases from generating a
    # second row for an already-existing official player.
    # --------------------------------------------------------

    existing_names = set()

    for row in result[
        [
            "game_id",
            "team_id",
            "player_name",
        ]
    ].itertuples(
        index=False
    ):
        existing_names.add(
            (
                row.game_id,
                row.team_id,
                normalize_name(
                    row.player_name
                ),
            )
        )

    name_column = (
        "player_name"
        if "player_name" in candidates.columns
        else "player_name_pbp"
    )

    candidates[
        "_name_norm"
    ] = candidates[
        name_column
    ].apply(
        normalize_name
    )

    candidates[
        "_name_collision"
    ] = [
        (
            game_id,
            team_id,
            name,
        )
        in existing_names
        for game_id, team_id, name
        in zip(
            candidates[
                "game_id"
            ],
            candidates[
                "pos_team_id"
            ],
            candidates[
                "_name_norm"
            ],
        )
    ]

    pass_quality = bool_series(
        candidates[
            "ctx_pbp_pass_tolerant"
        ]
    )

    box_closed = bool_series(
        candidates[
            "receiving_box_closed"
        ]
    )

    safe = candidates[
        pass_quality
        & box_closed
        & ~candidates[
            "_name_collision"
        ]
        & candidates[
            "_name_norm"
        ].ne("")
    ].copy()

    # --------------------------------------------------------
    # Build canonical zero-outcome rows.
    # --------------------------------------------------------

    rows = []

    for candidate in safe.itertuples(
        index=False
    ):
        values = {
            column: np.nan
            for column
            in result.columns
        }

        values[
            "game_id"
        ] = candidate.game_id

        values[
            "player_id"
        ] = candidate.player_id

        values[
            "player_name"
        ] = getattr(
            candidate,
            name_column,
        )

        values[
            "team_id"
        ] = candidate.pos_team_id

        for column in context_columns:
            values[
                column
            ] = getattr(
                candidate,
                f"ctx_{column}",
            )

        for column in OFFICIAL_ZERO_COLUMNS:
            if column in values:
                values[
                    column
                ] = 0

        for column in RECEIVING_FEATURES:
            if (
                column in values
                and hasattr(
                    candidate,
                    column,
                )
            ):
                values[
                    column
                ] = getattr(
                    candidate,
                    column,
                )

        # Preserve descriptive PBP metadata when present.
        if "player_name_pbp" in values:
            values[
                "player_name_pbp"
            ] = getattr(
                candidate,
                name_column,
            )

        if "team_pbp" in values:
            values[
                "team_pbp"
            ] = getattr(
                candidate,
                "team",
                np.nan,
            )

        if "opponent_pbp" in values:
            values[
                "opponent_pbp"
            ] = getattr(
                candidate,
                "opponent",
                np.nan,
            )

        values[
            "pbp_player_matched"
        ] = True

        values[
            "pbp_pass_features_usable"
        ] = False

        values[
            "pbp_rush_features_usable"
        ] = False

        values[
            "pbp_receiving_features_usable"
        ] = True

        values[
            "official_passer"
        ] = False

        values[
            "official_rusher"
        ] = False

        values[
            "official_receiver"
        ] = False

        values[
            "zero_catch_augmented"
        ] = True

        values[
            "receiving_outcome_source"
        ] = "derived_zero_closed_box"

        rows.append(
            values
        )

    additions = pd.DataFrame(
        rows
    )

    if not additions.empty:
        result = pd.concat(
            [
                result,
                additions,
            ],
            ignore_index=True,
            sort=False,
        )

    # --------------------------------------------------------
    # Receiving-model observation universe.
    #
    # Includes:
    #   - official receivers with >=1 reception
    #   - trusted targeted players with zero receptions
    # --------------------------------------------------------

    result[
        "zero_catch_augmented"
    ] = bool_series(
        result[
            "zero_catch_augmented"
        ]
    )

    targets = (
        pd.to_numeric(
            result.get(
                "targets_pbp",
                0,
            ),
            errors="coerce",
        )
        .fillna(0)
    )

    receiver_quality = bool_series(
        result[
            "pbp_receiving_features_usable"
        ]
    )

    official_receiver = bool_series(
        result[
            "official_receiver"
        ]
    )

    result[
        "receiving_training_observation"
    ] = (
        official_receiver
        | (
            receiver_quality
            & targets.gt(0)
        )
        | result[
            "zero_catch_augmented"
        ]
    )

    bool_columns = [
        "pbp_present",
        "pbp_player_matched",
        "pbp_pass_strict",
        "pbp_pass_tolerant",
        "pbp_rush_strict",
        "pbp_rush_tolerant",
        "pbp_receiving_tolerant",
        "pbp_full_tolerant",
        "pbp_pass_features_usable",
        "pbp_rush_features_usable",
        "pbp_receiving_features_usable",
        "official_passer",
        "official_rusher",
        "official_receiver",
        "zero_catch_augmented",
        "receiving_training_observation",
    ]

    for column in bool_columns:
        if column in result.columns:
            result[
                column
            ] = bool_series(
                result[
                    column
                ]
            )

    print()
    print("=" * 80)
    print(
        f"ZERO-CATCH RECEIVER AUGMENTATION — {season}"
    )
    print("=" * 80)

    print(
        "Pure PBP-only zero-catch candidates:",
        f"{initial_candidates:,}",
    )

    print(
        "Safe tolerant rows appended:",
        f"{len(additions):,}",
    )

    print(
        "Receiving training observations:",
        f"{int(result['receiving_training_observation'].sum()):,}",
    )

    return result
