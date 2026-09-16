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


def bool_col(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
        )

    series = frame[column]

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


def build_pbp_team_totals(
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

    frame["game_id"] = normalize_id(
        frame["game_id"]
    )

    frame["team_id"] = normalize_id(
        frame["pos_team_id"]
    )

    frame["_pass_attempt"] = bool_col(
        frame,
        "pass_attempt",
    )

    frame["_completion"] = bool_col(
        frame,
        "completion",
    )

    frame["_target"] = bool_col(
        frame,
        "target",
    )

    frame["_rush"] = bool_col(
        frame,
        "rush",
    )

    frame["_sack"] = bool_col(
        frame,
        "sack",
    )

    frame["_penalty_no_play"] = bool_col(
        frame,
        "penalty_no_play",
    )

    frame["_valid_play"] = (
        ~frame["_penalty_no_play"]
    )

    frame["_pass_attempt_valid"] = (
        frame["_pass_attempt"]
        & frame["_valid_play"]
    )

    frame["_completion_valid"] = (
        frame["_completion"]
        & frame["_valid_play"]
    )

    frame["_target_valid"] = (
        frame["_target"]
        & frame["_valid_play"]
    )

    frame["_rush_valid"] = (
        frame["_rush"]
        & frame["_valid_play"]
    )

    frame["_sack_valid"] = (
        frame["_sack"]
        & frame["_valid_play"]
    )

    usable = frame[
        frame["game_id"].notna()
        & frame["team_id"].notna()
    ].copy()

    grouped = (
        usable.groupby(
            [
                "game_id",
                "team_id",
            ],
            as_index=False,
        )
        .agg(
            pbp_team_name=(
                "pos_team",
                "first",
            ),
            pbp_rows=(
                "game_id",
                "size",
            ),

            pbp_pass_attempts_raw=(
                "_pass_attempt",
                "sum",
            ),
            pbp_completions_raw=(
                "_completion",
                "sum",
            ),

            pbp_pass_attempts_valid=(
                "_pass_attempt_valid",
                "sum",
            ),
            pbp_completions_valid=(
                "_completion_valid",
                "sum",
            ),

            pbp_targets_valid=(
                "_target_valid",
                "sum",
            ),
            pbp_rushes_valid=(
                "_rush_valid",
                "sum",
            ),
            pbp_sacks_valid=(
                "_sack_valid",
                "sum",
            ),

            penalty_no_plays=(
                "_penalty_no_play",
                "sum",
            ),
        )
    )

    return grouped


def build_audit(
    season: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    team_path = (
        PROCESSED_ROOT
        / f"team_games_{season}.parquet"
    )

    if not team_path.exists():
        raise RuntimeError(
            f"Team-game table not found: "
            f"{team_path}"
        )

    official = pd.read_parquet(
        team_path
    ).copy()

    official["game_id"] = normalize_id(
        official["game_id"]
    )

    official["team_id"] = normalize_id(
        official["team_id"]
    )

    pbp = build_pbp_team_totals(
        season
    )

    keep = [
        "game_id",
        "season",
        "week",
        "season_type",
        "team_id",
        "team",
        "opponent",
        "home_away",
        "completions",
        "pass_attempts",
        "rush_attempts",
        "rushing_yards",
    ]

    official = official[
        [
            column
            for column in keep
            if column in official.columns
        ]
    ].copy()

    audit = official.merge(
        pbp,
        on=[
            "game_id",
            "team_id",
        ],
        how="left",
        validate="one_to_one",
    )

    audit["pbp_present"] = (
        audit["pbp_rows"].notna()
    )

    count_columns = [
        "pbp_rows",
        "pbp_pass_attempts_raw",
        "pbp_completions_raw",
        "pbp_pass_attempts_valid",
        "pbp_completions_valid",
        "pbp_targets_valid",
        "pbp_rushes_valid",
        "pbp_sacks_valid",
        "penalty_no_plays",
    ]

    for column in count_columns:
        audit[column] = (
            pd.to_numeric(
                audit[column],
                errors="coerce",
            )
            .fillna(0)
        )

    # --------------------------------------------------------
    # RAW PBP vs official
    # --------------------------------------------------------

    audit[
        "pass_attempt_delta_raw"
    ] = (
        audit["pbp_pass_attempts_raw"]
        - audit["pass_attempts"]
    )

    audit[
        "completion_delta_raw"
    ] = (
        audit["pbp_completions_raw"]
        - audit["completions"]
    )

    # --------------------------------------------------------
    # Penalty-no-play filtered PBP vs official
    # --------------------------------------------------------

    audit[
        "pass_attempt_delta_valid"
    ] = (
        audit["pbp_pass_attempts_valid"]
        - audit["pass_attempts"]
    )

    audit[
        "completion_delta_valid"
    ] = (
        audit["pbp_completions_valid"]
        - audit["completions"]
    )

    # --------------------------------------------------------
    # Quality levels
    #
    # STRICT:
    # exact official pass attempts AND completions.
    #
    # TOLERANT:
    # both metrics within 1.
    #
    # We preserve both flags. We do NOT silently decide that
    # an inaccurate game is acceptable.
    # --------------------------------------------------------

    audit[
        "pbp_strict"
    ] = (
        audit["pbp_present"]
        & audit[
            "pass_attempt_delta_valid"
        ].eq(0)
        & audit[
            "completion_delta_valid"
        ].eq(0)
    )

    audit[
        "pbp_tolerant"
    ] = (
        audit["pbp_present"]
        & audit[
            "pass_attempt_delta_valid"
        ].abs().le(1)
        & audit[
            "completion_delta_valid"
        ].abs().le(1)
    )

    # Diagnostic classification only.
    audit[
        "pbp_issue"
    ] = "OK"

    audit.loc[
        ~audit["pbp_present"],
        "pbp_issue",
    ] = "MISSING"

    likely_extra = (
        audit["pbp_present"]
        & (
            audit[
                "pass_attempt_delta_valid"
            ].ge(5)
            | audit[
                "completion_delta_valid"
            ].ge(5)
        )
    )

    audit.loc[
        likely_extra,
        "pbp_issue",
    ] = "LIKELY_DUPLICATED_OR_EXTRA"

    likely_missing = (
        audit["pbp_present"]
        & (
            audit[
                "pass_attempt_delta_valid"
            ].le(-5)
            | audit[
                "completion_delta_valid"
            ].le(-5)
        )
    )

    audit.loc[
        likely_missing,
        "pbp_issue",
    ] = "LIKELY_PARTIAL"

    smaller_mismatch = (
        audit["pbp_present"]
        & ~audit["pbp_tolerant"]
        & ~likely_extra
        & ~likely_missing
    )

    audit.loc[
        smaller_mismatch,
        "pbp_issue",
    ] = "MISMATCH"

    # --------------------------------------------------------
    # GAME-LEVEL QUALITY
    #
    # Advanced game-state/PBP features should only treat the
    # game as fully clean when BOTH offenses validate.
    # --------------------------------------------------------

    game_quality = (
        audit.groupby(
            "game_id",
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
            team_rows=(
                "team_id",
                "size",
            ),
            teams_with_pbp=(
                "pbp_present",
                "sum",
            ),
            strict_teams=(
                "pbp_strict",
                "sum",
            ),
            tolerant_teams=(
                "pbp_tolerant",
                "sum",
            ),
            max_abs_pass_delta=(
                "pass_attempt_delta_valid",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    )
                    .abs()
                    .max()
                ),
            ),
            max_abs_completion_delta=(
                "completion_delta_valid",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    )
                    .abs()
                    .max()
                ),
            ),
        )
    )

    game_quality[
        "game_pbp_strict"
    ] = (
        game_quality[
            "team_rows"
        ].eq(2)
        & game_quality[
            "strict_teams"
        ].eq(2)
    )

    game_quality[
        "game_pbp_tolerant"
    ] = (
        game_quality[
            "team_rows"
        ].eq(2)
        & game_quality[
            "tolerant_teams"
        ].eq(2)
    )

    return (
        audit,
        game_quality,
    )


def pct(
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


def display_summary(
    *,
    season: int,
    audit: pd.DataFrame,
    games: pd.DataFrame,
) -> None:
    team_count = len(
        audit
    )

    game_count = len(
        games
    )

    team_present = int(
        audit[
            "pbp_present"
        ].sum()
    )

    team_strict = int(
        audit[
            "pbp_strict"
        ].sum()
    )

    team_tolerant = int(
        audit[
            "pbp_tolerant"
        ].sum()
    )

    game_strict = int(
        games[
            "game_pbp_strict"
        ].sum()
    )

    game_tolerant = int(
        games[
            "game_pbp_tolerant"
        ].sum()
    )

    print()
    print("=" * 80)
    print(
        f"PBP QUALITY AUDIT — {season}"
    )
    print("=" * 80)

    print()
    print(
        "Official team-games:",
        f"{team_count:,}",
    )

    print(
        "Official games:",
        f"{game_count:,}",
    )

    print()

    print(
        "Team-games with PBP:",
        f"{team_present:,} "
        f"({pct(team_present, team_count):.2f}%)",
    )

    print(
        "STRICT team-games:",
        f"{team_strict:,} "
        f"({pct(team_strict, team_count):.2f}%)",
    )

    print(
        "TOLERANT team-games:",
        f"{team_tolerant:,} "
        f"({pct(team_tolerant, team_count):.2f}%)",
    )

    print()

    print(
        "STRICT complete games:",
        f"{game_strict:,} "
        f"({pct(game_strict, game_count):.2f}%)",
    )

    print(
        "TOLERANT complete games:",
        f"{game_tolerant:,} "
        f"({pct(game_tolerant, game_count):.2f}%)",
    )

    print()
    print("ISSUE COUNTS")

    print(
        audit[
            "pbp_issue"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("=" * 80)
    print(
        "RAW VS NO-PLAY-FILTERED ACCURACY"
    )
    print("=" * 80)

    present = audit[
        audit[
            "pbp_present"
        ]
    ]

    for column, label in [
        (
            "pass_attempt_delta_raw",
            "PASS ATTEMPTS — RAW",
        ),
        (
            "pass_attempt_delta_valid",
            "PASS ATTEMPTS — NO-PLAY FILTERED",
        ),
        (
            "completion_delta_raw",
            "COMPLETIONS — RAW",
        ),
        (
            "completion_delta_valid",
            "COMPLETIONS — NO-PLAY FILTERED",
        ),
    ]:
        values = (
            pd.to_numeric(
                present[column],
                errors="coerce",
            )
            .dropna()
        )

        print()
        print(label)

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

    print()
    print("=" * 80)
    print("WORST TEAM-GAME MISMATCHES")
    print("=" * 80)

    worst = audit.copy()

    worst[
        "_severity"
    ] = (
        worst[
            "pass_attempt_delta_valid"
        ].abs()
        + worst[
            "completion_delta_valid"
        ].abs()
    )

    columns = [
        "game_id",
        "week",
        "team",
        "opponent",
        "pass_attempts",
        "pbp_pass_attempts_valid",
        "pass_attempt_delta_valid",
        "completions",
        "pbp_completions_valid",
        "completion_delta_valid",
        "pbp_rows",
        "pbp_issue",
    ]

    print(
        worst.sort_values(
            "_severity",
            ascending=False,
        )[
            columns
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


def save_outputs(
    *,
    season: int,
    audit: pd.DataFrame,
    games: pd.DataFrame,
) -> tuple[
    Path,
    Path,
]:
    REPORTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    team_path = (
        REPORTS_ROOT
        / f"pbp_quality_team_{season}.csv"
    )

    game_path = (
        REPORTS_ROOT
        / f"pbp_quality_game_{season}.csv"
    )

    audit.to_csv(
        team_path,
        index=False,
    )

    games.to_csv(
        game_path,
        index=False,
    )

    return (
        team_path,
        game_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    audit, games = build_audit(
        args.season
    )

    team_path, game_path = save_outputs(
        season=args.season,
        audit=audit,
        games=games,
    )

    display_summary(
        season=args.season,
        audit=audit,
        games=games,
    )

    print()
    print(
        "Team audit:",
        team_path,
    )

    print(
        "Game audit:",
        game_path,
    )


if __name__ == "__main__":
    main()
