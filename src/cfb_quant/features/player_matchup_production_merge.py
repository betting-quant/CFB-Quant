from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PLAYER_DATA = Path(
    "data/processed/cfb_pregame_features_through_2026_week_3_v0_1.parquet"
)

PLAYER_MANIFEST = Path(
    "reports/cfb_feature_manifest_through_2026_week_3_v0_1.json"
)

MATCHUP_OOF = Path(
    "data/processed/cfb_matchup_oof_features_v0_1.parquet"
)

MATCHUP_2026_WALKFORWARD = Path(
    "data/processed/cfb_matchup_walkforward_2026_weeks_1_3_v0_1.parquet"
)

OUTPUT_DATA = Path(
    "data/processed/cfb_pregame_features_matchup_through_2026_week_3_v0_1.parquet"
)

OUTPUT_MANIFEST = Path(
    "reports/cfb_feature_manifest_matchup_through_2026_week_3_v0_1.json"
)

COVERAGE_PATH = Path(
    "reports/player_matchup_production_coverage_through_2026_week_3_v0_1.csv"
)


WINNER_SOURCE_COLUMNS = {
    "matchup_proj_points":
        "matchup_ridge_points",

    "matchup_proj_pass_attempts":
        "matchup_extra_trees_pass_attempts",

    "matchup_proj_rush_attempts":
        "matchup_hist_gradient_boosting_rush_attempts",

    "matchup_proj_total_plays":
        "matchup_hist_gradient_boosting_total_plays",

    "matchup_proj_passing_yards":
        "matchup_hist_gradient_boosting_passing_yards",

    "matchup_proj_rushing_yards":
        "matchup_ridge_rushing_yards",

    "matchup_proj_possession_seconds":
        "matchup_hist_gradient_boosting_possession_seconds",
}


MATCHUP_FEATURES = list(WINNER_SOURCE_COLUMNS)


def _prepare_historical_oof(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if frame.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate historical OOF team-game rows."
        )

    missing = [
        source
        for source in WINNER_SOURCE_COLUMNS.values()
        if source not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"Historical OOF missing columns: {missing}"
        )

    compact = frame[
        [
            "game_id",
            "season",
            "week",
            "team",
            *WINNER_SOURCE_COLUMNS.values(),
        ]
    ].copy()

    compact = compact.rename(
        columns={
            source: dest
            for dest, source
            in WINNER_SOURCE_COLUMNS.items()
        }
    )

    return compact


def _prepare_2026_walkforward(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    if frame.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate 2026 walk-forward team-game rows."
        )

    missing = [
        feature
        for feature in MATCHUP_FEATURES
        if feature not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            f"2026 walk-forward missing columns: {missing}"
        )

    compact = frame[
        [
            "game_id",
            "season",
            "week",
            "team",
            *MATCHUP_FEATURES,
        ]
    ].copy()

    return compact


def main() -> None:
    print("=" * 100)
    print(
        "PLAYER + MATCHUP PRODUCTION TRAINING MERGE "
        "THROUGH 2026 WEEK 3"
    )
    print("=" * 100)

    players = pd.read_parquet(
        PLAYER_DATA
    )

    historical = pd.read_parquet(
        MATCHUP_OOF
    )

    walkforward = pd.read_parquet(
        MATCHUP_2026_WALKFORWARD
    )

    manifest = json.loads(
        PLAYER_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    original_rows = len(players)
    original_features = list(
        manifest["feature_columns"]
    )

    print(
        f"Player rows: {original_rows:,}"
    )
    print(
        f"Original features: "
        f"{len(original_features):,}"
    )

    hist = _prepare_historical_oof(
        historical
    )

    wf = _prepare_2026_walkforward(
        walkforward
    )

    print(
        f"Historical OOF team rows: "
        f"{len(hist):,}"
    )

    print(
        f"2026 walk-forward team rows: "
        f"{len(wf):,}"
    )

    if not hist["season"].between(
        2015,
        2025,
    ).all():
        raise RuntimeError(
            "Historical OOF contains season "
            "outside 2015-2025."
        )

    if not wf["season"].eq(
        2026
    ).all():
        raise RuntimeError(
            "Walk-forward file contains "
            "non-2026 rows."
        )

    if not set(
        wf["week"].dropna().unique()
    ).issubset({1, 2, 3}):
        raise RuntimeError(
            "Walk-forward file contains "
            "weeks beyond Week 3."
        )

    matchup = pd.concat(
        [
            hist,
            wf,
        ],
        ignore_index=True,
        sort=False,
    )

    if matchup.duplicated(
        ["game_id", "team"]
    ).any():
        duplicates = matchup.loc[
            matchup.duplicated(
                ["game_id", "team"],
                keep=False,
            ),
            [
                "game_id",
                "season",
                "week",
                "team",
            ],
        ]

        raise RuntimeError(
            "Duplicate combined matchup "
            "team-game rows:\n"
            + duplicates.head(
                20
            ).to_string(
                index=False
            )
        )

    merged = players.merge(
        matchup[
            [
                "game_id",
                "team",
                *MATCHUP_FEATURES,
            ]
        ],
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(merged) != original_rows:
        raise RuntimeError(
            f"Row multiplication: "
            f"{original_rows} -> "
            f"{len(merged)}"
        )

    overlap = (
        set(original_features)
        & set(MATCHUP_FEATURES)
    )

    if overlap:
        raise RuntimeError(
            f"Feature overlap: "
            f"{sorted(overlap)}"
        )

    enhanced_features = (
        original_features
        + MATCHUP_FEATURES
    )

    if len(
        enhanced_features
    ) != len(
        set(enhanced_features)
    ):
        raise RuntimeError(
            "Duplicate enhanced "
            "feature names."
        )

    direct_targets = {
        "pass_attempts",
        "completions",
        "passing_yards",
        "rush_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
    }

    leaks = (
        direct_targets
        & set(enhanced_features)
    )

    if leaks:
        raise RuntimeError(
            f"Direct target leakage: "
            f"{sorted(leaks)}"
        )

    if len(
        enhanced_features
    ) != 472:
        raise RuntimeError(
            f"Expected 472 features, "
            f"got {len(enhanced_features)}."
        )

    coverage_rows = []

    print()
    print(
        "MATCHUP COVERAGE BY SEASON"
    )
    print("-" * 100)

    for season, frame in merged.groupby(
        "season",
        sort=True,
    ):
        all_present = (
            frame[MATCHUP_FEATURES]
            .notna()
            .all(axis=1)
        )

        any_present = (
            frame[MATCHUP_FEATURES]
            .notna()
            .any(axis=1)
        )

        partial = (
            any_present
            & ~all_present
        )

        coverage_rows.append({
            "season":
                int(season),

            "rows":
                len(frame),

            "rows_with_all_matchup":
                int(all_present.sum()),

            "rows_with_any_matchup":
                int(any_present.sum()),

            "partial_matchup_rows":
                int(partial.sum()),

            "coverage_all":
                float(
                    all_present.mean()
                ),
        })

        print(
            f"{int(season)}: "
            f"{all_present.sum():,}"
            f"/{len(frame):,} "
            f"all 7 "
            f"({all_present.mean():.2%}) "
            f"| partial={partial.sum():,}"
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    row_2014 = coverage.loc[
        coverage["season"].eq(2014)
    ]

    if (
        not row_2014.empty
        and int(
            row_2014.iloc[0][
                "rows_with_any_matchup"
            ]
        ) != 0
    ):
        raise RuntimeError(
            "2014 unexpectedly has "
            "matchup projections."
        )

    row_2026 = coverage.loc[
        coverage["season"].eq(2026)
    ]

    if row_2026.empty:
        raise RuntimeError(
            "2026 player rows missing."
        )

    player_2026_games = (
        merged.loc[
            merged["season"].eq(2026),
            "game_id",
        ]
        .nunique()
    )

    matchup_2026_games = (
        matchup.loc[
            matchup["season"].eq(2026),
            "game_id",
        ]
        .nunique()
    )

    print()
    print(
        "2026 GAME COVERAGE"
    )
    print("-" * 100)

    print(
        "Player games:",
        player_2026_games,
    )

    print(
        "Walk-forward matchup games:",
        matchup_2026_games,
    )

    if player_2026_games != 260:
        raise RuntimeError(
            f"Expected 260 player games "
            f"in 2026, got "
            f"{player_2026_games}."
        )

    if matchup_2026_games != 260:
        raise RuntimeError(
            f"Expected 260 walk-forward "
            f"matchup games, got "
            f"{matchup_2026_games}."
        )

    week_game_counts = (
        merged.loc[
            merged["season"].eq(2026)
        ]
        .groupby("week")[
            "game_id"
        ]
        .nunique()
    )

    print()
    print(
        "2026 PLAYER GAMES BY WEEK"
    )
    print(
        week_game_counts
    )

    expected_week_games = {
        1: 99,
        2: 86,
        3: 75,
    }

    for week, expected in (
        expected_week_games.items()
    ):
        observed = int(
            week_game_counts.get(
                week,
                0,
            )
        )

        if observed != expected:
            raise RuntimeError(
                f"2026 Week {week}: "
                f"expected {expected} games, "
                f"got {observed}."
            )

    merged.to_parquet(
        OUTPUT_DATA,
        index=False,
    )

    enhanced_manifest = dict(
        manifest
    )

    enhanced_manifest[
        "feature_columns"
    ] = enhanced_features

    enhanced_manifest[
        "feature_count"
    ] = len(
        enhanced_features
    )

    enhanced_manifest[
        "matchup_features"
    ] = MATCHUP_FEATURES

    enhanced_manifest[
        "historical_matchup_source"
    ] = str(
        MATCHUP_OOF
    )

    enhanced_manifest[
        "walkforward_2026_matchup_source"
    ] = str(
        MATCHUP_2026_WALKFORWARD
    )

    enhanced_manifest[
        "matchup_model_winner_sources"
    ] = WINNER_SOURCE_COLUMNS

    enhanced_manifest[
        "base_player_feature_manifest"
    ] = str(
        PLAYER_MANIFEST
    )

    enhanced_manifest[
        "matchup_training_method"
    ] = (
        "2015-2025 chronological season-level "
        "OOF matchup predictions plus "
        "2026 Weeks 1-3 weekly walk-forward "
        "predictions. 2014 matchup features "
        "remain missing."
    )

    enhanced_manifest[
        "production_training_cutoff"
    ] = (
        "2026_week_3_completed"
    )

    OUTPUT_MANIFEST.write_text(
        json.dumps(
            enhanced_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    coverage.to_csv(
        COVERAGE_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "PRODUCTION MERGE COMPLETE"
    )
    print("=" * 100)

    print(
        f"Rows after: "
        f"{len(merged):,}"
    )

    print(
        f"Original features: "
        f"{len(original_features):,}"
    )

    print(
        f"Matchup features added: "
        f"{len(MATCHUP_FEATURES):,}"
    )

    print(
        f"Enhanced features: "
        f"{len(enhanced_features):,}"
    )

    for feature in MATCHUP_FEATURES:
        print(
            f"  {feature}"
        )

    print()
    print(
        f"Enhanced production data: "
        f"{OUTPUT_DATA}"
    )

    print(
        f"Enhanced production manifest: "
        f"{OUTPUT_MANIFEST}"
    )

    print(
        f"Coverage audit: "
        f"{COVERAGE_PATH}"
    )

    print(
        "Evaluation datasets: UNCHANGED"
    )


if __name__ == "__main__":
    main()
