from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cfb_quant.features.engine import (
    build_features,
    load_canonical_history,
)

from cfb_quant.live import predict as live_base


ARTIFACT_DIR = Path("models/artifacts")
REPORT_DIR = Path("reports")


ORIGINAL_ARTIFACTS = {
    "pass_attempts":
        "cfb_player_pass_attempts_production_v0_1.joblib",

    "completions":
        "cfb_player_completions_production_v0_1.joblib",

    "passing_yards":
        "cfb_player_passing_yards_production_v0_1.joblib",

    "rush_attempts":
        "cfb_player_rush_attempts_production_v0_1.joblib",

    "rushing_yards":
        "cfb_player_rushing_yards_production_v0_1.joblib",

    "receptions":
        "cfb_player_receptions_production_v0_1.joblib",

    "receiving_yards":
        "cfb_player_receiving_yards_production_v0_1.joblib",
}


MATCHUP_ARTIFACTS = {
    "pass_attempts":
        "cfb_player_pass_attempts_matchup_production_v0_1.joblib",

    "completions":
        "cfb_player_completions_matchup_production_v0_1.joblib",

    "passing_yards":
        "cfb_player_passing_yards_matchup_production_v0_1.joblib",

    "rush_attempts":
        "cfb_player_rush_attempts_matchup_production_v0_1.joblib",

    "rushing_yards":
        "cfb_player_rushing_yards_matchup_production_v0_1.joblib",

    "receptions":
        "cfb_player_receptions_matchup_production_v0_1.joblib",

    "receiving_yards":
        "cfb_player_receiving_yards_matchup_production_v0_1.joblib",
}


# Frozen from 2021-2024 validation.
# This is the weight applied to the matchup-enhanced projection.
MATCHUP_BLEND_WEIGHT = {
    "pass_attempts": 1.00,
    "completions": 0.60,
    "passing_yards": 0.50,
    "rush_attempts": 0.95,
    "rushing_yards": 1.00,
    "receptions": 0.55,
    "receiving_yards": 0.65,
}


MATCHUP_COLUMN_MAP = {
    "proj_points":
        "matchup_proj_points",

    "proj_pass_attempts":
        "matchup_proj_pass_attempts",

    "proj_rush_attempts":
        "matchup_proj_rush_attempts",

    "proj_total_plays":
        "matchup_proj_total_plays",

    "proj_passing_yards":
        "matchup_proj_passing_yards",

    "proj_rushing_yards":
        "matchup_proj_rushing_yards",

    "proj_possession_seconds":
        "matchup_proj_possession_seconds",
}


def _load_artifact_set(
    mapping: dict[str, str],
) -> dict[str, dict]:

    artifacts = {}

    for target, filename in mapping.items():

        path = ARTIFACT_DIR / filename

        if not path.exists():
            raise RuntimeError(
                f"Missing production artifact: {path}"
            )

        artifact = joblib.load(path)

        if not isinstance(artifact, dict):
            raise RuntimeError(
                f"Unexpected artifact format: {path}"
            )

        if artifact.get("target") != target:
            raise RuntimeError(
                f"Artifact target mismatch: {path}"
            )

        if artifact.get(
            "production_training_cutoff"
        ) != "2026_week_3_completed":
            raise RuntimeError(
                f"Unexpected training cutoff: {path}"
            )

        artifacts[target] = artifact

    return artifacts


def _predict_one(
    live_rows: pd.DataFrame,
    artifact: dict,
    projection_name: str,
) -> pd.DataFrame:

    target = str(
        artifact["target"]
    )

    eligible = live_base._pregame_eligibility(
        live_rows,
        artifact,
    )

    frame = live_rows.loc[
        eligible
    ].copy()

    if frame.empty:
        return pd.DataFrame()

    feature_columns = list(
        artifact["feature_columns"]
    )

    x = live_base._numeric_features(
        frame,
        feature_columns,
    )

    predictions = np.asarray(
        artifact["model"].predict(x),
        dtype=float,
    )

    bounds = artifact.get(
        "prediction_bounds"
    )

    if (
        bounds is not None
        and len(bounds) == 2
    ):
        predictions = np.clip(
            predictions,
            float(bounds[0]),
            float(bounds[1]),
        )

    frame["target"] = target
    frame[projection_name] = predictions

    frame[
        f"{projection_name}_model"
    ] = artifact.get(
        "model_name"
    )

    keep = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "player_id",
        "player_name",
        "player_games_before",
        "player_season_games_before",
        "target",
        projection_name,
        f"{projection_name}_model",
    ]

    keep = [
        column
        for column in keep
        if column in frame.columns
    ]

    return frame[keep].copy()


def main(
    *,
    season: int,
    week: int,
    game_id: int | None = None,
    team: str | None = None,
) -> pd.DataFrame:

    print("=" * 100)
    print(
        f"CFB PRODUCTION LIVE PLAYER PROJECTIONS "
        f"— {season} WEEK {week}"
    )
    print("=" * 100)

    matchup_path = (
        REPORT_DIR
        / f"matchup_predictions_{season}_week_{week}.csv"
    )

    if not matchup_path.exists():
        raise RuntimeError(
            f"Missing matchup projection file: "
            f"{matchup_path}"
        )

    matchup = pd.read_csv(
        matchup_path
    )

    expected_matchup_columns = {
        "game_id",
        "season",
        "week",
        "team",
        "opponent",
        *MATCHUP_COLUMN_MAP.keys(),
    }

    missing = (
        expected_matchup_columns
        - set(matchup.columns)
    )

    if missing:
        raise RuntimeError(
            "Matchup projection file missing: "
            + ", ".join(sorted(missing))
        )

    if matchup.duplicated(
        ["game_id", "team"]
    ).any():
        raise RuntimeError(
            "Duplicate matchup game/team rows."
        )

    matchup = matchup.rename(
        columns=MATCHUP_COLUMN_MAP
    )

    schedule = live_base._load_schedule(
        season
    )

    games = live_base._select_games(
        schedule,
        season=season,
        week=week,
        game_id=game_id,
        team=team,
    )

    if not games:
        raise RuntimeError(
            f"No uncompleted games found for "
            f"{season} Week {week}"
        )

    selected_game_ids = {
        int(game["id"])
        for game in games
    }

    matchup = matchup.loc[
        pd.to_numeric(
            matchup["game_id"],
            errors="coerce",
        ).isin(
            selected_game_ids
        )
    ].copy()

    print(
        f"Games selected: {len(games)}"
    )

    print(
        f"Matchup team rows selected: "
        f"{len(matchup)}"
    )

    expected_team_rows = (
        2 * len(games)
    )

    if len(matchup) != expected_team_rows:
        raise RuntimeError(
            f"Expected {expected_team_rows} "
            f"matchup team rows, "
            f"got {len(matchup)}"
        )

    history = load_canonical_history()

    print(
        f"Canonical history: "
        f"{len(history):,} rows"
    )

    synthetic_rows = []

    for game in games:

        print(
            f"  {game['awayTeam']} @ "
            f"{game['homeTeam']} "
            f"[{game['id']}]"
        )

        synthetic_rows.extend(
            live_base._make_game_rows(
                history,
                game,
                season=season,
                week=week,
            )
        )

    if not synthetic_rows:
        raise RuntimeError(
            "No live player candidates constructed."
        )

    print(
        f"Synthetic player rows: "
        f"{len(synthetic_rows):,}"
    )

    relevant_history = (
        live_base._prune_history(
            history,
            games=games,
            season=season,
            week=week,
        )
    )

    print(
        f"Relevant historical rows: "
        f"{len(relevant_history):,}"
    )

    synthetic = pd.DataFrame(
        synthetic_rows,
        columns=history.columns,
    )

    combined = pd.concat(
        [
            relevant_history,
            synthetic,
        ],
        ignore_index=True,
        sort=False,
    )

    print(
        f"Building base leakage-safe features "
        f"on {len(combined):,} rows..."
    )

    featured, base_feature_columns = (
        build_features(
            combined
        )
    )

    print(
        f"Base features: "
        f"{len(base_feature_columns)}"
    )

    if len(base_feature_columns) != 465:
        raise RuntimeError(
            f"Expected 465 base features, "
            f"got {len(base_feature_columns)}"
        )

    live_rows = featured.loc[
        pd.to_numeric(
            featured["game_id"],
            errors="coerce",
        ).isin(
            selected_game_ids
        )
    ].copy()

    print(
        f"Live player rows before matchup merge: "
        f"{len(live_rows):,}"
    )

    matchup_keep = [
        "game_id",
        "team",
        *MATCHUP_COLUMN_MAP.values(),
    ]

    live_matchup = live_rows.merge(
        matchup[
            matchup_keep
        ],
        on=[
            "game_id",
            "team",
        ],
        how="left",
        validate="many_to_one",
    )

    if len(live_matchup) != len(live_rows):
        raise RuntimeError(
            "Player rows multiplied during "
            "matchup merge."
        )

    matchup_features = list(
        MATCHUP_COLUMN_MAP.values()
    )

    complete_matchup = (
        live_matchup[
            matchup_features
        ]
        .notna()
        .all(axis=1)
    )

    print(
        "Player rows with all 7 matchup features: "
        f"{int(complete_matchup.sum()):,}"
        f"/{len(live_matchup):,}"
    )

    if not complete_matchup.all():
        bad = live_matchup.loc[
            ~complete_matchup,
            [
                "game_id",
                "team",
                "opponent",
                "player_name",
            ],
        ]

        raise RuntimeError(
            "Missing matchup projections for "
            "live player rows:\n"
            + bad.head(20).to_string(
                index=False
            )
        )

    original_artifacts = (
        _load_artifact_set(
            ORIGINAL_ARTIFACTS
        )
    )

    matchup_artifacts = (
        _load_artifact_set(
            MATCHUP_ARTIFACTS
        )
    )

    original_outputs = []
    matchup_outputs = []

    print()
    print(
        "RUNNING ORIGINAL 465-FEATURE MODELS"
    )
    print("-" * 100)

    for target in ORIGINAL_ARTIFACTS:

        result = _predict_one(
            live_rows,
            original_artifacts[target],
            "projection_original",
        )

        print(
            f"{target:18s}: "
            f"{len(result):,}"
        )

        if not result.empty:
            original_outputs.append(
                result
            )

    print()
    print(
        "RUNNING MATCHUP 472-FEATURE MODELS"
    )
    print("-" * 100)

    for target in MATCHUP_ARTIFACTS:

        result = _predict_one(
            live_matchup,
            matchup_artifacts[target],
            "projection_matchup",
        )

        print(
            f"{target:18s}: "
            f"{len(result):,}"
        )

        if not result.empty:
            matchup_outputs.append(
                result
            )

    original_predictions = pd.concat(
        original_outputs,
        ignore_index=True,
        sort=False,
    )

    matchup_predictions = pd.concat(
        matchup_outputs,
        ignore_index=True,
        sort=False,
    )

    keys = [
        "game_id",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "player_id",
        "player_name",
        "target",
    ]

    original_duplicates = (
        original_predictions
        .duplicated(keys)
        .sum()
    )

    matchup_duplicates = (
        matchup_predictions
        .duplicated(keys)
        .sum()
    )

    if original_duplicates:
        raise RuntimeError(
            f"Original duplicate predictions: "
            f"{original_duplicates}"
        )

    if matchup_duplicates:
        raise RuntimeError(
            f"Matchup duplicate predictions: "
            f"{matchup_duplicates}"
        )

    final = original_predictions.merge(
        matchup_predictions[
            keys
            + [
                "projection_matchup",
                "projection_matchup_model",
            ]
        ],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    mismatch = final.loc[
        final["_merge"].ne("both")
    ]

    if not mismatch.empty:
        raise RuntimeError(
            "Original/matchup eligibility mismatch:\n"
            + mismatch[
                keys + ["_merge"]
            ]
            .head(30)
            .to_string(index=False)
        )

    final = final.drop(
        columns="_merge"
    )

    final[
        "blend_weight_matchup"
    ] = final["target"].map(
        MATCHUP_BLEND_WEIGHT
    )

    if final[
        "blend_weight_matchup"
    ].isna().any():
        raise RuntimeError(
            "Missing blend weight for target."
        )

    final[
        "blend_weight_original"
    ] = (
        1.0
        - final[
            "blend_weight_matchup"
        ]
    )

    final[
        "projection_blended"
    ] = (
        final[
            "blend_weight_original"
        ]
        * final[
            "projection_original"
        ]
        + final[
            "blend_weight_matchup"
        ]
        * final[
            "projection_matchup"
        ]
    )

    for target in MATCHUP_BLEND_WEIGHT:

        artifact = matchup_artifacts[
            target
        ]

        bounds = artifact.get(
            "prediction_bounds"
        )

        if (
            bounds is not None
            and len(bounds) == 2
        ):
            mask = final[
                "target"
            ].eq(target)

            final.loc[
                mask,
                "projection_blended",
            ] = np.clip(
                final.loc[
                    mask,
                    "projection_blended",
                ],
                float(bounds[0]),
                float(bounds[1]),
            )

    start_lookup = {
        int(game["id"]):
            game.get("startDate")
        for game in games
    }

    final[
        "start_date"
    ] = final[
        "game_id"
    ].map(
        lambda value:
            start_lookup.get(
                int(value)
            )
    )

    output_columns = [
        "game_id",
        "start_date",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "player_id",
        "player_name",
        "target",
        "projection_original",
        "projection_matchup",
        "blend_weight_original",
        "blend_weight_matchup",
        "projection_blended",
        "projection_original_model",
        "projection_matchup_model",
        "player_games_before",
        "player_season_games_before",
    ]

    output_columns = [
        col
        for col in output_columns
        if col in final.columns
    ]

    final = final[
        output_columns
    ].copy()

    for column in [
        "projection_original",
        "projection_matchup",
        "projection_blended",
    ]:
        final[column] = (
            pd.to_numeric(
                final[column],
                errors="coerce",
            )
            .round(3)
        )

    final = final.sort_values(
        [
            "game_id",
            "team",
            "player_name",
            "target",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / (
            f"live_player_production_"
            f"{season}_week_{week}_v0_1.csv"
        )
    )

    final.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "CFB PRODUCTION LIVE PLAYER BUILD COMPLETE"
    )
    print("=" * 100)

    print(
        f"Games: "
        f"{final['game_id'].nunique():,}"
    )

    print(
        f"Players: "
        f"{final['player_id'].nunique():,}"
    )

    print(
        f"Projection rows: "
        f"{len(final):,}"
    )

    print()
    print(
        "PROJECTIONS BY TARGET"
    )

    print(
        final.groupby(
            "target"
        ).size()
    )

    print()
    print(
        f"Output: {output_path}"
    )

    return final


def cli() -> None:

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
        "--game-id",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--team",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    main(
        season=args.season,
        week=args.week,
        game_id=args.game_id,
        team=args.team,
    )


if __name__ == "__main__":
    cli()
