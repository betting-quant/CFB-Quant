from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT_GLOB = "data/processed/team_games_*.parquet"
DEFAULT_OUTPUT = "data/processed/cfb_matchup_features_v0_1.parquet"
DEFAULT_MANIFEST = "reports/cfb_matchup_feature_manifest_v0_1.json"
DEFAULT_KICKOFF_MAP = "reports/spread_total_schedule_chronology_v2/team_game_kickoff_map.csv"

SEASON_TYPE_ORDER = {
    "regular": 0,
    "postseason": 1,
    "spring_regular": 2,
    "spring_postseason": 3,
}

TARGET_COLUMNS = [
    "points",
    "pass_attempts",
    "rush_attempts",
    "total_plays",
    "passing_yards",
    "rushing_yards",
    "possession_seconds",
]

BASE_HISTORY_METRICS = [
    "points",
    "opponent_points",
    "pass_attempts",
    "rush_attempts",
    "total_plays",
    "passing_yards",
    "rushing_yards",
    "possession_seconds",
    "completions",
    "passing_touchdowns",
    "rushing_touchdowns",
    "interceptions",
    "turnovers",
    "fumbles_lost",
    "penalties",
    "penalty_yards",
    "yards_per_rush_attempt",
    "yards_per_pass",
    "pass_rate",
    "rush_rate",
    "yards_per_play",
    "completion_rate",
    "points_per_play",
]

OPTIONAL_HISTORY_METRICS = [
    "sacks",
    "tackles_for_loss",
    "qb_hurries",
]

FORBIDDEN_DIRECT_FEATURES = set(TARGET_COLUMNS)


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    num = pd.to_numeric(
        numerator,
        errors="coerce",
    )
    den = pd.to_numeric(
        denominator,
        errors="coerce",
    ).replace(
        0,
        np.nan,
    )

    return num / den


def _sort_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    out["_season_type_order"] = (
        out["season_type"]
        .astype(str)
        .str.lower()
        .map(SEASON_TYPE_ORDER)
        .fillna(9)
        .astype(int)
    )

    # The production loader requires a kickoff map. The legacy fallback
    # supports older unit tests that construct small frames directly.
    if "kickoff_utc" in out.columns:
        out["kickoff_utc"] = pd.to_datetime(out["kickoff_utc"], utc=True, errors="coerce")
        if out["kickoff_utc"].isna().any():
            raise ValueError("Missing/invalid kickoff_utc: refusing unsafe chronological sort")
        order = ["season", "kickoff_utc", "game_id", "team"]
    else:
        order = ["season", "_season_type_order", "week", "game_id", "team"]

    if "kickoff_utc" in out.columns:
        if out.duplicated(["team", "kickoff_utc"]).any():
            bad = out.loc[
                out.duplicated(
                    ["team", "kickoff_utc"],
                    keep=False
                ),
                ["team", "game_id", "kickoff_utc"]
            ].head(10)
            raise ValueError(
                "Duplicate team kickoff: unsafe chronology. "
                "Refusing pregame feature construction.\n"
                + bad.to_string(index=False)
            )

        if "opponent" in out.columns:
            if out.duplicated(
                ["opponent", "kickoff_utc"]
            ).any():
                raise ValueError(
                    "Duplicate defensive team kickoff: "
                    "unsafe chronology"
                )

    return out.sort_values(order, kind="mergesort").reset_index(drop=True)


def attach_verified_kickoff(
    data: pd.DataFrame,
    kickoff_map: str | Path = DEFAULT_KICKOFF_MAP,
) -> pd.DataFrame:
    """Fail closed unless every source team-game has a matching kickoff UTC."""
    path = Path(kickoff_map)
    if not path.is_file():
        raise FileNotFoundError(f"Missing kickoff map {path}; run audit_schedule_chronology_v2.py")
    sched = pd.read_csv(path)
    required = {"game_id", "season", "team", "home_away", "kickoff_utc"}
    if not required.issubset(sched.columns):
        raise ValueError(f"Schedule map missing {sorted(required - set(sched.columns))}")
    if not (required - {"kickoff_utc"}).issubset(data.columns):
        raise ValueError("Team history is missing identifying columns")
    out = data.drop(columns=["kickoff_utc"], errors="ignore").copy()
    sched = sched[list(required)].copy()
    for frame in (out, sched):
        frame["_chrono_id"] = frame["game_id"].astype("string").str.replace(r"\.0$", "", regex=True)
        frame["_chrono_side"] = frame["home_away"].astype("string").str.strip().str.lower()
        frame["_chrono_team"] = frame["team"].astype("string").str.strip()
        frame["_chrono_season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    sched["kickoff_utc"] = pd.to_datetime(sched["kickoff_utc"], utc=True, errors="coerce")
    if sched["kickoff_utc"].isna().any():
        raise ValueError("Schedule has missing kickoff UTC")
    if sched.groupby("_chrono_id")["kickoff_utc"].nunique().gt(1).any():
        raise ValueError("Conflicting kickoff times for a game")
    join_keys = ["_chrono_id", "_chrono_season", "_chrono_team", "_chrono_side"]
    if sched.duplicated(join_keys).any():
        raise ValueError("Duplicate schedule team-game rows")
    out = out.merge(sched[join_keys + ["kickoff_utc"]], on=join_keys,
                    how="left", validate="one_to_one")
    if out["kickoff_utc"].isna().any():
        sample = out.loc[out["kickoff_utc"].isna(), ["game_id", "team"]].head(5)
        raise ValueError(f"Unmatched kickoff; feature build stopped:\n{sample.to_string(index=False)}")
    return out.drop(columns=join_keys)


def load_team_history(
    input_glob: str = DEFAULT_INPUT_GLOB,
    *,
    kickoff_map: str = DEFAULT_KICKOFF_MAP,
) -> pd.DataFrame:
    files = sorted(
        glob.glob(
            input_glob
        )
    )

    if not files:
        raise RuntimeError(
            f"No team-game files found: "
            f"{input_glob}"
        )

    frames = [
        pd.read_parquet(
            file
        )
        for file in files
    ]

    data = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    required = {
        "game_id",
        "season",
        "week",
        "season_type",
        "team",
        "opponent",
        "home_away",
        "points",
        "opponent_points",
        "pass_attempts",
        "rush_attempts",
        "passing_yards",
        "rushing_yards",
        "possession_seconds",
    }

    missing = sorted(
        required
        - set(
            data.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Team-game data missing required "
            f"columns: {missing}"
        )

    data = attach_verified_kickoff(data, kickoff_map)
    data = _sort_frame(data)

    dupes = data.duplicated(
        [
            "game_id",
            "team",
        ],
        keep=False,
    )

    if dupes.any():
        sample = data.loc[
            dupes,
            [
                "game_id",
                "season",
                "week",
                "team",
                "opponent",
            ],
        ].head(
            20
        )

        raise RuntimeError(
            "Duplicate team-game rows "
            "detected.\n"
            + sample.to_string(
                index=False
            )
        )

    return data


def add_derived_game_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    out["total_plays"] = (
        pd.to_numeric(
            out["pass_attempts"],
            errors="coerce",
        )
        + pd.to_numeric(
            out["rush_attempts"],
            errors="coerce",
        )
    )

    out["pass_rate"] = _safe_divide(
        out["pass_attempts"],
        out["total_plays"],
    )

    out["rush_rate"] = _safe_divide(
        out["rush_attempts"],
        out["total_plays"],
    )

    total_yards = (
        pd.to_numeric(
            out["passing_yards"],
            errors="coerce",
        )
        + pd.to_numeric(
            out["rushing_yards"],
            errors="coerce",
        )
    )

    out["yards_per_play"] = _safe_divide(
        total_yards,
        out["total_plays"],
    )

    out["completion_rate"] = _safe_divide(
        out["completions"],
        out["pass_attempts"],
    )

    out["points_per_play"] = _safe_divide(
        out["points"],
        out["total_plays"],
    )

    return out


def _history_features(
    frame: pd.DataFrame,
    *,
    group_col: str,
    metrics: Iterable[str],
    prefix: str,
    season_col: str = "season",
    windows: tuple[int, ...] = (
        3,
        5,
    ),
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    out = frame.copy()

    group = out[
        group_col
    ]

    season_groupers = [
        out[
            group_col
        ],
        out[
            season_col
        ],
    ]

    generated: dict[
        str,
        pd.Series,
    ] = {}

    for metric in metrics:
        if metric not in out.columns:
            continue

        values = pd.to_numeric(
            out[
                metric
            ],
            errors="coerce",
        )

        # Leakage guard:
        # every pregame historical feature
        # must shift before rolling.
        shifted = values.groupby(
            group,
            sort=False,
        ).shift(
            1
        )

        generated[
            f"{prefix}{metric}_lag1"
        ] = shifted

        for window in windows:
            generated[
                f"{prefix}{metric}_avg{window}"
            ] = shifted.groupby(
                group,
                sort=False,
            ).transform(
                lambda s, w=window:
                s.rolling(
                    w,
                    min_periods=1,
                ).mean()
            )

        generated[
            f"{prefix}{metric}_career_avg"
        ] = shifted.groupby(
            group,
            sort=False,
        ).transform(
            lambda s:
            s.expanding(
                min_periods=1
            ).mean()
        )

        season_shifted = (
            values.groupby(
                season_groupers,
                sort=False,
            ).shift(
                1
            )
        )

        generated[
            f"{prefix}{metric}_season_avg"
        ] = season_shifted.groupby(
            season_groupers,
            sort=False,
        ).transform(
            lambda s:
            s.expanding(
                min_periods=1
            ).mean()
        )

    features = pd.DataFrame(
        generated,
        index=out.index,
    )

    out = pd.concat(
        [
            out,
            features,
        ],
        axis=1,
    )

    return (
        out,
        list(
            generated
        ),
    )


def add_offense_history(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    out = frame.copy()

    out["_team_key"] = (
        out["team"]
        .astype("string")
    )

    out["team_games_before"] = (
        out.groupby(
            "_team_key",
            sort=False,
        ).cumcount()
    )

    out["team_season_games_before"] = (
        out.groupby(
            [
                "_team_key",
                "season",
            ],
            sort=False,
        ).cumcount()
    )

    metrics = [
        c
        for c in (
            BASE_HISTORY_METRICS
            + OPTIONAL_HISTORY_METRICS
        )
        if c in out.columns
    ]

    out, hist_cols = (
        _history_features(
            out,
            group_col="_team_key",
            metrics=metrics,
            prefix="off_",
            windows=(
                3,
                5,
            ),
        )
    )

    feature_cols = [
        "team_games_before",
        "team_season_games_before",
        *hist_cols,
    ]

    return (
        out,
        feature_cols,
    )


def build_defense_history(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    defense = frame.copy()

    defense[
        "defense_team"
    ] = (
        defense[
            "opponent"
        ].astype(
            "string"
        )
    )

    source_metrics = [
        c
        for c in (
            BASE_HISTORY_METRICS
            + OPTIONAL_HISTORY_METRICS
        )
        if c in defense.columns
    ]

    rename = {
        metric:
        f"allowed_{metric}"
        for metric in source_metrics
    }

    defense = defense.rename(
        columns=rename
    )

    allowed_metrics = list(
        rename.values()
    )

    if "kickoff_utc" in defense.columns:
        if defense["kickoff_utc"].isna().any():
            raise ValueError("Cannot build defense history without kickoff UTC")
        defense_order = ["season", "kickoff_utc", "game_id", "defense_team"]
    else:
        defense_order = ["season", "_season_type_order", "week", "game_id", "defense_team"]
    defense = defense.sort_values(defense_order, kind="mergesort").reset_index(drop=True)

    defense[
        "def_games_before"
    ] = (
        defense.groupby(
            "defense_team",
            sort=False,
        ).cumcount()
    )

    defense[
        "def_season_games_before"
    ] = (
        defense.groupby(
            [
                "defense_team",
                "season",
            ],
            sort=False,
        ).cumcount()
    )

    defense, hist_cols = (
        _history_features(
            defense,
            group_col="defense_team",
            metrics=allowed_metrics,
            prefix="def_",
            windows=(
                3,
                5,
            ),
        )
    )

    feature_cols = [
        "def_games_before",
        "def_season_games_before",
        *hist_cols,
    ]

    keep = [
        "game_id",
        "defense_team",
        *feature_cols,
    ]

    return (
        defense[
            keep
        ].copy(),
        feature_cols,
    )


def build_features(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
]:
    frame = _sort_frame(
        add_derived_game_metrics(
            data
        )
    )

    frame["is_home"] = (
        frame["home_away"]
        .astype(str)
        .str.lower()
        .eq("home")
        .astype("int8")
    )

    frame["is_postseason"] = (
        frame["season_type"]
        .astype(str)
        .str.lower()
        .eq("postseason")
        .astype("int8")
    )

    frame["week_num"] = (
        pd.to_numeric(
            frame["week"],
            errors="coerce",
        )
        .astype(float)
    )

    base_feature_cols = [
        "is_home",
        "is_postseason",
        "week_num",
    ]

    frame, offense_cols = (
        add_offense_history(
            frame
        )
    )

    defense_hist, defense_cols = (
        build_defense_history(
            frame
        )
    )

    frame = frame.merge(
        defense_hist,
        left_on=[
            "game_id",
            "opponent",
        ],
        right_on=[
            "game_id",
            "defense_team",
        ],
        how="left",
        validate="one_to_one",
    )

    frame = frame.drop(
        columns=[
            "defense_team"
        ],
        errors="ignore",
    )

    feature_cols = list(
        dict.fromkeys(
            base_feature_cols
            + offense_cols
            + defense_cols
        )
    )

    leaked = sorted(
        FORBIDDEN_DIRECT_FEATURES
        .intersection(
            feature_cols
        )
    )

    if leaked:
        raise RuntimeError(
            "Current-game matchup targets "
            "leaked into features: "
            f"{leaked}"
        )

    missing_features = sorted(
        set(
            feature_cols
        )
        - set(
            frame.columns
        )
    )

    if missing_features:
        raise RuntimeError(
            "Generated matchup features "
            "missing from output: "
            f"{missing_features[:20]}"
        )

    frame = _sort_frame(frame)

    return (
        frame,
        feature_cols,
    )


def write_manifest(
    *,
    path: str | Path,
    frame: pd.DataFrame,
    feature_cols: list[str],
    input_glob: str,
    output_path: str,
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "artifact_version": "v0.1",
        "purpose": (
            "CFB leakage-safe pregame "
            "team matchup features"
        ),
        "input_glob": input_glob,
        "output": output_path,
        "rows": int(
            len(
                frame
            )
        ),
        "games": int(
            frame[
                "game_id"
            ].nunique()
        ),
        "seasons": sorted(
            int(
                x
            )
            for x in pd.Series(
                frame[
                    "season"
                ]
            ).dropna().unique()
        ),
        "feature_count": len(
            feature_cols
        ),
        "chronology_source": DEFAULT_KICKOFF_MAP,
        "chronology_sort": "season,kickoff_utc,game_id,team",
        "upstream_source_snapshot_verified": False,
        "feature_columns": feature_cols,
        "target_columns_retained_for_training_only": [
            c
            for c in TARGET_COLUMNS
            if c in frame.columns
        ],
        "forbidden_direct_features": sorted(
            FORBIDDEN_DIRECT_FEATURES
        ),
        "notes": [
            (
                "All offense history is shifted "
                "before rolling/expanding calculations."
            ),
            (
                "All opponent-defense history is shifted "
                "before merge to the current game."
            ),
            (
                "Current-game targets are retained only "
                "for training/evaluation and never appear "
                "in the manifest feature list."
            ),
            (
                "Sacks, TFL, and QB hurries are optional "
                "history inputs because historical coverage "
                "is incomplete."
            ),
        ],
    }

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe CFB "
            "pregame matchup features."
        )
    )

    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--kickoff-map",
        default=DEFAULT_KICKOFF_MAP,
    )

    args = parser.parse_args()

    print(
        "=" * 80
    )
    print(
        "CFB MATCHUP FEATURE ENGINE v0.1"
    )
    print(
        "=" * 80
    )

    data = load_team_history(
        args.input_glob,
        kickoff_map=args.kickoff_map,
    )

    frame, feature_cols = (
        build_features(
            data
        )
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_parquet(
        output,
        index=False,
    )

    write_manifest(
        path=args.manifest,
        frame=frame,
        feature_cols=feature_cols,
        input_glob=args.input_glob,
        output_path=str(
            output
        ),
    )

    print(
        f"Rows: {len(frame):,}"
    )

    print(
        "Games:",
        f"{frame['game_id'].nunique():,}",
    )

    print(
        "Seasons:",
        f"{int(frame['season'].min())}-"
        f"{int(frame['season'].max())}",
    )

    print(
        f"Features: {len(feature_cols):,}"
    )

    print(
        f"Output: {output}"
    )

    print(
        f"Manifest: {args.manifest}"
    )

    print()
    print(
        "LEAKAGE GUARD: train only with "
        "manifest feature_columns."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()
