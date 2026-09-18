from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT_GLOB = "data/processed/canonical_player_games_*.parquet"
DEFAULT_OUTPUT = "data/processed/cfb_pregame_features_v0_1.parquet"
DEFAULT_MANIFEST = "reports/cfb_feature_manifest_v0_1.json"

SEASON_TYPE_ORDER = {
    "regular": 0,
    "postseason": 1,
    "spring_regular": 2,
    "spring_postseason": 3,
}

# These columns are retained in the feature table so they can be used as labels,
# audits, or training-sample filters. They are NEVER included directly in the
# model feature list.
OUTCOME_COLUMNS = [
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
]

PLAYER_HISTORY_METRICS = [
    # Official box-score history.
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

    # Passing PBP history.
    "qbr",
    "dropbacks",
    "sacks_taken",
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

    # Rushing PBP history.
    "rush_plays_pbp",
    "non_kneel_carries",
    "kneels",
    "rush_epa",
    "rush_successes",
    "red_zone_carries",
    "short_yardage_carries",
    "power_carries",
    "explosive_rushes",
    "rush_success_rate",
    "rush_epa_per_play",

    # Receiving PBP history.
    "targets_pbp",
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

SHARE_METRICS = [
    ("pass_attempts", "team_pass_attempts"),
    ("completions", "team_completions"),
    ("passing_yards", "team_passing_yards"),
    ("rush_attempts", "team_rush_attempts"),
    ("rushing_yards", "team_rushing_yards"),
    ("receptions", "team_receptions"),
    ("receiving_yards", "team_receiving_yards"),
    ("targets_pbp", "team_targets_pbp"),
    ("red_zone_carries", "team_red_zone_carries"),
    ("red_zone_targets", "team_red_zone_targets"),
]

TEAM_METRICS = [
    "team_pass_attempts",
    "team_completions",
    "team_passing_yards",
    "team_rush_attempts",
    "team_rushing_yards",
    "team_receptions",
    "team_receiving_yards",
    "team_targets_pbp",
    "team_red_zone_carries",
    "team_red_zone_targets",
    "team_points_for",
    "team_points_against",
]

FORBIDDEN_DIRECT_FEATURES = set(
    OUTCOME_COLUMNS
    + [
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
        "has_passing_stats",
        "has_rushing_stats",
        "has_receiving_stats",
        "is_team_row",
        "pbp_rush_attempt_proxy",
        "pbp_rush_attempt_delta",
        "season_pbp",
        "week_pbp",
        "player_name_pbp",
        "team_pbp",
        "opponent_pbp",
        "zero_catch_augmented",
        "receiving_outcome_source",
        "receiving_training_observation",
        "targets",  # 0% populated in the audited 2025 canonical table.
    ]
    + PLAYER_HISTORY_METRICS
)


def _safe_sum(series: pd.Series) -> float:
    return series.sum(min_count=1)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return num / den


def _player_key(frame: pd.DataFrame) -> pd.Series:
    ids = frame["player_id"].astype("Int64")
    name = (
        frame["player_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    team = (
        frame["team"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    # Player ID preserves transfer history. Name+team is only a conservative
    # fallback if an old row lacks an ID, avoiding accidental same-name merges.
    values = np.where(
        ids.notna(),
        "id:" + ids.astype(str),
        "name:" + name + "|team:" + team,
    )
    return pd.Series(values, index=frame.index, dtype="string")


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_season_type_order"] = (
        out["season_type"]
        .astype(str)
        .str.lower()
        .map(SEASON_TYPE_ORDER)
        .fillna(9)
        .astype(int)
    )
    out["_player_key"] = _player_key(out)
    return out.sort_values(
        ["season", "_season_type_order", "week", "game_id", "team", "_player_key"],
        kind="mergesort",
    ).reset_index(drop=True)


def load_canonical_history(input_glob: str = DEFAULT_INPUT_GLOB) -> pd.DataFrame:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise RuntimeError(f"No canonical player-game files found: {input_glob}")

    frames = [pd.read_parquet(file) for file in files]
    data = pd.concat(frames, ignore_index=True, sort=False)

    required = {
        "game_id",
        "season",
        "week",
        "season_type",
        "player_id",
        "player_name",
        "team",
        "opponent",
        "home_away",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"Canonical data missing required columns: {missing}")

    data = _sort_frame(data)

    dupes = data.duplicated(["game_id", "_player_key"], keep=False)
    if dupes.any():
        sample = data.loc[
            dupes, ["game_id", "season", "week", "player_name", "team"]
        ].head(20)
        raise RuntimeError(
            "Duplicate player-game rows detected.\n" + sample.to_string(index=False)
        )

    return data


def build_team_game_table(data: pd.DataFrame) -> pd.DataFrame:
    agg_map: dict[str, tuple[str, object]] = {
        "team_pass_attempts": ("pass_attempts", _safe_sum),
        "team_completions": ("completions", _safe_sum),
        "team_passing_yards": ("passing_yards", _safe_sum),
        "team_rush_attempts": ("rush_attempts", _safe_sum),
        "team_rushing_yards": ("rushing_yards", _safe_sum),
        "team_receptions": ("receptions", _safe_sum),
        "team_receiving_yards": ("receiving_yards", _safe_sum),
        "team_targets_pbp": ("targets_pbp", _safe_sum),
        "team_red_zone_carries": ("red_zone_carries", _safe_sum),
        "team_red_zone_targets": ("red_zone_targets", _safe_sum),
        "team_points_for": ("team_points", "first"),
        "team_points_against": ("opponent_points", "first"),
    }
    available = {
        out_name: spec
        for out_name, spec in agg_map.items()
        if spec[0] in data.columns
    }

    grouped = (
        data.groupby(
            [
                "game_id",
                "season",
                "week",
                "season_type",
                "_season_type_order",
                "team",
                "opponent",
                "home_away",
            ],
            dropna=False,
            sort=False,
        )
        .agg(**available)
        .reset_index()
    )

    if grouped.duplicated(["game_id", "team"]).any():
        raise RuntimeError("Duplicate team-game rows created during aggregation.")

    return grouped


def _history_features(
    frame: pd.DataFrame,
    *,
    group_col: str,
    metrics: Iterable[str],
    prefix: str,
    season_col: str = "season",
    windows: tuple[int, ...] = (3, 5),
    include_std5: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    group = out[group_col]
    season_groupers = [out[group_col], out[season_col]]

    generated: dict[str, pd.Series] = {}

    for metric in metrics:
        if metric not in out.columns:
            continue

        values = pd.to_numeric(out[metric], errors="coerce")

        # CRITICAL: shift before every rolling/expanding calculation.
        shifted = values.groupby(group, sort=False).shift(1)

        generated[f"{prefix}{metric}_lag1"] = shifted

        for window in windows:
            generated[f"{prefix}{metric}_avg{window}"] = shifted.groupby(
                group, sort=False
            ).transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())

        if include_std5:
            generated[f"{prefix}{metric}_std5"] = shifted.groupby(
                group, sort=False
            ).transform(lambda s: s.rolling(5, min_periods=2).std())

        generated[f"{prefix}{metric}_career_avg"] = shifted.groupby(
            group, sort=False
        ).transform(lambda s: s.expanding(min_periods=1).mean())

        # Shift within the same season so Week 1 cannot inherit that season's
        # current game as its own "season-to-date" history.
        season_shifted = values.groupby(season_groupers, sort=False).shift(1)
        generated[f"{prefix}{metric}_season_avg"] = season_shifted.groupby(
            season_groupers, sort=False
        ).transform(lambda s: s.expanding(min_periods=1).mean())

    features = pd.DataFrame(generated, index=out.index)
    out = pd.concat([out, features], axis=1)
    return out, list(generated)


def add_team_pregame_features(
    team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    tg = team_games.copy()
    tg["_team_key"] = tg["team"].astype("string")
    tg = tg.sort_values(
        ["season", "_season_type_order", "week", "game_id", "team"],
        kind="mergesort",
    ).reset_index(drop=True)

    tg["team_games_before"] = tg.groupby("_team_key", sort=False).cumcount()
    tg["team_season_games_before"] = tg.groupby(
        ["_team_key", "season"], sort=False
    ).cumcount()

    tg, hist_cols = _history_features(
        tg,
        group_col="_team_key",
        metrics=[c for c in TEAM_METRICS if c in tg.columns],
        prefix="tm_",
        windows=(3, 5),
        include_std5=False,
    )

    feature_cols = ["team_games_before", "team_season_games_before"] + hist_cols
    return tg[["game_id", "team", *feature_cols]].copy(), feature_cols


def add_defense_pregame_features(
    team_games: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    # Each offensive row becomes a prior-performance row for the defense it faced.
    defense = team_games.copy()
    defense["defense_team"] = defense["opponent"].astype("string")

    rename = {
        metric: f"allowed_{metric.removeprefix('team_')}"
        for metric in TEAM_METRICS
        if metric in defense.columns
    }
    defense = defense.rename(columns=rename)
    allowed_metrics = list(rename.values())

    defense = defense.sort_values(
        ["season", "_season_type_order", "week", "game_id", "defense_team"],
        kind="mergesort",
    ).reset_index(drop=True)

    defense["def_games_before"] = defense.groupby(
        "defense_team", sort=False
    ).cumcount()
    defense["def_season_games_before"] = defense.groupby(
        ["defense_team", "season"], sort=False
    ).cumcount()

    defense, hist_cols = _history_features(
        defense,
        group_col="defense_team",
        metrics=allowed_metrics,
        prefix="opp_",
        windows=(3, 5),
        include_std5=False,
    )

    feature_cols = ["def_games_before", "def_season_games_before"] + hist_cols
    return defense[["game_id", "defense_team", *feature_cols]].copy(), feature_cols


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = _sort_frame(data)

    # Known before kickoff.
    frame["is_home"] = (
        frame["home_away"].astype(str).str.lower().eq("home").astype("int8")
    )
    frame["is_postseason"] = (
        frame["season_type"]
        .astype(str)
        .str.lower()
        .eq("postseason")
        .astype("int8")
    )
    frame["week_num"] = pd.to_numeric(frame["week"], errors="coerce").astype(float)

    frame["player_games_before"] = frame.groupby(
        "_player_key", sort=False
    ).cumcount()
    frame["player_season_games_before"] = frame.groupby(
        ["_player_key", "season"], sort=False
    ).cumcount()

    feature_cols: list[str] = [
        "is_home",
        "is_postseason",
        "week_num",
        "player_games_before",
        "player_season_games_before",
    ]

    team_games = build_team_game_table(frame)

    # Current-game totals are temporary only. They are used to calculate each
    # historical player's share, then ONLY shifted share history is retained.
    possible_team_current = [
        "team_pass_attempts",
        "team_completions",
        "team_passing_yards",
        "team_rush_attempts",
        "team_rushing_yards",
        "team_receptions",
        "team_receiving_yards",
        "team_targets_pbp",
        "team_red_zone_carries",
        "team_red_zone_targets",
    ]
    team_current_cols = [
        c for c in possible_team_current if c in team_games.columns
    ]
    team_current = team_games[["game_id", "team", *team_current_cols]].copy()

    frame = frame.merge(
        team_current,
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    share_names: list[str] = []
    for player_metric, team_metric in SHARE_METRICS:
        if player_metric not in frame.columns or team_metric not in frame.columns:
            continue
        share_name = f"share_{player_metric}"
        frame[share_name] = _safe_divide(frame[player_metric], frame[team_metric])
        share_names.append(share_name)

    player_metrics = [
        c for c in PLAYER_HISTORY_METRICS if c in frame.columns
    ] + share_names

    frame, player_hist_cols = _history_features(
        frame,
        group_col="_player_key",
        metrics=player_metrics,
        prefix="p_",
        windows=(3, 5),
        include_std5=True,
    )
    feature_cols.extend(player_hist_cols)

    # Remove all current-game share/total intermediates before the final output.
    frame = frame.drop(
        columns=[c for c in team_current_cols + share_names if c in frame.columns],
        errors="ignore",
    )

    team_hist, team_hist_cols = add_team_pregame_features(team_games)
    frame = frame.merge(
        team_hist,
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )
    feature_cols.extend(team_hist_cols)

    defense_hist, defense_hist_cols = add_defense_pregame_features(team_games)
    frame = frame.merge(
        defense_hist,
        left_on=["game_id", "opponent"],
        right_on=["game_id", "defense_team"],
        how="left",
        validate="many_to_one",
    )
    frame = frame.drop(columns=["defense_team"], errors="ignore")
    feature_cols.extend(defense_hist_cols)

    feature_cols = list(dict.fromkeys(feature_cols))

    leaked = sorted(FORBIDDEN_DIRECT_FEATURES.intersection(feature_cols))
    if leaked:
        raise RuntimeError(
            f"Forbidden current-game columns leaked into model features: {leaked}"
        )

    missing_features = sorted(set(feature_cols) - set(frame.columns))
    if missing_features:
        raise RuntimeError(
            f"Generated feature columns missing from output: {missing_features}"
        )

    frame = frame.sort_values(
        ["season", "_season_type_order", "week", "game_id", "team", "_player_key"],
        kind="mergesort",
    ).reset_index(drop=True)

    return frame, feature_cols


def write_manifest(
    *,
    path: str | Path,
    frame: pd.DataFrame,
    feature_cols: list[str],
    input_glob: str,
    output_path: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "artifact_version": "v0.1",
        "purpose": "CFB pregame leakage-safe player prop features",
        "input_glob": input_glob,
        "output": output_path,
        "rows": int(len(frame)),
        "seasons": sorted(
            int(x) for x in pd.Series(frame["season"]).dropna().unique()
        ),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "target_columns_retained_for_training_only": [
            c for c in OUTCOME_COLUMNS if c in frame.columns
        ],
        "forbidden_direct_features": sorted(FORBIDDEN_DIRECT_FEATURES),
        "notes": [
            "Player history is shifted before rolling/expanding calculations.",
            "Team history is shifted before merge to the current game.",
            "Opponent-defense history is shifted before merge to the current game.",
            "Current-game official/PBP values are retained only as labels/audit data, never direct model features.",
            "The canonical targets column is excluded because audited 2025 coverage is 0%; targets_pbp is used historically instead.",
        ],
    }

    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe pregame CFB player-prop features."
    )
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    print("=" * 80)
    print("CFB PREGAME FEATURE ENGINE v0.1")
    print("=" * 80)

    data = load_canonical_history(args.input_glob)
    frame, feature_cols = build_features(data)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    write_manifest(
        path=args.manifest,
        frame=frame,
        feature_cols=feature_cols,
        input_glob=args.input_glob,
        output_path=str(output),
    )

    print(f"Rows: {len(frame):,}")
    print(f"Seasons: {int(frame['season'].min())}-{int(frame['season'].max())}")
    print(f"Features: {len(feature_cols):,}")
    print(f"Output: {output}")
    print(f"Manifest: {args.manifest}")
    print()
    print("LEAKAGE GUARD: train only with manifest feature_columns.")
    print("=" * 80)


if __name__ == "__main__":
    main()