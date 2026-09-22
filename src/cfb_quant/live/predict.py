from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from cfb_quant.features.engine import build_features, load_canonical_history


ARTIFACT_DIR = Path("models/artifacts")
SCHEDULE_ROOT = Path("data/raw/cfbd/games")
REPORT_DIR = Path("reports")

TARGET_ARTIFACTS = {
    "pass_attempts": "cfb_pass_attempts_quant_v0_1.joblib",
    "completions": "cfb_completions_quant_v0_1.joblib",
    "passing_yards": "cfb_passing_yards_quant_v0_1.joblib",
    "rush_attempts": "cfb_rush_attempts_quant_v0_1.joblib",
    "rushing_yards": "cfb_rushing_yards_quant_v0_1.joblib",
    "receptions": "cfb_receptions_quant_v0_1.joblib",
    "receiving_yards": "cfb_receiving_yards_quant_v0_1.joblib",
}


def _load_schedule(season: int) -> list[dict]:
    path = SCHEDULE_ROOT / f"games_{season}.json"

    if not path.exists():
        raise RuntimeError(f"Schedule file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected schedule structure: {path}")

    return data


def _select_games(
    games: list[dict],
    *,
    season: int,
    week: int,
    game_id: int | None = None,
    team: str | None = None,
) -> list[dict]:

    selected = [
        game
        for game in games
        if int(game.get("season", -1)) == season
        and int(game.get("week", -1)) == week
        and not bool(game.get("completed", False))
    ]

    if game_id is not None:
        selected = [
            game
            for game in selected
            if int(game.get("id", -1)) == game_id
        ]

    if team:
        wanted = team.strip().casefold()

        selected = [
            game
            for game in selected
            if str(game.get("homeTeam", "")).casefold() == wanted
            or str(game.get("awayTeam", "")).casefold() == wanted
        ]

    return sorted(
        selected,
        key=lambda game: (
            str(game.get("startDate", "")),
            int(game.get("id", 0)),
        ),
    )


def _current_season_candidates(
    history: pd.DataFrame,
    *,
    season: int,
    team: str,
    week: int,
) -> pd.DataFrame:

    frame = history.loc[
        (pd.to_numeric(history["season"], errors="coerce") == season)
        & (pd.to_numeric(history["week"], errors="coerce") < week)
        & (history["team"].astype(str) == team)
    ].copy()

    if "is_team_row" in frame.columns:
        frame = frame.loc[
            ~frame["is_team_row"].fillna(False).astype(bool)
        ].copy()

    if frame.empty:
        return frame

    frame = frame.sort_values(
        ["season", "week", "game_id"],
        kind="mergesort",
    )

    return (
        frame.groupby("_player_key", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _make_game_rows(
    history: pd.DataFrame,
    game: dict,
    *,
    season: int,
    week: int,
) -> list[dict]:

    game_id = int(game["id"])

    sides = [
        {
            "team": str(game["homeTeam"]),
            "team_id": game.get("homeId"),
            "opponent": str(game["awayTeam"]),
            "opponent_team_id": game.get("awayId"),
            "home_away": "home",
        },
        {
            "team": str(game["awayTeam"]),
            "team_id": game.get("awayId"),
            "opponent": str(game["homeTeam"]),
            "opponent_team_id": game.get("homeId"),
            "home_away": "away",
        },
    ]

    rows: list[dict] = []

    for side in sides:

        candidates = _current_season_candidates(
            history,
            season=season,
            team=side["team"],
            week=week,
        )

        print(
            f"    {side['team']}: "
            f"{len(candidates)} current-season player candidates"
        )

        if candidates.empty:
            continue

        for _, player in candidates.iterrows():

            row = {
                column: pd.NA
                for column in history.columns
            }

            row.update(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "season_type": str(
                        game.get("seasonType", "regular")
                    ).lower(),
                    "player_id": player.get("player_id"),
                    "player_name": player.get("player_name"),
                    "team_id": side["team_id"],
                    "team": side["team"],
                    "opponent_team_id": side["opponent_team_id"],
                    "opponent": side["opponent"],
                    "home_away": side["home_away"],
                    "is_team_row": False,
                }
            )

            rows.append(row)

    return rows


def _prune_history(
    history: pd.DataFrame,
    *,
    games: list[dict],
    season: int,
    week: int,
) -> pd.DataFrame:
    """
    Keep complete historical games needed for:
      1. every selected team's offensive history,
      2. every selected team's defensive history,
      3. every current candidate player's full career history.

    Entire game IDs are retained so player-share/team-total features remain
    identical to what the full feature engine would calculate.
    """

    selected_teams: set[str] = set()

    for game in games:
        selected_teams.add(str(game["homeTeam"]))
        selected_teams.add(str(game["awayTeam"]))

    current_candidates = history.loc[
        (pd.to_numeric(history["season"], errors="coerce") == season)
        & (pd.to_numeric(history["week"], errors="coerce") < week)
        & history["team"].astype(str).isin(selected_teams)
    ].copy()

    if "is_team_row" in current_candidates.columns:
        current_candidates = current_candidates.loc[
            ~current_candidates["is_team_row"]
            .fillna(False)
            .astype(bool)
        ]

    candidate_keys = set(
        current_candidates["_player_key"]
        .dropna()
        .astype(str)
        .tolist()
    )

    team_context_mask = (
        history["team"].astype(str).isin(selected_teams)
        | history["opponent"].astype(str).isin(selected_teams)
    )

    player_context_mask = (
        history["_player_key"].astype(str).isin(candidate_keys)
    )

    relevant_seed = history.loc[
        team_context_mask | player_context_mask
    ]

    relevant_game_ids = set(
        pd.to_numeric(
            relevant_seed["game_id"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    game_ids_numeric = pd.to_numeric(
        history["game_id"],
        errors="coerce",
    )

    pruned = history.loc[
        game_ids_numeric.isin(relevant_game_ids)
    ].copy()

    return pruned.reset_index(drop=True)


def _load_artifacts() -> dict[str, dict]:

    artifacts: dict[str, dict] = {}

    for target, filename in TARGET_ARTIFACTS.items():

        path = ARTIFACT_DIR / filename

        if not path.exists():
            raise RuntimeError(f"Missing model artifact: {path}")

        artifact = joblib.load(path)

        if not isinstance(artifact, dict):
            raise RuntimeError(
                f"Unexpected artifact format: {path}"
            )

        if artifact.get("target") != target:
            raise RuntimeError(
                f"Artifact target mismatch: {path}"
            )

        artifacts[target] = artifact

    return artifacts


def _pregame_eligibility(
    frame: pd.DataFrame,
    artifact: dict,
) -> pd.Series:

    mask = pd.Series(True, index=frame.index)

    mask &= (
        pd.to_numeric(
            frame["player_games_before"],
            errors="coerce",
        )
        .fillna(0)
        .ge(1)
    )

    history_columns = artifact.get(
        "eligibility_history_columns",
        [],
    )

    available = [
        column
        for column in history_columns
        if column in frame.columns
    ]

    if not available:
        raise RuntimeError(
            f"No eligibility history columns for "
            f"{artifact['target']}"
        )

    history_signal = pd.Series(
        False,
        index=frame.index,
    )

    for column in available:
        history_signal |= (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .fillna(0)
            .gt(0.25)
        )

    mask &= history_signal

    if "is_team_row" in frame.columns:
        mask &= ~frame["is_team_row"].fillna(False).astype(bool)

    return mask


def _numeric_features(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:

    missing = sorted(
        set(columns) - set(frame.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing trained live features: "
            + ", ".join(missing[:20])
        )

    output = frame[columns].copy()

    for column in columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    return output.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _predict_target(
    live_rows: pd.DataFrame,
    artifact: dict,
) -> pd.DataFrame:

    target = str(artifact["target"])

    mask = _pregame_eligibility(
        live_rows,
        artifact,
    )

    candidates = live_rows.loc[mask].copy()

    if candidates.empty:
        return pd.DataFrame()

    feature_columns = list(
        artifact["feature_columns"]
    )

    x = _numeric_features(
        candidates,
        feature_columns,
    )

    predictions = np.asarray(
        artifact["model"].predict(x),
        dtype=float,
    )

    bounds = artifact.get(
        "prediction_bounds"
    )

    if bounds is not None and len(bounds) == 2:
        predictions = np.clip(
            predictions,
            float(bounds[0]),
            float(bounds[1]),
        )

    candidates["target"] = target
    candidates["projection"] = predictions
    candidates["model_name"] = artifact.get(
        "model_name"
    )
    candidates["validation_mean_mae"] = artifact.get(
        "validation_mean_mae"
    )
    candidates["holdout_mae"] = artifact.get(
        "holdout_mae"
    )
    candidates["holdout_bias"] = artifact.get(
        "holdout_bias"
    )

    return candidates


def run_live_predictions(
    *,
    season: int,
    week: int,
    game_id: int | None = None,
    team: str | None = None,
) -> pd.DataFrame:

    schedule = _load_schedule(season)

    games = _select_games(
        schedule,
        season=season,
        week=week,
        game_id=game_id,
        team=team,
    )

    if not games:
        raise RuntimeError(
            f"No uncompleted games found for "
            f"{season} week {week}"
        )

    print("=" * 80)
    print(
        f"CFB LIVE PLAYER PROJECTIONS — "
        f"{season} WEEK {week}"
    )
    print("=" * 80)

    print(
        f"Upcoming games selected: {len(games)}"
    )

    print("Loading canonical history...")

    history = load_canonical_history()

    print(
        f"Full history: {len(history):,} player-game rows"
    )

    synthetic_rows: list[dict] = []

    for game in games:

        print(
            f"  {game['awayTeam']} @ "
            f"{game['homeTeam']} [{game['id']}]"
        )

        synthetic_rows.extend(
            _make_game_rows(
                history,
                game,
                season=season,
                week=week,
            )
        )

    if not synthetic_rows:
        raise RuntimeError(
            "No live player candidates were constructed."
        )

    print("Pruning history to relevant games...")

    relevant_history = _prune_history(
        history,
        games=games,
        season=season,
        week=week,
    )

    print(
        f"Relevant history: "
        f"{len(relevant_history):,} rows "
        f"({len(relevant_history) / len(history):.1%} of full history)"
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
        f"Building 465 leakage-safe features on "
        f"{len(combined):,} rows..."
    )

    featured, feature_columns = build_features(
        combined
    )

    print(
        f"Feature build complete: "
        f"{len(feature_columns)} features"
    )

    selected_game_ids = {
        int(game["id"])
        for game in games
    }

    live_rows = featured.loc[
        pd.to_numeric(
            featured["game_id"],
            errors="coerce",
        ).isin(selected_game_ids)
    ].copy()

    artifacts = _load_artifacts()

    outputs: list[pd.DataFrame] = []

    print("Running trained models...")

    for target, artifact in artifacts.items():

        result = _predict_target(
            live_rows,
            artifact,
        )

        print(
            f"  {target}: "
            f"{len(result)} eligible projections"
        )

        if not result.empty:
            outputs.append(result)

    if not outputs:
        raise RuntimeError(
            "No players passed pregame eligibility."
        )

    predictions = pd.concat(
        outputs,
        ignore_index=True,
        sort=False,
    )

    schedule_lookup = {
        int(game["id"]): game
        for game in games
    }

    predictions["start_date"] = predictions[
        "game_id"
    ].map(
        lambda value:
        schedule_lookup[int(value)].get(
            "startDate"
        )
    )

    keep = [
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
        "projection",
        "model_name",
        "player_games_before",
        "player_season_games_before",
        "validation_mean_mae",
        "holdout_mae",
        "holdout_bias",
    ]

    keep = [
        column
        for column in keep
        if column in predictions.columns
    ]

    predictions = predictions[
        keep
    ].copy()

    predictions["projection"] = (
        pd.to_numeric(
            predictions["projection"],
            errors="coerce",
        )
        .round(2)
    )

    predictions = predictions.sort_values(
        [
            "game_id",
            "team",
            "player_name",
            "target",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / f"live_predictions_{season}_week_{week}.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("CFB LIVE PROJECTION BUILD COMPLETE")
    print("=" * 80)
    print(
        f"Projection rows: {len(predictions):,}"
    )
    print(
        f"Output: {output_path}"
    )

    return predictions


def main() -> None:

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

    run_live_predictions(
        season=args.season,
        week=args.week,
        game_id=args.game_id,
        team=args.team,
    )


if __name__ == "__main__":
    main()
