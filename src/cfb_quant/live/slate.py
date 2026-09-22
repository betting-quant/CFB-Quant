from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from cfb_quant.features.engine import build_features, load_canonical_history
from cfb_quant.live.predict import (
    _load_schedule,
    _make_game_rows,
    _prune_history,
    _load_artifacts,
    _predict_target,
)

REPORT_DIR = Path("reports")

PASS_TARGETS = {
    "pass_attempts",
    "completions",
    "passing_yards",
}

RUSH_TARGETS = {
    "rush_attempts",
    "rushing_yards",
}

RECEIVING_TARGETS = {
    "receptions",
    "receiving_yards",
}


def select_date_games(
    season: int,
    week: int,
    date: str,
    timezone: str,
) -> list[dict]:

    tz = ZoneInfo(timezone)

    games = _load_schedule(season)

    selected = []

    for game in games:

        if int(game.get("season", -1)) != season:
            continue

        if int(game.get("week", -1)) != week:
            continue

        if bool(game.get("completed", False)):
            continue

        start = pd.to_datetime(
            game["startDate"],
            utc=True,
        ).tz_convert(tz)

        if start.date().isoformat() != date:
            continue

        game = dict(game)
        game["_local_start"] = start

        selected.append(game)

    return sorted(
        selected,
        key=lambda g: g["_local_start"],
    )


def build_role_table(
    history: pd.DataFrame,
    *,
    season: int,
    week: int,
    teams: set[str],
) -> pd.DataFrame:

    frame = history.loc[
        (pd.to_numeric(history["season"], errors="coerce") == season)
        & (pd.to_numeric(history["week"], errors="coerce") < week)
        & history["team"].astype(str).isin(teams)
    ].copy()

    if "is_team_row" in frame.columns:
        frame = frame.loc[
            ~frame["is_team_row"]
            .fillna(False)
            .astype(bool)
        ].copy()

    for col in [
        "pass_attempts",
        "rush_attempts",
        "targets_pbp",
        "receptions",
    ]:
        frame[col] = pd.to_numeric(
            frame[col],
            errors="coerce",
        ).fillna(0)

    # Receiving-opportunity fallback:
    # if PBP target data is missing, receptions still provide
    # a conservative minimum opportunity count.
    frame["receiving_opportunities"] = frame[
        ["targets_pbp", "receptions"]
    ].max(axis=1)

    latest_week_by_team = (
        frame.groupby("team")["week"].transform("max")
    )

    frame["_latest_game"] = (
        pd.to_numeric(frame["week"], errors="coerce")
        .eq(latest_week_by_team)
    )

    rows = []

    for (team, player_key), player in frame.groupby(
        ["team", "_player_key"],
        sort=False,
    ):

        team_frame = frame.loc[
            frame["team"].astype(str).eq(str(team))
        ]

        # ----------------------------
        # PASSING ROLE
        # ----------------------------

        season_pa = player["pass_attempts"].sum()

        latest_pa = player.loc[
            player["_latest_game"],
            "pass_attempts",
        ].sum()

        team_season_pa = (
            team_frame["pass_attempts"].sum()
        )

        team_latest_pa = team_frame.loc[
            team_frame["_latest_game"],
            "pass_attempts",
        ].sum()

        pass_share = (
            season_pa / team_season_pa
            if team_season_pa
            else 0
        )

        latest_pass_share = (
            latest_pa / team_latest_pa
            if team_latest_pa
            else 0
        )

        pass_active = (
            (season_pa >= 15 or latest_pa >= 10)
            and (
                pass_share >= 0.25
                or latest_pass_share >= 0.35
            )
        )

        # ----------------------------
        # RUSHING ROLE
        # ----------------------------

        season_carries = (
            player["rush_attempts"].sum()
        )

        latest_carries = player.loc[
            player["_latest_game"],
            "rush_attempts",
        ].sum()

        team_season_carries = (
            team_frame["rush_attempts"].sum()
        )

        team_latest_carries = team_frame.loc[
            team_frame["_latest_game"],
            "rush_attempts",
        ].sum()

        rush_share = (
            season_carries / team_season_carries
            if team_season_carries
            else 0
        )

        latest_rush_share = (
            latest_carries / team_latest_carries
            if team_latest_carries
            else 0
        )

        rush_active = (
            pass_active
            or (
                season_carries >= 8
                and rush_share >= 0.10
            )
            or (
                latest_carries >= 6
                and latest_rush_share >= 0.12
            )
        )

        # ----------------------------
        # RECEIVING ROLE
        # ----------------------------

        season_targets = (
            player["receiving_opportunities"].sum()
        )

        latest_targets = player.loc[
            player["_latest_game"],
            "receiving_opportunities",
        ].sum()

        team_season_targets = (
            team_frame["receiving_opportunities"].sum()
        )

        team_latest_targets = team_frame.loc[
            team_frame["_latest_game"],
            "receiving_opportunities",
        ].sum()

        target_share = (
            season_targets / team_season_targets
            if team_season_targets
            else 0
        )

        latest_target_share = (
            latest_targets / team_latest_targets
            if team_latest_targets
            else 0
        )

        receive_active = (
            (
                season_targets >= 4
                and target_share >= 0.06
            )
            or (
                latest_targets >= 3
                and latest_target_share >= 0.08
            )
        )

        rows.append(
            {
                "team": str(team),
                "_player_key": str(player_key),
                "pass_active": bool(pass_active),
                "rush_active": bool(rush_active),
                "receive_active": bool(receive_active),
                "season_pass_attempts": float(season_pa),
                "pass_share": float(pass_share),
                "season_carries": float(season_carries),
                "rush_share": float(rush_share),
                "season_receiving_opportunities": float(
                    season_targets
                ),
                "target_share": float(target_share),
            }
        )

    return pd.DataFrame(rows)


def apply_role_filter(
    predictions: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:

    if predictions.empty:
        return predictions

    predictions = predictions.copy()

    predictions["_player_key"] = (
        predictions["_player_key"].astype(str)
    )

    predictions["team"] = (
        predictions["team"].astype(str)
    )

    merged = predictions.merge(
        roles,
        on=["team", "_player_key"],
        how="left",
        validate="many_to_one",
    )

    for col in [
        "pass_active",
        "rush_active",
        "receive_active",
    ]:
        merged[col] = (
            merged[col]
            .fillna(False)
            .astype(bool)
        )

    keep = pd.Series(
        False,
        index=merged.index,
    )

    keep |= (
        merged["target"].isin(PASS_TARGETS)
        & merged["pass_active"]
    )

    keep |= (
        merged["target"].isin(RUSH_TARGETS)
        & merged["rush_active"]
    )

    keep |= (
        merged["target"].isin(RECEIVING_TARGETS)
        & merged["receive_active"]
    )

    return merged.loc[keep].copy()


def run_date_slate(
    *,
    season: int,
    week: int,
    date: str,
    timezone: str,
    batch_size: int,
) -> pd.DataFrame:

    games = select_date_games(
        season,
        week,
        date,
        timezone,
    )

    if not games:
        raise RuntimeError(
            f"No uncompleted games found for {date}"
        )

    print("=" * 90)
    print(
        f"CFB LIVE SATURDAY SLATE — "
        f"{date} — {len(games)} GAMES"
    )
    print("=" * 90)

    history = load_canonical_history()

    print(
        f"Canonical history: "
        f"{len(history):,} rows"
    )

    teams = set()

    for game in games:
        teams.add(str(game["homeTeam"]))
        teams.add(str(game["awayTeam"]))

    print("Building current-role table...")

    roles = build_role_table(
        history,
        season=season,
        week=week,
        teams=teams,
    )

    print(
        f"Role rows: {len(roles):,}"
    )

    artifacts = _load_artifacts()

    all_raw = []

    total_batches = (
        len(games) + batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(0, len(games), batch_size),
        start=1,
    ):

        batch = games[
            start:start + batch_size
        ]

        print()
        print("=" * 90)
        print(
            f"BATCH {batch_number}/{total_batches} "
            f"— {len(batch)} games"
        )
        print("=" * 90)

        for game in batch:
            local = game["_local_start"]

            print(
                f"  {local.strftime('%I:%M %p')}  "
                f"{game['awayTeam']} @ "
                f"{game['homeTeam']}"
            )

        synthetic_rows = []

        for game in batch:
            synthetic_rows.extend(
                _make_game_rows(
                    history,
                    game,
                    season=season,
                    week=week,
                )
            )

        if not synthetic_rows:
            print("No candidate rows in batch.")
            continue

        relevant_history = _prune_history(
            history,
            games=batch,
            season=season,
            week=week,
        )

        print(
            f"Relevant history: "
            f"{len(relevant_history):,} rows"
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
            f"Building features on "
            f"{len(combined):,} rows..."
        )

        featured, feature_columns = (
            build_features(combined)
        )

        print(
            f"Feature build complete: "
            f"{len(feature_columns)} features"
        )

        game_ids = {
            int(game["id"])
            for game in batch
        }

        live_rows = featured.loc[
            pd.to_numeric(
                featured["game_id"],
                errors="coerce",
            ).isin(game_ids)
        ].copy()

        batch_outputs = []

        for target, artifact in artifacts.items():

            result = _predict_target(
                live_rows,
                artifact,
            )

            if not result.empty:
                batch_outputs.append(result)

        if batch_outputs:

            raw = pd.concat(
                batch_outputs,
                ignore_index=True,
                sort=False,
            )

            all_raw.append(raw)

            print(
                f"Raw projection rows: "
                f"{len(raw):,}"
            )

    if not all_raw:
        raise RuntimeError(
            "No slate projections were generated."
        )

    raw = pd.concat(
        all_raw,
        ignore_index=True,
        sort=False,
    )

    print()
    print("Applying current-role filter...")

    active = apply_role_filter(
        raw,
        roles,
    )

    schedule_lookup = {
        int(game["id"]): game
        for game in games
    }

    for frame in [raw, active]:

        frame["start_date"] = (
            frame["game_id"].map(
                lambda x:
                schedule_lookup[int(x)][
                    "startDate"
                ]
            )
        )

        frame["local_start"] = (
            frame["game_id"].map(
                lambda x:
                schedule_lookup[int(x)][
                    "_local_start"
                ].isoformat()
            )
        )

        frame["projection"] = (
            pd.to_numeric(
                frame["projection"],
                errors="coerce",
            ).round(2)
        )

    raw_path = (
        REPORT_DIR
        / f"live_predictions_{date}_raw.csv"
    )

    active_path = (
        REPORT_DIR
        / f"live_predictions_{date}_active.csv"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    active.to_csv(
        active_path,
        index=False,
    )

    print()
    print("=" * 90)
    print("SATURDAY SLATE COMPLETE")
    print("=" * 90)

    print(
        f"Games: {len(games)}"
    )

    print(
        f"Raw projection rows: "
        f"{len(raw):,}"
    )

    print(
        f"Active-role projection rows: "
        f"{len(active):,}"
    )

    print(
        f"Raw output: {raw_path}"
    )

    print(
        f"Active output: {active_path}"
    )

    print()
    print("ACTIVE PROJECTIONS BY MARKET")

    print(
        active["target"]
        .value_counts()
        .to_string()
    )

    return active


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
        "--date",
        required=True,
    )

    parser.add_argument(
        "--timezone",
        default="America/Edmonton",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    run_date_slate(
        season=args.season,
        week=args.week,
        date=args.date,
        timezone=args.timezone,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
