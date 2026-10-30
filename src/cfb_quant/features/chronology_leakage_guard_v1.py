"""Detect demonstrable future-result leakage in matchup lag features."""

from pathlib import Path
import pandas as pd
import numpy as np


def audit_future_result_leakage(dataset, kickoff_map):
    frame = pd.read_parquet(dataset)
    schedule = pd.read_csv(kickoff_map)

    keys = ["game_id", "season", "team", "home_away"]

    required = set(keys + [
        "week", "season_type", "points", "off_points_lag1"
    ])

    if missing := required - set(frame.columns):
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    if missing := set(keys + ["kickoff_utc"]) - set(schedule.columns):
        raise ValueError(f"Missing schedule columns: {sorted(missing)}")

    def normalize(df):
        df = df.copy()
        df["game_id"] = (
            df.game_id.astype("string")
            .str.replace(r"\.0$", "", regex=True)
        )
        df["season"] = pd.to_numeric(
            df.season, errors="raise"
        ).astype(int)
        df["team"] = df.team.astype(str).str.strip()
        df["home_away"] = (
            df.home_away.astype(str).str.strip().str.lower()
        )
        return df

    frame = normalize(frame)
    schedule = normalize(schedule)

    if frame.duplicated(keys).any() or schedule.duplicated(keys).any():
        raise ValueError("Duplicate team-game identities")

    frame = frame.drop(columns=["kickoff_utc"], errors="ignore").merge(
        schedule[keys + ["kickoff_utc"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    frame["kickoff_utc"] = pd.to_datetime(
        frame.kickoff_utc, utc=True, errors="coerce"
    )

    frame["_type_order"] = (
        frame.season_type.astype(str).str.lower()
        .map({"regular": 0, "postseason": 1})
        .fillna(9).astype(int)
    )

    frame["week_numeric"] = pd.to_numeric(
        frame.week, errors="raise"
    )

    frame = frame.sort_values(
        ["season", "_type_order", "week_numeric", "game_id", "team"],
        kind="mergesort"
    )

    grouped = frame.groupby("team", sort=False)

    frame["previous_kickoff"] = grouped.kickoff_utc.shift(1)
    frame["previous_points"] = grouped.points.shift(1)

    unsafe = (
        frame.kickoff_utc.notna()
        & frame.previous_kickoff.notna()
        & (frame.previous_kickoff >= frame.kickoff_utc)
    )

    flagged = frame.loc[unsafe].copy()

    future_match = (
        flagged.off_points_lag1.notna()
        & flagged.previous_points.notna()
        & np.isclose(
            pd.to_numeric(flagged.off_points_lag1),
            pd.to_numeric(flagged.previous_points),
            atol=1e-8,
            rtol=0,
        )
    )

    return {
        "total_rows": len(frame),
        "missing_kickoff_rows": int(frame.kickoff_utc.isna().sum()),
        "unsafe_predecessor_rows": len(flagged),
        "stored_lag_matches_future_result": int(future_match.sum()),
        "chronology_certified": False,
        "source_availability_certified": False,
    }


def require_clean_chronology(result):
    if result["missing_kickoff_rows"]:
        raise ValueError("Missing verified kickoff timestamps")

    if result["unsafe_predecessor_rows"]:
        raise ValueError("Unsafe future-game ordering detected")

    if result["stored_lag_matches_future_result"]:
        raise ValueError("Future-result feature leakage detected")

    # Passing this guard establishes only these checks, not full PIT.
    return True
