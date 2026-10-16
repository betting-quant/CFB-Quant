import pandas as pd
import numpy as np

path = "reports/live_player_production_2026_week_4_v0_1.csv"

df = pd.read_csv(path)

print("=" * 100)
print("WEEK 4 PLAYER PROJECTION SANITY AUDIT")
print("=" * 100)

keys = [
    "game_id",
    "team",
    "player_id",
    "target",
]

duplicates = int(
    df.duplicated(keys).sum()
)

print("Rows:", len(df))
print("Games:", df["game_id"].nunique())
print("Players:", df["player_id"].nunique())
print("Duplicate player-target rows:", duplicates)

if duplicates:
    raise RuntimeError(
        f"Found {duplicates} duplicate player-target rows."
    )

for col in [
    "projection_original",
    "projection_matchup",
    "projection_blended",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df["model_delta"] = (
    df["projection_matchup"]
    - df["projection_original"]
)

df["abs_model_delta"] = (
    df["model_delta"].abs()
)

df["blend_delta_vs_original"] = (
    df["projection_blended"]
    - df["projection_original"]
)

df["abs_blend_delta_vs_original"] = (
    df["blend_delta_vs_original"].abs()
)

print()
print("NULL PROJECTIONS")
print(
    df[
        [
            "projection_original",
            "projection_matchup",
            "projection_blended",
        ]
    ]
    .isna()
    .sum()
)

print()
print("=" * 100)
print("PROJECTION RANGES BY TARGET")
print("=" * 100)

summary = (
    df.groupby("target")
    .agg(
        rows=("projection_blended", "size"),
        min_proj=("projection_blended", "min"),
        median_proj=("projection_blended", "median"),
        mean_proj=("projection_blended", "mean"),
        max_proj=("projection_blended", "max"),
        mean_abs_model_delta=("abs_model_delta", "mean"),
        max_abs_model_delta=("abs_model_delta", "max"),
    )
)

print(
    summary.round(3).to_string()
)

print()
print("=" * 100)
print("TOP 30 ORIGINAL VS MATCHUP DISAGREEMENTS")
print("=" * 100)

cols = [
    "game_id",
    "team",
    "opponent",
    "player_name",
    "target",
    "projection_original",
    "projection_matchup",
    "projection_blended",
    "model_delta",
]

print(
    df.sort_values(
        "abs_model_delta",
        ascending=False,
    )[
        cols
    ]
    .head(30)
    .to_string(index=False)
)

print()
print("=" * 100)
print("TOP 30 BLEND MOVES VS ORIGINAL")
print("=" * 100)

cols2 = [
    "game_id",
    "team",
    "opponent",
    "player_name",
    "target",
    "projection_original",
    "projection_matchup",
    "blend_weight_matchup",
    "projection_blended",
    "blend_delta_vs_original",
]

print(
    df.sort_values(
        "abs_blend_delta_vs_original",
        ascending=False,
    )[
        cols2
    ]
    .head(30)
    .to_string(index=False)
)

print()
print("=" * 100)
print("HIGHEST 15 PROJECTIONS BY TARGET")
print("=" * 100)

for target in sorted(df["target"].unique()):

    sub = (
        df.loc[
            df["target"].eq(target)
        ]
        .sort_values(
            "projection_blended",
            ascending=False,
        )
        .head(15)
    )

    print()
    print(target.upper())
    print("-" * 100)

    print(
        sub[
            [
                "team",
                "opponent",
                "player_name",
                "projection_blended",
                "projection_original",
                "projection_matchup",
            ]
        ]
        .to_string(index=False)
    )

print()
print("=" * 100)
print("SANITY AUDIT COMPLETE")
print("=" * 100)
