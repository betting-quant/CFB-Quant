import pandas as pd

path = "data/processed/cfb_pregame_features_matchup_through_2026_week_3_v0_1.parquet"

df = pd.read_parquet(path)

features = [
    "matchup_proj_points",
    "matchup_proj_pass_attempts",
    "matchup_proj_rush_attempts",
    "matchup_proj_total_plays",
    "matchup_proj_passing_yards",
    "matchup_proj_rushing_yards",
    "matchup_proj_possession_seconds",
]

any_present = df[features].notna().any(axis=1)
all_present = df[features].notna().all(axis=1)
partial = df.loc[any_present & ~all_present].copy()

print("=" * 100)
print("PARTIAL MATCHUP FEATURE AUDIT")
print("=" * 100)

print("Partial player rows:", len(partial))
print(
    "Partial team-games:",
    partial[["game_id", "team"]].drop_duplicates().shape[0],
)

print()
print("PARTIAL PLAYER ROWS BY SEASON")
print(
    partial.groupby("season").size()
    if len(partial)
    else "NONE"
)

print()
print("MISSING VALUES BY FEATURE")
for feature in features:
    print(
        f"{feature}:",
        int(partial[feature].isna().sum()),
    )

team_games = (
    partial[
        [
            "game_id",
            "season",
            "week",
            "team",
            "opponent",
            *features,
        ]
    ]
    .drop_duplicates(["game_id", "team"])
    .copy()
)

for feature in features:
    team_games[f"missing__{feature}"] = team_games[feature].isna()

print()
print("PARTIAL TEAM-GAMES BY SEASON")
if len(team_games):
    print(team_games.groupby("season").size())
else:
    print("NONE")

print()
print("PARTIAL TEAM-GAME DETAILS")
cols = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
] + [
    f"missing__{feature}"
    for feature in features
]

print(
    team_games[cols]
    .sort_values(
        ["season", "week", "game_id", "team"]
    )
    .head(100)
    .to_string(index=False)
)

print()
print("=" * 100)
