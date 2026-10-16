import pandas as pd

paths = [
    "data/processed/cfb_matchup_oof_features_v0_1.parquet",
    "data/processed/cfb_matchup_walkforward_2026_weeks_1_3_v0_1.parquet",
]

for path in paths:
    df = pd.read_parquet(path)

    print("=" * 100)
    print(path)
    print("=" * 100)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print()
    print("COLUMN NAMES")
    for col in df.columns:
        print(col)

    print()
    print("SEASONS:", sorted(df["season"].dropna().unique().tolist()) if "season" in df else "NO SEASON")
    print("WEEKS:", sorted(df["week"].dropna().unique().tolist()) if "week" in df else "NO WEEK")
    print("GAMES:", df["game_id"].nunique() if "game_id" in df else "NO GAME_ID")

    print()
    print("FIRST 2 ROWS")
    print(df.head(2).to_string())
    print()
