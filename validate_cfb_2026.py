import json
import pandas as pd
from pathlib import Path

feature_path = Path("data/processed/cfb_pregame_features_through_2026_week_3_v0_1.parquet")
canonical_path = Path("data/processed/canonical_player_games_2026.parquet")
manifest_path = Path("reports/cfb_feature_manifest_through_2026_week_3_v0_1.json")

features = pd.read_parquet(feature_path)
canonical = pd.read_parquet(canonical_path)

f26 = features[pd.to_numeric(features["season"], errors="coerce").eq(2026)].copy()
c26 = canonical[pd.to_numeric(canonical["season"], errors="coerce").eq(2026)].copy()

print("=" * 90)
print("2026 PLAYER FEATURE VALIDATION")
print("=" * 90)

print("\nNEW FEATURE FILE")
print("Rows:", len(f26))
print("Games:", f26["game_id"].nunique())
print("Players:", f26["player_id"].nunique())
print("Weeks:", sorted(f26["week"].dropna().unique().tolist()))

print("\nROWS BY WEEK")
print(f26.groupby("week").size())

print("\nGAMES BY WEEK")
print(f26.groupby("week")["game_id"].nunique())

print("\nCANONICAL 2026 SOURCE")
print("Rows:", len(c26))
print("Games:", c26["game_id"].nunique())
print("Players:", c26["player_id"].nunique())
print("Weeks:", sorted(c26["week"].dropna().unique().tolist()))

player_key = (
    f26["player_id"].astype("Int64").astype(str)
    if "player_id" in f26.columns
    else f26["player_name"].astype(str)
)

dupes = pd.DataFrame({
    "game_id": f26["game_id"],
    "player_key": player_key,
}).duplicated().sum()

print("\nDUPLICATE PLAYER-GAMES:", dupes)

with open(manifest_path, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)

print("\nMANIFEST")
print("Feature count:", manifest.get("feature_count"))
print("Last season:", max(manifest.get("seasons", [])))
print("Has 2026:", 2026 in manifest.get("seasons", []))

print("\nTARGET 2026 COMPLETED-GAME EXPECTATION")
print("Expected completed games through Week 3: 260")
print("Observed feature games:", f26["game_id"].nunique())

print("=" * 90)
