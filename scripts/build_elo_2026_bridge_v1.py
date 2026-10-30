"""2026 reconstructed Elo bridge — research only."""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math

import pandas as pd

from cfb_quant.spread_total.elo_engine_v58 import EloConfig
from cfb_quant.spread_total.elo_replay_v58 import ReplayGame, replay_games
from cfb_quant.spread_total.elo_incremental_v58 import merge_fbs_captures

ROOT = Path.cwd()
HIST = ROOT / "data/processed/cfb_matchup_features_opponent_v45.parquet"
CAP = ROOT / (
    "reports/v58_development/elo_live_captures_v58/"
    "20261030T152316258421Z"
)
PREFLIGHT = ROOT / (
    "reports/research/elo_shadow_challenger_v1/"
    "2026_coverage_preflight.json"
)
OUT = ROOT / "reports/research/elo_2026_rating_bridge_v1"
CSV = OUT / "pregame_ratings.csv"
MANIFEST = OUT / "manifest.json"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(
    not CSV.exists() and not MANIFEST.exists(),
    "Bridge output already exists; refusing overwrite"
)

preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))

raw_path = CAP / "games_2026.raw.json"
capture_manifest_path = CAP / "manifest.json"
raw = raw_path.read_bytes()
capture_manifest = json.loads(
    capture_manifest_path.read_text(encoding="utf-8")
)

require(
    digest(raw_path) == capture_manifest["payload_sha256"],
    "Source capture SHA256 mismatch"
)
require(
    digest(raw_path) == preflight["source_sha256"],
    "Preflight source SHA256 mismatch"
)

merged = merge_fbs_captures([(raw, capture_manifest)])
observed = merged["latest_capture_utc"]
current = datetime.now(timezone.utc)

require(
    observed <= current,
    "Capture timestamp is in the future"
)

columns = [
    "game_id", "season", "team", "home_away",
    "points", "opponent_points", "kickoff_utc"
]
df = pd.read_parquet(HIST, columns=columns)

require(
    not df.duplicated(["game_id", "home_away"]).any(),
    "Duplicate historical game sides"
)

games = []

for gid, group in df.groupby("game_id", sort=False):
    require(
        len(group) == 2
        and set(group.home_away) == {"home", "away"},
        f"Invalid historical game pairing: {gid}"
    )

    home = group.loc[group.home_away == "home"].iloc[0]
    away = group.loc[group.home_away == "away"].iloc[0]

    require(
        home.team != away.team,
        f"Invalid same-team game: {gid}"
    )
    require(
        int(home.season) == int(away.season),
        f"Season mismatch: {gid}"
    )

    kickoff = pd.Timestamp(home.kickoff_utc).to_pydatetime()
    away_kickoff = pd.Timestamp(away.kickoff_utc).to_pydatetime()

    require(
        kickoff.tzinfo is not None and kickoff == away_kickoff,
        f"Kickoff mismatch: {gid}"
    )

    hp = float(home.points)
    ap = float(away.points)

    require(
        all(math.isfinite(x) and x >= 0 for x in [hp, ap]),
        f"Invalid historical score: {gid}"
    )
    require(
        hp == float(away.opponent_points)
        and ap == float(home.opponent_points),
        f"Historical score disagreement: {gid}"
    )

    games.append(ReplayGame(
        game_id=str(gid),
        kickoff_utc=kickoff,
        home=str(home.team),
        away=str(away.team),
        home_points=hp,
        away_points=ap,
        result_available_utc=kickoff + timedelta(hours=24),
    ))

historical_count = len(games)
require(historical_count == 10369, "Historical coverage changed")

completed_2026 = 0

for game in merged["games"]:
    kickoff = game.kickoff_utc
    require(kickoff.tzinfo is not None, "Naive 2026 kickoff")

    completed = (
        game.home_points is not None
        and game.away_points is not None
    )

    if completed:
        completed_2026 += 1

        require(
            math.isfinite(float(game.home_points))
            and math.isfinite(float(game.away_points)),
            f"Invalid 2026 score: {game.game_id}"
        )
        require(
            kickoff < observed,
            f"Completed game starts at or after capture: {game.game_id}"
        )

    games.append(ReplayGame(
        game_id=game.game_id,
        kickoff_utc=kickoff,
        home=game.home,
        away=game.away,
        home_points=game.home_points if completed else None,
        away_points=game.away_points if completed else None,
        result_available_utc=(
            observed if completed else None
        ),
    ))

require(
    completed_2026 == 450,
    "Expected 2026 completed-game coverage has changed"
)

require(
    len({g.game_id for g in games}) == len(games),
    "Duplicate historical/2026 IDs"
)

print("\n=== REPLAYING 2014–2026 GAME HISTORY ===")

replay = replay_games(
    games,
    config=EloConfig(
        k_factor=24,
        home_advantage=0,
        scale=400,
    ),
    initial_rating=1500.0,
)

require(
    len(replay.snapshots) == len(games),
    "Incomplete replay snapshots"
)

require(
    replay.updates_applied == historical_count + completed_2026,
    "Unexpected update count"
)

eligible = set(preflight["eligible_game_ids"])
source_by_id = {g.game_id: g for g in merged["games"]}

rows = []
skipped_since_preflight = []

for snap in replay.snapshots:
    if snap.game_id not in eligible:
        continue

    original = source_by_id[snap.game_id]

    # Recheck timing at the moment the bridge is generated.
    if original.kickoff_utc <= current + timedelta(minutes=10):
        skipped_since_preflight.append(snap.game_id)
        continue

    require(
        snap.home == original.home
        and snap.away == original.away
        and snap.kickoff_utc == original.kickoff_utc,
        f"2026 snapshot identity mismatch: {snap.game_id}"
    )

    require(
        original.home_points is None
        and original.away_points is None,
        f"Future target has observed outcome: {snap.game_id}"
    )

    rows.append({
        "game_id": snap.game_id,
        "kickoff_utc": snap.kickoff_utc.isoformat(),
        "home_team": snap.home,
        "away_team": snap.away,
        "home_elo": snap.home_elo,
        "away_elo": snap.away_elo,
        "elo_difference": snap.home_elo - snap.away_elo,
        "elo_home_win_probability": snap.expected_home_win,
    })

result = pd.DataFrame(rows)
require(
    not result.empty and result.game_id.is_unique,
    "No valid unique future ratings generated"
)

require(
    len(result) + len(skipped_since_preflight) == len(eligible),
    "Eligible forecast coverage mismatch"
)

OUT.mkdir(parents=True, exist_ok=True)
result.sort_values(["kickoff_utc", "game_id"]).to_csv(
    CSV, index=False
)

manifest = {
    "schema": "elo_2026_rating_bridge_v1",
    "created_at_utc": current.isoformat(),
    "capture_observed_utc": observed.isoformat(),
    "historical_games": historical_count,
    "completed_2026_games": completed_2026,
    "all_replay_snapshots": len(replay.snapshots),
    "rating_updates": replay.updates_applied,
    "preflight_eligible_games": len(eligible),
    "exported_future_games": len(result),
    "skipped_since_preflight": skipped_since_preflight,
    "historical_source_sha256": digest(HIST),
    "capture_source_sha256": digest(raw_path),
    "preflight_sha256": digest(PREFLIGHT),
    "output_sha256": digest(CSV),
    "historical_result_timing_assumption_hours": 24,
    "result_2026_availability_policy": "first_observed_at_capture",
    "2026_completed_results_available_utc": observed.isoformat(),
    "2026_weekly_result_timestamps_verified": False,
    "historical_point_in_time_certified": False,
    "prospective_predictions_generated": False,
    "production_approved": False,
    "research_only": True,
}

with MANIFEST.open("x", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")

print("\n=== 2026 ELO RATING BRIDGE COMPLETE ===")
print("Historical games:", historical_count)
print("Completed 2026 games:", completed_2026)
print("Rating updates:", replay.updates_applied)
print("Eligible at preflight:", len(eligible))
print("Future ratings exported:", len(result))
print("Expired since preflight:", len(skipped_since_preflight))
print("Output:", CSV.relative_to(ROOT))
print("Historical PIT certified: NO")
print("Production modified: NO")
