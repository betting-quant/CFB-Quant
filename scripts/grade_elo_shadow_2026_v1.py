"""Step 39: Immutable prospective Ridge vs Elo grading.

No fitting, no forecast changes, no sportsbook assumptions.
Only observed completed games may be graded.
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys

import pandas as pd

from cfb_quant.spread_total.elo_incremental_v58 import (
    merge_fbs_captures,
)

ROOT = Path(__file__).resolve().parents[1]

LEDGER_DIR = ROOT / (
    "reports/research/elo_shadow_challenger_v1/"
    "2026_shadow_ledger"
)

LEDGER = LEDGER_DIR / "paired_forecasts.csv"
LEDGER_MANIFEST = LEDGER_DIR / "manifest.json"

CAPTURE_ROOT = ROOT / (
    "reports/v58_development/elo_live_captures_v58"
)

OUTPUT_ROOT = ROOT / (
    "reports/research/elo_shadow_grading_v1"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value):
    stamp = pd.Timestamp(value)

    require(
        not pd.isna(stamp) and stamp.tzinfo is not None,
        f"Missing or naive timestamp: {value}"
    )

    return stamp.tz_convert("UTC")


def metrics(frame, pred_col):
    if frame.empty:
        return None

    actual = frame["actual_home_margin"]
    predicted = frame[pred_col]
    error = actual - predicted

    non_ties = frame.loc[
        frame["actual_home_margin"] != 0
    ]

    winner_accuracy = None

    if not non_ties.empty:
        winner_accuracy = float(
            (
                (non_ties[pred_col] > 0)
                == (non_ties["actual_home_margin"] > 0)
            ).mean()
        )

    return {
        "games": int(len(frame)),
        "mae": float(error.abs().mean()),
        "rmse": float((error.pow(2).mean()) ** 0.5),
        "bias_actual_minus_pred": float(error.mean()),
        "winner_accuracy_excluding_ties": winner_accuracy,
    }


def main():
    require(
        LEDGER.exists() and LEDGER_MANIFEST.exists(),
        "Committed shadow ledger missing"
    )

    ledger_info = json.loads(
        LEDGER_MANIFEST.read_text(encoding="utf-8")
    )

    require(
        sha(LEDGER) == ledger_info["output_sha256"],
        "Ledger SHA256 mismatch"
    )

    require(
        ledger_info["frozen_correction_applied"] is False,
        "Unexpected blended forecasts"
    )

    require(
        ledger_info["production_approved"] is False,
        "Unexpected production approval"
    )

    if len(sys.argv) > 1:
        capture_dir = Path(sys.argv[1]).resolve()
    else:
        candidates = sorted(
            p for p in CAPTURE_ROOT.iterdir()
            if p.is_dir()
            and (p / "games_2026.raw.json").is_file()
            and (p / "manifest.json").is_file()
        )

        require(candidates, "No saved captures available")
        capture_dir = candidates[-1]

    raw_path = capture_dir / "games_2026.raw.json"
    manifest_path = capture_dir / "manifest.json"

    require(
        raw_path.is_file() and manifest_path.is_file(),
        "Capture files missing"
    )

    capture_info = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    require(
        sha(raw_path) == capture_info["payload_sha256"],
        "Capture payload integrity failure"
    )

    merged = merge_fbs_captures([
        (raw_path.read_bytes(), capture_info)
    ])

    observed = utc(merged["latest_capture_utc"])

    require(
        observed <= pd.Timestamp.now(tz="UTC"),
        "Capture observed in the future"
    )

    source = {
        str(game.game_id): game
        for game in merged["games"]
    }

    ledger = pd.read_csv(
        LEDGER, dtype={"game_id": str}
    )

    require(
        ledger.game_id.is_unique,
        "Duplicate ledger game IDs"
    )

    require(
        len(ledger) == ledger_info["paired_forecasts"],
        "Ledger coverage mismatch"
    )

    require(
        ledger[[
            "ridge_home_margin",
            "elo_home_margin"
        ]].notna().all().all(),
        "Missing original forecasts"
    )

    results = []
    pending = []

    for row in ledger.itertuples(index=False):
        game = source.get(str(row.game_id))

        require(
            game is not None,
            f"Game missing from capture: {row.game_id}"
        )

        kickoff = utc(row.kickoff_utc)
        elo_created = utc(row.elo_created_utc)

        require(
            game.home == row.home_team
            and game.away == row.away_team
            and utc(game.kickoff_utc) == kickoff,
            f"Identity or kickoff mismatch: {row.game_id}"
        )

        require(
            elo_created < kickoff,
            f"Original Elo forecast not pregame: {row.game_id}"
        )

        hp = game.home_points
        ap = game.away_points

        if hp is None or ap is None:
            require(
                hp is None and ap is None,
                f"Partial score: {row.game_id}"
            )
            pending.append(str(row.game_id))
            continue

        # Results are usable only in a capture taken after
        # both kickoff and the saved Elo prediction.
        require(
            observed > kickoff and observed > elo_created,
            f"Result capture is not post-kickoff: {row.game_id}"
        )

        hp = float(hp)
        ap = float(ap)

        require(
            math.isfinite(hp)
            and math.isfinite(ap)
            and hp >= 0
            and ap >= 0,
            f"Invalid final score: {row.game_id}"
        )

        actual = hp - ap

        ridge_margin = float(row.ridge_home_margin)
        elo_margin = float(row.elo_home_margin)

        results.append({
            "game_id": str(row.game_id),
            "kickoff_utc": kickoff.isoformat(),
            "away_team": row.away_team,
            "home_team": row.home_team,
            "capture_observed_utc": observed.isoformat(),
            "actual_away_points": ap,
            "actual_home_points": hp,
            "actual_home_margin": actual,
            "ridge_home_margin": ridge_margin,
            "elo_home_margin": elo_margin,
            "ridge_absolute_error": abs(actual - ridge_margin),
            "elo_absolute_error": abs(actual - elo_margin),
            "elo_improvement": (
                abs(actual - ridge_margin)
                - abs(actual - elo_margin)
            ),
        })

    graded = pd.DataFrame(results)

    require(
        len(graded) + len(pending) == len(ledger),
        "Grading coverage accounting mismatch"
    )

    if not graded.empty:
        graded = graded.sort_values(
            ["kickoff_utc", "game_id"]
        ).reset_index(drop=True)

    ridge_metrics = metrics(
        graded, "ridge_home_margin"
    )
    elo_metrics = metrics(
        graded, "elo_home_margin"
    )

    improvement = None

    if ridge_metrics is not None:
        improvement = (
            ridge_metrics["mae"] - elo_metrics["mae"]
        )

    # One immutable grading run per source capture.
    destination = OUTPUT_ROOT / capture_dir.name

    require(
        not destination.exists(),
        "This capture was already graded; refusing overwrite"
    )

    destination.mkdir(parents=True, exist_ok=False)

    csv_path = destination / "graded_games.csv"
    manifest_out = destination / "manifest.json"

    columns = [
        "game_id", "kickoff_utc", "away_team", "home_team",
        "capture_observed_utc", "actual_away_points",
        "actual_home_points", "actual_home_margin",
        "ridge_home_margin", "elo_home_margin",
        "ridge_absolute_error", "elo_absolute_error",
        "elo_improvement",
    ]

    graded.reindex(columns=columns).to_csv(
        csv_path, index=False
    )

    report = {
        "schema": "elo_shadow_grading_v1",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "capture_observed_utc": observed.isoformat(),
        "capture_payload_sha256": sha(raw_path),
        "capture_manifest_sha256": sha(manifest_path),
        "original_ledger_sha256": sha(LEDGER),
        "graded_csv_sha256": sha(csv_path),
        "forecast_games": len(ledger),
        "graded_games": len(graded),
        "pending_games": len(pending),
        "pending_game_ids": sorted(pending),
        "ridge_metrics": ridge_metrics,
        "elo_metrics": elo_metrics,
        "elo_mae_improvement_over_ridge": improvement,
        "ridge_issuance_time_certified": False,
        "historical_elo_pit_certified": False,
        "frozen_blend_graded": False,
        "production_approved": False,
        "research_only": True,
    }

    with manifest_out.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print("\n=== STEP 39: SHADOW GRADING ===")
    print("Capture:", capture_dir.name)
    print("Ledger games:", len(ledger))
    print("Completed and graded:", len(graded))
    print("Pending:", len(pending))

    if ridge_metrics is None:
        print("\nNo completed games eligible for grading.")
    else:
        print("\n=== PAIRED RESULTS ===")
        print("Ridge MAE:", round(ridge_metrics["mae"], 4))
        print("Elo MAE:", round(elo_metrics["mae"], 4))
        print(
            "Elo improvement:",
            round(improvement, 4)
        )
        print(
            "Ridge RMSE:",
            round(ridge_metrics["rmse"], 4)
        )
        print(
            "Elo RMSE:",
            round(elo_metrics["rmse"], 4)
        )
        print(
            "Ridge winner accuracy:",
            ridge_metrics["winner_accuracy_excluding_ties"]
        )
        print(
            "Elo winner accuracy:",
            elo_metrics["winner_accuracy_excluding_ties"]
        )

    print("\nRidge issuance certified: NO")
    print("Historical Elo PIT certified: NO")
    print("Frozen blend applied: NO")
    print("Production modified: NO")
    print("Output:", destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
