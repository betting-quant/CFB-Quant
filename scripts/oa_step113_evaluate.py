"""Step 113: Evaluate frozen OA hybrid forecasts against completed games.

Default: read-only readiness check.
--score: create a new, versioned result artifact from completed games only.
Never modifies the original forecast, snapshot, or scoring protocol.
"""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FORECAST = ROOT / "reports/oa110c/20261030T205857872958Z"
PROTOCOL = ROOT / "reports/oa112_protocol_v1"
SCHEDULE = ROOT / "data/raw/cfbd/games/games_2026.json"
OUT_ROOT = ROOT / "reports/oa113_scores"

EXPECTED_PROTOCOL_HASH = (
    "0652697fcb33c2c22b4e3ebab320ea8e023c96fa43b11b44c9cc74878924e0f4"
)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def game_id(value):
    return str(int(value))

def score_game(row, home_actual, away_actual, model, columns):
    home_pred = float(getattr(row, columns[0]))
    away_pred = float(getattr(row, columns[1]))

    actual_margin = home_actual - away_actual
    predicted_margin = home_pred - away_pred

    actual_total = home_actual + away_actual
    predicted_total = home_pred + away_pred

    actual_winner = np.sign(actual_margin)
    predicted_winner = np.sign(predicted_margin)

    return {
        "game_id": game_id(row.game_id),
        "away_team": row.away_team,
        "home_team": row.home_team,
        "model": model,
        "actual_away": away_actual,
        "actual_home": home_actual,
        "predicted_away": away_pred,
        "predicted_home": home_pred,
        "actual_margin": actual_margin,
        "predicted_margin": predicted_margin,
        "actual_total": actual_total,
        "predicted_total": predicted_total,
        "spread_ae": abs(predicted_margin - actual_margin),
        "total_ae": abs(predicted_total - actual_total),
        "team_score_ae": (
            abs(home_pred - home_actual)
            + abs(away_pred - away_actual)
        ) / 2,
        "winner_correct": (
            None
            if actual_winner == 0 or predicted_winner == 0
            else bool(actual_winner == predicted_winner)
        ),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score",
        action="store_true",
        help="Publish a new evaluation of currently completed games",
    )
    args = parser.parse_args()

    now = pd.Timestamp(datetime.now(timezone.utc))

    print("\n=== STEP 113: FROZEN EVALUATION ===")
    print("Inspection UTC:", now.isoformat())

    protocol_path = PROTOCOL / "protocol.json"
    protocol_marker = PROTOCOL / "COMPLETE.json"

    forecast_path = FORECAST / "forecast.csv"
    forecast_manifest = FORECAST / "manifest.json"
    forecast_marker = FORECAST / "COMPLETE.json"

    for path in (
        protocol_path, protocol_marker, forecast_path,
        forecast_manifest, forecast_marker, SCHEDULE
    ):
        require(path.is_file(), f"Required file missing: {path}")

    protocol = read_json(protocol_path)
    pm = read_json(protocol_marker)
    fm = read_json(forecast_manifest)
    fc = read_json(forecast_marker)

    require(
        digest(protocol_path) == EXPECTED_PROTOCOL_HASH,
        "Locked protocol hash changed",
    )
    require(
        pm["protocol_sha256"] == digest(protocol_path),
        "Protocol completion marker invalid",
    )
    require(
        protocol["schema"] == "oa_step112_scoring_protocol_v1",
        "Unexpected scoring protocol",
    )
    require(
        protocol["expected_games"] == 10,
        "Locked sample size changed",
    )
    require(
        protocol["forecast_sha256"] == digest(forecast_path),
        "Forecast changed since protocol lock",
    )
    require(
        protocol["forecast_manifest_sha256"] == digest(forecast_manifest),
        "Forecast manifest changed since protocol lock",
    )
    require(
        fc["manifest_sha256"] == digest(forecast_manifest)
        and fc["forecast_sha256"] == digest(forecast_path),
        "Forecast completion hashes invalid",
    )
    require(
        fm["research_only"] and not fm["production_approved"],
        "Research-only contract violated",
    )

    forecasts = pd.read_csv(forecast_path)
    require(len(forecasts) == 10, "Forecast game count changed")

    forecasts["id_key"] = forecasts.game_id.map(game_id)
    require(
        not forecasts.id_key.duplicated().any(),
        "Duplicate forecast game ID",
    )
    require(
        set(forecasts.id_key) == set(protocol["game_ids"]),
        "Game IDs differ from locked protocol",
    )

    raw_schedule = read_json(SCHEDULE)

    require(
        isinstance(raw_schedule, list),
        "Unexpected schedule JSON structure",
    )

    schedule = {}

    for item in raw_schedule:
        if int(item.get("season", -1)) != 2026:
            continue

        key = game_id(item["id"])

        require(key not in schedule, f"Duplicate schedule ID: {key}")
        schedule[key] = item

    print("Locked protocol: VERIFIED")
    print("Frozen forecast: VERIFIED")
    print("Game identities:", len(forecasts))

    statuses = []
    completed_rows = []

    for row in forecasts.itertuples(index=False):
        key = row.id_key

        require(key in schedule, f"Schedule game missing: {key}")
        game = schedule[key]

        require(
            str(game["homeTeam"]) == str(row.home_team)
            and str(game["awayTeam"]) == str(row.away_team),
            f"Home/away team identities differ: {key}",
        )

        kickoff = pd.Timestamp(row.kickoff_utc)
        require(kickoff.tzinfo is not None, "Naive kickoff")
        require(
            pd.Timestamp(fm["inference_completed_utc"]) < kickoff,
            f"Forecast was not completed before kickoff: {key}",
        )

        is_final = game.get("completed") is True

        home_points = game.get("homePoints")
        away_points = game.get("awayPoints")

        if is_final:
            require(
                kickoff <= now,
                f"Completed game has future kickoff: {key}",
            )
            require(
                home_points is not None and away_points is not None,
                f"Final score missing: {key}",
            )

            home_points = float(home_points)
            away_points = float(away_points)

            require(
                np.isfinite([home_points, away_points]).all()
                and home_points >= 0
                and away_points >= 0
                and home_points.is_integer()
                and away_points.is_integer(),
                f"Invalid final score: {key}",
            )

            completed_rows.append(
                (row, int(home_points), int(away_points))
            )
            status = "completed"
        else:
            status = "pending"

        statuses.append({
            "game_id": key,
            "away_team": row.away_team,
            "home_team": row.home_team,
            "kickoff_utc": row.kickoff_utc,
            "status": status,
        })

    status_frame = pd.DataFrame(statuses)
    pending = status_frame[status_frame.status.eq("pending")]

    print("\n=== RESULT AVAILABILITY ===")
    print("Completed:", len(completed_rows))
    print("Pending:", len(pending))

    if len(pending):
        print("\nGames not yet completed in local schedule:")
        print(
            pending[["away_team", "home_team", "kickoff_utc"]]
            .to_string(index=False)
        )

    if not args.score:
        print("\n=== STEP 113A CHECK COMPLETE ===")
        print("Evaluation protocol: VERIFIED")
        print("Forecast unchanged: VERIFIED")
        print("Evaluation outputs written: NO")
        return

    require(
        completed_rows,
        "No completed games available. Scoring not published.",
    )

    # Model columns are obtained from the frozen protocol.
    model_map = protocol["model_predictions"]

    require(
        set(model_map) == {
            "baseline", "original_oa", "feasible_hybrid"
        },
        "Locked model definitions differ",
    )

    rows = []

    for forecast, home, away in completed_rows:
        for model, columns in model_map.items():
            rows.append(
                score_game(forecast, home, away, model, columns)
            )

    scored = pd.DataFrame(rows)

    require(
        len(scored) == len(completed_rows) * 3,
        "Paired scoring rows incomplete",
    )

    require(
        scored.groupby("game_id").model.nunique().eq(3).all(),
        "Models scored on different game universes",
    )

    summary = []

    for model, group in scored.groupby("model"):
        valid_winner = group.winner_correct.dropna()

        summary.append({
            "model": model,
            "games": len(group),
            "spread_mae": float(group.spread_ae.mean()),
            "total_mae": float(group.total_ae.mean()),
            "team_score_mae": float(group.team_score_ae.mean()),
            "winner_accuracy": (
                float(valid_winner.astype(float).mean())
                if len(valid_winner) else None
            ),
            "winner_evaluated_games": len(valid_winner),
        })

    summary_frame = pd.DataFrame(summary)

    # Paired improvements are calculated within each game.
    wide = scored.pivot(
        index="game_id",
        columns="model",
        values=["spread_ae", "total_ae", "team_score_ae"],
    )

    deltas = []

    for challenger, reference in (
        ("original_oa", "baseline"),
        ("feasible_hybrid", "baseline"),
        ("feasible_hybrid", "original_oa"),
    ):
        for metric in ("spread_ae", "total_ae", "team_score_ae"):
            difference = (
                wide[(metric, challenger)]
                - wide[(metric, reference)]
            )

            deltas.append({
                "challenger": challenger,
                "reference": reference,
                "metric": metric,
                "mean_paired_delta": float(difference.mean()),
                "games": len(difference),
            })

    # New output directory every time: no historical overwrite.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dest = OUT_ROOT / now.strftime("%Y%m%dT%H%M%S%fZ")

    require(not dest.exists(), "Evaluation output collision")
    dest.mkdir()

    try:
        scored.to_csv(dest / "per_game.csv", index=False)
        summary_frame.to_csv(dest / "summary.csv", index=False)
        pd.DataFrame(deltas).to_csv(dest / "paired_deltas.csv", index=False)
        status_frame.to_csv(dest / "game_status.csv", index=False)

        output_files = (
            "per_game.csv", "summary.csv",
            "paired_deltas.csv", "game_status.csv"
        )

        report = {
            "schema": "oa_step113_prospective_evaluation_v1",
            "evaluated_at_utc": now.isoformat(),
            "research_only": True,
            "production_approved": False,
            "publication_pit_certified": False,
            "locked_protocol_sha256": digest(protocol_path),
            "locked_forecast_sha256": digest(forecast_path),
            "result_schedule_sha256": digest(SCHEDULE),
            "completed_games": len(completed_rows),
            "pending_games": len(pending),
            "expected_games": 10,
            "output_sha256": {
                name: digest(dest / name)
                for name in output_files
            },
            "warnings": [
                "Historical source publication PIT is uncertified.",
                "Only completed games are included in aggregate MAE.",
                "This is a small exploratory prospective sample.",
                "Result correctness depends on the refreshed schedule source.",
            ],
        }

        manifest_path = dest / "manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        marker_path = dest / "COMPLETE.json"
        require(
            digest(protocol_path) == EXPECTED_PROTOCOL_HASH
            and digest(forecast_path) == protocol["forecast_sha256"],
            "Locked input changed during evaluation",
        )
        require(
            digest(SCHEDULE) == report["result_schedule_sha256"],
            "Results source changed during evaluation",
        )

        marker_path.write_text(
            json.dumps({
                "schema": "oa_step113_completion_v1",
                "manifest_sha256": digest(manifest_path),
                "research_only": True,
            }, indent=2),
            encoding="utf-8",
        )

    except Exception:
        print("EVALUATION INCOMPLETE — DO NOT USE:", dest)
        raise

    print("\n=== STEP 113 SCORING COMPLETE ===")
    print("Evaluation directory:", dest.relative_to(ROOT))
    print(summary_frame.to_string(index=False))
    print("\nPaired differences (negative is better):")
    print(pd.DataFrame(deltas).to_string(index=False))
    print("\nOriginal forecast modified: NO")

if __name__ == "__main__":
    main()
