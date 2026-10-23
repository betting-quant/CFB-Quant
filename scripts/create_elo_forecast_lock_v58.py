"""Create a sealed, research-only Elo forecast lock."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cfb_quant.spread_total.elo_incremental_v58 import (
    merge_fbs_captures,
)
from cfb_quant.spread_total.elo_forecast_lock_v58 import (
    prepare_lock,
    canonical_bytes,
    content_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "reports/v58_development/elo_live_captures_v58/"
    "20261023T173047828175Z"
)
DEST_BASE = ROOT / (
    "reports/v58_development/elo_forecast_locks_v58"
)


def main():
    raw = (SOURCE / "games_2026.raw.json").read_bytes()
    manifest = json.loads(
        (SOURCE / "manifest.json").read_text(encoding="utf-8")
    )

    merged = merge_fbs_captures([(raw, manifest)])

    # Lock time is explicitly distinct from the older API capture time.
    lock_time = datetime.now(timezone.utc)
    payload = prepare_lock(
        merged,
        lock_time,
        horizon_hours=72,
        min_lead_minutes=10,
    )

    if payload["forecast_count"] == 0:
        raise RuntimeError("No eligible future games to lock")

    # Avoid pretending a just-started game was predicted in advance.
    current = datetime.now(timezone.utc)
    threshold = current + timedelta(minutes=5)

    if any(
        datetime.fromisoformat(row["kickoff_utc"]) <= threshold
        for row in payload["forecasts"]
    ):
        raise RuntimeError("Forecast lock timing margin violated")

    forecast_bytes = canonical_bytes(payload)
    digest = content_sha256(forecast_bytes)

    lock_manifest = {
        "lock_created_at_utc": payload["locked_at_utc"],
        "source_capture_observed_at_utc": (
            payload["source_capture_cutoff_utc"]
        ),
        "source_payload_sha256": manifest["payload_sha256"],
        "forecast_payload_sha256": digest,
        "forecast_count": payload["forecast_count"],
        "replay_scope": "FBS_vs_FBS",
        "experimental": True,
        "production_approved": False,
        "historical_pit_certified": False,
        "market_inputs_used": False,
    }

    stamp = lock_time.strftime("%Y%m%dT%H%M%S%fZ")
    destination = DEST_BASE / stamp
    destination.mkdir(parents=True, exist_ok=False)

    forecasts_path = destination / "locked_forecasts.json"
    manifest_path = destination / "lock_manifest.json"

    with forecasts_path.open("xb") as f:
        f.write(forecast_bytes)

    with manifest_path.open("xb") as f:
        f.write(canonical_bytes(lock_manifest))

    assert content_sha256(
        forecasts_path.read_bytes()
    ) == digest

    print("=" * 72)
    print("STEP 36 - EXPERIMENTAL ELO FORECAST LOCK")
    print("=" * 72)
    print("Lock directory:", destination.relative_to(ROOT))
    print("Locked at:", payload["locked_at_utc"])
    print("Capture cutoff:", payload["source_capture_cutoff_utc"])
    print("Eligible FBS games:", payload["eligible_fbs_games"])
    print("Prior rating updates:", payload["rating_updates_applied"])
    print("Forecasts locked:", payload["forecast_count"])
    print("Forecast SHA256:", digest)

    print("\nLOCKED GAMES")
    for row in payload["forecasts"]:
        print(
            row["kickoff_utc"],
            "|", row["away"], "@", row["home"],
            "| Home Elo:", round(row["home_elo"], 2),
            "| Away Elo:", round(row["away_elo"], 2),
            "| Home win %:",
            round(row["home_win_probability"] * 100, 1),
        )

    print("\nHistorical PIT certification: NO")
    print("Model configuration: EXPERIMENTAL")
    print("Production model changed: NO")
    print("STEP 36 COMPLETE")


if __name__ == "__main__":
    main()
