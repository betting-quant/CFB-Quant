"""Conservative chronology gate for archived research forecasts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from cfb_quant.live.acquisition_receipt_v58 import parse_time


def check(bridge_audit, forecast_audit, snapshot):
    bridge = json.loads(Path(bridge_audit).read_text(encoding="utf-8"))
    forecast = json.loads(Path(forecast_audit).read_text(encoding="utf-8"))

    frame = pd.read_parquet(snapshot)
    if frame.empty or "kickoff_utc" not in frame:
        raise ValueError("Missing pregame kickoff evidence")

    kickoffs = pd.to_datetime(
        frame["kickoff_utc"], utc=True, errors="raise"
    )

    if kickoffs.isna().any():
        raise ValueError("Null kickoff timestamp")

    if "game_id" not in frame:
        raise ValueError("Missing game ID")

    asof = parse_time(bridge["asof_utc"])
    created = parse_time(bridge["created_at_utc"])
    generated = parse_time(forecast["generated_at_utc"])
    earliest = kickoffs.min().to_pydatetime()

    metadata = forecast.get("snapshot_metadata", {})
    validation = forecast.get("snapshot_validation", {})

    if metadata.get("asof_utc") != bridge.get("asof_utc"):
        raise ValueError("Bridge/forecast as-of mismatch")

    if validation.get("snapshot_asof_utc") != bridge.get("asof_utc"):
        raise ValueError("Snapshot/bridge as-of mismatch")

    if generated < created:
        raise ValueError("Forecast predates recorded feature creation")

    if asof > created:
        raise ValueError("Declared as-of exceeds snapshot creation time")

    reasons = []
    if asof >= earliest:
        reasons.append("DECLARED_ASOF_NOT_BEFORE_KICKOFF")
    if created >= earliest:
        reasons.append("SNAPSHOT_CREATED_AFTER_KICKOFF")
    if generated >= earliest:
        reasons.append("FORECAST_GENERATED_AFTER_KICKOFF")

    if reasons:
        chronology = "REJECT"
    else:
        chronology = "PASS"

    return {
        "schema": "cfb_v58_pregame_chronology_gate_v1",
        "chronology_status": chronology,
        "lineage_status": "UNVERIFIED",
        "research_only": True,
        "game_ids": sorted(set(frame["game_id"].astype(str))),
        "earliest_kickoff_utc": earliest.isoformat(),
        "declared_asof_utc": asof.isoformat(),
        "snapshot_created_utc": created.isoformat(),
        "forecast_generated_utc": generated.isoformat(),
        "reasons": reasons,
        "source_availability_certified": False,
        "historical_point_in_time_certified": False,
        "production_forecast_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-audit", required=True)
    parser.add_argument("--forecast-audit", required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    print(json.dumps(check(
        args.bridge_audit,
        args.forecast_audit,
        args.snapshot,
    ), indent=2))


if __name__ == "__main__":
    main()
