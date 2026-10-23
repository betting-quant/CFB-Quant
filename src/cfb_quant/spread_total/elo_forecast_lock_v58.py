"""Immutable prospective Elo forecast-lock preparation.

Only games strictly after the lock time and within the chosen
horizon are included. No market inputs or historical backdating.
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

from .elo_prospective_gate_v58 import prospective_elo_candidates


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def prepare_lock(
    merged,
    locked_at_utc,
    horizon_hours=72,
    min_lead_minutes=10,
):
    if (
        not isinstance(locked_at_utc, datetime)
        or locked_at_utc.tzinfo is None
        or locked_at_utc.utcoffset() is None
    ):
        raise ValueError("Lock timestamp must be timezone-aware")

    if horizon_hours <= 0 or min_lead_minutes < 0:
        raise ValueError("Invalid forecast horizon or lead time")

    locked = locked_at_utc.astimezone(timezone.utc)
    gate = prospective_elo_candidates(merged)
    cutoff = gate["capture_cutoff_utc"]

    if locked < cutoff:
        raise ValueError("Cannot lock before the source observation")

    minimum_kickoff = locked + timedelta(minutes=min_lead_minutes)
    maximum_kickoff = locked + timedelta(hours=horizon_hours)

    rows = []

    for snap in gate["candidates"]:
        if not (minimum_kickoff < snap.kickoff_utc <= maximum_kickoff):
            continue

        rows.append({
            "game_id": snap.game_id,
            "kickoff_utc": snap.kickoff_utc.isoformat(),
            "home": snap.home,
            "away": snap.away,
            "home_elo": snap.home_elo,
            "away_elo": snap.away_elo,
            "home_win_probability": snap.expected_home_win,
            "away_win_probability": 1.0 - snap.expected_home_win,
        })

    rows.sort(key=lambda r: (r["kickoff_utc"], r["game_id"]))

    if len({r["game_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate games in forecast lock")

    return {
        "locked_at_utc": locked.isoformat(),
        "source_capture_cutoff_utc": cutoff.isoformat(),
        "forecast_horizon_hours": horizon_hours,
        "minimum_lead_minutes": min_lead_minutes,
        "eligible_fbs_games": gate["eligible_games"],
        "rating_updates_applied": gate["updates_applied"],
        "forecast_count": len(rows),
        "forecasts": rows,
        "model": {
            "name": "experimental_elo_v58",
            "initial_rating": 1500.0,
            "k_factor": 24.0,
            "home_advantage": 0.0,
            "elo_scale": 400.0,
            "eligibility": "FBS_vs_FBS_only",
            "configuration_status": "experimental_not_optimized",
        },
        "historical_pit_certified": False,
        "production_approved": False,
    }


def content_sha256(data):
    return hashlib.sha256(data).hexdigest()
