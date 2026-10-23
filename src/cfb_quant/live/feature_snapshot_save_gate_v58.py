"""Fail-closed timing guard for research feature snapshot saves."""

import pandas as pd


def utc(value, label):
    try:
        value = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {label}") from exc

    if pd.isna(value) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")

    return value.tz_convert("UTC")


def assess(metadata, frame, save_time, *, minimum_lead_seconds=60):
    if not isinstance(metadata, dict):
        raise ValueError("Missing feature metadata")

    if frame is None or frame.empty:
        raise ValueError("Cannot save empty feature snapshot")

    if "kickoff_utc" not in frame or "game_id" not in frame:
        raise ValueError("Missing game or kickoff fields")

    if (
        isinstance(minimum_lead_seconds, bool)
        or not isinstance(minimum_lead_seconds, int)
        or minimum_lead_seconds < 0
    ):
        raise ValueError("Invalid minimum lead time")

    asof = utc(metadata.get("asof_utc"), "asof")
    now = utc(save_time, "actual save time")

    kickoffs = pd.to_datetime(
        frame["kickoff_utc"],
        utc=True,
        errors="raise",
    )

    if kickoffs.isna().any():
        raise ValueError("Null kickoff timestamps")

    earliest = kickoffs.min()
    seconds_remaining = (earliest - now).total_seconds()

    reasons = []

    if asof > now:
        reasons.append("ASOF_AFTER_ACTUAL_SAVE_TIME")

    if asof >= earliest:
        reasons.append("ASOF_NOT_BEFORE_KICKOFF")

    if now >= earliest:
        reasons.append("ACTUAL_SAVE_AT_OR_AFTER_KICKOFF")
    elif seconds_remaining < minimum_lead_seconds:
        reasons.append("INSUFFICIENT_PREKICKOFF_SAVE_BUFFER")

    if metadata.get("no_current_game_outcomes") is not True:
        reasons.append("OUTCOME_ABSENCE_NOT_CONFIRMED")

    return {
        "schema": "cfb_v58_feature_snapshot_save_gate_v1",
        "status": "REJECT" if reasons else "PASS",
        "reasons": reasons,
        "actual_save_check_utc": now.isoformat(),
        "declared_asof_utc": asof.isoformat(),
        "earliest_kickoff_utc": earliest.isoformat(),
        "lead_seconds": seconds_remaining,
        "minimum_lead_seconds": minimum_lead_seconds,
        "source_availability_certified": False,
        "historical_point_in_time_certified": False,
        "production_forecast_authorized": False,
        "research_only": True,
    }
