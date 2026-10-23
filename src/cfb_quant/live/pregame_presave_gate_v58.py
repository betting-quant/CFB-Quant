"""Research-only, fail-closed pre-save forecast chronology assessment."""

import pandas as pd


def timestamp(value, label):
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {label}") from exc

    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{label} must be timezone aware")

    return result.tz_convert("UTC")


def assess(metadata, frame, generated_at, *, allow_started=False):
    if not isinstance(metadata, dict):
        raise ValueError("Missing feature metadata")
    if frame.empty or "kickoff_utc" not in frame:
        raise ValueError("Missing snapshot kickoff evidence")
    if not isinstance(allow_started, bool):
        raise ValueError("allow_started must be boolean")

    asof = timestamp(metadata.get("asof_utc"), "asof")
    created = timestamp(
        metadata.get("created_at_utc"), "feature creation"
    )
    generated = timestamp(generated_at, "forecast generation")

    kicks = pd.to_datetime(
        frame["kickoff_utc"], utc=True, errors="raise"
    )
    if kicks.isna().any():
        raise ValueError("Null kickoff")

    earliest = kicks.min()
    reasons = []

    if asof > created:
        reasons.append("ASOF_AFTER_FEATURE_CREATION")
    if created > generated:
        reasons.append("FEATURE_CREATED_AFTER_FORECAST_GENERATION")
    if asof >= earliest:
        reasons.append("ASOF_NOT_BEFORE_KICKOFF")

    if not allow_started:
        if created >= earliest:
            reasons.append("FEATURE_CREATED_AFTER_KICKOFF")
        if generated >= earliest:
            reasons.append("FORECAST_GENERATED_AFTER_KICKOFF")

    status = "REJECT" if reasons else "PASS"

    # An archived run is never classified as prospective, even
    # if the supplied timestamps happen to precede kickoff.
    classification = (
        "REJECTED"
        if status == "REJECT"
        else (
            "RETROSPECTIVE_RESEARCH"
            if allow_started
            else "CHRONOLOGY_VALID_RESEARCH"
        )
    )

    return {
        "schema": "cfb_v58_presave_chronology_v1",
        "status": status,
        "classification": classification,
        "archive_override_requested": allow_started,
        "earliest_kickoff_utc": earliest.isoformat(),
        "declared_asof_utc": asof.isoformat(),
        "feature_created_utc": created.isoformat(),
        "forecast_generated_utc": generated.isoformat(),
        "reasons": reasons,
        "source_lineage_status": "UNVERIFIED",
        "source_availability_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
        "research_only": True,
    }
