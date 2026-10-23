"""Strict research forecast publication lineage validation."""

from datetime import datetime, timezone
from pathlib import Path

from cfb_quant.live.forecast_input_freeze_v58 import (
    inside,
    verify_unchanged,
)


def validate_lineage(root, audit):
    """Reverify inputs and bind fingerprints to this forecast's snapshot."""
    root = Path(root).resolve()

    if not isinstance(audit, dict):
        raise ValueError("Missing forecast audit")

    fingerprints = audit.get("execution_input_fingerprints")
    execution_check = audit.get("execution_input_verification")

    if not isinstance(fingerprints, dict):
        raise ValueError("Missing execution input fingerprints")

    if not isinstance(execution_check, dict):
        raise ValueError("Missing execution input verification")

    if (
        fingerprints.get("schema")
        != "cfb_v58_execution_input_fingerprints_v1"
        or fingerprints.get("dependency_count") != 10
        or execution_check.get("status") != "UNCHANGED"
        or execution_check.get("verified_dependency_count") != 10
    ):
        raise ValueError("Invalid execution lineage certification")

    snapshot_validation = audit.get("snapshot_validation")
    if not isinstance(snapshot_validation, dict):
        raise ValueError("Missing snapshot validation")

    folder = snapshot_validation.get("snapshot_folder")
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError("Missing snapshot folder binding")

    snapshot = Path(folder)
    if not snapshot.is_absolute():
        snapshot = root / snapshot

    snapshot, _ = inside(root, snapshot)

    dependencies = fingerprints.get("dependencies", {})

    for key, filename in (
        ("snapshot_parquet", "pregame_features.parquet"),
        ("snapshot_audit", "audit.json"),
    ):
        expected_path, relative = inside(root, snapshot / filename)
        record = dependencies.get(key)

        if not isinstance(record, dict):
            raise ValueError(f"Missing {key} fingerprint")

        if record.get("path") != relative:
            raise ValueError(f"Snapshot binding mismatch: {key}")

    expected_snapshot_hash = snapshot_validation.get("snapshot_sha256")

    if (
        not isinstance(expected_snapshot_hash, str)
        or dependencies["snapshot_parquet"].get("sha256")
        != expected_snapshot_hash
    ):
        raise ValueError("Snapshot hash binding mismatch")

    check = verify_unchanged(root, fingerprints)

    if check.get("status") != "UNCHANGED":
        raise ValueError("Publication dependency verification failed")

    return {
        "schema": "cfb_v58_forecast_publication_lineage_v1",
        "research_only": True,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_folder": snapshot.relative_to(root).as_posix(),
        "snapshot_sha256": expected_snapshot_hash,
        "dependency_count": 10,
        "dependencies": fingerprints["dependencies"],
        "execution_check": execution_check,
        "publication_check": check,
        "source_availability_certified": False,
        "historical_point_in_time_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
    }
