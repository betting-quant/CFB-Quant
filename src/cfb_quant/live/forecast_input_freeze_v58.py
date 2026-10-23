"""Research forecast input fingerprints; not PIT certification."""

import hashlib
import json
from pathlib import Path


DEPENDENCIES = {
    "v3_training":
        "data/processed/cfb_matchup_features_kickoff_v3.parquet",
    "v45_training":
        "data/processed/cfb_matchup_features_opponent_v45.parquet",
    "v3_feature_manifest":
        "reports/cfb_matchup_feature_manifest_kickoff_v3.json",
    "v45_feature_manifest":
        "reports/cfb_matchup_feature_manifest_opponent_v45.json",
    "residual_pairing":
        "reports/spread_total_opponent_v45/paired_game_errors.csv",
    "residual_universe":
        "reports/spread_total_fbs_calibration_v52/universe_audit.csv",
    "schedule_2026":
        "data/raw/cfbd/games/games_2026.json",
    "advanced_2026":
        "data/raw/cfbd/advanced_game_v4/advanced_2026.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def inside(root, candidate):
    root = Path(root).resolve()
    candidate = Path(candidate).resolve()

    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "Dependency resolves outside repository"
        ) from exc

    return candidate, relative.as_posix()


def take(root, snapshot):
    """Hash each required physical input before model fitting."""

    root = Path(root).resolve()
    snapshot = Path(snapshot)

    if not snapshot.is_absolute():
        snapshot = root / snapshot

    snapshot, _ = inside(root, snapshot)

    mapping = dict(DEPENDENCIES)
    mapping["snapshot_parquet"] = (
        snapshot / "pregame_features.parquet"
    )
    mapping["snapshot_audit"] = snapshot / "audit.json"

    records = {}

    for name, reference in mapping.items():
        candidate = Path(reference)
        if not candidate.is_absolute():
            candidate = root / candidate

        path, relative = inside(root, candidate)

        if not path.is_file():
            raise FileNotFoundError(
                f"Required forecast input missing: {relative}"
            )

        records[name] = {
            "path": relative,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    audit_file = snapshot / "audit.json"
    metadata = json.loads(
        audit_file.read_text(encoding="utf-8-sig")
    )

    expected = {
        "schedule_2026": metadata.get("schedule_sha256"),
        "advanced_2026": metadata.get(
            "advanced_2026_sha256"
        ),
    }

    for name, expected_hash in expected.items():
        if (
            not isinstance(expected_hash, str)
            or records[name]["sha256"] != expected_hash
        ):
            raise ValueError(
                f"Snapshot source identity mismatch: {name}"
            )

    return {
        "schema": "cfb_v58_execution_input_fingerprints_v1",
        "research_only": True,
        "dependencies": records,
        "dependency_count": len(records),
        "snapshot_source_hashes_verified": True,
        "source_availability_certified": False,
        "historical_point_in_time_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
    }


def verify_unchanged(root, original):
    """Detect missing, altered or replaced execution inputs."""

    root = Path(root).resolve()
    records = original["dependencies"]

    if not isinstance(records, dict) or len(records) != 10:
        raise ValueError("Incomplete dependency fingerprint set")

    if set(records) != set(DEPENDENCIES) | {
        "snapshot_parquet", "snapshot_audit"
    }:
        raise ValueError("Unexpected dependency fingerprint set")

    problems = []

    for name, record in records.items():
        path, relative = inside(root, root / record["path"])

        if relative != record["path"]:
            problems.append(f"{name}:PATH_CHANGED")
        elif not path.is_file():
            problems.append(f"{name}:MISSING")
        elif (
            sha256(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            problems.append(f"{name}:CHANGED")

    if problems:
        raise ValueError(
            "Forecast execution inputs changed: "
            + ", ".join(problems)
        )

    return {
        "status": "UNCHANGED",
        "verified_dependency_count": len(records),
        "source_availability_certified": False,
        "historical_point_in_time_certified": False,
    }
