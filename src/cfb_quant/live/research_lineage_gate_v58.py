"""Read-only, research-only assessment of 2026 input lineage."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CRITICAL_INPUTS = {
    "schedule": "data/raw/cfbd/games/games_2026.json",
    "history": "data/processed/team_games_2026.parquet",
    "advanced": "data/raw/cfbd/advanced_game_v4/advanced_2026.json",
}


def timestamp(value, name):
    try:
        result = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc

    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(
            f"{name} must be a valid timezone-aware timestamp"
        )

    return result.tz_convert("UTC")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(workspace, forecast_cutoff_utc=None, feature_audit=None):
    workspace = Path(workspace)
    manifest_path = workspace / "research_input_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(manifest, dict):
        raise ValueError("Research manifest must be an object")

    expected = manifest.get("input_sha256")
    if not isinstance(expected, dict):
        raise ValueError("Missing input fingerprint mapping")

    issues = []
    observations = []
    inputs = {}

    for name, relative in CRITICAL_INPUTS.items():
        path = workspace / relative

        # Research manifest keys use Windows-style separators.
        normalized = {
            str(key).replace("\\", "/"): value
            for key, value in expected.items()
        }
        claimed = normalized.get(relative)

        if not isinstance(claimed, str) or len(claimed) != 64:
            issues.append(f"{name}: missing/invalid recorded SHA-256")
            inputs[name] = {"fingerprint_matches": False}
            continue

        if not path.is_file():
            issues.append(f"{name}: source file missing")
            inputs[name] = {"fingerprint_matches": False}
            continue

        actual = sha256(path)
        matches = actual.lower() == claimed.lower()

        if not matches:
            issues.append(f"{name}: fingerprint mismatch")

        inputs[name] = {
            "sha256": actual,
            "fingerprint_matches": matches,
        }

    research_asof = timestamp(
        manifest.get("research_asof_utc"), "research_asof_utc"
    )
    created_at = timestamp(
        manifest.get("created_at_utc"), "created_at_utc"
    )

    if created_at < research_asof:
        issues.append(
            "Research manifest creation precedes its recorded asof"
        )

    cutoff = (
        timestamp(forecast_cutoff_utc, "forecast_cutoff_utc")
        if forecast_cutoff_utc is not None
        else research_asof
    )

    if research_asof > cutoff:
        issues.append("Research asof is after forecast cutoff")

    if created_at > cutoff:
        observations.append(
            "Research manifest created after forecast cutoff"
        )

    feature_created = None

    if feature_audit is not None:
        audit = json.loads(
            Path(feature_audit).read_text(encoding="utf-8-sig")
        )
        feature_created_ts = timestamp(
            audit.get("created_at_utc"),
            "feature_audit.created_at_utc",
        )
        feature_created = feature_created_ts.isoformat()

        if feature_created_ts > cutoff:
            observations.append(
                "Feature snapshot created after forecast cutoff"
            )

        audit_asof = timestamp(
            audit.get("asof_utc"), "feature_audit.asof_utc"
        )
        if audit_asof != research_asof:
            issues.append("Feature audit/research asof mismatch")

        for name, key in (
            ("schedule", "schedule_sha256"),
            ("advanced", "advanced_2026_sha256"),
        ):
            recorded = audit.get(key)
            if recorded != inputs.get(name, {}).get("sha256"):
                issues.append(f"Feature audit {name} SHA mismatch")

    observations.append(
        "Original schedule/advanced acquisition times and "
        "history derivation time are not certified by this manifest"
    )

    return {
        "schema": "cfb_v58_research_lineage_gate_v1",
        "status": "INVALID" if issues else "UNVERIFIED",
        "research_only": True,
        "pregame_certified": False,
        "production_forecast_authorized": False,
        "research_asof_utc": research_asof.isoformat(),
        "forecast_cutoff_utc": cutoff.isoformat(),
        "manifest_created_utc": created_at.isoformat(),
        "feature_created_utc": feature_created,
        "inputs": inputs,
        "issues": issues,
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--forecast-cutoff-utc")
    parser.add_argument("--feature-audit")
    args = parser.parse_args()

    result = assess(
        workspace=args.workspace,
        forecast_cutoff_utc=args.forecast_cutoff_utc,
        feature_audit=args.feature_audit,
    )

    print(json.dumps(result, indent=2))
    if result["status"] == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
