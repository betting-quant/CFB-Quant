"""Fail-closed validation for local v5.8 research source captures."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = {
    "schedule_2026",
    "advanced_2026",
    "team_games_2026",
}

EXPECTED_NAMES = {
    "schedule_2026": "games_2026.json",
    "advanced_2026": "advanced_2026.json",
    "team_games_2026": "team_games_2026.parquet",
}


def parse_utc(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(capture_dir, asof_utc=None):
    folder = Path(capture_dir).resolve(strict=True)
    manifest_path = folder / "capture_manifest.json"

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing or symlinked capture manifest")

    document = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if document.get("schema") != "cfb_v58_source_capture_v1":
        raise ValueError("Unsupported capture schema")

    if document.get("capture_kind") != "local_file_copy":
        raise ValueError("Unsupported capture type")

    for key, expected in (
        ("research_only", True),
        ("historical_pit_certified", False),
        ("provider_publication_times_verified", False),
        ("independent_external_timestamp_verified", False),
        ("safe_for_forecast_asof_before_capture", False),
        ("production_modified", False),
    ):
        if document.get(key) is not expected:
            raise ValueError(f"Invalid provenance claim: {key}")

    completed = parse_utc(document["capture_completed_at_utc"])
    if completed > datetime.now(timezone.utc):
        raise ValueError("Capture timestamp is in the future")

    if asof_utc is not None:
        proposed = parse_utc(asof_utc)
        if proposed < completed:
            raise ValueError(
                "Forecast as-of precedes source capture"
            )
        if proposed > datetime.now(timezone.utc):
            raise ValueError("Forecast as-of is in the future")

    sources = document.get("sources")
    if not isinstance(sources, dict) or set(sources) != REQUIRED:
        raise ValueError("Unexpected source inventory")

    verified = {}

    for label in sorted(REQUIRED):
        entry = sources[label]
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed source entry: {label}")

        filename = entry.get("captured_filename")
        if filename != EXPECTED_NAMES[label]:
            raise ValueError(f"Invalid source filename: {label}")

        path = folder / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/symlinked source: {label}")

        if path.resolve() .parent != folder:
            raise ValueError(f"Source escapes capture: {label}")

        captured = parse_utc(entry["captured_at_utc"])
        if captured > completed:
            raise ValueError(
                f"Source capture time exceeds completion: {label}"
            )

        if entry.get("provider_publication_time_verified") is not False:
            raise ValueError(
                f"Unsupported provider timestamp claim: {label}"
            )
        if entry.get("pre_capture_availability_certified") is not False:
            raise ValueError(
                f"Unsupported availability claim: {label}"
            )

        expected_size = entry.get("bytes")
        if type(expected_size) is not int or expected_size <= 0:
            raise ValueError(f"Invalid source size: {label}")

        if path.stat().st_size != expected_size:
            raise ValueError(f"Source size mismatch: {label}")

        expected_hash = entry.get("sha256")
        actual_hash = sha256(path)

        if actual_hash != expected_hash:
            raise ValueError(f"Source SHA256 mismatch: {label}")

        verified[label] = {
            "path": str(path),
            "sha256": actual_hash,
            "bytes": expected_size,
        }

    return {
        "capture_dir": str(folder),
        "capture_completed_at_utc": completed.isoformat(),
        "source_count": len(verified),
        "sources": verified,
        "verified": True,
        "historical_pit_certified": False,
        "research_only": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--asof-utc")
    args = parser.parse_args()

    result = verify(args.capture, args.asof_utc)

    print("CFB v5.8 SOURCE CAPTURE: VERIFIED")
    print("Capture:", result["capture_dir"])
    print("Source count:", result["source_count"])
    print("Captured:", result["capture_completed_at_utc"])
    print("Historical PIT certified: NO")
    print("Production modified: NO")


if __name__ == "__main__":
    main()
