"""Read-only verification of portable CFBD schedule archives."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from cfb_quant.live.acquisition_receipt_v58 import (
    parse_time,
    verify_receipt,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(archive_dir):
    archive = Path(archive_dir).resolve(strict=True)
    if not archive.is_dir():
        raise ValueError("Schedule archive must be a directory")

    # A schedule archive is a dated child of the output root.
    output_root = archive.parent

    audit_path = archive / "audit.json"
    snapshot_path = archive / "cfbd_games.json"

    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if not isinstance(audit, dict):
        raise ValueError("Invalid schedule audit")

    if sha256(snapshot_path) != audit.get("snapshot_sha256"):
        raise ValueError("Schedule snapshot fingerprint mismatch")

    acquisition = audit.get("http_acquisition")
    if not isinstance(acquisition, dict):
        raise ValueError("Missing HTTP acquisition linkage")

    relative = acquisition.get("receipt_relative_to_output")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing portable receipt path")

    # Reject traversal, absolute paths and Windows drive paths on
    # all operating systems, not just the current host OS.
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)

    if (
        "\\" in relative
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in ("..", ".", "") for part in relative.split("/"))
    ):
        raise ValueError("Unsafe portable receipt path")

    receipt_dir = (output_root / Path(relative)).resolve()

    if not receipt_dir.is_relative_to(output_root.resolve()):
        raise ValueError("Receipt path escapes archive output root")

    # Even when symlinks are present, the resolved evidence must
    # remain within the portable output root.
    receipt = verify_receipt(receipt_dir)

    if receipt.get("endpoint") != "/games":
        raise ValueError("Unexpected HTTP endpoint")

    if receipt.get("source") != "CFBD":
        raise ValueError("Unexpected acquisition source")

    expected_params = audit.get("parameters")
    if receipt.get("parameters") != expected_params:
        raise ValueError("Request parameter mismatch")

    if receipt.get("response_sha256") != acquisition.get("response_sha256"):
        raise ValueError("HTTP response SHA linkage mismatch")

    for name in ("request_started_utc", "request_completed_utc"):
        if parse_time(receipt[name]) != parse_time(acquisition[name]):
            raise ValueError(f"Acquisition timestamp mismatch: {name}")

    if parse_time(receipt["request_completed_utc"]) > parse_time(
        audit["request_completed_utc"]
    ):
        raise ValueError("Acquisition completed after schedule audit timestamp")

    original = json.loads(
        (receipt_dir / "response.json").read_bytes()
    )
    snapshot = json.loads(snapshot_path.read_bytes())

    if not isinstance(original, list) or not isinstance(snapshot, list):
        raise ValueError("Schedule payload must be a JSON list")

    if original != snapshot:
        raise ValueError("HTTP response and schedule snapshot differ")

    if len(snapshot) != audit.get("fresh_games"):
        raise ValueError("Schedule game count mismatch")

    if (
        acquisition.get("production_forecast_authorized") is not False
        or acquisition.get("historical_point_in_time_certified") is not False
        or receipt.get("production_forecast_authorized") is not False
    ):
        raise ValueError("Unexpected acquisition authorization claims")

    return {
        "schema": "cfb_v58_portable_schedule_verification_v1",
        "status": "PASS",
        "research_only": True,
        "archive": str(archive),
        "schedule_games": len(snapshot),
        "schedule_sha256": sha256(snapshot_path),
        "response_sha256": receipt["response_sha256"],
        "receipt_relative_to_output": relative,
        "historical_point_in_time_certified": False,
        "production_forecast_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.archive), indent=2))


if __name__ == "__main__":
    main()
