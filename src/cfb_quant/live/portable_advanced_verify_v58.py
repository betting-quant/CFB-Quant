"""Read-only, portable verification of CFBD advanced caches."""

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath

from cfb_quant.live.acquisition_receipt_v58 import (
    parse_time,
    verify_receipt,
)
from cfb_quant.spread_total.advanced_v4 import _cache_path


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def verify_advanced(root, year):
    root = Path(root).resolve(strict=True)
    cache = _cache_path(root, int(year))
    metadata_path = cache.with_suffix(".meta.json")

    cache_bytes = cache.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if digest(cache_bytes) != metadata.get("sha256"):
        raise ValueError("Advanced cache fingerprint mismatch")

    link = metadata.get("http_acquisition")
    if not isinstance(link, dict):
        raise ValueError("Missing acquisition receipt linkage")

    relative = link.get("receipt_relative_to_root")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing portable receipt path")

    if (
        "\\" in relative
        or relative.startswith("/")
        or PureWindowsPath(relative).drive
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError("Unsafe portable receipt path")

    directory = (root / Path(relative)).resolve(strict=True)
    if not directory.is_relative_to(root):
        raise ValueError("Receipt escapes workspace")

    receipt = verify_receipt(directory)

    expected_params = {
        "year": int(year),
        "seasonType": "both",
        "excludeGarbageTime": "false",
    }
    if (
        receipt.get("source") != "CFBD"
        or receipt.get("endpoint") != "/stats/game/advanced"
        or receipt.get("parameters") != expected_params
        or metadata.get("parameters") != expected_params
        or receipt.get("status_code") != 200
    ):
        raise ValueError("Advanced HTTP source mismatch")

    if receipt["response_sha256"] != link.get("response_sha256"):
        raise ValueError("Acquisition response hash mismatch")

    for field in ("request_started_utc", "request_completed_utc"):
        if parse_time(receipt[field]) != parse_time(link[field]):
            raise ValueError("Acquisition timestamp mismatch")

    if parse_time(receipt["request_completed_utc"]) > parse_time(
        metadata["downloaded_at_utc"]
    ):
        raise ValueError("Cache timestamp precedes HTTP acquisition")

    original = json.loads((directory / "response.json").read_bytes())
    cached = json.loads(cache_bytes)

    if original != cached:
        raise ValueError("Acquisition response and advanced cache differ")
    if not isinstance(cached, list) or len(cached) != metadata.get("record_count"):
        raise ValueError("Advanced cache record count mismatch")

    if (
        link.get("historical_point_in_time_certified") is not False
        or link.get("production_forecast_authorized") is not False
        or receipt.get("production_forecast_authorized") is not False
        or receipt.get("historical_point_in_time_certified") is not False
    ):
        raise ValueError("Invalid research authorization claims")

    return {
        "status": "PASS",
        "research_only": True,
        "year": int(year),
        "records": len(cached),
        "cache_sha256": digest(cache_bytes),
        "response_sha256": receipt["response_sha256"],
        "receipt_relative_to_root": relative,
        "historical_point_in_time_certified": False,
        "production_forecast_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--year", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(verify_advanced(args.root, args.year), indent=2))


if __name__ == "__main__":
    main()
