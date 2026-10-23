"""Immutable research acquisition receipts for successful HTTP responses."""

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_time(value):
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid acquisition timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("Acquisition timestamps must be timezone-aware")
    return result.tz_convert("UTC")


def fingerprint(payload):
    return hashlib.sha256(payload).hexdigest()


def capture_response(
    archive_root,
    *,
    endpoint,
    parameters,
    request_started_utc,
    request_completed_utc,
    status_code,
    response_bytes,
):
    """Archive original HTTP response bytes without modifying source caches."""
    if not isinstance(response_bytes, bytes) or not response_bytes:
        raise ValueError("Nonempty HTTP response bytes required")
    if not isinstance(status_code, int) or not 200 <= status_code < 300:
        raise ValueError("Only successful HTTP responses may be captured")
    if not isinstance(endpoint, str) or not endpoint.startswith("/"):
        raise ValueError("Expected relative CFBD endpoint")
    if not isinstance(parameters, dict):
        raise ValueError("Parameters must be a dictionary")

    forbidden = ("key", "token", "auth", "secret", "password")
    if any(
        any(word in str(key).lower() for word in forbidden)
        for key in parameters
    ):
        raise ValueError("Potential secret in request parameters")

    for value in parameters.values():
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("Only scalar request parameters are permitted")

    started = parse_time(request_started_utc)
    completed = parse_time(request_completed_utc)
    if completed < started:
        raise ValueError("Request completed before it started")

    try:
        parsed = json.loads(response_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON response") from exc
    if not isinstance(parsed, (list, dict)):
        raise ValueError("Expected JSON object or array")

    safe_endpoint = re.sub(r"[^a-z0-9_-]+", "_", endpoint.lower()).strip("_")
    if not safe_endpoint:
        raise ValueError("Invalid endpoint")

    metadata = {
        "schema": "cfb_v58_http_acquisition_receipt_v1",
        "source": "CFBD",
        "capture_type": "http_response_bytes",
        "endpoint": endpoint,
        "parameters": parameters,
        "request_started_utc": started.isoformat(),
        "request_completed_utc": completed.isoformat(),
        "status_code": status_code,
        "response_filename": "response.json",
        "response_bytes": len(response_bytes),
        "response_sha256": fingerprint(response_bytes),
        "historical_point_in_time_certified": False,
        "provider_publication_times_verified": False,
        "production_forecast_authorized": False,
    }

    root = Path(archive_root)
    root.mkdir(parents=True, exist_ok=True)

    unique = completed.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex
    destination = root / safe_endpoint / unique
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending")

    if destination.exists() or pending.exists():
        raise FileExistsError("Acquisition archive collision")

    try:
        pending.mkdir()
        (pending / "response.json").write_bytes(response_bytes)
        (pending / "receipt.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if fingerprint((pending / "response.json").read_bytes()) != metadata["response_sha256"]:
            raise RuntimeError("Response read-back fingerprint mismatch")
        pending.rename(destination)
    except Exception:
        if pending.exists():
            shutil.rmtree(pending)
        raise

    return destination


def verify_receipt(directory):
    directory = Path(directory)
    receipt = json.loads(
        (directory / "receipt.json").read_text(encoding="utf-8")
    )
    if receipt.get("schema") != "cfb_v58_http_acquisition_receipt_v1":
        raise ValueError("Unrecognized receipt schema")
    payload = (directory / "response.json").read_bytes()
    if fingerprint(payload) != receipt.get("response_sha256"):
        raise ValueError("Acquisition response fingerprint mismatch")
    if len(payload) != receipt.get("response_bytes"):
        raise ValueError("Acquisition response length mismatch")
    started = parse_time(receipt["request_started_utc"])
    completed = parse_time(receipt["request_completed_utc"])
    if completed < started:
        raise ValueError("Invalid acquisition chronology")
    return receipt
