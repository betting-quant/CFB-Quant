"""Tests for offline, research-only acquisition receipts."""

import json

import pytest

from cfb_quant.live.acquisition_receipt_v58 import (
    capture_response,
    verify_receipt,
)


def make_capture(tmp_path, **overrides):
    arguments = {
        "endpoint": "/games",
        "parameters": {"year": 2026, "classification": "fbs"},
        "request_started_utc": "2026-10-22T15:00:00Z",
        "request_completed_utc": "2026-10-22T15:00:01Z",
        "status_code": 200,
        "response_bytes": b'[{"id":123}]',
    }
    arguments.update(overrides)
    return capture_response(tmp_path, **arguments)


def test_exact_bytes_archived_and_verified(tmp_path):
    directory = make_capture(tmp_path)
    assert (directory / "response.json").read_bytes() == b'[{"id":123}]'
    receipt = verify_receipt(directory)
    assert receipt["status_code"] == 200
    assert receipt["production_forecast_authorized"] is False


def test_tampering_detected(tmp_path):
    directory = make_capture(tmp_path)
    (directory / "response.json").write_bytes(b'[{"id":456}]')
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_receipt(directory)


def test_invalid_chronology_rejected(tmp_path):
    with pytest.raises(ValueError, match="before it started"):
        make_capture(
            tmp_path,
            request_completed_utc="2026-10-22T14:59:59Z",
        )


def test_failed_http_response_not_archived(tmp_path):
    with pytest.raises(ValueError, match="successful"):
        make_capture(tmp_path, status_code=500)
    assert not list(tmp_path.rglob("receipt.json"))


def test_secret_parameter_rejected(tmp_path):
    with pytest.raises(ValueError, match="secret"):
        make_capture(
            tmp_path,
            parameters={"year": 2026, "api_key": "redacted"},
        )


def test_distinct_immutable_archive_directories(tmp_path):
    first = make_capture(tmp_path)
    second = make_capture(tmp_path)
    assert first != second
    assert verify_receipt(first)["response_sha256"] == verify_receipt(second)["response_sha256"]
