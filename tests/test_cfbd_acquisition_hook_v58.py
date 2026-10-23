"""Offline regression tests for optional CFBD response capture."""

import json

import pytest

from cfb_quant.data import download_history
from cfb_quant.live.acquisition_receipt_v58 import (
    capture_response,
    verify_receipt,
)


class FakeResponse:
    status_code = 200
    content = b'[{"id":123456}]'

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self.content)


def install_fake(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse()

    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")
    monkeypatch.setattr(download_history.requests, "get", fake_get)
    return calls


def test_default_client_remains_compatible(monkeypatch):
    calls = install_fake(monkeypatch)

    result = download_history.cfbd_get(
        "/games",
        params={"year": 2026},
        attempts=1,
    )

    assert result == [{"id": 123456}]
    assert len(calls) == 1


def test_opt_in_capture_preserves_exact_response(tmp_path, monkeypatch):
    calls = install_fake(monkeypatch)
    archives = []

    def capture(**kwargs):
        archives.append(
            capture_response(tmp_path, **kwargs)
        )

    result = download_history.cfbd_get(
        "/games",
        params={"year": 2026},
        attempts=1,
        response_capture=capture,
    )

    assert result == [{"id": 123456}]
    assert len(calls) == 1
    assert len(archives) == 1

    receipt = verify_receipt(archives[0])
    assert receipt["endpoint"] == "/games"
    assert receipt["response_bytes"] == len(FakeResponse.content)
    assert receipt["production_forecast_authorized"] is False
    assert (archives[0] / "response.json").read_bytes() == FakeResponse.content
    assert "test-only-secret" not in json.dumps(receipt)


def test_capture_failure_does_not_retry(monkeypatch):
    calls = install_fake(monkeypatch)

    def broken_capture(**kwargs):
        raise OSError("Simulated archive write failure")

    with pytest.raises(RuntimeError, match="capture failed"):
        download_history.cfbd_get(
            "/games",
            params={"year": 2026},
            attempts=4,
            response_capture=broken_capture,
        )

    assert len(calls) == 1
