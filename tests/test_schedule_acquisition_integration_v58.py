"""Offline checks for opt-in schedule acquisition integration."""

import inspect
import json
from pathlib import Path

import pytest

from cfb_quant.live import daily_schedule_sync_v58 as sync
from cfb_quant.live.acquisition_receipt_v58 import verify_receipt


def test_default_run_stays_opt_out():
    sig = inspect.signature(sync.run)
    assert sig.parameters["capture_http"].default is False


def test_schedule_uses_capture_hook_only_when_opted_in(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("[]", encoding="utf-8")

    calls = []

    def fake_cfbd_get(endpoint, *, params, response_capture=None):
        calls.append({
            "endpoint": endpoint,
            "params": params,
            "capture_enabled": response_capture is not None,
        })

        payload = b"[]"

        if response_capture is not None:
            response_capture(
                endpoint=endpoint,
                parameters=params,
                request_started_utc="2026-10-22T15:00:00Z",
                request_completed_utc="2026-10-22T15:00:01Z",
                status_code=200,
                response_bytes=payload,
            )
        return json.loads(payload)

    monkeypatch.setattr(sync, "cfbd_get", fake_cfbd_get)

    # The validator may reject an empty schedule. We only need to
    # establish that the HTTP acquisition happens before validation.
    with pytest.raises((ValueError, RuntimeError)):
        sync.run(
            season=2026,
            start="2026-10-22",
            end="2026-10-22",
            baseline=baseline,
            output=tmp_path / "out",
            capture_http=True,
        )

    assert len(calls) == 1
    assert calls[0]["capture_enabled"] is True
    assert calls[0]["endpoint"] == "/games"

    receipts = list(
        (tmp_path / "out" / "_http_acquisitions").rglob("receipt.json")
    )
    assert len(receipts) == 1

    receipt = verify_receipt(receipts[0].parent)
    assert receipt["response_sha256"]
    assert receipt["production_forecast_authorized"] is False


def test_uncaptured_default_has_no_new_api_argument(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    baseline.write_text("[]", encoding="utf-8")

    calls = []

    def original_signature_fake(endpoint, *, params):
        calls.append((endpoint, params))
        return []

    monkeypatch.setattr(sync, "cfbd_get", original_signature_fake)

    with pytest.raises((ValueError, RuntimeError)):
        sync.run(
            season=2026,
            start="2026-10-22",
            end="2026-10-22",
            baseline=baseline,
            output=tmp_path / "out",
        )

    assert len(calls) == 1
    assert not list(tmp_path.rglob("receipt.json"))
