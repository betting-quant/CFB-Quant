"""Offline tests for opt-in advanced-statistics HTTP acquisition."""

import json

import pytest

from cfb_quant.spread_total import advanced_v4
from cfb_quant.live.acquisition_receipt_v58 import (
    capture_response,
    verify_receipt,
)


PAYLOAD = b'[{"gameId":123,"team":"Synthetic Team","offense":{},"defense":{}}]'


class FakeResponse:
    status_code = 200
    content = PAYLOAD

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return FakeResponse()


def test_capture_archives_exact_bytes_before_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")
    session = FakeSession()
    archives = []

    def capture(**kwargs):
        assert not advanced_v4._cache_path(tmp_path, 2026).exists()
        archives.append(
            capture_response(
                tmp_path / "research_acquisitions",
                **kwargs,
            )
        )

    count, status = advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=session,
        response_capture=capture,
    )

    assert (count, status) == (1, "downloaded")
    assert session.calls == 1
    assert len(archives) == 1

    receipt = verify_receipt(archives[0])
    assert receipt["endpoint"] == "/stats/game/advanced"
    assert receipt["parameters"]["year"] == 2026
    assert receipt["production_forecast_authorized"] is False
    assert (archives[0] / "response.json").read_bytes() == PAYLOAD
    assert "test-only-secret" not in json.dumps(receipt)

    cache = advanced_v4._cache_path(tmp_path, 2026)
    assert json.loads(cache.read_bytes()) == json.loads(PAYLOAD)


def test_default_fetch_remains_compatible(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")
    session = FakeSession()

    count, status = advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=session,
    )

    assert (count, status) == (1, "downloaded")
    assert session.calls == 1
    assert not list(tmp_path.rglob("receipt.json"))


def test_capture_failure_does_not_write_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")
    session = FakeSession()

    def failed_capture(**kwargs):
        raise OSError("Simulated archive failure")

    with pytest.raises(OSError, match="archive failure"):
        advanced_v4.fetch_one(
            tmp_path,
            2026,
            session=session,
            response_capture=failed_capture,
        )

    assert session.calls == 1
    assert not advanced_v4._cache_path(tmp_path, 2026).exists()


def test_cached_data_does_not_redownload(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")
    session = FakeSession()

    advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=session,
    )

    def unexpected_capture(**kwargs):
        raise AssertionError("Cached data should not be recaptured")

    count, status = advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=session,
        response_capture=unexpected_capture,
    )

    assert (count, status) == (1, "cached")
    assert session.calls == 1
