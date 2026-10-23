"""Validate advanced-statistics cache acquisition linkage."""

import hashlib
import json

import pytest

from cfb_quant.spread_total import advanced_v4
from cfb_quant.live.acquisition_receipt_v58 import (
    capture_response,
    verify_receipt,
)

PAYLOAD = (
    b'[{"gameId":123,"team":"Synthetic Team",'
    b'"offense":{},"defense":{}}]'
)


class FakeResponse:
    status_code = 200
    content = PAYLOAD

    def raise_for_status(self):
        return None

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def get(self, *args, **kwargs):
        return FakeResponse()


def capture_into(directory):
    def capture(**kwargs):
        return capture_response(directory, **kwargs)
    return capture


def test_receipt_link_saved_in_cache_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")

    count, status = advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=FakeSession(),
        response_capture=capture_into(
            tmp_path / "research_acquisitions"
        ),
    )

    assert (count, status) == (1, "downloaded")

    cache = advanced_v4._cache_path(tmp_path, 2026)
    metadata = json.loads(
        cache.with_suffix(".meta.json").read_text(
            encoding="utf-8"
        )
    )

    link = metadata["http_acquisition"]
    receipt = verify_receipt(link["receipt_directory"])

    expected_sha = hashlib.sha256(PAYLOAD).hexdigest()

    assert link["response_sha256"] == expected_sha
    assert receipt["response_sha256"] == expected_sha
    assert metadata["sha256"] == hashlib.sha256(
        cache.read_bytes()
    ).hexdigest()

    assert receipt["endpoint"] == "/stats/game/advanced"
    assert receipt["parameters"]["year"] == 2026
    assert link["production_forecast_authorized"] is False
    assert "test-only-secret" not in json.dumps(metadata)


def test_wrong_receipt_blocks_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")

    def wrong_capture(**kwargs):
        kwargs["response_bytes"] = b'[{"different":true}]'
        return capture_response(
            tmp_path / "research_acquisitions",
            **kwargs,
        )

    with pytest.raises(ValueError, match="does not match"):
        advanced_v4.fetch_one(
            tmp_path,
            2026,
            session=FakeSession(),
            response_capture=wrong_capture,
        )

    assert not advanced_v4._cache_path(
        tmp_path, 2026
    ).exists()


def test_old_callback_without_return_still_works(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")
    directories = []

    def legacy_capture(**kwargs):
        directories.append(
            capture_response(
                tmp_path / "research_acquisitions",
                **kwargs,
            )
        )

    count, status = advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=FakeSession(),
        response_capture=legacy_capture,
    )

    assert (count, status) == (1, "downloaded")
    assert len(directories) == 1

    cache = advanced_v4._cache_path(tmp_path, 2026)
    metadata = json.loads(
        cache.with_suffix(".meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert "http_acquisition" not in metadata


def test_default_cache_metadata_unchanged(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")

    advanced_v4.fetch_one(
        tmp_path,
        2026,
        session=FakeSession(),
    )

    cache = advanced_v4._cache_path(tmp_path, 2026)
    metadata = json.loads(
        cache.with_suffix(".meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert "http_acquisition" not in metadata
