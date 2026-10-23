"""Offline tests for portable CFBD advanced receipt verification."""

import hashlib
import json
import shutil

import pytest

from cfb_quant.spread_total import advanced_v4
from cfb_quant.live.acquisition_receipt_v58 import capture_response
from cfb_quant.live.portable_advanced_verify_v58 import verify_advanced


PAYLOAD = b'[{"gameId":123,"team":"Synthetic","offense":{},"defense":{}}]'


class FakeResponse:
    status_code = 200
    content = PAYLOAD

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def get(self, *args, **kwargs):
        return FakeResponse()


def create_workspace(root, monkeypatch):
    monkeypatch.setenv("CFBD_API_KEY", "test-only-secret")

    count, status = advanced_v4.fetch_one(
        root,
        2026,
        session=FakeSession(),
        response_capture=lambda **kwargs: capture_response(
            root / "research_acquisitions",
            **kwargs,
        ),
    )
    assert (count, status) == (1, "downloaded")
    return advanced_v4._cache_path(root, 2026)


def test_verified_cache(tmp_path, monkeypatch):
    create_workspace(tmp_path, monkeypatch)
    result = verify_advanced(tmp_path, 2026)
    assert result["status"] == "PASS"
    assert result["records"] == 1


def test_copy_to_different_pc(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    create_workspace(original, monkeypatch)

    moved = tmp_path / "different_pc"
    shutil.copytree(original, moved)

    assert verify_advanced(moved, 2026)["status"] == "PASS"


def test_tampered_cache_rejected(tmp_path, monkeypatch):
    cache = create_workspace(tmp_path, monkeypatch)
    cache.write_bytes(b"[]")

    with pytest.raises(ValueError, match="fingerprint"):
        verify_advanced(tmp_path, 2026)


def test_path_traversal_rejected(tmp_path, monkeypatch):
    cache = create_workspace(tmp_path, monkeypatch)
    meta_path = cache.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["http_acquisition"]["receipt_relative_to_root"] = "../outside"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe"):
        verify_advanced(tmp_path, 2026)


def test_response_cache_mismatch_rejected(tmp_path, monkeypatch):
    cache = create_workspace(tmp_path, monkeypatch)
    replacement = b'[{"gameId":999,"team":"Other"}]'
    cache.write_bytes(replacement)

    meta_path = cache.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sha256"] = hashlib.sha256(replacement).hexdigest()
    meta["record_count"] = 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="differ"):
        verify_advanced(tmp_path, 2026)
