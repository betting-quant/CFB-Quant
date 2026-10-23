"""Tests for research-only input lineage validation."""

import hashlib
import json

import pytest

from cfb_quant.live.research_lineage_gate_v58 import (
    CRITICAL_INPUTS,
    assess,
)


def fixture_workspace(tmp_path):
    hashes = {}

    for relative in CRITICAL_INPUTS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test data")
        hashes[relative.replace("/", "\\")] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    manifest = {
        "research_asof_utc": "2026-10-21T21:55:42Z",
        "created_at_utc": "2026-10-21T22:09:54Z",
        "input_sha256": hashes,
    }
    (tmp_path / "research_input_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return tmp_path


def test_intact_inputs_remain_unverified(tmp_path):
    work = fixture_workspace(tmp_path)
    result = assess(work)

    assert result["status"] == "UNVERIFIED"
    assert result["pregame_certified"] is False
    assert result["production_forecast_authorized"] is False
    assert all(
        item["fingerprint_matches"]
        for item in result["inputs"].values()
    )


def test_tampered_input_is_invalid(tmp_path):
    work = fixture_workspace(tmp_path)
    path = work / CRITICAL_INPUTS["advanced"]
    path.write_bytes(b"tampered")

    result = assess(work)

    assert result["status"] == "INVALID"
    assert any("fingerprint mismatch" in x for x in result["issues"])


def test_missing_input_is_invalid(tmp_path):
    work = fixture_workspace(tmp_path)
    (work / CRITICAL_INPUTS["history"]).unlink()

    result = assess(work)

    assert result["status"] == "INVALID"
    assert any("source file missing" in x for x in result["issues"])


def test_post_cutoff_feature_generation_reported(tmp_path):
    work = fixture_workspace(tmp_path)
    audit_path = work / "audit.json"

    audit_path.write_text(json.dumps({
        "created_at_utc": "2026-10-22T14:17:36Z",
        "asof_utc": "2026-10-21T21:55:42Z",
        "schedule_sha256": hashlib.sha256(
            (work / CRITICAL_INPUTS["schedule"]).read_bytes()
        ).hexdigest(),
        "advanced_2026_sha256": hashlib.sha256(
            (work / CRITICAL_INPUTS["advanced"]).read_bytes()
        ).hexdigest(),
    }), encoding="utf-8")

    result = assess(
        work,
        forecast_cutoff_utc="2026-10-21T23:00:00Z",
        feature_audit=audit_path,
    )

    assert result["status"] == "UNVERIFIED"
    assert any(
        "Feature snapshot created after forecast cutoff" in x
        for x in result["observations"]
    )
    assert result["pregame_certified"] is False


def test_naive_cutoff_rejected(tmp_path):
    work = fixture_workspace(tmp_path)

    with pytest.raises(ValueError, match="timezone-aware"):
        assess(work, forecast_cutoff_utc="2026-10-21T23:00:00")
