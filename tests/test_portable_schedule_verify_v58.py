"""Regression tests for portable schedule archive validation."""

import hashlib
import json
from pathlib import Path

import pytest

from cfb_quant.live.acquisition_receipt_v58 import capture_response
from cfb_quant.live.portable_schedule_verify_v58 import verify_archive


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fixture_archive(tmp_path):
    output = tmp_path / "research"
    archive = output / "20261022T160000Z"
    archive.mkdir(parents=True)

    game = {
        "id": 123,
        "season": 2026,
        "homeTeam": "Home",
        "awayTeam": "Away",
    }
    params = {
        "year": 2026,
        "seasonType": "both",
        "classification": "fbs",
    }

    receipt_dir = capture_response(
        output / "_http_acquisitions",
        endpoint="/games",
        parameters=params,
        request_started_utc="2026-10-22T15:00:00Z",
        request_completed_utc="2026-10-22T15:00:01Z",
        status_code=200,
        response_bytes=json.dumps([game]).encode("utf-8"),
    )

    snapshot = archive / "cfbd_games.json"
    write_json(snapshot, [game])

    receipt = json.loads(
        (receipt_dir / "receipt.json").read_text(encoding="utf-8")
    )

    audit = {
        "research_only": True,
        "fresh_games": 1,
        "parameters": params,
        "snapshot_sha256": hashlib.sha256(
            snapshot.read_bytes()
        ).hexdigest(),
        "request_completed_utc": "2026-10-22T15:00:02Z",
        "http_acquisition": {
            "receipt_relative_to_output": (
                receipt_dir.relative_to(output).as_posix()
            ),
            "response_sha256": receipt["response_sha256"],
            "request_started_utc": receipt["request_started_utc"],
            "request_completed_utc": receipt["request_completed_utc"],
            "historical_point_in_time_certified": False,
            "production_forecast_authorized": False,
        },
    }

    write_json(archive / "audit.json", audit)
    return output, archive


def test_valid_archive_passes(tmp_path):
    _, archive = fixture_archive(tmp_path)
    result = verify_archive(archive)
    assert result["status"] == "PASS"
    assert result["schedule_games"] == 1
    assert result["production_forecast_authorized"] is False


def test_relocated_archive_passes(tmp_path):
    import shutil
    output, archive = fixture_archive(tmp_path)
    relocated = tmp_path / "different_pc"
    shutil.copytree(output, relocated)
    result = verify_archive(relocated / archive.name)
    assert result["status"] == "PASS"


def test_tampered_schedule_rejected(tmp_path):
    _, archive = fixture_archive(tmp_path)
    (archive / "cfbd_games.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_archive(archive)


def test_unsafe_traversal_rejected(tmp_path):
    _, archive = fixture_archive(tmp_path)
    path = archive / "audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    audit["http_acquisition"]["receipt_relative_to_output"] = (
        "../outside"
    )
    write_json(path, audit)
    with pytest.raises(ValueError, match="Unsafe"):
        verify_archive(archive)


def test_missing_receipt_rejected(tmp_path):
    output, archive = fixture_archive(tmp_path)
    import shutil
    shutil.rmtree(output / "_http_acquisitions")
    with pytest.raises(FileNotFoundError):
        verify_archive(archive)


def test_response_snapshot_mismatch_rejected(tmp_path):
    _, archive = fixture_archive(tmp_path)
    snapshot = archive / "cfbd_games.json"
    different = [{"id": 999}]
    write_json(snapshot, different)
    audit_path = archive / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["snapshot_sha256"] = hashlib.sha256(
        snapshot.read_bytes()
    ).hexdigest()
    write_json(audit_path, audit)

    with pytest.raises(ValueError, match="differ"):
        verify_archive(archive)


def test_bad_receipt_linkage_rejected(tmp_path):
    _, archive = fixture_archive(tmp_path)
    audit_path = archive / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["http_acquisition"]["response_sha256"] = "0" * 64
    write_json(audit_path, audit)

    with pytest.raises(ValueError, match="linkage mismatch"):
        verify_archive(archive)
