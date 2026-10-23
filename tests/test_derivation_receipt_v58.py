"""Offline validation of source-to-Parquet derivation receipts."""

import shutil

import pytest

from cfb_quant.live.derivation_receipt_v58 import (
    make_receipt,
    verify_derivation,
)

START = "2026-10-22T15:00:00Z"
END = "2026-10-22T15:00:02Z"


def create(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "reports").mkdir()

    (tmp_path / "raw" / "schedule.json").write_bytes(b'[{"id":123}]')
    (tmp_path / "raw" / "box.json").write_bytes(b'[{"gameId":123}]')
    # Synthetic bytes only: this test validates provenance mechanics,
    # not Parquet decoding or the history builder.
    (tmp_path / "processed" / "team_games_2026.parquet").write_bytes(
        b"synthetic-parquet-placeholder"
    )

    receipt = make_receipt(
        tmp_path,
        input_paths=["raw/schedule.json", "raw/box.json"],
        output_path="processed/team_games_2026.parquet",
        started_utc=START,
        completed_utc=END,
        receipt_path="reports/derivation.json",
    )
    return receipt


def test_valid_derivation_passes(tmp_path):
    create(tmp_path)
    result = verify_derivation(tmp_path, "reports/derivation.json")
    assert result["status"] == "PASS"
    assert result["input_count"] == 2
    assert result["production_forecast_authorized"] is False


def test_workspace_relocation_passes(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    create(original)
    copied = tmp_path / "different_pc"
    shutil.copytree(original, copied)
    assert verify_derivation(
        copied, "reports/derivation.json"
    )["status"] == "PASS"


def test_modified_source_rejected(tmp_path):
    create(tmp_path)
    (tmp_path / "raw" / "box.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="Source fingerprint mismatch"):
        verify_derivation(tmp_path, "reports/derivation.json")


def test_modified_output_rejected(tmp_path):
    create(tmp_path)
    (tmp_path / "processed" / "team_games_2026.parquet").write_bytes(
        b"changed"
    )
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        verify_derivation(tmp_path, "reports/derivation.json")


def test_receipt_cannot_overwrite_existing(tmp_path):
    create(tmp_path)
    with pytest.raises(FileExistsError):
        make_receipt(
            tmp_path,
            input_paths=["raw/schedule.json"],
            output_path="processed/team_games_2026.parquet",
            started_utc=START,
            completed_utc=END,
            receipt_path="reports/derivation.json",
        )


def test_invalid_derivation_chronology_rejected(tmp_path):
    create(tmp_path)
    with pytest.raises(ValueError, match="before it started"):
        make_receipt(
            tmp_path,
            input_paths=["raw/schedule.json"],
            output_path="processed/team_games_2026.parquet",
            started_utc=END,
            completed_utc=START,
            receipt_path="reports/other.json",
        )
