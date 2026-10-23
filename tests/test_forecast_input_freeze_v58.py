"""Tests for execution-time forecast input fingerprints."""

import hashlib
import json
from pathlib import Path

import pytest

from cfb_quant.live.forecast_input_freeze_v58 import (
    DEPENDENCIES,
    take,
    verify_unchanged,
)


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def setup_inputs(root):
    for name, relative in DEPENDENCIES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))

    snapshot = root / "reports/synthetic_snapshot"
    snapshot.mkdir(parents=True)

    (snapshot / "pregame_features.parquet").write_bytes(
        b"synthetic-feature-bytes"
    )

    metadata = {
        "schedule_sha256": hash_bytes(b"schedule_2026"),
        "advanced_2026_sha256": hash_bytes(
            b"advanced_2026"
        ),
        "no_current_game_outcomes": True,
    }

    (snapshot / "audit.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    return snapshot


def test_complete_fingerprint_set(tmp_path):
    snapshot = setup_inputs(tmp_path)
    receipt = take(tmp_path, snapshot)

    assert receipt["dependency_count"] == 10
    assert len(receipt["dependencies"]) == 10
    assert receipt["source_availability_certified"] is False
    assert verify_unchanged(
        tmp_path, receipt
    )["status"] == "UNCHANGED"


def test_training_file_mutation_detected(tmp_path):
    snapshot = setup_inputs(tmp_path)
    receipt = take(tmp_path, snapshot)

    path = tmp_path / DEPENDENCIES["v3_training"]
    path.write_bytes(b"modified-during-execution")

    with pytest.raises(ValueError, match="v3_training:CHANGED"):
        verify_unchanged(tmp_path, receipt)


def test_missing_calibration_file_detected(tmp_path):
    snapshot = setup_inputs(tmp_path)
    receipt = take(tmp_path, snapshot)

    (tmp_path / DEPENDENCIES["residual_pairing"]).unlink()

    with pytest.raises(
        ValueError, match="residual_pairing:MISSING"
    ):
        verify_unchanged(tmp_path, receipt)


def test_snapshot_source_mismatch_rejected(tmp_path):
    snapshot = setup_inputs(tmp_path)

    path = tmp_path / DEPENDENCIES["advanced_2026"]
    path.write_bytes(b"wrong-source-version")

    with pytest.raises(
        ValueError, match="Snapshot source identity mismatch"
    ):
        take(tmp_path, snapshot)


def test_outside_repository_snapshot_rejected(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    with pytest.raises(
        ValueError, match="outside repository"
    ):
        take(root, tmp_path / "external_snapshot")
