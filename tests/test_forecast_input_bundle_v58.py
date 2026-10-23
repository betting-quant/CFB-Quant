"""Tests for verified research input bundles."""

import json
import pytest

from cfb_quant.live.forecast_input_bundle_v58 import (
    create,
    verify,
)
from cfb_quant.live.forecast_input_freeze_v58 import (
    DEPENDENCIES,
)
from tests.test_forecast_input_freeze_v58 import setup_inputs


def test_bundle_created_with_ten_files(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)

    destination = create(
        root, snapshot, tmp_path / "bundles"
    )

    result = verify(destination)

    assert result["status"] == "VERIFIED"
    assert result["dependency_count"] == 10
    assert result["prospective_performance_eligible"] is False

    manifest = json.loads(
        (destination / "bundle_manifest.json").read_text()
    )

    assert len(manifest["dependencies"]) == 10
    assert (
        destination / DEPENDENCIES["v3_training"]
    ).is_file()


def test_bundle_survives_original_mutation(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)

    destination = create(
        root, snapshot, tmp_path / "bundles"
    )

    (root / DEPENDENCIES["v3_training"]).write_bytes(
        b"changed-original"
    )

    assert verify(destination)["status"] == "VERIFIED"


def test_modified_bundle_rejected(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)

    destination = create(
        root, snapshot, tmp_path / "bundles"
    )

    (destination / DEPENDENCIES["v3_training"]).write_bytes(
        b"tampered-bundle"
    )

    with pytest.raises(
        ValueError, match="Bundle integrity mismatch"
    ):
        verify(destination)


def test_duplicate_bundle_never_overwrites(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)
    parent = tmp_path / "bundles"

    first = create(root, snapshot, parent)

    with pytest.raises(
        FileExistsError, match="already exists"
    ):
        create(root, snapshot, parent)

    assert verify(first)["status"] == "VERIFIED"


def test_source_mismatch_leaves_no_bundle(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)
    parent = tmp_path / "bundles"

    (root / DEPENDENCIES["schedule_2026"]).write_bytes(
        b"invalid-new-schedule"
    )

    with pytest.raises(
        ValueError, match="Snapshot source identity mismatch"
    ):
        create(root, snapshot, parent)

    assert not parent.exists() or not list(parent.iterdir())
