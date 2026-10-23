"""Validation tests for forecast-to-snapshot publication lineage."""

import pytest

from cfb_quant.live.forecast_input_freeze_v58 import (
    take,
)
from cfb_quant.live.forecast_publication_lineage_v58 import (
    validate_lineage,
)
from tests.test_forecast_input_freeze_v58 import (
    setup_inputs,
)


def fixture(root):
    snapshot = setup_inputs(root)
    fingerprints = take(root, snapshot)

    audit = {
        "snapshot_validation": {
            "snapshot_folder": str(snapshot),
            "snapshot_sha256": fingerprints[
                "dependencies"
            ]["snapshot_parquet"]["sha256"],
        },
        "execution_input_fingerprints": fingerprints,
        "execution_input_verification": {
            "status": "UNCHANGED",
            "verified_dependency_count": 10,
        },
    }
    return snapshot, audit


def test_valid_lineage(tmp_path):
    _, audit = fixture(tmp_path)
    result = validate_lineage(tmp_path, audit)

    assert result["dependency_count"] == 10
    assert result["publication_check"]["status"] == "UNCHANGED"
    assert result["prospective_performance_eligible"] is False


def test_changed_training_data_blocks_publication(tmp_path):
    _, audit = fixture(tmp_path)

    path = tmp_path / (
        "data/processed/cfb_matchup_features_kickoff_v3.parquet"
    )
    path.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="v3_training:CHANGED"):
        validate_lineage(tmp_path, audit)


def test_swapped_snapshot_binding_rejected(tmp_path):
    _, audit = fixture(tmp_path)
    audit["snapshot_validation"]["snapshot_folder"] = (
        str(tmp_path / "reports/another_snapshot")
    )

    with pytest.raises(ValueError, match="Snapshot binding mismatch"):
        validate_lineage(tmp_path, audit)


def test_forged_snapshot_hash_rejected(tmp_path):
    _, audit = fixture(tmp_path)
    audit["snapshot_validation"]["snapshot_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="Snapshot hash binding mismatch"):
        validate_lineage(tmp_path, audit)


def test_missing_execution_verification_rejected(tmp_path):
    _, audit = fixture(tmp_path)
    del audit["execution_input_verification"]

    with pytest.raises(ValueError, match="Missing execution"):
        validate_lineage(tmp_path, audit)
