"""Strict three-artifact atomic forecast publication tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cfb_quant.live.forecast_input_freeze_v58 import take
from cfb_quant.live.forecast_publication_v58 import (
    publish_verified,
)
from tests.test_forecast_input_freeze_v58 import setup_inputs


TIME = "2026-10-21T22:02:00Z"
KICKOFF = "2026-10-22T00:00:00Z"


def inputs(root):
    snapshot = setup_inputs(root)
    fingerprints = take(root, snapshot)

    frame = pd.DataFrame({
        "game_id": ["100"],
        "kickoff_utc": [KICKOFF],
        "fair_total": [52.5],
    })

    audit = {
        "snapshot_metadata": {
            "slate_date_eastern": "2026-10-21",
            "asof_utc": "2026-10-21T21:00:00Z",
            "created_at_utc": "2026-10-21T21:30:00Z",
        },
        "generated_at_utc": "2026-10-21T22:00:00Z",
        "snapshot_validation": {
            "snapshot_folder": str(snapshot),
            "snapshot_sha256": fingerprints[
                "dependencies"
            ]["snapshot_parquet"]["sha256"],
        },
        "chronology_gate": {
            "status": "PASS",
            "archive_override_requested": False,
        },
        "forecast_classification": "CHRONOLOGY_VALID_RESEARCH",
        "source_availability_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
        "execution_input_fingerprints": fingerprints,
        "execution_input_verification": {
            "status": "UNCHANGED",
            "verified_dependency_count": 10,
        },
    }

    return frame, audit


def entries(parent):
    return list(parent.iterdir()) if parent.exists() else []


def test_three_artifacts_published_together(tmp_path):
    frame, audit = inputs(tmp_path)
    parent = tmp_path / "reports/forecast_test"

    out = publish_verified(
        tmp_path, parent, frame, audit,
        clock=lambda: TIME,
    )

    assert out.is_dir()
    assert sorted(p.name for p in out.iterdir()) == [
        "forecast_input_lineage_v1.json",
        "model_audit.json",
        "slate_forecasts.csv",
    ]

    lineage = json.loads(
        (out / "forecast_input_lineage_v1.json").read_text()
    )

    assert lineage["dependency_count"] == 10
    assert lineage["publication_check"]["status"] == "UNCHANGED"
    assert lineage["prospective_performance_eligible"] is False


def test_missing_lineage_creates_no_output(tmp_path):
    frame, audit = inputs(tmp_path)
    del audit["execution_input_fingerprints"]
    parent = tmp_path / "reports/forecast_test"

    with pytest.raises(ValueError, match="Missing execution input"):
        publish_verified(
            tmp_path, parent, frame, audit,
            clock=lambda: TIME,
        )

    assert entries(parent) == []


def test_mutation_before_publication_creates_no_output(tmp_path):
    frame, audit = inputs(tmp_path)
    parent = tmp_path / "reports/forecast_test"

    changed = tmp_path / (
        "data/processed/cfb_matchup_features_kickoff_v3.parquet"
    )
    changed.write_bytes(b"modified-after-model-run")

    with pytest.raises(ValueError, match="v3_training:CHANGED"):
        publish_verified(
            tmp_path, parent, frame, audit,
            clock=lambda: TIME,
        )

    assert entries(parent) == []


def test_mutation_during_staging_cleans_up(
    tmp_path, monkeypatch
):
    frame, audit = inputs(tmp_path)
    parent = tmp_path / "reports/forecast_test"

    original = pd.DataFrame.to_csv

    def write_then_mutate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        changed = tmp_path / (
            "data/processed/cfb_matchup_features_kickoff_v3.parquet"
        )
        changed.write_bytes(b"changed-during-staging")
        return result

    monkeypatch.setattr(
        pd.DataFrame, "to_csv", write_then_mutate
    )

    with pytest.raises(ValueError, match="v3_training:CHANGED"):
        publish_verified(
            tmp_path, parent, frame, audit,
            clock=lambda: TIME,
        )

    assert entries(parent) == []
