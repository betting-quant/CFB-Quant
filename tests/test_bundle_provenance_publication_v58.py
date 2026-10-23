"""Atomic four-artifact research publication tests."""

import json

import pandas as pd
import pytest

from cfb_quant.live.forecast_publication_v58 import publish
from cfb_quant.live.forecast_publication_lineage_v58 import (
    validate_lineage,
)
from cfb_quant.live import execution_provenance_v58 as prov
from tests.test_verified_forecast_publication_v58 import inputs


CLOCK = "2026-10-21T22:02:00Z"


def setup_case(tmp_path):
    frame, audit = inputs(tmp_path)

    source = tmp_path / "src/cfb_quant/models"
    source.mkdir(parents=True)
    model = source / "model.py"
    model.write_text("VALUE = 1\n", encoding="utf-8")

    records = prov.source_files(tmp_path)
    manifest = {
        "schema": prov.SCHEMA,
        "research_only": True,
        "source_files": records,
        "source_file_count": len(records),
        "code_identity": prov.identity(records),
        "environment_identity": "b" * 64,
        "execution_identity": "c" * 64,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
    }

    return frame, audit, model, manifest


def test_atomic_four_artifact_publication(tmp_path):
    frame, audit, model, manifest = setup_case(tmp_path)
    parent = tmp_path / "reports/four_artifacts"

    def check():
        prov.verify_sources(tmp_path, manifest)
        return manifest

    output = publish(
        parent,
        frame,
        audit,
        clock=lambda: CLOCK,
        lineage_check=lambda: validate_lineage(tmp_path, audit),
        provenance_check=check,
    )

    assert sorted(p.name for p in output.iterdir()) == [
        "execution_provenance_v1.json",
        "forecast_input_lineage_v1.json",
        "model_audit.json",
        "slate_forecasts.csv",
    ]

    saved = json.loads(
        (output / "execution_provenance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["code_identity"] == manifest["code_identity"]


def test_changed_code_leaves_no_completed_forecast(
    tmp_path, monkeypatch
):
    frame, audit, model, manifest = setup_case(tmp_path)
    parent = tmp_path / "reports/four_artifacts"

    original_csv = pd.DataFrame.to_csv

    def change_model_after_csv(self, *args, **kwargs):
        result = original_csv(self, *args, **kwargs)
        model.write_text("VALUE = 999\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        pd.DataFrame, "to_csv", change_model_after_csv
    )

    def check():
        prov.verify_sources(tmp_path, manifest)
        return manifest

    with pytest.raises(ValueError, match="Source identity changed"):
        publish(
            parent,
            frame,
            audit,
            clock=lambda: CLOCK,
            lineage_check=lambda: validate_lineage(tmp_path, audit),
            provenance_check=check,
        )

    assert parent.exists()
    assert list(parent.iterdir()) == []
