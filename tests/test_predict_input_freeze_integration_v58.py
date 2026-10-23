"""Execution-order and mutation tests for the prediction pipeline."""

import numpy as np
import pandas as pd
import pytest

from cfb_quant.live import predict_v57 as predict
from cfb_quant.live import forecast_input_freeze_v58 as freeze
from tests.test_predict_chronology_integration_v58 import (
    setup_pipeline,
)


class Model:
    def predict(self, frame):
        return np.ones(len(frame), dtype=float)


def setup_completed_run(monkeypatch):
    # Reuse the existing synthetic prediction-entry fixture.
    from pathlib import Path

    events = setup_pipeline(
        monkeypatch,
        created="2026-10-21T22:00:00Z",
        archive=False,
    )

    fingerprint = {
        "schema": "cfb_v58_execution_input_fingerprints_v1",
        "dependency_count": 10,
        "dependencies": {"synthetic": {"sha256": "a" * 64}},
        "research_only": True,
        "source_availability_certified": False,
        "production_forecast_authorized": False,
    }

    def take(*args, **kwargs):
        events.append("take")
        return fingerprint

    def train(*args, **kwargs):
        events.append("train_points")
        return Model(), {"training_rows": 2}

    def game_forecasts(*args, **kwargs):
        events.append("game_forecasts")
        return pd.DataFrame({
            "game_id": ["401871070"],
            "kickoff_utc": ["2026-10-22T00:00:00Z"],
        })

    def residuals(*args, **kwargs):
        events.append("residuals")
        return np.ones(2), {"residual_games": 2}

    def probabilities(frame, *args, **kwargs):
        events.append("probabilities")
        return frame.copy()

    monkeypatch.setattr(freeze, "take", take)
    monkeypatch.setattr(predict, "train_points", train)
    monkeypatch.setattr(predict, "game_forecasts", game_forecasts)
    monkeypatch.setattr(
        predict, "historical_fbs_residuals", residuals
    )
    monkeypatch.setattr(
        predict, "add_probabilities", probabilities
    )

    return events, fingerprint


def test_fingerprints_enclose_model_execution(
    tmp_path, monkeypatch
):
    events, fingerprint = setup_completed_run(monkeypatch)

    def verify(root, original):
        events.append("verify_unchanged")
        assert original is fingerprint
        return {
            "status": "UNCHANGED",
            "verified_dependency_count": 10,
        }

    monkeypatch.setattr(freeze, "verify_unchanged", verify)

    frame, audit = predict.run(
        tmp_path,
        tmp_path / "synthetic_snapshot",
        now=pd.Timestamp("2026-10-21T22:01:00Z"),
    )

    assert len(frame) == 1
    assert events == [
        "load_manifest",
        "load_manifest",
        "verify_snapshot",
        "take",
        "train_points",
        "train_points",
        "game_forecasts",
        "residuals",
        "probabilities",
        "verify_unchanged",
    ]

    assert audit["execution_input_fingerprints"] is fingerprint
    assert audit["execution_input_verification"]["status"] == (
        "UNCHANGED"
    )
    assert audit["production_forecast_authorized"] is False


def test_mutation_blocks_return_of_forecast(
    tmp_path, monkeypatch
):
    events, _ = setup_completed_run(monkeypatch)

    def reject(*args, **kwargs):
        events.append("verify_unchanged")
        raise ValueError(
            "Forecast execution inputs changed: "
            "v3_training:CHANGED"
        )

    monkeypatch.setattr(freeze, "verify_unchanged", reject)

    with pytest.raises(
        ValueError, match="v3_training:CHANGED"
    ):
        predict.run(
            tmp_path,
            tmp_path / "synthetic_snapshot",
            now=pd.Timestamp("2026-10-21T22:01:00Z"),
        )

    assert events[-1] == "verify_unchanged"
