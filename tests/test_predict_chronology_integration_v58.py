"""Integration: prediction chronology guard precedes model fitting."""

import pandas as pd
import pytest

from cfb_quant.live import predict_v57 as predict


KICKOFF = "2026-10-22T00:00:00Z"
ASOF = "2026-10-21T21:00:00Z"


def setup_pipeline(monkeypatch, *, created, archive):
    events = []

    frame = pd.DataFrame({
        "game_id": ["401871070", "401871070"],
        "kickoff_utc": [KICKOFF, KICKOFF],
        "synthetic_feature": [1.0, 2.0],
    })

    metadata = {
        "asof_utc": ASOF,
        "created_at_utc": created,
    }

    def fake_load_manifest(*args, **kwargs):
        events.append("load_manifest")
        return ["synthetic_feature"]

    def fake_verify(*args, **kwargs):
        events.append("verify_snapshot")
        assert kwargs.get("allow_started") is archive
        return frame, metadata, {}

    def forbid_training(*args, **kwargs):
        events.append("train_points")
        raise RuntimeError("TRAINING_BOUNDARY_REACHED")

    monkeypatch.setattr(
        predict, "load_manifest", fake_load_manifest
    )
    monkeypatch.setattr(
        predict, "verify_snapshot", fake_verify
    )
    monkeypatch.setattr(
        predict, "train_points", forbid_training
    )

    from cfb_quant.live import forecast_input_freeze_v58 as freeze

    monkeypatch.setattr(
        freeze,
        "take",
        lambda *args, **kwargs: {
            "dependency_count": 10,
            "dependencies": {},
        },
    )
    monkeypatch.setattr(
        freeze,
        "verify_unchanged",
        lambda *args, **kwargs: {
            "status": "UNCHANGED",
            "verified_dependency_count": 10,
        },
    )

    return events


def test_postkickoff_creation_blocks_before_training(
    tmp_path, monkeypatch
):
    events = setup_pipeline(
        monkeypatch,
        created="2026-10-22T14:00:00Z",
        archive=False,
    )

    with pytest.raises(
        ValueError, match="Pregame chronology rejected"
    ):
        predict.run(
            tmp_path,
            tmp_path / "unused",
            now=pd.Timestamp("2026-10-22T14:01:00Z"),
        )

    assert events == [
        "load_manifest",
        "load_manifest",
        "verify_snapshot",
    ]


def test_valid_timing_reaches_training(
    tmp_path, monkeypatch
):
    events = setup_pipeline(
        monkeypatch,
        created="2026-10-21T22:00:00Z",
        archive=False,
    )

    with pytest.raises(
        RuntimeError, match="TRAINING_BOUNDARY_REACHED"
    ):
        predict.run(
            tmp_path,
            tmp_path / "unused",
            now=pd.Timestamp("2026-10-21T22:01:00Z"),
        )

    assert events[-2:] == [
        "verify_snapshot",
        "train_points",
    ]


def test_archived_mode_reaches_training_but_is_explicit(
    tmp_path, monkeypatch
):
    events = setup_pipeline(
        monkeypatch,
        created="2026-10-22T14:00:00Z",
        archive=True,
    )

    with pytest.raises(
        RuntimeError, match="TRAINING_BOUNDARY_REACHED"
    ):
        predict.run(
            tmp_path,
            tmp_path / "unused",
            allow_started=True,
            now=pd.Timestamp("2026-10-22T14:01:00Z"),
        )

    assert events[-2:] == [
        "verify_snapshot",
        "train_points",
    ]
