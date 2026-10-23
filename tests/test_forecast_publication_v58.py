"""Offline forecast publication tests."""

import json

import pandas as pd
import pytest

from cfb_quant.live.forecast_publication_v58 import publish


KICKOFF = "2026-10-22T00:00:00Z"


def fixture(archive=False):
    frame = pd.DataFrame({
        "game_id": ["100"],
        "kickoff_utc": [KICKOFF],
        "fair_total": [52.5],
    })

    audit = {
        "snapshot_metadata": {
            "slate_date_eastern": "2026-10-21",
            "asof_utc": "2026-10-21T21:00:00Z",
            "created_at_utc": "2026-10-21T22:00:00Z",
        },
        "generated_at_utc": "2026-10-21T22:01:00Z",
        "chronology_gate": {
            "status": "PASS",
            "archive_override_requested": archive,
        },
        "forecast_classification": (
            "RETROSPECTIVE_RESEARCH"
            if archive else "CHRONOLOGY_VALID_RESEARCH"
        ),
        "source_availability_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
    }
    return frame, audit


def ticking(values):
    values = iter(values)
    return lambda: next(values)


def entries(parent):
    return list(parent.iterdir()) if parent.exists() else []


def test_valid_atomic_publication(tmp_path):
    frame, audit = fixture()
    clock = ticking(["2026-10-21T22:02:00Z"] * 4)

    dest = publish(tmp_path / "out", frame, audit, clock=clock)

    assert dest.is_dir()
    assert (dest / "slate_forecasts.csv").is_file()
    saved = json.loads(
        (dest / "model_audit.json").read_text()
    )
    assert saved["publication"]["research_only"] is True
    assert saved["prospective_performance_eligible"] is False


def test_started_game_rejected_before_output(tmp_path):
    frame, audit = fixture()
    parent = tmp_path / "out"

    with pytest.raises(ValueError, match="already started"):
        publish(
            parent, frame, audit,
            clock=lambda: "2026-10-22T00:01:00Z",
        )

    assert entries(parent) == []


def test_kickoff_during_publication_cleans_stage(tmp_path):
    frame, audit = fixture()
    parent = tmp_path / "out"

    clock = ticking([
        "2026-10-21T22:02:00Z",
        "2026-10-21T22:02:01Z",
        "2026-10-21T22:02:02Z",
        "2026-10-22T00:00:01Z",
    ])

    with pytest.raises(ValueError, match="already started"):
        publish(parent, frame, audit, clock=clock)

    assert entries(parent) == []


def test_archive_remains_retro(tmp_path):
    frame, audit = fixture(archive=True)

    dest = publish(
        tmp_path / "out",
        frame, audit,
        clock=lambda: "2026-10-22T14:00:00Z",
    )

    saved = json.loads(
        (dest / "model_audit.json").read_text()
    )
    assert saved["forecast_classification"] == "RETROSPECTIVE_RESEARCH"
    assert saved["prospective_performance_eligible"] is False


def test_serialization_failure_does_not_publish(
    tmp_path, monkeypatch
):
    frame, audit = fixture()
    parent = tmp_path / "out"

    def fail(*args, **kwargs):
        raise OSError("SYNTHETIC_CSV_WRITE_FAILURE")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail)

    with pytest.raises(OSError, match="SYNTHETIC_CSV_WRITE_FAILURE"):
        publish(
            parent, frame, audit,
            clock=lambda: "2026-10-21T22:02:00Z",
        )

    assert entries(parent) == []
