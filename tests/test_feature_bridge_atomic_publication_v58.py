"""Offline integration tests of staged feature publication."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from cfb_quant.live import feature_bridge_v56 as bridge


KICKOFF = "2026-10-22T00:00:00Z"
ASOF = "2026-10-21T21:55:00Z"


def setup(monkeypatch, root, *, times):
    values = iter(times)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            try:
                result = pd.Timestamp(next(values)).to_pydatetime()
            except StopIteration:
                raise AssertionError("Unexpected extra clock read")
            return result.astimezone(tz) if tz else result

    def prepare(*args, **kwargs):
        return pd.DataFrame({
            "game_id": ["100", "100"],
            "kickoff_utc": [KICKOFF, KICKOFF],
            "team": ["Home", "Away"],
            "opponent": ["Away", "Home"],
            "home_away": ["home", "away"],
            "points": [float("nan"), float("nan")],
        }), {
            "asof_utc": ASOF,
            "no_current_game_outcomes": True,
            "slate_games": 1,
            "slate_team_rows": 2,
            "slate_date_eastern": "2026-10-21",
        }

    monkeypatch.setattr(bridge, "datetime", Clock)
    monkeypatch.setattr(bridge, "prepare", prepare)
    monkeypatch.setattr(sys, "argv", [
        "feature_bridge_v56", "--root", str(root),
        "--date", "2026-10-21", "--asof-utc", ASOF, "--save",
    ])

    for relative in (
        "data/raw/cfbd/games/games_2026.json",
        "data/raw/cfbd/advanced_game_v4/advanced_2026.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")


def times(*values):
    return list(values)


def published(root):
    parent = root / bridge.OUT_DIR
    return (
        list(parent.iterdir()) if parent.exists() else []
    )


def test_valid_snapshot_published_atomically(tmp_path, monkeypatch):
    setup(monkeypatch, tmp_path, times=times(
        "2026-10-21T23:00:00Z",
        "2026-10-21T23:00:01Z",
        "2026-10-21T23:00:02Z",
        "2026-10-21T23:00:03Z",
        "2026-10-21T23:00:04Z",
    ))
    bridge.main()
    dirs = published(tmp_path)
    assert len(dirs) == 1
    assert not dirs[0].name.startswith(".pending_")
    assert (dirs[0] / "pregame_features.parquet").is_file()
    audit = json.loads((dirs[0] / "audit.json").read_text())
    assert audit["snapshot_save_gate"]["status"] == "PASS"
    assert audit["source_availability_certified"] is False


def test_kickoff_during_serialization_leaves_no_snapshot(
    tmp_path, monkeypatch
):
    setup(monkeypatch, tmp_path, times=times(
        "2026-10-21T23:00:00Z",
        "2026-10-21T23:00:01Z",
        "2026-10-21T23:00:02Z",
        "2026-10-22T00:00:01Z",
    ))
    with pytest.raises(ValueError, match="publication rejected"):
        bridge.main()
    assert published(tmp_path) == []


def test_serialization_exception_removes_stage(tmp_path, monkeypatch):
    setup(monkeypatch, tmp_path, times=times(
        "2026-10-21T23:00:00Z",
        "2026-10-21T23:00:01Z",
        "2026-10-21T23:00:02Z",
    ))

    def fail(*args, **kwargs):
        raise OSError("SYNTHETIC_SERIALIZATION_FAILURE")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="SYNTHETIC_SERIALIZATION_FAILURE"):
        bridge.main()
    assert published(tmp_path) == []
