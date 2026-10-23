"""Offline integration tests for the real feature-bridge save entry point."""

import json
import sys
from datetime import datetime, timezone

import pandas as pd
import pytest

from cfb_quant.live import feature_bridge_v56 as bridge


KICKOFF = "2026-10-22T00:00:00Z"
ASOF = "2026-10-21T21:55:00Z"


def setup_bridge(monkeypatch, root, clock):
    events = []

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = pd.Timestamp(clock).to_pydatetime()
            if tz is not None:
                value = value.astimezone(tz)
            return value

    frame = pd.DataFrame({
        "game_id": ["100", "100"],
        "kickoff_utc": [KICKOFF, KICKOFF],
        "team": ["Home", "Away"],
        "opponent": ["Away", "Home"],
        "home_away": ["home", "away"],
        "points": [float("nan"), float("nan")],
    })

    metadata = {
        "asof_utc": ASOF,
        "no_current_game_outcomes": True,
        "slate_games": 1,
        "slate_team_rows": 2,
        "slate_date_eastern": "2026-10-21",
    }

    def fake_prepare(*args, **kwargs):
        events.append("prepare")
        return frame.copy(), metadata.copy()

    monkeypatch.setattr(bridge, "prepare", fake_prepare)
    monkeypatch.setattr(bridge, "datetime", FrozenDatetime)

    monkeypatch.setattr(sys, "argv", [
        "feature_bridge_v56",
        "--root", str(root),
        "--date", "2026-10-21",
        "--asof-utc", ASOF,
        "--save",
    ])

    return events


def test_postkickoff_rejection_creates_no_snapshot(
    tmp_path, monkeypatch
):
    events = setup_bridge(
        monkeypatch,
        tmp_path,
        "2026-10-22T14:00:00Z",
    )

    with pytest.raises(
        ValueError,
        match="Feature snapshot save rejected",
    ):
        bridge.main()

    assert events == ["prepare"]
    assert not (tmp_path / bridge.OUT_DIR).exists()


def test_buffer_rejection_creates_no_snapshot(
    tmp_path, monkeypatch
):
    setup_bridge(
        monkeypatch,
        tmp_path,
        "2026-10-21T23:59:30Z",
    )

    with pytest.raises(
        ValueError,
        match="INSUFFICIENT_PREKICKOFF_SAVE_BUFFER",
    ):
        bridge.main()

    assert not (tmp_path / bridge.OUT_DIR).exists()


def test_valid_save_writes_feature_and_audit(
    tmp_path, monkeypatch
):
    setup_bridge(
        monkeypatch,
        tmp_path,
        "2026-10-21T23:00:00Z",
    )

    # The real writer hashes these sources. Synthetic bytes are sufficient
    # because feature computation is explicitly mocked for this test.
    for relative in (
        "data/raw/cfbd/games/games_2026.json",
        "data/raw/cfbd/advanced_game_v4/advanced_2026.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")

    bridge.main()

    directories = list((tmp_path / bridge.OUT_DIR).iterdir())
    assert len(directories) == 1

    saved_dir = directories[0]
    feature_path = saved_dir / "pregame_features.parquet"
    audit_path = saved_dir / "audit.json"

    assert feature_path.is_file()
    assert audit_path.is_file()

    frame = pd.read_parquet(feature_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert len(frame) == 2
    assert audit["slate_games"] == 1
    assert audit["no_current_game_outcomes"] is True

    created = pd.Timestamp(audit["created_at_utc"])
    kickoff = pd.Timestamp(KICKOFF)

    assert created < kickoff
    assert created.tzinfo is not None
