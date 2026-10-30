"""Regression tests for the 2026 Elo shadow grader."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/grade_elo_shadow_2026_v1.py"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_grader():
    spec = importlib.util.spec_from_file_location(
        "elo_shadow_grader_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_case(
    tmp_path,
    monkeypatch,
    *,
    completed=False,
    game_home="Home",
    observed="2026-11-01T00:00:00+00:00",
    home_points=24,
    away_points=17,
):
    module = load_grader()

    # Freeze the clock after all synthetic capture timestamps.
    # This changes test behavior only, never production behavior.
    real_timestamp = pd.Timestamp

    class ControlledTimestamp:
        def __call__(self, *args, **kwargs):
            return real_timestamp(*args, **kwargs)

        @staticmethod
        def now(tz=None):
            return real_timestamp(
                "2026-11-02T12:00:00Z"
            ).tz_convert(tz or "UTC")

    monkeypatch.setattr(
        module.pd,
        "Timestamp",
        ControlledTimestamp()
    )

    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()

    ledger = ledger_dir / "paired_forecasts.csv"
    pd.DataFrame([{
        "game_id": "12345",
        "kickoff_utc": "2026-10-31T18:00:00Z",
        "home_team": "Home",
        "away_team": "Away",
        "ridge_home_margin": 3.0,
        "elo_home_margin": 5.0,
        "elo_created_utc": "2026-10-30T15:00:00Z",
    }]).to_csv(ledger, index=False)

    ledger_manifest = ledger_dir / "manifest.json"
    ledger_manifest.write_text(
        json.dumps({
            "output_sha256": sha(ledger),
            "paired_forecasts": 1,
            "frozen_correction_applied": False,
            "production_approved": False,
        }),
        encoding="utf-8",
    )

    capture_dir = tmp_path / "capture"
    capture_dir.mkdir()

    raw = capture_dir / "games_2026.raw.json"
    raw.write_bytes(b"synthetic-test-payload")

    capture_manifest = capture_dir / "manifest.json"
    capture_manifest.write_text(
        json.dumps({
            "payload_sha256": sha(raw),
            "captured_at_utc": observed,
        }),
        encoding="utf-8",
    )

    game = SimpleNamespace(
        game_id="12345",
        kickoff_utc=pd.Timestamp(
            "2026-10-31T18:00:00Z"
        ).to_pydatetime(),
        home=game_home,
        away="Away",
        home_points=home_points if completed else None,
        away_points=away_points if completed else None,
    )

    monkeypatch.setattr(module, "LEDGER", ledger)
    monkeypatch.setattr(
        module, "LEDGER_MANIFEST", ledger_manifest
    )
    monkeypatch.setattr(
        module, "OUTPUT_ROOT", tmp_path / "output"
    )

    # The grader prints its destination relative to ROOT.
    # Keep both paths inside the isolated temporary workspace.
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def fake_merge(captures):
        assert len(captures) == 1
        return {
            "latest_capture_utc": pd.Timestamp(
                observed
            ).to_pydatetime(),
            "games": [game],
        }

    monkeypatch.setattr(
        module, "merge_fbs_captures", fake_merge
    )
    monkeypatch.setattr(
        sys, "argv", ["grader", str(capture_dir)]
    )

    return module, ledger, raw, capture_dir


def test_pending_game_is_not_graded(tmp_path, monkeypatch):
    module, _, _, _ = setup_case(
        tmp_path, monkeypatch
    )

    module.main()

    report = json.loads(
        next(
            (tmp_path / "output").glob("*/manifest.json")
        ).read_text(encoding="utf-8")
    )

    assert report["forecast_games"] == 1
    assert report["graded_games"] == 0
    assert report["pending_games"] == 1
    assert report["ridge_metrics"] is None
    assert report["elo_metrics"] is None


def test_completed_game_is_graded_correctly(
    tmp_path, monkeypatch
):
    module, _, _, _ = setup_case(
        tmp_path, monkeypatch, completed=True
    )

    module.main()

    report = json.loads(
        next(
            (tmp_path / "output").glob("*/manifest.json")
        ).read_text(encoding="utf-8")
    )

    assert report["graded_games"] == 1
    assert report["pending_games"] == 0
    assert report["ridge_metrics"]["mae"] == 4.0
    assert report["elo_metrics"]["mae"] == 2.0
    assert report["elo_mae_improvement_over_ridge"] == 2.0


def test_tampered_ledger_is_rejected(
    tmp_path, monkeypatch
):
    module, ledger, _, _ = setup_case(
        tmp_path, monkeypatch
    )

    with ledger.open("a", encoding="utf-8") as file:
        file.write("\nTAMPERED")

    with pytest.raises(RuntimeError, match="Ledger SHA256"):
        module.main()


def test_tampered_capture_is_rejected(
    tmp_path, monkeypatch
):
    module, _, raw, _ = setup_case(
        tmp_path, monkeypatch
    )

    raw.write_bytes(b"altered-capture")

    with pytest.raises(
        RuntimeError, match="Capture payload integrity"
    ):
        module.main()


def test_team_identity_mismatch_is_rejected(
    tmp_path, monkeypatch
):
    module, _, _, _ = setup_case(
        tmp_path, monkeypatch, game_home="Different Home"
    )

    with pytest.raises(
        RuntimeError, match="Identity or kickoff mismatch"
    ):
        module.main()


def test_capture_before_kickoff_cannot_grade(
    tmp_path, monkeypatch
):
    module, _, _, _ = setup_case(
        tmp_path,
        monkeypatch,
        completed=True,
        observed="2026-10-31T17:00:00+00:00",
    )

    with pytest.raises(
        RuntimeError, match="not post-kickoff"
    ):
        module.main()


def test_repeat_grading_cannot_overwrite(
    tmp_path, monkeypatch
):
    module, _, _, _ = setup_case(
        tmp_path, monkeypatch
    )

    module.main()

    with pytest.raises(
        RuntimeError, match="already graded"
    ):
        module.main()
