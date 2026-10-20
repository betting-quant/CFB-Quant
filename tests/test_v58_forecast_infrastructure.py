"""Offline regression tests for CFB-Quant v5.8 infrastructure."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from cfb_quant.live import daily_schedule_sync_v58 as sync
from cfb_quant.live import forecast_coverage_v58 as coverage
from cfb_quant.live import forecast_authenticity_v58 as auth


GAME_ID = "401000001"


def synthetic_game(*, tbd=False, home="Kansas"):
    future = datetime.now(timezone.utc) + timedelta(days=10)
    # Select by Eastern date, rather than assuming UTC calendar date.
    return {
        "id": int(GAME_ID),
        "season": 2026,
        "startDate": future.isoformat(),
        "startTimeTBD": tbd,
        "completed": False,
        "homeTeam": home,
        "awayTeam": "Baylor",
        "homeId": 100,
        "awayId": 200,
        "homeClassification": "fbs",
        "awayClassification": "fbs",
    }


def make_inputs(tmp_path, *, home="Kansas"):
    game = synthetic_game(home=home)
    date = sync.kickoff(game).astimezone(
        sync.EASTERN
    ).date().isoformat()

    snapshot = tmp_path / "schedule.json"
    snapshot.write_text(json.dumps([game]), encoding="utf-8")

    csv = tmp_path / "paired_forecasts.csv"
    pd.DataFrame([{
        "game_id": GAME_ID,
        "kickoff_utc": sync.kickoff(game).isoformat(),
        "away": "Baylor",
        "home": "Kansas",
        "fair_home_spread": -3.0,
        "fair_total": 51.0,
        "blend_home_spread": -3.2,
    }]).to_csv(csv, index=False)

    return snapshot, csv, date


def fake_authentication(monkeypatch, *, status="GIT_PREKICKOFF_SUPPORTED"):
    def verify(_root, archives):
        return {
            "remote_head": "synthetic-test-only",
            "results": [
                {
                    "status": status,
                    "errors": (
                        [] if status == "GIT_PREKICKOFF_SUPPORTED"
                        else ["Synthetic authenticity failure"]
                    ),
                    "csv_sha256": coverage.sha(path),
                }
                for path in archives
            ],
        }

    monkeypatch.setattr(auth, "verify_all", verify)


def test_duplicate_schedule_game_rejected():
    game = synthetic_game()
    with pytest.raises(ValueError, match="Duplicate game ID"):
        sync.index_games([game, dict(game)])


def test_schedule_classifies_fbs_fcs():
    game = synthetic_game()
    game["awayClassification"] = "fcs"
    assert sync.matchup_type(game) == "FBS_FCS"


def test_schedule_tbd_not_treated_as_confirmed():
    game = synthetic_game(tbd=True)
    assert game["startTimeTBD"] is True
    # Inclusion in a schedule registry does not establish an exact kickoff.
    selected = sync.select_window(
        {GAME_ID: game},
        "2026-01-01",
        "2026-12-31",
    )
    assert GAME_ID in selected
    assert selected[GAME_ID]["startTimeTBD"] is True


def test_duplicate_forecast_archive_rejected(tmp_path):
    _, csv, _ = make_inputs(tmp_path)
    with pytest.raises(ValueError, match="multiple forecast archives"):
        coverage.load_archives([csv, csv])


def test_nonfinite_forecast_rejected(tmp_path):
    _, csv, _ = make_inputs(tmp_path)
    frame = pd.read_csv(csv)
    frame.loc[0, "fair_total"] = np.inf
    frame.to_csv(csv, index=False)

    with pytest.raises(ValueError, match="Nonfinite forecasts"):
        coverage.load_archives([csv])


def test_valid_coverage_accepted(tmp_path, monkeypatch):
    snapshot, csv, day = make_inputs(tmp_path)
    fake_authentication(monkeypatch)

    result = coverage.run(
        snapshot, [csv], day, day, tmp_path / "valid"
    )

    assert result["total_games"] == 1
    assert result["statuses"]["ARCHIVED"] == 1
    assert result["all_fbs_fbs_forecasted"] is True


def test_missing_forecast_is_reported(tmp_path, monkeypatch):
    snapshot, _, day = make_inputs(tmp_path)
    fake_authentication(monkeypatch)

    result = coverage.run(
        snapshot, [], day, day, tmp_path / "missing"
    )

    assert result["statuses"]["MISSING_PREGAME_FORECAST"] == 1
    assert result["all_fbs_fbs_forecasted"] is False


def test_incorrect_team_identity_rejected(tmp_path, monkeypatch):
    snapshot, csv, day = make_inputs(tmp_path, home="WRONG TEAM")
    fake_authentication(monkeypatch)

    with pytest.raises(RuntimeError, match="team identity mismatch"):
        coverage.run(
            snapshot, [csv], day, day, tmp_path / "wrong_team"
        )


def test_failed_authenticity_rejected(tmp_path, monkeypatch):
    snapshot, csv, day = make_inputs(tmp_path)
    fake_authentication(
        monkeypatch, status="AUTHENTICITY_UNVERIFIED"
    )

    with pytest.raises(RuntimeError, match="Forecast authenticity rejected"):
        coverage.run(
            snapshot, [csv], day, day, tmp_path / "bad_auth"
        )


def test_invalid_schedule_timestamp_rejected():
    game = synthetic_game()
    game["startDate"] = "2026-10-24T12:00:00"

    with pytest.raises(ValueError, match="Naive kickoff"):
        sync.kickoff(game)
