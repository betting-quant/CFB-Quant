import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from cfb_quant.spread_total.advanced_v4 import (
    flatten_records, make_lagged_features, years_from_arg, fetch_one,
)


def source(gid, team, ppa, year=2024):
    return {"gameId": gid, "team": team, "offense": {"ppa": ppa, "successRate": .5},
            "defense": {"ppa": -ppa, "successRate": .4}}


def base_frame():
    return pd.DataFrame([
        {"game_id": "10", "team":"Team A", "season":2024, "kickoff_utc":"2024-09-08T18:00:00Z", "x": 1},
        {"game_id": "9", "team":"Team A", "season":2024, "kickoff_utc":"2024-09-01T18:00:00Z", "x": 2},
        {"game_id": "11", "team":"Team A", "season":2025, "kickoff_utc":"2025-08-30T18:00:00Z", "x": 3},
        {"game_id": "10", "team":"Team B", "season":2024, "kickoff_utc":"2024-09-08T18:00:00Z", "x": 4},
    ])


def test_kickoff_before_gameid_order_and_season_shift():
    base = base_frame()
    advanced = flatten_records([source(9,"Team A", 2.0),source(10,"Team A", 9.0),source(11,"Team A", 40.0),source(10,"Team B", 3.0)])
    out, feats, coverage = make_lagged_features(base, advanced)
    assert "adv_offense_ppa_lag1" in feats
    a = out.set_index(["game_id", "team"])
    assert np.isnan(a.loc[("9", "Team A"), "adv_offense_ppa_lag1"])
    assert a.loc[("10", "Team A"), "adv_offense_ppa_lag1"] == 2.0
    assert a.loc[("11", "Team A"), "adv_offense_ppa_lag1"] == 9.0
    assert np.isnan(a.loc[("11", "Team A"), "adv_offense_ppa_season_avg"])
    assert "raw_adv_offense_ppa" not in out.columns
    assert len(out)==len(base)


def test_future_result_does_not_change_prior_features():
    base = base_frame()
    a = flatten_records([source(9,"Team A", 2.0),source(10,"Team A", 9.0),source(11,"Team A", 40.0)])
    b = flatten_records([source(9,"Team A", 2.0),source(10,"Team A", 9.0),source(11,"Team A", 999.0)])
    outa, cols, _ = make_lagged_features(base, a)
    outb, _, _ = make_lagged_features(base, b)
    past_a = outa[outa.game_id.isin(["9", "10"])].sort_values(["game_id","team"])[cols].reset_index(drop=True)
    past_b = outb[outb.game_id.isin(["9", "10"])].sort_values(["game_id","team"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(past_a, past_b)


def test_duplicate_source_rows_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        flatten_records([source(1,"A", 1), source(1,"A", 2)])


def test_no_missing_kickoff():
    base = base_frame()
    base.loc[0,"kickoff_utc"] = None
    with pytest.raises(ValueError, match="Null kickoff"):
        make_lagged_features(base, flatten_records([source(9,"Team A", 2.0)]))


def test_year_range():
    assert years_from_arg("2022-2024")==[2022,2023,2024]
    assert years_from_arg("2024,2022,2024")==[2022,2024]


def test_cached_download_avoids_network(tmp_path):
    cache = tmp_path / "data/raw/cfbd/advanced_game_v4/advanced_2023.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps([source(1,"Team A", 1.0)]))
    fake = MagicMock()
    n, mode = fetch_one(tmp_path, 2023, session=fake)
    assert (n,mode)==(1,"cached")
    fake.get.assert_not_called()
