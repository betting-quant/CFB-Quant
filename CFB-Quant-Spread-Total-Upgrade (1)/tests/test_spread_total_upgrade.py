import numpy as np
import pandas as pd
import pytest

from cfb_quant.spread_total.core import (
    ScoreDistribution, build_game_errors, prior_errors, simulate_scores,
    stable_seed, settle, expected_roi, empirical_crps, fair_summary,
)
from cfb_quant.spread_total.cli import _read_live_games, _read_markets


def points_oof():
    data = []
    for season in (2020, 2021, 2022, 2023, 2024):
        for i in range(4):
            game_id = season * 100 + i
            for side, pred, act in (("home", 28 + i, 30 + i), ("away", 20 + i, 21 + i)):
                data.append(dict(game_id=game_id, season=season, week=i+1,
                                 team=f"{side}-{game_id}", home_away=side,
                                 target="points", model="ridge", prediction=float(pred),
                                 actual=float(act), train_end_season=season-1))
    return pd.DataFrame(data)


def test_pair_and_priors_are_strictly_prior():
    out = build_game_errors(points_oof())
    assert len(out) == 20
    assert np.allclose(out.home_error, 2)
    assert np.allclose(out.away_error, 1)
    pool = prior_errors(out, before_season=2023, min_games=8, min_seasons=2)
    assert pool.season.max() == 2022
    assert len(pool) == 12


def test_chronology_guard_rejects_current_and_future_training():
    data = points_oof()
    data.loc[0, "train_end_season"] = data.loc[0, "season"]
    with pytest.raises(ValueError, match="Chronology"):
        build_game_errors(data)


def test_incomplete_pair_and_duplicate_rejected():
    data = points_oof()
    with pytest.raises(ValueError, match="one home and one away"):
        build_game_errors(data.iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        build_game_errors(pd.concat([data, data.iloc[:1]], ignore_index=True))


def test_future_results_cannot_change_prior_distribution():
    base = build_game_errors(points_oof())
    altered = base.copy()
    altered.loc[altered.season >= 2023, "home_error"] = 1000
    p1 = prior_errors(base, before_season=2023, min_games=8, min_seasons=2)
    p2 = prior_errors(altered, before_season=2023, min_games=8, min_seasons=2)
    s1 = simulate_scores(25, 24, p1, n=200, seed=99)
    s2 = simulate_scores(25, 24, p2, n=200, seed=99)
    np.testing.assert_array_equal(s1.home, s2.home)
    np.testing.assert_array_equal(s1.away, s2.away)


def test_coupled_errors_remain_coupled():
    pool = pd.DataFrame({"home_error": [-12, -4, 8], "away_error": [-12, -4, 8]})
    sim = simulate_scores(35, 35, pool, n=2000, seed=2)
    np.testing.assert_array_equal(sim.home, sim.away)


def test_spread_selection_orientation_and_push():
    # 20-17 (+3 margin), 21-21 (margin 0), 14-24 (-10 margin)
    d = ScoreDistribution(home=np.array([20,21,14]), away=np.array([17,21,24]))
    assert settle(d, market="spread", selection="home", line=-3) == pytest.approx((0, 1/3, 2/3))
    assert settle(d, market="spread", selection="away", line=3) == pytest.approx((2/3, 1/3, 0))
    assert settle(d, market="total", selection="under", line=37) == pytest.approx((0, 1/3, 2/3))
    assert settle(d, market="total", selection="over", line=37.5) == pytest.approx((2/3,0,1/3))


def test_ev_and_push():
    assert expected_roi(.60, 0, .40, -130) == pytest.approx(.60 * 100/130 - .40)
    assert expected_roi(.50, .25, .25, 150) == pytest.approx(.5*1.5-.25)
    with pytest.raises(ValueError):
        expected_roi(.8,.4,.4,-110)


def test_crps_is_zero_for_perfect_distribution():
    assert empirical_crps(np.array([44]*500), 44) == 0
    assert empirical_crps(np.array([44]*500), 45) == 1


def test_seed_is_repeatable():
    assert stable_seed(123) == stable_seed('123')
    assert stable_seed(123) != stable_seed(124)


def test_pregame_asof_and_market_quote_checks(tmp_path):
    games = tmp_path / "games.csv"
    pd.DataFrame([{
        "game_id": "123", "start_date": "2026-10-10T18:00:00Z",
        "home_team": "A", "away_team": "B", "home_points": 29,
        "away_points": 23,
    }]).to_csv(games,index=False)
    as_of = pd.Timestamp("2026-10-10T17:00:00Z")
    live = _read_live_games(str(games), as_of, 2026)
    assert len(live) == 1
    with pytest.raises(ValueError,match="started"):
        _read_live_games(str(games), pd.Timestamp("2026-10-10T19:00:00Z"), 2026)
    markets = tmp_path / "markets.csv"
    pd.DataFrame([{
        "game_id": "123", "market": "total", "selection": "under", "line": 60.5,
        "american_odds": -110, "bookmaker": "example", "quoted_at_utc": "2026-10-10T17:30:00Z",
    }]).to_csv(markets,index=False)
    with pytest.raises(ValueError,match="no later than"):
        _read_markets(str(markets), game_ids={"123"}, as_of_utc=as_of,
                      starts={"123": pd.Timestamp("2026-10-10T18:00:00Z")})
