import pandas as pd
import pytest

from cfb_quant.spread_total.ablation_v4 import family, variant_sets, metrics_for_predictions


def test_families():
    assert family('adv_offense_passing_ppa_lag1') == 'efficiency'
    assert family('adv_defense_rushing_success_rate_avg3') == 'efficiency'
    assert family('adv_defense_passing_explosiveness_season_avg') == 'explosiveness'
    assert family('adv_offense_line_yards_avg5') == 'trenches'
    assert family('adv_defense_stuff_rate_lag1') == 'trenches'
    assert family('adv_offense_plays_avg3') == 'tempo'
    with pytest.raises(ValueError):
        family('adv_undocumented_mystery_lag1')


def test_variant_sets_preserve_baseline():
    basic = ['off_points_avg3','def_allowed_points_avg3']
    advanced = ['adv_offense_ppa_lag1','adv_offense_explosiveness_lag1',
                'adv_offense_line_yards_lag1','adv_offense_plays_lag1']
    v = variant_sets(basic, advanced)
    assert v['baseline'] == basic
    assert len(v['all_advanced']) == 6
    assert v['efficiency_trenches'] == basic + [advanced[0],advanced[2]]
    assert len(v) == 7


def test_variant_sets_reject_overlap():
    with pytest.raises(ValueError):
        variant_sets(['adv_offense_ppa_lag1'], ['adv_offense_ppa_lag1'])


def test_metrics_pair_and_scoring():
    rows = []
    for side, team, prediction, actual in [('home','H',31,28),('away','A',21,17)]:
        rows.append(dict(game_id=123,season=2019,week=4,team=team,home_away=side,
                         target='points',model='ridge',prediction=prediction,
                         actual=actual,train_end_season=2018))
    out = metrics_for_predictions(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0].actual_margin == 11
    assert out.iloc[0].pred_margin == 10
    assert out.iloc[0].margin_abs_error == 1
    assert out.iloc[0].total_abs_error == 7
