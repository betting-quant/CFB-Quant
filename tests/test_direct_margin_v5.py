import numpy as np
import pandas as pd
import pytest

from cfb_quant.spread_total.direct_margin_v5 import (
    paired_game_frame, season_blocked_margin_oof, compare_to_saved_baselines,
)


def rows_for_years():
    rows=[]
    for year in range(2014,2022):
        for j in range(4):
            g=f'{year}{j}'
            home=f'H{j}'; away=f'A{j}'
            kick=f'{year}-09-{j+1:02d}T16:00:00Z'
            for side,team,opp,pts,strength in (
                ('home',home,away,21+year%5+j,2+j),
                ('away',away,home,14+year%3+j,-1-j),
            ):
                rows.append(dict(game_id=g,season=year,week=j+1,kickoff_utc=kick,team=team,
                                 opponent=opp,home_away=side,points=pts,team_games_before=1,
                                 fo=float(strength),bar=float(j)*0.1))
    return pd.DataFrame(rows)


def test_pairing_correct_difference_and_margin():
    d=rows_for_years()
    x=paired_game_frame(d,['fo','bar'])
    g=x.loc[x.game_id.eq('20190')].iloc[0]
    assert g.home_team=='H0' and g.away_team=='A0'
    assert g.diff_fo==3.0
    assert g.actual_margin==(21+2019%5)-(14+2019%3)
    assert len(x)==len(d)//2


def test_incomplete_pair_and_opponent_mismatch_rejected():
    d=rows_for_years()
    with pytest.raises(ValueError,match='Incomplete'):
        paired_game_frame(d.drop(index=d.index[0]),['fo'])
    broken=d.copy()
    broken.loc[0,'opponent']='WRONG'
    with pytest.raises(ValueError,match='identity mismatch'):
        paired_game_frame(broken,['fo'])


def test_current_game_points_not_allowed_as_feature():
    with pytest.raises(ValueError,match='Forbidden'):
        paired_game_frame(rows_for_years(),['fo','points'])


def test_fold_is_chronological_and_future_y_does_not_affect_prior_predictions():
    d=rows_for_years()
    games=paired_game_frame(d,['fo','bar'])
    columns=['diff_fo','diff_bar']
    a=season_blocked_margin_oof(games,columns,first_year=2019,last_year=2021)
    altered=games.copy()
    altered.loc[altered.season.eq(2021),'actual_margin'] += 1000
    b=season_blocked_margin_oof(altered,columns,first_year=2019,last_year=2021)
    assert a.train_end_season.lt(a.season).all()
    np.testing.assert_allclose(a.v5_pred_margin, b.v5_pred_margin,atol=1e-10)


def test_compare_requires_matching_games_and_scores():
    d=rows_for_years()
    games=paired_game_frame(d,['fo','bar'])
    a=season_blocked_margin_oof(games,['diff_fo','diff_bar'],first_year=2019,last_year=2020)
    old=pd.DataFrame({
        'game_id':a.game_id, 'season':a.season,
        'actual_margin_v3':a.actual_margin, 'actual_margin_v45':a.actual_margin,
        'pred_margin_v3':a.v5_pred_margin+2, 'pred_margin_v45':a.v5_pred_margin+1,
    })
    pair,by,overall=compare_to_saved_baselines(a,old)
    assert len(pair)==8 and len(by)==2 and overall.games.iloc[0]==8
    bad=old.iloc[1:].copy()
    with pytest.raises(ValueError,match='Different evaluation games'):
        compare_to_saved_baselines(a,bad)
    bad=old.copy();bad.loc[bad.index[0],'actual_margin_v3']=999
    with pytest.raises(ValueError,match='Actual scores differ'):
        compare_to_saved_baselines(a,bad)


def test_kickoff_required_and_no_same_game_feature_leakage():
    d=rows_for_years()
    d.loc[d.index[0],'kickoff_utc']=None
    with pytest.raises(ValueError,match='Kickoff'):
        paired_game_frame(d,['fo','bar'])
