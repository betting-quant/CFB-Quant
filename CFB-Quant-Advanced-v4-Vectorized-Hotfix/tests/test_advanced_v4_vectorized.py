"""Regression tests for fast v4 feature construction and temporal isolation."""
import numpy as np
import pandas as pd
from cfb_quant.spread_total.advanced_v4 import make_lagged_features


def fixture():
    base=[]
    advanced=[]
    for team in ['A','B']:
        for i, (year, date) in enumerate([(2023,'2023-09-01'),(2023,'2023-09-08'),(2023,'2023-09-15'),(2023,'2023-09-22'),(2024,'2024-09-01'),(2024,'2024-09-08')]):
            gid=str(i+10)
            base.append(dict(game_id=gid, team=team, season=year, kickoff_utc=date+'T19:00:00Z'))
            value = np.nan if i==1 else (100 if team=='B' else 0) + i + 1
            advanced.append(dict(game_id=gid,team=team,raw_adv_offense_ppa=value,raw_adv_defense_ppa=value*2))
    base=pd.DataFrame(base).sample(frac=1,random_state=51).reset_index(drop=True)
    return base,pd.DataFrame(advanced)


def test_vector_features_match_reference_pandas():
    base,advanced=fixture()
    output,cols,coverage=make_lagged_features(base,advanced)
    actual=output.set_index(['team','game_id'])
    for team in ['A','B']:
        for metric in ['offense_ppa','defense_ppa']:
            original=advanced.loc[advanced.team.eq(team), 'raw_adv_'+metric].reset_index(drop=True)
            seasons=base.loc[base.team.eq(team),'season']
            # Align known fixture order by game_id instead of shuffled base input.
            season_order=pd.Series([2023]*4+[2024]*2)
            shifted=original.shift(1)
            ref3=shifted.rolling(3,min_periods=1).mean()
            ref5=shifted.rolling(5,min_periods=1).mean()
            season_shift=original.groupby(season_order).shift(1)
            season_avg=season_shift.groupby(season_order).expanding(min_periods=1).mean().reset_index(level=0,drop=True)
            for i in range(6):
                key=(team,str(i+10))
                for suffix,expect in [('lag1',shifted[i]),('avg3',ref3[i]),('avg5',ref5[i]),('season_avg',season_avg[i])]:
                    got=actual.loc[key,f'adv_{metric}_{suffix}']
                    assert np.isclose(got,expect,equal_nan=True), (key,metric,suffix,got,expect)
    assert len(cols)==8
    assert len(output)==12
    assert not any(x.startswith('raw_adv_') for x in output.columns)
    assert coverage.team_games.sum()==12


def test_future_stat_change_does_not_affect_earlier_games():
    base,advanced=fixture()
    initial,cols,_=make_lagged_features(base,advanced)
    advanced.loc[(advanced.team=='A') & (advanced.game_id=='15'),'raw_adv_offense_ppa']=10000000
    altered,_,_=make_lagged_features(base,advanced)
    earlier=['10','11','12','13','14','15']
    a=initial[(initial.team=='A') & initial.game_id.isin(earlier)].sort_values('game_id')
    b=altered[(altered.team=='A') & altered.game_id.isin(earlier)].sort_values('game_id')
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True),b[cols].reset_index(drop=True))


def test_expanding_average_keeps_prior_history_after_null():
    base,advanced=fixture()
    out,_,_=make_lagged_features(base,advanced)
    rows=out.set_index(['team','game_id'])
    assert rows.loc[('A','12'),'adv_offense_ppa_season_avg']==1
    assert rows.loc[('A','13'),'adv_offense_ppa_season_avg']==2
    assert np.isnan(rows.loc[('A','14'),'adv_offense_ppa_season_avg'])
