import numpy as np
import pandas as pd
import pytest
from cfb_quant.spread_total.phase_volatility_v53 import prepare_fbs, sample_indices, evaluate, summarize


def make_data():
    rows=[]; schedule=[]
    for year in (2019,2020,2021):
        for i in range(80):
            rows.append({'game_id':f'{year}{i:03d}', 'season':year,
                         'actual_margin_v3':3.0+i%5, 'actual_margin_v45':3.0+i%5,
                         'actual_total_v3':48.0+i%5, 'actual_total_v45':48.0+i%5,
                         'pred_margin_v3':3.0, 'pred_margin_v45':3.2,
                         'pred_total_v3':50.0, 'pred_total_v45':52.0})
            phase='postseason' if i<6 else 'regular'
            schedule.append({'season':year,'game_id':f'{year}{i:03d}',
                             'season_type':phase, 'scheduled_week':1 if i<35 else 8,
                             'universe_status':'fbs_fbs'})
    return pd.DataFrame(rows),pd.DataFrame(schedule)


def test_fbs_phase_and_postseason_precedence():
    d,u=make_data()
    u.loc[u.index[0],'universe_status']='other'
    z=prepare_fbs(d,u)
    assert len(z)==239
    assert (z.loc[z.game_id.eq('2019001'),'phase']=='postseason').all()
    assert (z.loc[z.game_id.eq('2019015'),'phase']=='regular_weeks_1_4').all()
    assert (z.loc[z.game_id.eq('2019060'),'phase']=='regular_week_5_plus').all()


def test_fails_on_missing_universe_record():
    d,u=make_data()
    with pytest.raises(ValueError,match='Not every prediction'):
        prepare_fbs(d,u.iloc[1:])


def test_phase_sampling_and_fallback():
    d,u=make_data()
    fbs=prepare_fbs(d,u)
    pool=fbs.loc[fbs.season==2019].reset_index(drop=True)
    test=fbs.loc[fbs.season==2020].reset_index(drop=True)
    a,b,audit=sample_indices(pool,test,samples=400,phase_share=.7,seed=11)
    assert a.shape==b.shape==(80,400)
    assert not np.array_equal(a,b)
    rows=np.flatnonzero(test.phase.to_numpy()=='postseason')
    assert np.array_equal(a[rows],b[rows]) # only 6 historical postseason examples; fallback
    assert audit.loc[audit.phase.eq('postseason'),'phase_sampling_applied'].eq(False).all()
    a2,b2,_=sample_indices(pool,test,samples=400,phase_share=.7,seed=11)
    assert np.array_equal(a,a2) and np.array_equal(b,b2)


def test_chronology_and_identity_of_means():
    d,u=make_data()
    fbs=prepare_fbs(d,u)
    games,audit=evaluate(fbs,samples=200,start_year=2020,end_year=2021)
    assert len(games)==320
    assert games.max_residual_season.lt(games.season).all()
    assert games.groupby(['season','game_id']).size().eq(2).all()
    pooled=games.loc[games.variant.eq('pooled')].sort_values(['season','game_id']).reset_index(drop=True)
    mixed=games.loc[games.variant.eq('phase_mixed')].sort_values(['season','game_id']).reset_index(drop=True)
    assert np.array_equal(pooled.pred_margin, mixed.pred_margin)
    assert np.array_equal(pooled.pred_total, mixed.pred_total)
    assert len(summarize(games,['variant','phase']))==6
    assert set(audit.max_residual_season)=={2019,2020}
