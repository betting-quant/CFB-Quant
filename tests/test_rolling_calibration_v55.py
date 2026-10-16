import numpy as np
import pandas as pd
import pytest
from cfb_quant.spread_total.rolling_calibration_v55 import (EVENTS,prepare,fit_prior,
                                                      transform,walk_forward,summarize)


def mock_games():
    rows=[]
    for year in (2020,2021,2022):
        for j in range(180):
            # ordered cumulative event probabilities +7.5 >= win >= -7.5
            p1=0.76+0.001*(j%10)
            p2=0.51+0.001*(j%10)
            p3=0.29+0.001*(j%10)
            actual=float((j%5)-2)*6
            rows.append({'game_id':str(year*1000+j),'season':year,'phase':'regular_week_5_plus',
                         'variant':'pooled','max_residual_season':year-1,
                         'actual_margin':actual,'p_home_covers_plus_7p5':p1,
                         'p_home_win':p2,'p_home_covers_minus_7p5':p3})
    return pd.DataFrame(rows)


def test_monotonicity_and_identity():
    p=np.array([[0.9,0.55,0.2],[0.7,0.3,0.1]])
    np.testing.assert_allclose(transform(p,0,1),p)
    adj=transform(p,0.13,1.3)
    assert (np.diff(adj,axis=1)<=0).all()


def test_fit_and_chronological_train():
    d=mock_games()
    games,audit=walk_forward(d)
    assert len(games)==540
    assert not bool(audit.iloc[0].calibration_applied)
    assert audit.iloc[1].max_calibration_season==2020
    assert audit.iloc[2].max_calibration_season==2021
    assert audit.iloc[2].prior_training_games==360
    assert np.isfinite(audit.slope).all()


def test_reject_future_pool(tmp_path):
    d=mock_games();d.loc[0,'max_residual_season']=2020
    path=tmp_path/'data.csv';d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='chronological'):prepare(path)


def test_reject_nonmonotone(tmp_path):
    d=mock_games();d.loc[0,'p_home_covers_minus_7p5']=0.99
    path=tmp_path/'data.csv';d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='order'):prepare(path)


def test_reject_duplicate_games(tmp_path):
    d=mock_games();d=pd.concat([d,d.iloc[:1]],ignore_index=True)
    path=tmp_path/'data.csv';d.to_csv(path,index=False)
    with pytest.raises(ValueError,match='Duplicate'):prepare(path)


def test_all_event_summaries():
    games,audit=walk_forward(mock_games())
    result=summarize(games,['season'])
    assert len(result)==9
    assert result.games.sum()==540*3
    assert np.isfinite(result.brier_delta_adjusted_minus_raw).all()


def test_shape_mismatch():
    with pytest.raises(ValueError,match='matching'):
        fit_prior(np.ones((150,3)),np.ones((149,3)))
