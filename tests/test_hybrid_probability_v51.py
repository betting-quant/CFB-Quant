import numpy as np
import pandas as pd
import pytest

from cfb_quant.spread_total.hybrid_probability_v51 import (
    paired_scoring_residuals, simulate_scores, empirical_crps,run_experiment
)


def fake_games():
    a=[]
    for year in (2019,2020,2021):
        for i in range(60):
            am=float((i%7)*3-6)
            at=float(35+i%20)
            a.append(dict(game_id=year*1000+i,season=year,
                actual_margin_v3=am,actual_margin_v45=am,actual_total_v3=at,actual_total_v45=at,
                pred_margin_v3=am+3,pred_margin_v45=am+1,
                pred_total_v3=at+2,pred_total_v45=at+4))
    return pd.DataFrame(a)


def test_paired_residuals_keep_corresponding_game_rows():
    d=fake_games()
    r=paired_scoring_residuals(d,'pred_margin_v45','pred_total_v3')
    assert r.shape==(len(d),2)
    assert np.allclose(r.sum(axis=1),-2.0)
    assert np.allclose(r[:,0]-r[:,1],-1.0)


def test_joint_scores_are_coherent_nonnegative():
    h=np.array([[0.,-90.],[10.,20.]])
    pm=np.array([-5., 6.]); pt=np.array([15., 40.])
    m,t=simulate_scores(pm,pt,h,np.array([[1,0],[0,1]]))
    assert m.shape==(2,2)
    assert np.all(t>=np.abs(m)-1e-10)
    assert np.isfinite(t).all()


def test_crps_perfect_point_forecast():
    sample=np.tile(np.array([5.]),(1,20))
    assert np.allclose(empirical_crps(sample,np.array([5.])),0.0)
    assert np.allclose(empirical_crps(sample,np.array([6.])),1.0)


def test_only_past_residuals_used():
    d=fake_games()
    g1,_,_=run_experiment(d,start_year=2020,end_year=2021,samples=200,seed=123)
    changed=d.copy()
    # Changing 2021 actual results must not change the 2021 simulated probabilities.
    idx=changed.season==2021
    changed.loc[idx,['actual_margin_v3','actual_margin_v45']]+=25
    changed.loc[idx,['actual_total_v3','actual_total_v45']]+=25
    g2,_,_=run_experiment(changed,start_year=2020,end_year=2021,samples=200,seed=123)
    a=g1.loc[g1.season==2021,['variant','game_id','p_home_win','p_over_55p5']].reset_index(drop=True)
    b=g2.loc[g2.season==2021,['variant','game_id','p_home_win','p_over_55p5']].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    assert (g1.residual_pool_seasons_max<g1.season).all()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        run_experiment(fake_games(),start_year=2019,end_year=2020)
