import numpy as np
import pandas as pd
import pytest
from cfb_quant.spread_total.calibration_audit_v54 import _wilson, load_games, event_tables, margin_bias_bins, LINES


def data():
    r=[]
    for variant in ['pooled','phase_mixed']:
        for i in range(12):
            x={'game_id':str(i),'season':2023,'variant':variant,
               'phase':'regular_weeks_1_4' if i<6 else 'regular_week_5_plus',
               'max_residual_season':2022,'pred_margin':float(i-6),
               'actual_margin':float(i-5),'pred_total':50.0,
               'actual_total':54.0,'margin_80_covered':True,'total_80_covered':True}
            for event in LINES: x['p_'+event]=0.5
            r.append(x)
    return pd.DataFrame(r)


def test_wilson():
    a,b=_wilson(50,100)
    assert 0<a<.5<b<1


def test_event_gap_and_counts():
    s,b=event_tables(data())
    r=s.query("variant == 'pooled' and event == 'home_win'").iloc[0]
    assert r.games==12
    assert r.signed_gap_observed_minus_predicted==pytest.approx(0.0)
    assert b.query("variant=='pooled' and event=='home_win'").games.sum()==12


def test_unique_margin_bias():
    r=margin_bias_bins(data())
    assert r.query('phase == "ALL"').games.sum()==12


def test_reject_future_pool(tmp_path):
    d=data();d.loc[0,'max_residual_season']=2023
    fn=tmp_path/'bad.csv';d.to_csv(fn,index=False)
    with pytest.raises(ValueError,match='leakage'):load_games(fn)


def test_reject_unpaired(tmp_path):
    d=data().iloc[:-1];fn=tmp_path/'bad.csv';d.to_csv(fn,index=False)
    with pytest.raises(ValueError,match='paired'):load_games(fn)
