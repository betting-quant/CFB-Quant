import json
import numpy as np
import pandas as pd
import pytest
from cfb_quant.spread_total.fbs_calibration_v52 import (
    load_schedule, attach_universe, bin_calibration, season_phase,
)
from cfb_quant.spread_total.hybrid_probability_v51 import run_experiment


def test_cached_classification_and_status(tmp_path):
    d = tmp_path/'data/raw/cfbd/games'
    d.mkdir(parents=True)
    (d/'games_2020.json').write_text(json.dumps([
        {'id':11,'homeClassification':'fbs','awayClassification':'fbs','week':1,'seasonType':'regular'},
        {'id':12,'homeClassification':'fbs','awayClassification':'fcs','week':2,'seasonType':'regular'},
        {'id':13,'homeClassification':'fbs','week':3,'seasonType':'regular'},
    ]))
    sch=load_schedule([2020],tmp_path)
    fr=attach_universe(pd.DataFrame({'game_id':[11,12,13,14], 'season':[2020]*4}),sch)
    assert fr.universe_status.tolist()==['fbs_fbs','other','unknown','unknown']


def test_duplicate_schedule_rejected():
    d=pd.DataFrame({'game_id':['1','1'], 'season':[2020,2020]})
    with pytest.raises(ValueError):
        attach_universe(pd.DataFrame({'game_id':['1'],'season':[2020]}),d)


def test_bins_no_future_outcomes_leak_into_probability():
    d=pd.DataFrame({'variant':['hybrid']*2,'season':[2020,2020],
                    'actual_margin':[3.,-5.],'actual_total':[70.,40.],
                    'p_home_win':[.7,.2],'p_home_covers_minus_7p5':[.3,.1],
                    'p_home_covers_plus_7p5':[.9,.6],
                    'p_over_45p5':[.9,.1], 'p_over_55p5':[.8,.1], 'p_over_65p5':[.6,.1]})
    cal=bin_calibration(d)
    row=cal[(cal.event=='home_win')&(cal.bin==7)].iloc[0]
    assert row.observed_frequency==1 and row["count"]==1
    assert cal.small_bin.all()


def test_phase_classification():
    g=pd.DataFrame({'variant':['hybrid']*3,'season':[2020]*3,'game_id':['1','2','3'],
                    'margin_abs_error':[1,2,3], 'total_abs_error':[1,2,3],
                    'margin_crps':[1,2,3], 'total_crps':[1,2,3],
                    'margin_80_covered':[1,1,0], 'total_80_covered':[1,0,1]})
    s=pd.DataFrame({'season':[2020]*3,'game_id':['1','2','3'],
                    'season_type':['regular','regular','postseason'], 'scheduled_week':[1,7,1]})
    p=season_phase(g,s)
    assert set(p.phase)=={'regular_weeks_1_4','regular_week_5_plus','postseason'}


def test_fbs_only_calibration_pool():
    data=[]
    for year in [2019,2020,2021]:
        for i in range(70):
            actual=float(i%20-5)
            total=float(45+i%15)
            data.append({'game_id':year*1000+i,'season':year,
                'actual_margin_v3':actual,'actual_margin_v45':actual,
                'actual_total_v3':total,'actual_total_v45':total,
                'pred_margin_v3':actual+3,'pred_margin_v45':actual+2,
                'pred_total_v3':total+4,'pred_total_v45':total+5,
                'is_fbs':i<60})
    all_games=pd.DataFrame(data)
    filtered=all_games[all_games.is_fbs].drop(columns='is_fbs')
    games,_,_=run_experiment(filtered,start_year=2020,end_year=2021,samples=200)
    assert games.loc[games.season==2020, 'calibration_pool_games'].eq(60).all()
    assert games.loc[games.season==2021, 'calibration_pool_games'].eq(120).all()
    assert len(games)==60*2*3
