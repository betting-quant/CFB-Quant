from __future__ import annotations

import hashlib
import json
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cfb_quant.live.predict_v57 import (
    game_key, game_forecasts, add_probabilities, load_manifest,
    verify_snapshot, train_points, historical_fbs_residuals,
    V3_MANIFEST,V45_MANIFEST,V3_DATA, PAIRED,UNIVERSE,SCHEDULE,ADVANCED,
)



@pytest.fixture(autouse=True)
def parquet_fixture_io(monkeypatch):
    """Pure logic tests can run in test environments lacking optional pyarrow.

    Actual deployment uses pandas' real Parquet implementation in CFB-Quant.
    """
    def save(self,path,*args,**kwargs):
        return self.to_pickle(path)
    def read(path,*args,**kwargs):
        return pd.read_pickle(path)
    monkeypatch.setattr(pd.DataFrame,'to_parquet',save)
    monkeypatch.setattr(pd,'read_parquet',read)


def test_game_id_normalization():
    assert game_key(401123) == '401123'
    assert game_key('401123.0') == '401123'
    with pytest.raises(ValueError):game_key('Texas')
    with pytest.raises(ValueError):game_key(None)


def test_sign_convention_and_hybrid_scores():
    teams=pd.DataFrame([
        {'game_id':'401123','kickoff_utc':'2026-10-10T19:30:00Z','team':'Alabama','opponent':'Georgia','home_away':'home'},
        {'game_id':'401123','kickoff_utc':'2026-10-10T19:30:00Z','team':'Georgia','opponent':'Alabama','home_away':'away'},
    ])
    f=game_forecasts(teams,np.array([30,20]),np.array([28,21]))
    assert f.fair_margin_home.iloc[0]==7
    assert f.fair_home_spread.iloc[0]==-7
    assert f.fair_total.iloc[0]==50
    assert f.projected_home_points.iloc[0]==28.5
    assert f.projected_away_points.iloc[0]==21.5


def test_negative_projection_rejected():
    teams=pd.DataFrame([
        {'game_id':1,'kickoff_utc':'2026-10-10Z','team':'A','opponent':'B','home_away':'home'},
        {'game_id':1,'kickoff_utc':'2026-10-10Z','team':'B','opponent':'A','home_away':'away'},
    ])
    with pytest.raises(ValueError,match='negative team scores'):
        game_forecasts(teams,np.array([1,1]),np.array([70,0]))


def test_probabilities_are_ordered_and_repeatable():
    f=pd.DataFrame([{'game_id':'1','projected_home_points':30.,'projected_away_points':21.}])
    r=np.random.default_rng(12).normal(0,10,(1100,2))
    p=add_probabilities(f,r,samples=5000,seed=12,home_line=14.5,total_line=55.5)
    q=add_probabilities(f,r,samples=5000,seed=12,home_line=14.5,total_line=55.5)
    assert p.equals(q)
    assert 0<=p.p_home_win.iloc[0]<=1
    assert p.p_home_covers_plus_7p5.iloc[0]>=p.p_home_win.iloc[0]>=p.p_home_covers_minus_7p5.iloc[0]
    assert p.p_over_45p5.iloc[0]>=p.p_over_55p5.iloc[0]>=p.p_over_65p5.iloc[0]
    assert p.p_home_covers_custom_line.iloc[0]>=p.p_home_win.iloc[0]
    assert p.margin_p10.iloc[0]<p.margin_p90.iloc[0]


def test_only_half_point_market_lines_allowed():
    f=pd.DataFrame([{'game_id':'1','projected_home_points':28.,'projected_away_points':21.}])
    r=np.zeros((120,2))
    with pytest.raises(ValueError,match='half-point'):
        add_probabilities(f,r,samples=1000,home_line=7.0)
    with pytest.raises(ValueError,match='half-point'):
        add_probabilities(f,r,samples=1000,total_line=40.0)


def _snapshot_fixture(tmp_path):
    root=tmp_path
    (root/SCHEDULE).parent.mkdir(parents=True)
    (root/ADVANCED).parent.mkdir(parents=True)
    (root/SCHEDULE).write_text('[{"id":401123}]')
    (root/ADVANCED).write_text('[]')
    a=[f'f{i}' for i in range(267)]
    b=a+[f'oa{i}' for i in range(20)]
    folder=root/'snapshot';folder.mkdir()
    kickoff='2026-10-10T19:30:00Z'
    payload=[]
    for side,team,opp,tid in [('home','Alabama','Georgia',1),('away','Georgia','Alabama',2)]:
        payload.append(dict(game_id=401123,season=2026,kickoff_utc=kickoff,team=team,
                            opponent=opp,home_away=side,team_id=tid,points=np.nan,
                            opponent_points=np.nan,**{n:float(i+1) for i,n in enumerate(b)}))
    pd.DataFrame(payload).to_parquet(folder/'pregame_features.parquet')
    audit=dict(v3_historical_parity=True,v45_historical_parity=True,no_current_game_outcomes=True,
               feature_count=287,slate_team_rows=2,slate_games=1,
               asof_utc='2026-10-10T08:28:00Z',created_at_utc='2026-10-10T08:28:01Z',
               slate_date_eastern='2026-10-10',
               schedule_sha256=hashlib.sha256((root/SCHEDULE).read_bytes()).hexdigest(),
               advanced_2026_sha256=hashlib.sha256((root/ADVANCED).read_bytes()).hexdigest())
    (folder/'audit.json').write_text(json.dumps(audit))
    return root,folder,a,b


def test_snapshot_validates_and_captures_hash(tmp_path):
    root,folder,a,b=_snapshot_fixture(tmp_path)
    frame,audit,details=verify_snapshot(root,folder,pd.Timestamp('2026-10-10T10:00:00Z'),a,b)
    assert len(frame)==2 and details['eligible_games']==1
    assert details['excluded_started_games']==0
    assert len(details['snapshot_sha256'])==64


def test_snapshot_rejects_mutated_source(tmp_path):
    root,folder,a,b=_snapshot_fixture(tmp_path)
    (root/SCHEDULE).write_text('[{"id":401124}]')
    with pytest.raises(ValueError,match='schedule_sha256'):
        verify_snapshot(root,folder,pd.Timestamp('2026-10-10T10:00:00Z'),a,b)


def test_snapshot_rejects_current_game_results(tmp_path):
    root,folder,a,b=_snapshot_fixture(tmp_path)
    data=pd.read_parquet(folder/'pregame_features.parquet')
    data.loc[0,'points']=17.0
    data.to_parquet(folder/'pregame_features.parquet')
    with pytest.raises(ValueError,match='same-game outcomes'):
        verify_snapshot(root,folder,pd.Timestamp('2026-10-10T10:00:00Z'),a,b)


def test_snapshot_skips_started_games_by_default(tmp_path):
    root,folder,a,b=_snapshot_fixture(tmp_path)
    with pytest.raises(ValueError,match='already kicked off'):
        verify_snapshot(root,folder,pd.Timestamp('2026-10-10T20:00:00Z'),a,b)
    data,_,details=verify_snapshot(root,folder,pd.Timestamp('2026-10-10T20:00:00Z'),a,b,allow_started=True)
    assert len(data)==2
    assert details['started_games_included_as_archive']==1


def test_snapshot_rejects_bad_feature_values(tmp_path):
    root,folder,a,b=_snapshot_fixture(tmp_path)
    data=pd.read_parquet(folder/'pregame_features.parquet')
    data.loc[0,'oa19']=np.inf
    data.to_parquet(folder/'pregame_features.parquet')
    with pytest.raises(ValueError,match='infinite features'):
        verify_snapshot(root,folder,pd.Timestamp('2026-10-10T10:00:00Z'),a,b)


def test_training_ignores_extra_seasons_and_never_fits_2026(tmp_path, monkeypatch):
    from cfb_quant.live import predict_v57 as live

    path=tmp_path/V3_DATA
    path.parent.mkdir(parents=True)
    historical=[{'game_id':i, 'team':'A', 'season':2014+i%12,
                 'team_games_before':1, 'points':20., 'f1':1.}
                for i in range(10080)]
    extra=[{'game_id':10080+i, 'team':'A', 'season':year,
            'team_games_before':1, 'points':1000., 'f1':1.}
           for i,year in enumerate([2013,2026,2026])]
    pd.DataFrame(historical+extra).to_parquet(path)

    class SpyModel:
        def fit(self,x,y):
            self.train_rows=len(x)
            self.max_target=float(y.max())
            return self
    monkeypatch.setattr(live,'_make_ridge',lambda target:SpyModel())
    model,audit=live.train_points(tmp_path,['f1'],V3_DATA)
    assert model.train_rows==10080
    assert model.max_target==20.
    assert audit['excluded_outside_2014_2025']=={'2013':1,'2026':2}
    assert audit['training_seasons']=='2014-2025'


def test_training_fails_if_one_historical_year_missing(tmp_path):
    path=tmp_path/V3_DATA
    path.parent.mkdir(parents=True)
    rows=[{'game_id':i,'team':'A','season':2014+i%11,
           'team_games_before':1,'points':20.,'f1':1.}
          for i in range(10120)]
    pd.DataFrame(rows).to_parquet(path)
    with pytest.raises(ValueError,match='Incomplete historical 2014-2025 training years'):
        train_points(tmp_path,['f1'],V3_DATA)


def test_fbs_only_residual_pool(tmp_path):
    root=tmp_path
    (root/PAIRED).parent.mkdir(parents=True)
    (root/UNIVERSE).parent.mkdir(parents=True)
    rows=[];look=[]
    for i in range(4200):
        year=2019+(i%6)
        rows.append(dict(game_id=i+1000,season=year,pred_margin_v45=7.,pred_total_v3=55.,
                         actual_margin_v3=8.,actual_total_v3=58.))
        look.append(dict(game_id=i+1000,season=year,universe_status='fbs_fbs'))
    rows.append(dict(game_id=99999,season=2024,pred_margin_v45=7.,pred_total_v3=55.,
                         actual_margin_v3=300.,actual_total_v3=400.))
    look.append(dict(game_id=99999,season=2024,universe_status='other'))
    pd.DataFrame(rows).to_csv(root/PAIRED,index=False)
    pd.DataFrame(look).to_csv(root/UNIVERSE,index=False)
    arr,audit=historical_fbs_residuals(root)
    assert arr.shape==(4200,2)
    assert np.allclose(arr[0],[2,1])
    assert audit['confirmed_fbs_fbs'] is True
