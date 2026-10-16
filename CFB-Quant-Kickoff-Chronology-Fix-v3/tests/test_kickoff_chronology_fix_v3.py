from pathlib import Path
import pandas as pd
import pytest
from cfb_quant.features.matchup_engine import attach_verified_kickoff, build_features

def fake_data():
 rows=[]; dates=[]
 # Later match has lower game ID. Week same. Higher risk of future leak.
 for gid,stamp,other,pts,opppts in [(101,'2025-09-02T19:00:00Z','Beta',99,11),(202,'2025-08-25T19:00:00Z','Gamma',10,20)]:
  for team,opponent,side,p,op in [('Alpha',other,'home',pts,opppts),(other,'Alpha','away',opppts,pts)]:
   d={'game_id':gid,'season':2025,'season_type':'regular','week':1,'team':team,'opponent':opponent,'home_away':side,'points':p,'opponent_points':op,'pass_attempts':20,'rush_attempts':25,'passing_yards':170,'rushing_yards':85,'possession_seconds':1800,'completions':12}
   rows.append(d)
   dates.append({k:d[k] for k in ('game_id','season','team','home_away')}|{'kickoff_utc':stamp})
 return pd.DataFrame(rows),pd.DataFrame(dates)

def test_offense_and_defense_true_chronology(tmp_path):
 raw,dates=fake_data(); p=tmp_path/'map.csv'; dates.to_csv(p,index=False)
 hist=attach_verified_kickoff(raw,p)
 feat,col=build_features(hist)
 first=feat[(feat.game_id==202)&(feat.team=='Alpha')].iloc[0]
 second=feat[(feat.game_id==101)&(feat.team=='Alpha')].iloc[0]
 assert pd.isna(first.off_points_lag1)
 assert second.off_points_lag1==10
 assert feat[(feat.game_id==101)&(feat.team=='Beta')].iloc[0].def_allowed_points_lag1==20
 assert feat.loc[feat.team=='Alpha','game_id'].tolist()==[202,101]
 assert 'points' not in col

def test_missing_game_fails(tmp_path):
 raw,dates=fake_data();p=tmp_path/'map.csv'; dates[dates.game_id==101].to_csv(p,index=False)
 with pytest.raises(ValueError,match='Unmatched kickoff'):
  attach_verified_kickoff(raw,p)

def test_conflicting_kickoffs_fail(tmp_path):
 raw,dates=fake_data();p=tmp_path/'map.csv'
 dates.loc[dates.team=='Beta','kickoff_utc']='2025-09-03T19:00:00Z'
 dates.to_csv(p,index=False)
 with pytest.raises(ValueError,match='Conflicting kickoff'):
  attach_verified_kickoff(raw,p)

def test_missing_map_fails(tmp_path):
 raw,_=fake_data()
 with pytest.raises(FileNotFoundError):
  attach_verified_kickoff(raw,tmp_path/'not.csv')
