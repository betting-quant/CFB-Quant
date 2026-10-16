from datetime import datetime, timezone

import pandas as pd
import pytest

from cfb_quant.live.feature_bridge_v56 import (
    gid, kickoff, done, fbs_both, select_upcoming, make_placeholders,
)


def game(id=111,home='Alabama',away='Georgia',home_id=1,away_id=2,date='2026-10-10T18:00:00Z',**kwargs):
    return dict(id=id,season=2026,week=6,seasonType='regular',startDate=date,
                homeTeam=home,awayTeam=away,homeId=home_id,awayId=away_id,
                homeConference='SEC',awayConference='SEC',homeClassification='fbs',
                awayClassification='fbs',completed=False,startTimeTBD=False,
                homePoints=None,awayPoints=None,**kwargs)


def test_valid_selection():
    a=select_upcoming([game()], '2026-10-10',pd.Timestamp('2026-10-10T08:00Z'))
    assert len(a)==1


def test_wrong_calendar_day_and_started_excluded():
    games=[game(),game(id=222,home_id=3,away_id=4,date='2026-10-11T18:00Z')]
    assert len(select_upcoming(games,'2026-10-10',pd.Timestamp('2026-10-10T18:00Z')))==0


def test_only_fbs():
    g=game();g['awayClassification']='fcs'
    assert not select_upcoming([g],'2026-10-10',pd.Timestamp('2026-10-10T08:00Z'))


def test_double_booking_rejected():
    games=[game(),game(id=222,home='Alabama',away='Texas',home_id=1,away_id=3)]
    with pytest.raises(ValueError,match='twice'):
        select_upcoming(games,'2026-10-10',pd.Timestamp('2026-10-10T08:00Z'))


def test_duplicate_id_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        select_upcoming([game(),game()],'2026-10-10',pd.Timestamp('2026-10-10T08:00Z'))


def test_placeholders_have_no_game_outcomes():
    g=game()
    hist=pd.DataFrame([{'game_id':10,'season':2026,'week':1,'season_type':'regular',
                       'team_id':1,'team':'Alabama','opponent':'Georgia','home_away':'home',
                       'points':17.,'opponent_points':10.,'passing_yards':200.,
                       'kickoff_utc':pd.Timestamp('2026-09-01T18:00Z')},
                       {'game_id':10,'season':2026,'week':1,'season_type':'regular',
                       'team_id':2,'team':'Georgia','opponent':'Alabama','home_away':'away',
                       'points':10.,'opponent_points':17.,'passing_yards':220.,
                       'kickoff_utc':pd.Timestamp('2026-09-01T18:00Z')}])
    future=make_placeholders([g],hist)
    assert len(future)==2
    assert future.points.isna().all() and future.passing_yards.isna().all()
    assert set(future.home_away)=={'home','away'}


def test_bad_game_ids():
    assert gid(401111111)=='401111111'
    with pytest.raises(ValueError):gid('nope')


def test_completed_never_selected():
    g=game();g.update(completed=True,homePoints=14,awayPoints=10)
    assert done(g)
    assert not select_upcoming([g],'2026-10-10',pd.Timestamp('2026-10-10T08:00Z'))


def test_historical_parity_detects_changed_past(tmp_path, monkeypatch):
    from cfb_quant.live.feature_bridge_v56 import assert_historical_parity
    old=pd.DataFrame([{'game_id':1,'season':2025,'team':'A','my_feature':1.0},
                      {'game_id':2,'season':2026,'team':'A','my_feature':8.0}])
    new=pd.DataFrame([{'game_id':1,'season':2025,'team':'A','my_feature':1.2},
                      {'game_id':2,'season':2026,'team':'A','my_feature':9.0}])
    path=tmp_path/'old.parquet'; monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: old.copy())
    with pytest.raises(ValueError,match='historical feature mismatch'):
        assert_historical_parity(new,path,['my_feature'],name='test')


def test_historical_parity_ignores_live_rows(tmp_path, monkeypatch):
    from cfb_quant.live.feature_bridge_v56 import assert_historical_parity
    old=pd.DataFrame([{'game_id':1,'season':2025,'team':'A','my_feature':1.0},
                      {'game_id':2,'season':2026,'team':'A','my_feature':8.0}])
    new=pd.DataFrame([{'game_id':1,'season':2025,'team':'A','my_feature':1.0},
                      {'game_id':2,'season':2026,'team':'A','my_feature':9.0}])
    path=tmp_path/'old.parquet'; monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: old.copy())
    assert assert_historical_parity(new,path,['my_feature'],name='test')['historical_parity'] is True


def test_tbd_game_rejected():
    g=game();g['startTimeTBD']=True
    assert not select_upcoming([g],'2026-10-10',pd.Timestamp('2026-10-10T08:00Z'))
