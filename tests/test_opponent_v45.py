import pandas as pd
import numpy as np
import pytest
from cfb_quant.spread_total.opponent_v45 import build_adjustments, feature_names, METRICS


def make_data():
    # Deliberately shuffled: chronologically second game listed first.
    data = [
        ('2','2024-09-08T20:00:00Z','A','C','home',2024),
        ('2','2024-09-08T20:00:00Z','C','A','away',2024),
        ('1','2024-09-01T20:00:00Z','A','B','home',2024),
        ('1','2024-09-01T20:00:00Z','B','A','away',2024),
    ]
    base = pd.DataFrame(data,columns=['game_id','kickoff_utc','team','opponent','home_away','season'])
    obs=[]
    for row in base.itertuples():
        item={'game_id':row.game_id,'team':row.team}
        for m in METRICS:
            item[f'raw_adv_offense_{m}']=0.6 if m.endswith('success_rate') else 0.30
        obs.append(item)
    return base,pd.DataFrame(obs)


def test_output_and_initial_prior():
    base,raw=make_data()
    out,coverage=build_adjustments(base,raw)
    assert len(out)==4 and len(feature_names())==20
    assert out.game_id.tolist()==base.game_id.tolist()
    g1=out[out.game_id.eq('1')]
    assert (g1.oa_ppa_attack == 0).all()
    assert (g1.oa_ppa_opp_def_weakness == 0).all()
    assert (g1.oa_team_prior_games_season == 0).all()
    g2=out[(out.game_id=='2') & (out.team=='A')].iloc[0]
    assert g2.oa_team_prior_games_season == 1
    assert g2.oa_ppa_attack > 0
    assert g2.oa_ppa_opp_def_weakness == 0  # C's defense is unseen
    assert int(coverage.matched.iloc[0])==4


def test_future_edits_do_not_change_past():
    base,raw=make_data()
    x,_=build_adjustments(base,raw)
    altered=raw.copy()
    altered.loc[altered.game_id=='2','raw_adv_offense_ppa']=-0.7
    y,_=build_adjustments(base,altered)
    pd.testing.assert_frame_equal(x.loc[x.game_id=='1',feature_names()].reset_index(drop=True),
                                  y.loc[y.game_id=='1',feature_names()].reset_index(drop=True))
    # No current-game stats: 2024-09-08 pregame ratings also remain unchanged.
    pd.testing.assert_frame_equal(x.loc[x.game_id=='2',feature_names()].reset_index(drop=True),
                                  y.loc[y.game_id=='2',feature_names()].reset_index(drop=True))


def test_invalid_pair_detected():
    base,raw=make_data()
    base.loc[base.game_id=='1','opponent']='WRONG'
    with pytest.raises(ValueError,match='Opponent'):
        build_adjustments(base,raw)


def test_missing_kickoff_refused():
    base,raw=make_data()
    base.loc[0,'kickoff_utc']=None
    with pytest.raises(ValueError,match='kickoff'):
        build_adjustments(base,raw)


def test_low_source_coverage_is_rejected():
    base,raw=make_data()
    raw=raw.loc[~((raw.game_id=='2') & (raw.team=='C'))].copy()
    with pytest.raises(ValueError,match='below 90%'):
        build_adjustments(base,raw)


def test_same_kickoff_independent_update():
    base,raw=make_data()
    later=base.loc[base.game_id=='2'].copy()
    later.game_id='3'
    later.team=['D','E'];later.opponent=['E','D']
    base=pd.concat([base,later],ignore_index=True)
    obs=raw.loc[raw.game_id=='2'].copy()
    obs.game_id='3';obs.team=['D','E']
    raw=pd.concat([raw,obs],ignore_index=True)
    out,_=build_adjustments(base,raw)
    g3=out[out.game_id=='3']
    assert (g3.oa_team_prior_games_season==0).all()
    assert (g3.oa_ppa_attack==0).all()
