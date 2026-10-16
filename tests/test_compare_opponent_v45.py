import pandas as pd
import pytest
from cfb_quant.spread_total.compare_opponent_v45 import paired_metrics


def make_preds():
    rows = []
    for gid,year,home,away,h,a in [('1',2020,'A','B',24,14),('2',2021,'C','D',21,24)]:
        rows.append({'game_id':gid,'season':year,'week':1,'team':home,'home_away':'home','target':'points','model':'ridge','prediction':float(h+2),'actual':h,'train_end_season':year-1})
        rows.append({'game_id':gid,'season':year,'week':1,'team':away,'home_away':'away','target':'points','model':'ridge','prediction':float(a+1),'actual':a,'train_end_season':year-1})
    return pd.DataFrame(rows)


def test_identical_predictions_have_zero_delta():
    d=make_preds()
    paired,by=paired_metrics(d,d.copy())
    assert len(paired)==2 and len(by)==2
    assert paired.margin_delta_error.eq(0).all()
    assert paired.total_delta_error.eq(0).all()


def test_mismatched_games_are_rejected():
    old=make_preds();new=old[new_mask(old)].copy()
    with pytest.raises(ValueError,match='game sets differ'):
        paired_metrics(old,new)


def new_mask(df):
    return df.game_id=='1'


def test_changed_scores_are_rejected():
    old=make_preds();new=old.copy()
    new.loc[new.team=='A','actual']=44
    with pytest.raises(ValueError,match='Observed final scores'):
        paired_metrics(old,new)
