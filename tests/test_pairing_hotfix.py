import pandas as pd
import pytest
from cfb_quant.spread_total.core import audit_game_pairs, build_game_errors


def sample():
    rows=[]
    for gid,season,week,sides in ((101,2020,2,("home","away")),
                                  (102,2020,3,("home",)),
                                  (103,2021,1,("away",)),
                                  (104,2021,2,("home","away"))):
        for i,side in enumerate(sides):
            rows.append(dict(game_id=gid, season=season, week=week, team=f"t{gid}-{side}",
                             home_away=side, target="points", model="ridge",
                             prediction=float(25+i), actual=float(27+i),
                             train_end_season=season-1))
    return pd.DataFrame(rows)


def test_incomplete_games_explicitly_excluded():
    df=sample()
    audit=audit_game_pairs(df)
    assert audit.status.value_counts().to_dict()=={"complete":2,"missing_away":1,"missing_home":1}
    with pytest.raises(ValueError,match="Every OOF game"):
        build_game_errors(df)
    result=build_game_errors(df,allow_incomplete=True)
    assert set(result.game_id)=={101,104}
    assert (result.pred_total==51).all()


def test_invalid_metadata_not_silently_dropped():
    df=sample()
    df.loc[df.game_id.eq(101)&df.home_away.eq("away"),"week"]=999
    with pytest.raises(ValueError,match="Invalid game pairing metadata"):
        build_game_errors(df,allow_incomplete=True)


def test_duplicate_rejected_even_if_incomplete_allowed():
    df=sample()
    with pytest.raises(ValueError,match="Duplicate"):
        build_game_errors(pd.concat([df,df.iloc[:1]],ignore_index=True),allow_incomplete=True)
