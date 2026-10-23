"""Frozen-snapshot, pregame-only experimental v3/v4.5 hybrid projections.

Trains point Ridge models on 2014--2025 historical data, using v3's frozen
267-feature manifest for totals, and v4.5's frozen 287-feature manifest for
margins. Uses ONLY saved 2026 pregame features. The 2026 outcomes are never
used for fitting, and no APIs or odds feeds are accessed.

Historical (2019--2024) confirmed FBS-FBS, season-blocked joint scoring errors
provide the v5.2-style pooled bootstrap. This is NOT prospectively calibrated,
not an EV system, and must not be construed as betting guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_quant.models.matchup_tournament import _make_ridge, _target_mask, _to_numeric_frame

V3_DATA=Path('data/processed/cfb_matchup_features_kickoff_v3.parquet')
V45_DATA=Path('data/processed/cfb_matchup_features_opponent_v45.parquet')
V3_MANIFEST=Path('reports/cfb_matchup_feature_manifest_kickoff_v3.json')
V45_MANIFEST=Path('reports/cfb_matchup_feature_manifest_opponent_v45.json')
PAIRED=Path('reports/spread_total_opponent_v45/paired_game_errors.csv')
UNIVERSE=Path('reports/spread_total_fbs_calibration_v52/universe_audit.csv')
SCHEDULE=Path('data/raw/cfbd/games/games_2026.json')
ADVANCED=Path('data/raw/cfbd/advanced_game_v4/advanced_2026.json')
OUTPUT=Path('reports/live_predictions_v57')
YEAR=2026


def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def game_key(val) -> str:
    if val is None or isinstance(val,bool):
        raise ValueError('Missing/invalid game ID')
    s=str(val)
    if s.endswith('.0') and s[:-2].isdigit():s=s[:-2]
    if not s.isdigit():raise ValueError(f'Non-numeric CFBD game ID: {s!r}')
    return s


def load_manifest(root:Path,path:Path) -> list[str]:
    doc=json.loads((root/path).read_text(encoding='utf-8'))
    cols=list(doc['feature_columns'])
    if not cols or len(cols)!=len(set(cols)):
        raise ValueError(f'Duplicate/empty model features in {path}')
    if {'points','opponent_points','actual_margin','actual_total'} & set(cols):
        raise ValueError(f'Postgame targets in manifest {path}')
    return cols


def verify_snapshot(root:Path,snapshot:Path,now:pd.Timestamp, v3_cols:list[str],v45_cols:list[str],
                    *,allow_started:bool=False) -> tuple[pd.DataFrame,dict,dict]:
    """Fail closed if metadata/source fingerprints/timestamps are inconsistent."""
    folder=(snapshot if snapshot.is_absolute() else root/snapshot).resolve()
    meta=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
    data_path=folder/'pregame_features.parquet'
    if not meta.get('v3_historical_parity') or not meta.get('v45_historical_parity'):
        raise ValueError('Historical feature parity did not pass')
    if meta.get('no_current_game_outcomes') is not True:
        raise ValueError('Snapshot did not certify blank same-game outcomes')
    if meta.get('feature_count') != len(v45_cols) or meta.get('slate_team_rows') != 2*meta.get('slate_games',-1):
        raise ValueError('Snapshot metadata feature count / row count mismatch')
    if v45_cols[:len(v3_cols)]!=v3_cols or len(v3_cols)!=267 or len(v45_cols)!=287:
        raise ValueError('Manifest columns incompatible with established 267/287 feature architectures')
    for name,path in [('schedule_sha256',SCHEDULE),('advanced_2026_sha256',ADVANCED)]:
        if meta.get(name)!=file_sha256(root/path):
            raise ValueError(f'{name} differs from saved snapshot; rerun feature_bridge_v56 --save')
    asof=pd.to_datetime(meta['asof_utc'],utc=True,errors='raise')
    created=pd.to_datetime(meta['created_at_utc'],utc=True,errors='raise')
    if created<asof or asof>now or created>now:
        raise ValueError('Invalid/future snapshot as-of/creation times')
    df=pd.read_parquet(data_path)
    if len(df)!=int(meta['slate_team_rows']):
        raise ValueError('Snapshot row count does not agree with audit')
    required={'game_id','season','kickoff_utc','team','team_id','home_away','opponent','points','opponent_points',*v45_cols}
    if missing:=required-set(df.columns):
        raise ValueError(f'Snapshot missing columns: {sorted(missing)}')
    df=df.copy()
    df['game_id']=df.game_id.map(game_key)
    df['kickoff_utc']=pd.to_datetime(df.kickoff_utc,utc=True,errors='coerce')
    if df.kickoff_utc.isna().any() or (df.kickoff_utc<=asof).any():
        raise ValueError('Pregame rows not after original snapshot as-of')
    if not df.season.eq(YEAR).all():
        raise ValueError('Snapshot includes non-2026 games')
    if df[['points','opponent_points']].notna().any().any():
        raise ValueError('Snapshot contains same-game outcomes')
    if df.duplicated(['game_id','home_away']).any():
        raise ValueError('Duplicate game/side rows')
    for game,grp in df.groupby('game_id'):
        if len(grp)!=2 or set(grp.home_away)!={'home','away'}:
            raise ValueError(f'Incomplete scheduled game pair {game}')
        if grp.kickoff_utc.nunique()!=1 or grp.team_id.nunique()!=2 or grp.team.nunique()!=2:
            raise ValueError(f'Invalid team identities or kickoff for game {game}')
        h=grp.loc[grp.home_away=='home'].iloc[0]
        a=grp.loc[grp.home_away=='away'].iloc[0]
        if h.opponent!=a.team or a.opponent!=h.team:
            raise ValueError(f'Inconsistent opponent names in game {game}')
    if df.team_id.duplicated().any():
        raise ValueError('A team appears in multiple selected slate games')
    raw=_to_numeric_frame(df,v45_cols)
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError('Pregame input contains missing or infinite features; fail closed')
    all_games=int(df.game_id.nunique())
    started=df.loc[df.kickoff_utc<=now,'game_id'].unique().tolist()
    if started and not allow_started:
        df=df.loc[df.kickoff_utc>now].copy()
    if df.empty:
        raise ValueError('All frozen-snapshot games have already kicked off. No live forecasts.')
    return df,meta,{'snapshot_folder':str(folder),'snapshot_sha256':file_sha256(data_path),
                    'original_games':all_games,'excluded_started_games':len(started) if not allow_started else 0,
                    'started_games_included_as_archive':len(started) if allow_started else 0,
                    'eligible_games':int(df.game_id.nunique()),
                    'snapshot_asof_utc':asof.isoformat()}


def train_points(root:Path, cols:list[str],data_path:Path) -> tuple[object,dict]:
    data=pd.read_parquet(root/data_path)
    if not set(cols)<=(set(data.columns)):
        raise ValueError(f'Missing frozen manifest columns in historical dataset {data_path}')
    if data.duplicated(['game_id','team']).any():
        raise ValueError('Duplicate historical game/team rows')
    # Historical Parquet files may contain extra seasons, including 2026
    # placeholder rows, without those rows ever being eligible for training.
    # Select training years FIRST and verify the selected rows explicitly.
    season_values=pd.to_numeric(data.season,errors='raise')
    if not np.isfinite(season_values.to_numpy(dtype=float)).all() or (season_values%1!=0).any():
        raise ValueError(f'Invalid historical season labels: {data_path}')
    seasons=season_values.astype(int)
    eligible=seasons.between(2014,2025)
    dropped=seasons.loc[~eligible].value_counts().sort_index().to_dict()
    if dropped:
        print(f'{data_path}: excluded out-of-window seasons (NOT FIT): {dropped}',flush=True)
    mask=eligible & _target_mask(data,'points')
    train=data.loc[mask]
    selected_seasons=set(seasons.loc[mask].tolist())
    expected_seasons=set(range(2014,2026))
    if selected_seasons!=expected_seasons:
        raise ValueError(f'Incomplete historical 2014-2025 training years: missing={sorted(expected_seasons-selected_seasons)}; unexpected={sorted(selected_seasons-expected_seasons)}')
    if len(train)<10000:
        raise ValueError(f'Unexpectedly small historical 2014-2025 fit: {len(train)} team-games')
    if not seasons.loc[train.index].between(2014,2025).all():
        raise ValueError('FATAL: training rows include season outside 2014-2025')
    y=pd.to_numeric(train.points,errors='coerce')
    if not np.isfinite(y).all():raise ValueError('Nonfinite historical scoring targets')
    mdl=_make_ridge('points')
    mdl.fit(_to_numeric_frame(train,cols),y)
    return mdl,{'training_rows':len(train),'training_seasons':'2014-2025',
                'features':len(cols),'excluded_outside_2014_2025':{str(k):int(v) for k,v in dropped.items()},'historical_sha256':file_sha256(root/data_path)}


def historical_fbs_residuals(root:Path) -> tuple[np.ndarray,dict]:
    """Use matching game-level observed-vs-OOF hybrid scores on confirmed FBS matchups."""
    paired=pd.read_csv(root/PAIRED)
    universe=pd.read_csv(root/UNIVERSE)
    must={'game_id','season','pred_margin_v45','pred_total_v3','actual_margin_v3','actual_total_v3'}
    if missing:=must-set(paired):raise ValueError(f'Missing paired OOF fields: {sorted(missing)}')
    if not {'game_id','season','universe_status'}<=set(universe):
        raise ValueError('Missing FBS classification audit data')
    for d in (paired,universe):
        d['game_id']=d.game_id.map(game_key)
        d['season']=pd.to_numeric(d.season,errors='raise').astype(int)
        if d.duplicated(['season','game_id']).any():
            raise ValueError('Duplicate historical game identifiers')
    joined=paired.merge(universe[['game_id','season','universe_status']],on=['game_id','season'],how='left',validate='one_to_one',indicator=True)
    if not joined._merge.eq('both').all() or joined.universe_status.isna().any() or joined.universe_status.eq('unknown').any():
        raise ValueError('Unclassified historical game in residual pool')
    fbs=joined.loc[joined.universe_status.eq('fbs_fbs')].copy()
    if fbs.empty or fbs.season.min()!=2019 or fbs.season.max()!=2024 or fbs.season.nunique()!=6:
        raise ValueError('Expected 2019-2024 FBS-only OOF residuals')
    if len(fbs)<4000:raise ValueError('Insufficient FBS-only historical residuals')
    cols=['pred_margin_v45','pred_total_v3','actual_margin_v3','actual_total_v3']
    num=fbs[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(num).all():raise ValueError('Nonfinite historical residual values')
    me=num[:,2]-num[:,0]
    te=num[:,3]-num[:,1]
    residuals=np.column_stack([(te+me)/2.0,(te-me)/2.0])
    return residuals,{'residual_games':len(fbs),'residual_seasons':'2019-2024',
                      'confirmed_fbs_fbs':True,
                      'pairing_sha256':file_sha256(root/PAIRED),
                      'universe_sha256':file_sha256(root/UNIVERSE)}


def game_forecasts(rows:pd.DataFrame, pred_v3:np.ndarray,pred_v45:np.ndarray) -> pd.DataFrame:
    """Home spread is NEGATIVE of expected home margin. Preserves joint means."""
    if len(rows)!=len(pred_v3) or len(rows)!=len(pred_v45):
        raise ValueError('Prediction length mismatch')
    df=rows[['game_id','kickoff_utc','team','opponent','home_away']].copy()
    df['v3_score']=np.clip(np.asarray(pred_v3,dtype=float),0.0,100.0)
    df['v45_score']=np.clip(np.asarray(pred_v45,dtype=float),0.0,100.0)
    if not np.isfinite(df[['v3_score','v45_score']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite model forecasts')
    h=df.loc[df.home_away.eq('home')].set_index('game_id')
    a=df.loc[df.home_away.eq('away')].set_index('game_id')
    if set(h.index)!=set(a.index):raise ValueError('Forecast game pairs missing')
    a=a.reindex(h.index)
    frame=pd.DataFrame({'game_id':h.index,
        'kickoff_utc':h.kickoff_utc.to_numpy(),
        'home':h.team.to_numpy(),'away':a.team.to_numpy(),
        'fair_margin_home':h.v45_score.to_numpy()-a.v45_score.to_numpy(),
        'fair_total':h.v3_score.to_numpy()+a.v3_score.to_numpy()})
    frame['fair_home_spread']=-frame.fair_margin_home
    frame['projected_home_points']=(frame.fair_total+frame.fair_margin_home)/2
    frame['projected_away_points']=(frame.fair_total-frame.fair_margin_home)/2
    if (frame[['projected_home_points','projected_away_points']]<0).any().any():
        raise ValueError('Hybrid margin/total imply negative team scores: refuse incoherent forecasts')
    if frame.duplicated('game_id').any():raise ValueError('Duplicate forecast games')
    return frame.sort_values(['kickoff_utc','game_id']).reset_index(drop=True)


def add_probabilities(forecasts:pd.DataFrame,residuals:np.ndarray,*,samples:int=10000,seed:int=57,
                      home_line:float|None=None,total_line:float|None=None)->pd.DataFrame:
    if samples<1000 or samples>100000:raise ValueError('Samples must be 1,000-100,000')
    if len(residuals)<100 or residuals.shape[1]!=2 or not np.isfinite(residuals).all():
        raise ValueError('Insufficient finite paired historical scoring errors')
    if home_line is not None and (not np.isfinite(home_line) or home_line%1!=.5):
        raise ValueError('Use a half-point home handicap, e.g. --home-spread-line 14.5')
    if total_line is not None and (not np.isfinite(total_line) or total_line%1!=.5):
        raise ValueError('Use a half-point total, e.g. --total-line 55.5')
    d=forecasts.copy()
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(residuals),size=(len(d),samples))
    err=residuals[idx]
    home=np.maximum(0.0,d.projected_home_points.to_numpy()[:,None]+err[:,:,0])
    away=np.maximum(0.0,d.projected_away_points.to_numpy()[:,None]+err[:,:,1])
    margin=home-away
    total=home+away
    d['margin_p10']=np.quantile(margin,.1,axis=1)
    d['margin_p90']=np.quantile(margin,.9,axis=1)
    d['total_p10']=np.quantile(total,.1,axis=1)
    d['total_p90']=np.quantile(total,.9,axis=1)
    d['p_home_win']=(margin>0).mean(axis=1)
    d['p_home_covers_minus_7p5']=(margin>7.5).mean(axis=1)
    d['p_home_covers_plus_7p5']=(margin>-7.5).mean(axis=1)
    for line in [45.5,55.5,65.5]:
        d[f'p_over_{str(line).replace(".","p")}']=(total>line).mean(axis=1)
    if home_line is not None:
        d['custom_home_spread_line']=home_line
        d['p_home_covers_custom_line']=(margin+home_line>0).mean(axis=1)
    if total_line is not None:
        d['custom_total_line']=total_line
        d['p_over_custom_total']=(total>total_line).mean(axis=1)
    return d


def run(root:Path,snapshot:Path,*,samples:int=10000,seed:int=57,
        home_line:float|None=None,total_line:float|None=None,
        allow_started:bool=False,now:pd.Timestamp|None=None) -> tuple[pd.DataFrame,dict]:
    now=pd.Timestamp(datetime.now(timezone.utc)) if now is None else pd.to_datetime(now,utc=True)
    v3_cols=load_manifest(root,V3_MANIFEST)
    v45_cols=load_manifest(root,V45_MANIFEST)
    future,meta,stamp=verify_snapshot(root,snapshot,now,v3_cols,v45_cols,allow_started=allow_started)
    from cfb_quant.live.pregame_presave_gate_v58 import assess
    chronology_gate = assess(
        meta,
        future,
        now,
        allow_started=allow_started,
    )
    if chronology_gate["status"] == "REJECT":
        raise ValueError(
            "Pregame chronology rejected before forecasting: "
            + ", ".join(chronology_gate["reasons"])
        )

    from cfb_quant.live import forecast_input_freeze_v58 as freeze
    input_fingerprints = freeze.take(root, snapshot)

    v3_model,v3_audit=train_points(root,v3_cols,V3_DATA)
    print(f'v3 total model fitted on {v3_audit["training_rows"]:,} team games through 2025',flush=True)
    v45_model,v45_audit=train_points(root,v45_cols,V45_DATA)
    print(f'v4.5 spread model fitted on {v45_audit["training_rows"]:,} team games through 2025',flush=True)
    p3=v3_model.predict(_to_numeric_frame(future,v3_cols))
    p45=v45_model.predict(_to_numeric_frame(future,v45_cols))
    f=game_forecasts(future,p3,p45)
    residuals,residual_audit=historical_fbs_residuals(root)
    print(f'v5.2 pooled FBS-only residuals: {len(residuals):,} games, 2019-2024',flush=True)
    f=add_probabilities(f,residuals,samples=samples,seed=seed,home_line=home_line,total_line=total_line)
    input_verification = freeze.verify_unchanged(
        root, input_fingerprints
    )

    audit={'version':'v5.7 research-only','generated_at_utc':now.isoformat(),
           'snapshot_metadata':meta,'snapshot_validation':stamp,
           'v3_training':v3_audit,'v45_training':v45_audit,
           'residual_calibration':residual_audit,
           'samples':samples,'seed':seed,
           'disclaimer':'Exploratory probabilities, not prospectively calibrated; no market odds or expected-value claims; historical source timestamps not fully audited.'}
    audit["execution_input_fingerprints"] = input_fingerprints
    audit["execution_input_verification"] = input_verification
    audit["chronology_gate"] = chronology_gate
    audit["forecast_classification"] = chronology_gate["classification"]
    audit["prospective_performance_eligible"] = False
    audit["source_availability_certified"] = False
    audit["production_forecast_authorized"] = False
    return f,audit


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True,help='Saved v5.6 snapshot folder (not the Parquet file)')
    p.add_argument('--root',default='.')
    p.add_argument('--samples',type=int,default=10000)
    p.add_argument('--seed',type=int,default=57)
    p.add_argument('--home-spread-line',type=float,default=None,help='Optional SAME home handicap for all games, e.g. +14.5')
    p.add_argument('--total-line',type=float,default=None,help='Optional SAME game total for all games, e.g. 55.5')
    p.add_argument('--include-started-archived',action='store_true',help='Research only: allow forecasting from pregame snapshots after games have begun; clearly marked non-live')
    p.add_argument('--audit-only',action='store_true',help='Run forecasting/validation, do not save files')
    a=p.parse_args()
    root=Path(a.root).resolve()
    frame,audit=run(root,Path(a.snapshot),samples=a.samples,seed=a.seed,
                    home_line=a.home_spread_line,total_line=a.total_line,
                    allow_started=a.include_started_archived)
    print('\n2026 LIVE MODEL — PRELIMINARY RESEARCH PROJECTIONS')
    cols=['away','home','fair_home_spread','fair_total','projected_away_points','projected_home_points',
          'p_home_win','p_home_covers_plus_7p5','p_home_covers_minus_7p5']
    print(frame[cols].round(3).to_string(index=False))
    print('\nConventions: fair_home_spread < 0 means HOME is favored. P(home +7.5) means margin > -7.5.')
    print('The probabilities are exploratory and should not be treated as calibrated betting edges.')
    if a.audit_only:
        print('AUDIT ONLY: nothing saved')
        return
    from cfb_quant.live.forecast_publication_v58 import publish_verified
    out=publish_verified(root,root/OUTPUT,frame,audit)
    target=out/'slate_forecasts.csv'
    print('Predictions:',target)
    print('Audit:',out/'model_audit.json')
    print('Not sportsbook recommendations. No bet-size or EV outputs.')


if __name__=='__main__':main()
