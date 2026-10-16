"""Experimental direct game-margin forecast (v5), compared against saved v3/v4.5.

- Reads point-in-time v4.5 team-game feature table and feature manifest.
- Pairs home/away team rows into one game *before* fitting.
- Uses signed HOME MINUS AWAY feature differences; Ridge(alpha=20), median
  imputation, scaling and missingness indicators, same family as v3.
- Season-blocked backtesting: no game from prediction season in training.
- Strict comparison with saved v3/v4.5 predictions on identical games.
- Writes new files only; never fetches API, never modifies earlier outputs.

These retrospective seasons were repeatedly inspected during development.
Results are exploratory and NOT proof of live betting advantage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_quant.models.matchup_tournament import _make_ridge, _to_numeric_frame, _target_mask

DATA = Path('data/processed/cfb_matchup_features_opponent_v45.parquet')
MANIFEST = Path('reports/cfb_matchup_feature_manifest_opponent_v45.json')
BASELINE = Path('reports/spread_total_opponent_v45/paired_game_errors.csv')
OUT = Path('reports/spread_total_direct_margin_v5')


def paired_game_frame(team_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """One row per game. Reject malformed source; never impute current scores.

    All features must come from the supplied reviewed manifest, not inferred
    from data columns. Rows with insufficient PRIOR team history are excluded,
    matching the historical per-team-points baseline.
    """
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError('Feature manifest is empty or has duplicates')
    forbidden = {'points','opponent_points','actual_margin','actual_total','actual','prediction','game_id','season','week','kickoff_utc'}
    if forbidden.intersection(feature_columns):
        raise ValueError(f'Forbidden target/metadata feature(s): {sorted(forbidden.intersection(feature_columns))}')
    necessary = {'game_id','season','week','home_away','team','opponent','points','kickoff_utc'} | set(feature_columns)
    missing = necessary.difference(team_data.columns)
    if missing:
        raise ValueError(f'Missing game or feature columns: {sorted(missing)}')
    d = team_data.copy()
    if d.duplicated(['game_id','home_away']).any() or d.duplicated(['game_id','team']).any():
        raise ValueError('Duplicate team-game or side records')
    if not d.home_away.isin(('home','away')).all():
        raise ValueError('Invalid home_away tag')
    counts = d.groupby('game_id').home_away.agg(['size','nunique'])
    if (counts['size'].ne(2) | counts['nunique'].ne(2)).any():
        raise ValueError('Incomplete/invalid home-away pair')
    d['kickoff_utc'] = pd.to_datetime(d['kickoff_utc'],utc=True,errors='coerce')
    if d.kickoff_utc.isna().any():
        raise ValueError('Kickoff timestamps missing')
    d['points'] = pd.to_numeric(d.points,errors='coerce')
    if not np.isfinite(d.points).all():
        raise ValueError('Missing/nonfinite final score')
    eligible = _target_mask(d,'points')
    home = d.loc[d.home_away.eq('home')].copy()
    away = d.loc[d.home_away.eq('away')].copy()
    keep = ['game_id','season','week','team','opponent','points','kickoff_utc', *feature_columns]
    # Eligibility is determined at TEAM level (as for build_points_oof)
    home['eligible'] = eligible.loc[home.index].to_numpy(dtype=bool)
    away['eligible'] = eligible.loc[away.index].to_numpy(dtype=bool)
    games = home[keep+['eligible']].merge(
        away[keep+['eligible']],on='game_id',suffixes=('_home','_away'),validate='one_to_one',sort=False)
    if len(games) != len(counts):
        raise ValueError('Home/away join lost a game')
    if not games.season_home.eq(games.season_away).all() or not games.week_home.eq(games.week_away).all():
        raise ValueError('Home/away season/week disagreement')
    if not games.kickoff_utc_home.eq(games.kickoff_utc_away).all():
        raise ValueError('Home/away kickoff disagreement')
    if not games.team_home.eq(games.opponent_away).all() or not games.team_away.eq(games.opponent_home).all():
        raise ValueError('Home/away opponent identity mismatch')
    games = games.loc[games.eligible_home & games.eligible_away].copy()
    out = pd.DataFrame({
        'game_id': games.game_id.to_numpy(),
        'season': games.season_home.to_numpy(dtype=int),
        'week': games.week_home.to_numpy(),
        'kickoff_utc': games.kickoff_utc_home.to_numpy(),
        'home_team': games.team_home.to_numpy(),
        'away_team': games.team_away.to_numpy(),
        'actual_margin': (games.points_home - games.points_away).to_numpy(dtype=float),
    })
    # Antisymmetric inputs: a stronger HOME team relative to AWAY is positive.
    # _to_numeric_frame safely converts numeric feature columns and maps inf to NaN.
    home_feat = _to_numeric_frame(games.rename(columns={f'{f}_home':f for f in feature_columns}), feature_columns)
    away_feat = _to_numeric_frame(games.rename(columns={f'{f}_away':f for f in feature_columns}), feature_columns)
    # Subtraction preserves missingness for train-only imputation/indicators.
    deltas = home_feat.to_numpy(dtype=float) - away_feat.to_numpy(dtype=float)
    delta_frame = pd.DataFrame(deltas, columns=[f'diff_{f}' for f in feature_columns], index=out.index)
    return pd.concat([out,delta_frame],axis=1).sort_values(['season','kickoff_utc','game_id']).reset_index(drop=True)


def season_blocked_margin_oof(games: pd.DataFrame, features: list[str], first_year: int = 2019,
                              last_year: int = 2024, min_train_year: int = 2014) -> pd.DataFrame:
    """Same Ridge family as v3; fit on earlier seasons only."""
    if first_year <= min_train_year or last_year < first_year:
        raise ValueError('Invalid chronological study window')
    if games.duplicated(['game_id','season']).any():
        raise ValueError('Duplicate games')
    if games[features].shape[1] != len(features):
        raise ValueError('Missing design features')
    outputs = []
    for year in range(first_year, last_year+1):
        tr = games.loc[games.season.ge(min_train_year) & games.season.lt(year)]
        te = games.loc[games.season.eq(year)]
        if tr.empty or te.empty:
            raise ValueError(f'No training or evaluation games for season {year}')
        if tr.season.max() >= year:
            raise ValueError('Future training information')
        model = _make_ridge('margin')
        model.fit(_to_numeric_frame(tr,features), tr.actual_margin.astype(float))
        pred = np.asarray(model.predict(_to_numeric_frame(te,features)),dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite prediction')
        item = te[['game_id','season','week','kickoff_utc','home_team','away_team','actual_margin']].copy()
        item['v5_pred_margin'] = pred
        item['train_end_season'] = int(tr.season.max())
        outputs.append(item)
        print(f'{year}: direct margin games={len(item)}, training through {int(tr.season.max())}',flush=True)
    return pd.concat(outputs,ignore_index=True)


def compare_to_saved_baselines(oof: pd.DataFrame, saved: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    required = {'game_id','season','actual_margin_v3','actual_margin_v45','pred_margin_v3','pred_margin_v45'}
    missing = required.difference(saved.columns)
    if missing:
        raise ValueError(f'Missing v3/v4.5 baseline columns: {sorted(missing)}')
    s = saved.copy()
    x = oof.copy()
    # Use string identifier to protect leading zeros, integer-like/CSV inference.
    s['game_id'] = s.game_id.astype(str).str.replace(r'\.0$','',regex=True)
    x['game_id'] = x.game_id.astype(str).str.replace(r'\.0$','',regex=True)
    for t in (s,x):
        if t.duplicated(['game_id','season']).any():
            raise ValueError('Duplicate historical game keys')
    bkeys = set(zip(s.game_id,s.season))
    okeys = set(zip(x.game_id,x.season))
    if bkeys != okeys:
        raise ValueError(f'Different evaluation games: v5-only={len(okeys-bkeys)}, baseline-only={len(bkeys-okeys)}')
    z = x.merge(s[['game_id','season','actual_margin_v3','actual_margin_v45','pred_margin_v3','pred_margin_v45']],
                on=['game_id','season'],how='inner',validate='one_to_one')
    if not np.allclose(z.actual_margin,z.actual_margin_v3,rtol=0,atol=0) or not np.allclose(z.actual_margin,z.actual_margin_v45,rtol=0,atol=0):
        raise ValueError('Actual scores differ between comparison models')
    for name,pred in [('v3','pred_margin_v3'),('v45','pred_margin_v45'),('v5','v5_pred_margin')]:
        z[f'ae_{name}'] = (z.actual_margin - z[pred]).abs()
    z['delta_v5_vs_v45'] = z.ae_v5 - z.ae_v45
    z['delta_v5_vs_v3'] = z.ae_v5 - z.ae_v3
    by = z.groupby('season',as_index=False).agg(
        games=('game_id','size'),v3_margin_mae=('ae_v3','mean'),
        v45_margin_mae=('ae_v45','mean'),v5_margin_mae=('ae_v5','mean'),
        delta_v5_vs_v45=('delta_v5_vs_v45','mean'),delta_v5_vs_v3=('delta_v5_vs_v3','mean'),
    )
    overall = pd.DataFrame([{
        'games':len(z),
        'v3_margin_mae':float(z.ae_v3.mean()),
        'v45_margin_mae':float(z.ae_v45.mean()),
        'v5_margin_mae':float(z.ae_v5.mean()),
        'delta_v5_vs_v45':float(z.delta_v5_vs_v45.mean()),
        'delta_v5_vs_v3':float(z.delta_v5_vs_v3.mean()),
        'seasons_v5_beats_v45':int((by.delta_v5_vs_v45<0).sum()),
        'seasons_v5_beats_v3':int((by.delta_v5_vs_v3<0).sum()),
    }])
    return z,by,overall


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default=str(DATA))
    p.add_argument('--manifest',default=str(MANIFEST))
    p.add_argument('--baseline',default=str(BASELINE))
    p.add_argument('--out-dir',default=str(OUT))
    p.add_argument('--first-year',type=int,default=2019)
    p.add_argument('--last-year',type=int,default=2024)
    args=p.parse_args()
    if not (2019 <= args.first_year <= args.last_year <= 2024):
        p.error('Default exploratory window is limited to 2019-2024; do not reuse 2025 as an untouched validation.')
    dest=Path(args.out_dir)
    existing=[str(dest/fn) for fn in ('overall.csv','by_season.csv','paired_game_errors.csv') if (dest/fn).exists()]
    if existing:
        raise FileExistsError(f'Existing v5 outputs; refusing to overwrite: {existing}')
    for name,value in [('v4.5 dataset',args.data),('v4.5 manifest',args.manifest),('saved comparison',args.baseline)]:
        if not Path(value).is_file():
            raise FileNotFoundError(f'Missing {name}: {value}. Do not rerun historical downloads.')
    data=pd.read_parquet(args.data)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    feats=list(manifest['feature_columns'])
    games=paired_game_frame(data,feats)
    diffs=[f'diff_{x}' for x in feats]
    print(f'Paired eligible games: {len(games)}; direct-margin features: {len(diffs)}',flush=True)
    preds=season_blocked_margin_oof(games,diffs,first_year=args.first_year,last_year=args.last_year)
    baselines=pd.read_csv(args.baseline)
    baselines=baselines.loc[baselines.season.between(args.first_year,args.last_year)]
    pair,by,overall=compare_to_saved_baselines(preds,baselines)
    dest.mkdir(parents=True,exist_ok=True)
    files={
        'overall.csv':overall,
        'by_season.csv':by,
        'paired_game_errors.csv':pair,
    }
    if any((dest / fn).exists() for fn in files):
        raise FileExistsError(f'v5 output already exists at {dest}; refusing to overwrite. Move old report folder before rerun.')
    for fn,frame in files.items():
        frame.to_csv(dest/fn,index=False)
    print('\nEXPLORATORY DIRECT MARGIN v5 (lower MAE better)')
    print(overall.round(4).to_string(index=False))
    print('\nBY SEASON')
    print(by.round(4).to_string(index=False))
    print(f'\nSaved to {dest}. No API calls, no v3/v4.5 modifications.')
    print('DEVELOPMENT SAMPLE ONLY. No +EV/profitability claim or untouched holdout.')

if __name__=='__main__':
    main()
