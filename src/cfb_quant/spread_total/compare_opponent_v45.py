"""Side-by-side historical v3 versus experimental opponent v4.5 point predictions.

No API calls, no betting recommendations. Uses the exact same eligible historical
team-games and Ridge factory for both variants. Results are exploratory only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import build_game_errors
from .oof_points import build_points_oof
from .opponent_v45 import MANIFEST, OUTPUT, V3_MANIFEST, feature_names


def paired_metrics(base_oof: pd.DataFrame, challenger_oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old=build_game_errors(base_oof, allow_incomplete=True)
    new=build_game_errors(challenger_oof, allow_incomplete=True)
    key=['game_id','season']
    if old.duplicated(key).any() or new.duplicated(key).any():
        raise ValueError('Duplicate game keys in paired comparison')
    if set(zip(old.game_id.astype(str),old.season)) != set(zip(new.game_id.astype(str),new.season)):
        raise ValueError('Evaluation game sets differ across models')
    keep=['game_id','season','week','actual_margin','actual_total','pred_margin','pred_total']
    pair=old[keep].merge(new[keep],on=key,suffixes=('_v3','_v45'),validate='one_to_one')
    if not (pair.actual_margin_v3.eq(pair.actual_margin_v45).all() and pair.actual_total_v3.eq(pair.actual_total_v45).all()):
        raise ValueError('Observed final scores changed between models')
    for name in ('margin','total'):
        for label in ('v3','v45'):
            pair[f'{name}_absolute_error_{label}']=(pair[f'actual_{name}_{label}'] - pair[f'pred_{name}_{label}']).abs()
        pair[f'{name}_delta_error']=pair[f'{name}_absolute_error_v45']-pair[f'{name}_absolute_error_v3']
    by=pair.groupby('season',as_index=False).agg(
        games=('game_id','size'),
        v3_margin_mae=('margin_absolute_error_v3','mean'),
        v45_margin_mae=('margin_absolute_error_v45','mean'),
        delta_margin_mae=('margin_delta_error','mean'),
        v3_total_mae=('total_absolute_error_v3','mean'),
        v45_total_mae=('total_absolute_error_v45','mean'),
        delta_total_mae=('total_delta_error','mean'),
    )
    return pair,by


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default=str(OUTPUT))
    ap.add_argument('--manifest',default=str(MANIFEST))
    ap.add_argument('--baseline-manifest',default=str(V3_MANIFEST))
    ap.add_argument('--first-year',type=int,default=2019)
    ap.add_argument('--last-year',type=int,default=2024)
    ap.add_argument('--out-dir',default='reports/spread_total_opponent_v45')
    args=ap.parse_args()
    if args.first_year < 2015 or args.last_year > 2025 or args.last_year < args.first_year:
        ap.error('Allowed chronological evaluation: 2015–2025')
    d=pd.read_parquet(args.data)
    baseline=json.loads(Path(args.baseline_manifest).read_text(encoding='utf-8'))
    challenger=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    a=list(baseline['feature_columns']);b=list(challenger['feature_columns'])
    if b != a + feature_names():
        raise ValueError('Challenger manifest does not exactly extend v3 with verified opponent-rating features')
    if d.duplicated(['game_id','team']).any():
        raise ValueError('Duplicate team-game rows')
    print(f'BASELINE: {len(a)} v3 input features',flush=True)
    old=build_points_oof(d,a,first_year=args.first_year,last_year=args.last_year)
    print(f'CHALLENGER: {len(b)} v4.5 input features',flush=True)
    new=build_points_oof(d,b,first_year=args.first_year,last_year=args.last_year)
    pairs,by=paired_metrics(old,new)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    for f in ('overall.csv','by_season.csv','paired_game_errors.csv'):
        if (out / f).exists():
            raise FileExistsError(f'Refusing to overwrite previous comparison: {out / f}')
    n=len(pairs)
    overall=pd.DataFrame([{
        'games':n,
        'v3_margin_mae':pairs.margin_absolute_error_v3.mean(),
        'v45_margin_mae':pairs.margin_absolute_error_v45.mean(),
        'delta_margin_mae':pairs.margin_delta_error.mean(),
        'v3_total_mae':pairs.total_absolute_error_v3.mean(),
        'v45_total_mae':pairs.total_absolute_error_v45.mean(),
        'delta_total_mae':pairs.total_delta_error.mean(),
        'seasons_margin_improved':int((by.delta_margin_mae<0).sum()),
        'seasons_total_improved':int((by.delta_total_mae<0).sum()),
    }])
    overall.to_csv(out / 'overall.csv',index=False)
    by.to_csv(out / 'by_season.csv',index=False)
    pairs.to_csv(out / 'paired_game_errors.csv',index=False)
    print('\nOVERALL — v4.5 minus v3, negative is better')
    print(overall.round(4).to_string(index=False))
    print('\nBY SEASON')
    print(by.round(4).to_string(index=False))
    print(f'\nSaved to {out}. No sportsbook EV or unbiased future performance is established.')

if __name__=='__main__':
    main()
