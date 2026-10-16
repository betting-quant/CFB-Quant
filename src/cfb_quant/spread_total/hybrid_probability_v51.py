"""Experimental v5.1 hybrid spread/total probability baseline.

Compares (a) v3 margin+total, (b) v4.5 margin+total, and
(c) v4.5 margin + v3 total on identical historical games.

Paired home/away residuals are sampled from PRIOR seasons only, preserving
margin/total correlation and yielding physically nonnegative team scores.
This is NOT a model of game possessions and NOT a betting/EV engine.

Known limitations: history repeatedly used in development, unverified
point-in-time source availability, FBS/FCS eligibility unverified, no odds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path('reports/spread_total_opponent_v45/paired_game_errors.csv')
OUT = Path('reports/spread_total_hybrid_probability_v51')

REQUIRED = [
    'game_id', 'season', 'actual_margin_v3', 'actual_margin_v45',
    'actual_total_v3', 'actual_total_v45', 'pred_margin_v3',
    'pred_margin_v45', 'pred_total_v3', 'pred_total_v45',
]
MODELS = {
    'v3': ('pred_margin_v3', 'pred_total_v3'),
    'v45': ('pred_margin_v45', 'pred_total_v45'),
    'hybrid': ('pred_margin_v45', 'pred_total_v3'),
}
LINES = [
    ('home_win', 'margin', 0.5),
    ('home_covers_minus_7p5', 'margin', 7.5),
    ('home_covers_plus_7p5', 'margin', -7.5),
    ('over_45p5', 'total', 45.5),
    ('over_55p5', 'total', 55.5),
    ('over_65p5', 'total', 65.5),
]


def read_and_validate(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    missing = set(REQUIRED) - set(d.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if d.duplicated(['season', 'game_id']).any():
        raise ValueError('Duplicate season/game identifier')
    for c in REQUIRED[2:]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    if not np.isfinite(d[REQUIRED[2:]].to_numpy(dtype=float)).all():
        raise ValueError('Missing or nonfinite scored/predicted values')
    if not (d.actual_margin_v3 == d.actual_margin_v45).all():
        raise ValueError('Actual margins mismatch between models')
    if not (d.actual_total_v3 == d.actual_total_v45).all():
        raise ValueError('Actual totals mismatch between models')
    if (d.actual_total_v3 < d.actual_margin_v3.abs()).any():
        raise ValueError('Impossible observed scores')
    if d.season.isna().any():
        raise ValueError('Missing season')
    return d.sort_values(['season', 'game_id']).reset_index(drop=True)


def paired_scoring_residuals(rows: pd.DataFrame, margin_col: str, total_col: str) -> np.ndarray:
    """Historic home/away errors created from the same game's M and T errors."""
    me = (rows.actual_margin_v3 - rows[margin_col]).to_numpy(dtype=float)
    te = (rows.actual_total_v3 - rows[total_col]).to_numpy(dtype=float)
    return np.column_stack([(te+me)/2.0, (te-me)/2.0])


def simulate_scores(pred_m: np.ndarray, pred_t: np.ndarray, history: np.ndarray,
                    sample_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All sampled errors paired within the same historical game.

    A minimum score of zero is a simplifying assumption; this is not a
    fully calibrated possession-level scoring process.
    """
    ph = (pred_t+pred_m)/2.0
    pa = (pred_t-pred_m)/2.0
    er = history[sample_indices]
    home = np.maximum(0.0, ph[:,None] + er[:,:,0])
    away = np.maximum(0.0, pa[:,None] + er[:,:,1])
    return home - away, home + away


def empirical_crps(predictions: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Exact CRPS of a finite equal-weight empirical distribution (vectorized)."""
    n = predictions.shape[1]
    if n < 2:
        raise ValueError('At least two simulation samples required')
    x = np.sort(predictions,axis=1)
    j = np.arange(1, n+1, dtype=float)
    weighted_gini_half = (x * (2*j - n - 1)).sum(axis=1) / (n*n)
    return np.abs(predictions - actual[:,None]).mean(axis=1) - weighted_gini_half


def run_experiment(d: pd.DataFrame, *, start_year: int = 2020,
                   end_year: int = 2024, samples: int = 1200,
                   seed: int = 45) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if samples < 200:
        raise ValueError('Use at least 200 samples for interval/probability estimates')
    if start_year > end_year or end_year >= 2025 or start_year < 2020:
        raise ValueError('Historical experimental evaluation restricted to 2020–2024')
    if d.loc[d.season < start_year].empty:
        raise ValueError('No prior-season residuals for start year')
    outputs = []
    for season in range(start_year, end_year+1):
        pool = d.loc[d.season < season]
        test = d.loc[d.season == season]
        if len(pool) < 50 or test.empty:
            raise ValueError(f'Missing previous-year calibration data or evaluation games in {season}')
        assert int(pool.season.max()) < season
        # Resample historical game indices identically across variants for lower Monte Carlo noise.
        rng = np.random.default_rng(seed + season)
        idx = rng.integers(0, len(pool), size=(len(test), samples))
        actual_m = test.actual_margin_v3.to_numpy(dtype=float)
        actual_t = test.actual_total_v3.to_numpy(dtype=float)
        for model, (mcol,tcol) in MODELS.items():
            history = paired_scoring_residuals(pool,mcol,tcol)
            pm = test[mcol].to_numpy(dtype=float)
            pt = test[tcol].to_numpy(dtype=float)
            sm, st = simulate_scores(pm,pt,history,idx)
            frame = pd.DataFrame({
                'game_id':test.game_id.to_numpy(), 'season':season, 'variant':model,
                'residual_pool_seasons_max':int(pool.season.max()),
                'calibration_pool_games':len(pool),
                'pred_margin':pm, 'pred_total':pt,
                'actual_margin':actual_m, 'actual_total':actual_t,
                'margin_abs_error':np.abs(actual_m-pm),
                'total_abs_error':np.abs(actual_t-pt),
                'margin_crps':empirical_crps(sm,actual_m),
                'total_crps':empirical_crps(st,actual_t),
                'margin_80_covered':(actual_m>=np.quantile(sm,0.1,axis=1)) & (actual_m<=np.quantile(sm,0.9,axis=1)),
                'total_80_covered':(actual_t>=np.quantile(st,0.1,axis=1)) & (actual_t<=np.quantile(st,0.9,axis=1)),
            })
            for name,target,line in LINES:
                vals, actual = (sm,actual_m) if target=='margin' else (st,actual_t)
                prob = (vals>line).mean(axis=1)
                obs = (actual>line).astype(float)
                frame[f'p_{name}'] = prob
                frame[f'brier_{name}'] = np.square(prob-obs)
            outputs.append(frame)
        print(f'{season}: evaluated {len(test)} games; prior-season residual pool {len(pool)} through {int(pool.season.max())}',flush=True)
    games = pd.concat(outputs,ignore_index=True)
    metrics = ['margin_abs_error','total_abs_error','margin_crps','total_crps','margin_80_covered','total_80_covered'] + [f'brier_{name}' for name,_,_ in LINES]
    by = games.groupby(['variant','season'],as_index=False).agg(games=('game_id','size'),**{c:(c,'mean') for c in metrics})
    overall=games.groupby('variant',as_index=False).agg(games=('game_id','size'),**{c:(c,'mean') for c in metrics})
    return games,by,overall


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default=str(SOURCE))
    p.add_argument('--out-dir',default=str(OUT))
    p.add_argument('--start-year',type=int,default=2020)
    p.add_argument('--end-year',type=int,default=2024)
    p.add_argument('--samples',type=int,default=1200)
    p.add_argument('--seed',type=int,default=45)
    a=p.parse_args()
    dest=Path(a.out_dir)
    files=['overall.csv','by_season.csv','game_predictions.csv']
    if any((dest / f).exists() for f in files):
        p.error('Output already exists. Refusing to overwrite an earlier experiment; choose a new --out-dir.')
    if not Path(a.input).exists():
        p.error(f'Missing historical comparison: {a.input}. No API download needed.')
    d=read_and_validate(a.input)
    games,by,overall=run_experiment(d,start_year=a.start_year,end_year=a.end_year,samples=a.samples,seed=a.seed)
    dest.mkdir(parents=True,exist_ok=True)
    for filename,df in [('overall.csv',overall),('by_season.csv',by),('game_predictions.csv',games)]:
        df.to_csv(dest/filename,index=False)
    print('\nHYBRID JOINT SCORE PROBABILITY EXPERIMENT — NOT A BETTING BACKTEST')
    print(overall[['variant','games','margin_abs_error','total_abs_error','margin_crps','total_crps','margin_80_covered','total_80_covered']].round(4).to_string(index=False))
    print('\nSaved:', dest)
    print('WARNING: 2019 is used only as prior calibration for 2020; no 2025 holdout used.')
    print('FBS-vs-FBS status and upstream point-in-time sources remain unverified. No sportsbook EV data.')

if __name__=='__main__':
    main()
