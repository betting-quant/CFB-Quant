"""v5.5 — Strictly walk-forward, shared monotone recalibration for FBS spread events.

Research only: reuses 2020-2024 development games and applies the SAME
monotone logit transform at home +7.5 / home win / home -7.5 to preserve
cross-line probability order. Never changes underlying forecasts or totals.

No financial recommendation and no claim of profitability. No API calls.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

DEFAULT_SOURCE = Path('reports/spread_total_phase_volatility_v53/game_predictions.csv')
DEFAULT_DIR = Path('reports/spread_total_rolling_calibration_v55')
EVENTS = (
    ('home_covers_plus_7p5', -7.5),
    ('home_win', 0.5),
    ('home_covers_minus_7p5', 7.5),
)
EPS = 1e-5


def prepare(path: str | Path, variant: str = 'pooled') -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {'game_id', 'season', 'variant', 'phase', 'max_residual_season', 'actual_margin'} | {'p_' + e for e, _ in EVENTS}
    missing = required - set(raw)
    if missing:
        raise ValueError(f'Input is missing fields: {sorted(missing)}')
    d = raw.loc[raw.variant == variant].copy()
    if d.empty:
        raise ValueError(f'Variant {variant!r} not found')
    d['season'] = pd.to_numeric(d.season, errors='raise').astype(int)
    d['max_residual_season'] = pd.to_numeric(d.max_residual_season, errors='raise').astype(int)
    if d.duplicated(['season', 'game_id']).any():
        raise ValueError('Duplicate game predictions')
    if not (d.max_residual_season < d.season).all():
        raise ValueError('Original residual predictions violate chronological isolation')
    if not (d.season.between(2020, 2024)).all():
        raise ValueError('This research version only supports existing 2020-2024 development predictions')
    if d.phase.isna().any():
        raise ValueError('Missing game phase')
    for col in ['actual_margin'] + ['p_' + e for e, _ in EVENTS]:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not np.isfinite(d[col].to_numpy()).all():
            raise ValueError(f'Missing/nonfinite {col}')
    ps = d[['p_'+e for e,_ in EVENTS]].to_numpy(dtype=float)
    if np.any((ps < 0) | (ps > 1)):
        raise ValueError('Invalid probability outside 0..1')
    if np.any(np.diff(ps,axis=1) > 1e-10):
        raise ValueError('Input spread probabilities violate required order: +7.5 >= win >= -7.5')
    return d.sort_values(['season','game_id']).reset_index(drop=True)


def to_arrays(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probabilities = d[['p_' + e for e,_ in EVENTS]].to_numpy(dtype=float)
    margins = d.actual_margin.to_numpy(dtype=float)
    outcomes = np.column_stack([(margins > threshold).astype(float) for _,threshold in EVENTS])
    return probabilities, outcomes


def transform(probs: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope <= 0:
        raise ValueError('Monotone transform requires a finite intercept and positive slope')
    p = np.clip(np.asarray(probs,dtype=float), EPS, 1-EPS)
    return expit(intercept + slope*logit(p))


def fit_prior(p: np.ndarray, y: np.ndarray, *, penalty: float = 0.002) -> tuple[float,float]:
    """Fit one shared transform for all three thresholds using prior games only.

    Each game contributes the average Brier over its three correlated events;
    no artificial claim of 3 independent observations per game.
    """
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != len(EVENTS):
        raise ValueError('Expect matching (games,3) matrices')
    if len(p) < 150:
        raise ValueError('Insufficient previous-season games for calibration')
    def objective(params: np.ndarray) -> float:
        a,b = params
        corrected = transform(p,float(a),float(b))
        return float(np.mean((corrected-y)**2) + penalty*(a*a+(b-1)*(b-1)))
    result=minimize(objective,[0.,1.],bounds=[(-1.0,1.0),(0.5,2.5)],method='L-BFGS-B', options={'maxiter':300})
    if not result.success:
        raise RuntimeError(f'Calibration solver failed: {result.message}')
    return float(result.x[0]),float(result.x[1])


def walk_forward(d: pd.DataFrame, min_train_games: int=150) -> tuple[pd.DataFrame,pd.DataFrame]:
    if min_train_games < 150:
        raise ValueError('Refusing small calibration pools')
    forecast_rows=[]
    audit=[]
    for year in sorted(d.season.unique()):
        test=d.loc[d.season == year].copy()
        history=d.loc[d.season < year]
        if len(history)>=min_train_games:
            p_train,y_train=to_arrays(history)
            a,b=fit_prior(p_train,y_train)
            max_training=int(history.season.max())
        else:
            a,b=0.,1.
            max_training=None
        p,y=to_arrays(test)
        adjusted=transform(p,a,b)
        if np.any(np.diff(adjusted,axis=1)>1e-10):
            raise AssertionError('Probability monotonicity unexpectedly violated')
        forecast_rows.append(pd.DataFrame({
            'season':int(year),'game_id':test.game_id.to_numpy(),'phase':test.phase.to_numpy(),
            'variant':test.variant.to_numpy(),'training_games':len(history),
            'max_calibration_season':max_training,
            'actual_margin':test.actual_margin.to_numpy(),
            **{f'p_raw_{event}':p[:,j] for j,(event,_) in enumerate(EVENTS)},
            **{f'p_adjusted_{event}':adjusted[:,j] for j,(event,_) in enumerate(EVENTS)},
            **{f'y_{event}':y[:,j] for j,(event,_) in enumerate(EVENTS)},
        }))
        audit.append({'season':int(year),'evaluation_games':len(test), 'prior_training_games':len(history),
                      'max_calibration_season':max_training, 'intercept':a,'slope':b,
                      'calibration_applied':bool(len(history)>=min_train_games)})
    games=pd.concat(forecast_rows,ignore_index=True)
    history_audit=pd.DataFrame(audit)
    active=history_audit.calibration_applied
    if (history_audit.loc[active,'max_calibration_season'] >= history_audit.loc[active,'season']).any():
        raise AssertionError('Future-season training contamination')
    return games,history_audit


def summarize(games: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records=[]
    for group, sub in games.groupby(group_columns,dropna=False,sort=True):
        group=group if isinstance(group,tuple) else (group,)
        label=dict(zip(group_columns,group))
        for event,_ in EVENTS:
            y=sub[f'y_{event}'].to_numpy(float)
            raw=sub[f'p_raw_{event}'].to_numpy(float)
            adj=sub[f'p_adjusted_{event}'].to_numpy(float)
            records.append({**label,'event':event,'games':len(sub),
                            'observed_rate':float(y.mean()),'raw_mean_probability':float(raw.mean()),
                            'adjusted_mean_probability':float(adj.mean()),
                            'brier_raw':float(np.mean((raw-y)**2)),
                            'brier_adjusted':float(np.mean((adj-y)**2)),
                            'brier_delta_adjusted_minus_raw':float(np.mean((adj-y)**2)-np.mean((raw-y)**2))})
    return pd.DataFrame(records)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',default=str(DEFAULT_SOURCE))
    ap.add_argument('--out-dir',default=str(DEFAULT_DIR))
    ap.add_argument('--variant', choices=['pooled','phase_mixed'],default='pooled')
    args=ap.parse_args()
    input_path=Path(args.input)
    if not input_path.is_file():
        ap.error(f'Missing {input_path}. Expected file from prior v5.3 run; no API download needed.')
    target=Path(args.out_dir)
    files=['by_season.csv','overall.csv','by_phase.csv','calibration_coefficients.csv','game_predictions.csv']
    if any((target/f).exists() for f in files):
        ap.error('Output files already exist; supply a new --out-dir to preserve earlier experiments')
    source=prepare(input_path, args.variant)
    games, audit=walk_forward(source)
    reports={'by_season.csv':summarize(games,['season']),
             'overall.csv':summarize(games,['variant']),
             'by_phase.csv':summarize(games,['phase']),
             'calibration_coefficients.csv':audit,'game_predictions.csv':games}
    target.mkdir(parents=True,exist_ok=True)
    for filename,data in reports.items():
        data.to_csv(target/filename,index=False)
    print('v5.5 WALK-FORWARD SPREAD CALIBRATION — DEVELOPMENT RESEARCH ONLY')
    print('Season-specific training (prior seasons only):')
    print(audit.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nOverall probabilities; negative Brier delta = improvement:')
    print(reports['overall.csv'][['event','games','brier_raw','brier_adjusted','brier_delta_adjusted_minus_raw']].round(6).to_string(index=False))
    print(f'Outputs saved to: {target}')
    print('NOT VALIDATED FOR BETTING: 2020-24 repeatedly examined during development; no odds, no unexamined future games.')
    print('This transform changes spread probabilities only; it does NOT change totals, point forecasts, or score simulation.')

if __name__ == '__main__':
    main()
