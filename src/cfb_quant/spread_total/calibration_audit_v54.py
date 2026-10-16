"""v5.4 read-only diagnostics: distinguish point-forecast bias from probability calibration.

Works with the *existing* v5.3 `game_predictions.csv`. Never retrains,
recalibrates, requests API data, or changes a model. Development research only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path('reports/spread_total_phase_volatility_v53/game_predictions.csv')
DEFAULT_OUTPUT = Path('reports/spread_total_calibration_audit_v54')
LINES = {
    'home_win': ('margin', 0.5),
    'home_covers_minus_7p5': ('margin', 7.5),
    'home_covers_plus_7p5': ('margin', -7.5),
    'over_45p5': ('total', 45.5),
    'over_55p5': ('total', 55.5),
    'over_65p5': ('total', 65.5),
}
REQUIRED = {'game_id','season','variant','phase','max_residual_season',
            'pred_margin','actual_margin','pred_total','actual_total',
            'margin_80_covered','total_80_covered'} | {'p_' + k for k in LINES}


def load_games(path: str | Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    miss = REQUIRED - set(d.columns)
    if miss:
        raise ValueError(f'Missing columns: {sorted(miss)}')
    if set(d.variant) != {'pooled', 'phase_mixed'}:
        raise ValueError('Expected both pooled and phase_mixed variants')
    if d.duplicated(['season','game_id','variant']).any():
        raise ValueError('Duplicate game/season/variant records')
    d['season'] = pd.to_numeric(d.season, errors='raise').astype(int)
    d['max_residual_season'] = pd.to_numeric(d.max_residual_season, errors='raise').astype(int)
    if not (d.max_residual_season < d.season).all():
        raise ValueError('Chronological residual-pool leakage found')
    for col in ['pred_margin','actual_margin','pred_total','actual_total'] + ['p_'+k for k in LINES]:
        d[col] = pd.to_numeric(d[col], errors='raise')
        if not np.isfinite(d[col].to_numpy()).all():
            raise ValueError(f'Nonfinite {col}')
    for name in LINES:
        if not d['p_'+name].between(0,1).all():
            raise ValueError(f'Invalid probability p_{name}')
    if (d.actual_total < d.actual_margin.abs()).any():
        raise ValueError('Impossible actual score/margin combinations')
    variants = d.pivot(index=['season','game_id'],columns='variant',values='pred_margin')
    if variants.isna().any().any() or not np.allclose(variants.pooled, variants.phase_mixed):
        raise ValueError('Variants are not paired on identical games/point projections')
    for col in ['pred_total','actual_margin','actual_total','phase']:
        p = d.pivot(index=['season','game_id'],columns='variant',values=col)
        if col == 'phase':
            same = bool((p.pooled == p.phase_mixed).all())
        else:
            same = bool(np.allclose(p.pooled, p.phase_mixed))
        if not same:
            raise ValueError(f'Variants differ on {col}')
    return d


def _wilson(s: int, n: int, z: float = 1.95996398454) -> tuple[float,float]:
    if n <= 0:
        return float('nan'), float('nan')
    p = s/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    margin = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return center-margin, center+margin


def event_tables(d: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    overall = []
    by_bin = []
    for variant, df in d.groupby('variant',sort=True):
        for event,(target,line) in LINES.items():
            p = df['p_'+event].to_numpy(dtype=float)
            y = (df['actual_'+target].to_numpy(dtype=float)>line).astype(int)
            n = len(df); successes = int(y.sum()); lo,hi = _wilson(successes,n)
            overall.append({'variant':variant,'event':event,'games':n,
                            'average_probability':float(p.mean()),'observed_frequency':float(y.mean()),
                            'signed_gap_observed_minus_predicted':float((y-p).mean()),
                            'brier':float(np.mean((p-y)**2)),
                            'observed_wilson_95_low':lo,'observed_wilson_95_high':hi})
            bins = np.minimum((10*p).astype(int),9)
            for b in range(10):
                sub=bins==b
                if not sub.any(): continue
                m=int(sub.sum()); wins=int(y[sub].sum()); a,z=_wilson(wins,m)
                by_bin.append({'variant':variant,'event':event,'bin':b,'games':m,
                              'average_probability':float(p[sub].mean()),
                              'observed_frequency':float(y[sub].mean()),
                              'signed_gap_observed_minus_predicted':float((y[sub]-p[sub]).mean()),
                              'observed_wilson_95_low':a,'observed_wilson_95_high':z,
                              'small_bin':m<50})
    return pd.DataFrame(overall),pd.DataFrame(by_bin)


def phase_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v,ph),g in d.groupby(['variant','phase'],sort=True):
        rows.append({'variant':v,'phase':ph,'games':len(g),
                     'signed_margin_error_actual_minus_predicted':float((g.actual_margin-g.pred_margin).mean()),
                     'margin_mae':float((g.actual_margin-g.pred_margin).abs().mean()),
                     'margin_rmse':float(np.sqrt(np.mean((g.actual_margin-g.pred_margin)**2))),
                     'margin_80_coverage':float(g.margin_80_covered.mean()),
                     'total_80_coverage':float(g.total_80_covered.mean())})
    return pd.DataFrame(rows)


def margin_bias_bins(d: pd.DataFrame) -> pd.DataFrame:
    # Only one row per game. Prevents falsely counting v5.3's two variants twice.
    x = d.loc[d.variant.eq('pooled')].copy()
    x['predicted_favorite'] = np.select([x.pred_margin>7.5,x.pred_margin< -7.5],
                                        ['home_by_8_plus','away_by_8_plus'],default='close_game')
    x['pred_margin_quartile'] = pd.qcut(x.pred_margin.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
    rows=[]
    for (phase,b),g in x.groupby(['phase','predicted_favorite'],observed=True,sort=True):
        rows.append({'phase':phase,'bucket':str(b),'games':len(g),
                     'mean_predicted_margin':float(g.pred_margin.mean()),
                     'mean_actual_margin':float(g.actual_margin.mean()),
                     'signed_error_actual_minus_predicted':float((g.actual_margin-g.pred_margin).mean()),
                     'margin_mae':float((g.actual_margin-g.pred_margin).abs().mean())})
    for (q,),g in x.groupby(['pred_margin_quartile'],observed=True,sort=True):
        rows.append({'phase':'ALL','bucket':'predicted_margin_'+str(q),'games':len(g),
                     'mean_predicted_margin':float(g.pred_margin.mean()),
                     'mean_actual_margin':float(g.actual_margin.mean()),
                     'signed_error_actual_minus_predicted':float((g.actual_margin-g.pred_margin).mean()),
                     'margin_mae':float((g.actual_margin-g.pred_margin).abs().mean())})
    return pd.DataFrame(rows)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default=str(DEFAULT_SOURCE))
    p.add_argument('--out-dir',default=str(DEFAULT_OUTPUT))
    args=p.parse_args()
    source=Path(args.input); output=Path(args.out_dir)
    if not source.is_file():p.error(f'Expected previous v5.3 game_predictions.csv: {source}')
    if output.exists() and any(output.iterdir()):
        p.error(f'Output directory already contains files: {output}; choose another --out-dir')
    d=load_games(source)
    overall,bins=event_tables(d)
    reports={'event_summary.csv':overall,'event_calibration_bins.csv':bins,
             'by_phase.csv':phase_table(d),'margin_bias_by_matchup.csv':margin_bias_bins(d)}
    output.mkdir(parents=True,exist_ok=True)
    for filename,df in reports.items():df.to_csv(output/filename,index=False)
    print('v5.4 READ-ONLY CALIBRATION DIAGNOSTICS')
    print(f'Games: {len(d)//2}; forecasts: {len(d)}; prior-season residual check: passed')
    print(overall.loc[overall.event.isin(['home_covers_minus_7p5','home_covers_plus_7p5'])][
          ['variant','event','games','average_probability','observed_frequency',
           'signed_gap_observed_minus_predicted','brier']].round(4).to_string(index=False))
    print('Phase error analysis:');print(reports['by_phase.csv'].round(4).to_string(index=False))
    print(f'Outputs: {output}')
    print('No parameters fitted. NOT independent validation, no odds, no +EV claim.')

if __name__ == '__main__': main()
