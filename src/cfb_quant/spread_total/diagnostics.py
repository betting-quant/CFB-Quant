"""Read-only diagnostics for the historical CFB spread/total validation CSV.

This does not train, modify upstream data, access CFBD, or imply profitable EV.
It checks paired game results and reports where forecast errors and probability
calibration may be worse than headline means.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    'game_id', 'season', 'week', 'actual_margin', 'actual_total',
    'original_margin', 'original_total', 'fair_home_margin_mean',
    'fair_total_mean', 'margin_p10', 'margin_p90', 'total_p10', 'total_p90',
    'margin_80_covered', 'total_80_covered', 'prior_max_season', 'p_home_win',
}
NUMERIC = REQUIRED - {'game_id'}


def load_validation(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.empty:
        raise ValueError('Empty historical validation file')
    df['game_id'] = df['game_id'].astype(str)
    for key in NUMERIC:
        df[key] = pd.to_numeric(df[key], errors='coerce')
    if df[list(NUMERIC)].isna().any().any():
        raise ValueError('Invalid or missing numeric values')
    if not np.isfinite(df[list(NUMERIC)].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite numeric values')
    if df.duplicated('game_id').any():
        raise ValueError('Duplicate game_id in historical validation')
    if (df['prior_max_season'] >= df['season']).any():
        raise ValueError('Chronological residual leakage detected')
    if not df.p_home_win.between(0, 1).all():
        raise ValueError('Home win probabilities outside [0,1]')
    if ((df.actual_total < 0) | (df.actual_total + 1e-8 < df.actual_margin.abs())).any():
        raise ValueError('Actual scores inconsistent with margin/total')
    if ((df.margin_p90 < df.margin_p10) | (df.total_p90 < df.total_p10)).any():
        raise ValueError('Reversed predictive interval bounds')
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in ('margin', 'total'):
        out[f'{t}_bias'] = out[f'fair_{"home_margin_mean" if t == "margin" else "total_mean"}'] - out[f'actual_{t}']
        out[f'{t}_abs_error'] = out[f'{t}_bias'].abs()
        out[f'raw_{t}_abs_error'] = (out[f'original_{t}'] - out[f'actual_{t}']).abs()
        out[f'{t}_below_p10'] = (out[f'actual_{t}'] < out[f'{t}_p10']).astype(float)
        out[f'{t}_above_p90'] = (out[f'actual_{t}'] > out[f'{t}_p90']).astype(float)
        out[f'{t}_80_covered_check'] = (out[f'{t}_below_p10'].eq(0) & out[f'{t}_above_p90'].eq(0)).astype(float)
        if not np.allclose(out[f'{t}_80_covered_check'], out[f'{t}_80_covered']):
            raise ValueError(f'{t} interval coverage column disagrees with saved quantiles')
    out['home_won'] = (out.actual_margin > 0).astype(float)
    out['home_brier'] = (out.p_home_win - out.home_won) ** 2
    out['home_climatology'] = float(out.home_won.mean())
    out['home_brier_constant'] = (out.home_climatology - out.home_won) ** 2
    return out


def metric_table(df: pd.DataFrame, group: str) -> pd.DataFrame:
    groups = df.groupby(group, dropna=False, observed=True)
    result = groups.agg(
        games=('game_id', 'size'),
        margin_mae=('margin_abs_error', 'mean'),
        total_mae=('total_abs_error', 'mean'),
        original_margin_mae=('raw_margin_abs_error', 'mean'),
        original_total_mae=('raw_total_abs_error', 'mean'),
        margin_bias=('margin_bias', 'mean'),
        total_bias=('total_bias', 'mean'),
        margin_80_covered=('margin_80_covered', 'mean'),
        total_80_covered=('total_80_covered', 'mean'),
        margin_lower_tail=('margin_below_p10', 'mean'),
        margin_upper_tail=('margin_above_p90', 'mean'),
        total_lower_tail=('total_below_p10', 'mean'),
        total_upper_tail=('total_above_p90', 'mean'),
        home_win_brier=('home_brier', 'mean'),
    ).reset_index()
    return result


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, values in (
        ('projected_total_quartile', out.original_total),
        ('projected_mismatch_quartile', out.original_margin.abs()),
    ):
        ranked = values.rank(method='first')
        out[name] = pd.qcut(ranked, 4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high']).astype(str)
    out['season_period'] = np.where(out.week <= 4, 'early_weeks_1_to_4', 'later_week_5_plus')
    return out


def home_probability_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, 1, 11)
    out = df.copy()
    out['p_bin'] = pd.cut(out.p_home_win, bins=bins, include_lowest=True)
    summary = out.groupby('p_bin', observed=False).agg(
        games=('game_id', 'size'),
        mean_predicted=('p_home_win', 'mean'),
        observed_win_rate=('home_won', 'mean'),
        brier=('home_brier', 'mean'),
    ).reset_index()
    summary['p_bin'] = summary.p_bin.astype(str)
    return summary


def save_reports(df: pd.DataFrame, outdir: Path) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    df = add_groups(enrich(df))
    paths = {}
    for name, group in (
        ('overall', 'all_games'),
        ('by_season', 'season'),
        ('by_total', 'projected_total_quartile'),
        ('by_mismatch', 'projected_mismatch_quartile'),
        ('by_season_period', 'season_period'),
    ):
        frame = df.assign(all_games='all') if group == 'all_games' else df
        destination = outdir / f'validation_{name}_v2.csv'
        metric_table(frame, group).round(4).to_csv(destination, index=False)
        paths[name] = destination
    dest = outdir / 'validation_home_win_calibration_v2.csv'
    home_probability_bins(df).round(4).to_csv(dest, index=False)
    paths['home_win_calibration'] = dest
    return paths


def main():
    ap = argparse.ArgumentParser(description='Read-only reliability and error diagnostics on historical CFB spreads/totals')
    ap.add_argument('--infile', default='reports/spread_total_historical_validation_v1.csv')
    ap.add_argument('--outdir', default='reports/spread_total_diagnostics_v2')
    args = ap.parse_args()
    data = load_validation(args.infile)
    enrich_data = enrich(data)
    paths = save_reports(data, Path(args.outdir))
    n = len(enrich_data)
    print(f'Checked {n:,} unique, chronological historical games ({int(data.season.min())}-{int(data.season.max())}).')
    print('WEIGHTED ALL-SEASON METRICS')
    print(f'  Original margin MAE: {enrich_data.raw_margin_abs_error.mean():.3f}')
    print(f'  Simulated margin MAE: {enrich_data.margin_abs_error.mean():.3f}')
    print(f'  Original total MAE: {enrich_data.raw_total_abs_error.mean():.3f}')
    print(f'  Simulated total MAE: {enrich_data.total_abs_error.mean():.3f}')
    print(f'  Margin interval coverage: {enrich_data.margin_80_covered.mean():.1%} (target ~80%)')
    print(f'  Total interval coverage: {enrich_data.total_80_covered.mean():.1%} (target ~80%)')
    print(f'  Margin lower/upper tails: {enrich_data.margin_below_p10.mean():.1%} / {enrich_data.margin_above_p90.mean():.1%}')
    print(f'  Total lower/upper tails: {enrich_data.total_below_p10.mean():.1%} / {enrich_data.total_above_p90.mean():.1%}')
    print(f'  Home-win Brier: {enrich_data.home_brier.mean():.4f}')
    print(f'  Home-win constant baseline Brier: {enrich_data.home_brier_constant.mean():.4f} (not a market benchmark)')
    print('\nSaved read-only diagnostic files:')
    for name, path in paths.items():
        print(f'  {name}: {path}')
    print('\nThese metrics DO NOT establish wagering profitability or unbiased upstream features.')


if __name__ == '__main__':
    main()
