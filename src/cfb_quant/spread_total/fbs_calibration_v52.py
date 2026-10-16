"""Read-only v5.2 FBS-only joint probability and calibration audit.

Requires existing historical paired predictions and locally cached CFBD schedules.
The FBS filter is applied to BOTH evaluation games and earlier-season residual
pools. This is exploratory retrospective validation, NOT betting verification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hybrid_probability_v51 import read_and_validate, run_experiment, LINES

SOURCE = Path('reports/spread_total_opponent_v45/paired_game_errors.csv')
DEST = Path('reports/spread_total_fbs_calibration_v52')
EVENTS = [name for name, _, _ in LINES]


def _game_key(values: pd.Series) -> pd.Series:
    """Normalise integer-like CFBD game ids without altering actual IDs."""
    return values.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)


def load_schedule(years: list[int], root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for year in years:
        paths = [
            root / f'data/raw/cfbd/chronology_v2/games_{year}.json',
            root / f'data/raw/cfbd/games/games_{year}.json',
        ]
        match = next((p for p in paths if p.is_file()), None)
        if match is None:
            raise FileNotFoundError(f'No cached CFBD schedule for {year}; checked: {paths}')
        payload = json.loads(match.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, list):
            raise ValueError(f'Expected a list of game records in {match}')
        for game in payload:
            if not isinstance(game, dict) or game.get('id') is None:
                continue
            rows.append({
                'game_id': str(game['id']),
                'season': year,
                'home_classification': str(game.get('homeClassification') or '').lower().strip(),
                'away_classification': str(game.get('awayClassification') or '').lower().strip(),
                'scheduled_week': game.get('week'),
                'season_type': str(game.get('seasonType') or '').lower().strip(),
                'home_team': str(game.get('homeTeam') or ''),
                'away_team': str(game.get('awayTeam') or ''),
            })
    if not rows:
        raise ValueError('No usable schedule records')
    schedule = pd.DataFrame(rows)
    if schedule.duplicated(['game_id', 'season']).any():
        duplicates = schedule[schedule.duplicated(['game_id', 'season'], keep=False)]
        raise ValueError(f'Duplicate schedule keys: {duplicates[["season","game_id"]].head().to_dict("records")}')
    return schedule


def attach_universe(data: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d['game_id'] = _game_key(d['game_id'])
    s = schedule.copy()
    s['game_id'] = _game_key(s['game_id'])
    if d.duplicated(['season','game_id']).any():
        raise ValueError('Duplicate prediction keys')
    if s.duplicated(['season','game_id']).any():
        raise ValueError('Duplicate cached schedule keys')
    joined = d.merge(s, on=['season','game_id'], how='left', validate='one_to_one', indicator=True)
    has_info = (joined['_merge'].eq('both') &
                joined['home_classification'].notna() &
                joined['away_classification'].notna() &
                joined['home_classification'].ne('') &
                joined['away_classification'].ne(''))
    fbs = (joined['home_classification'].eq('fbs') & joined['away_classification'].eq('fbs'))
    joined['universe_status'] = np.select([~has_info, fbs], ['unknown', 'fbs_fbs'], default='other')
    return joined.drop(columns='_merge')


def bin_calibration(games: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for event in EVENTS:
        col = f'p_{event}'
        if col not in games:
            raise ValueError(f'Missing probability {col}')
        vals = pd.to_numeric(games[col], errors='raise')
        if vals.isna().any() or not vals.between(0, 1).all():
            raise ValueError(f'Invalid probabilities for {event}')
        if event == 'home_win':
            observed = games['actual_margin'] > 0.5
        elif event.startswith('home_covers_minus_'):
            observed = games['actual_margin'] > 7.5
        elif event.startswith('home_covers_plus_'):
            observed = games['actual_margin'] > -7.5
        elif event.startswith('over_'):
            raw = event.split('_')[1].replace('p', '.')
            observed = games['actual_total'] > float(raw)
        else:
            raise ValueError(f'Unrecognised event: {event}')
        t = games[['variant','season']].copy()
        t['event'] = event
        t['p'] = vals.to_numpy()
        t['observed'] = observed.astype(float).to_numpy()
        t['bin'] = np.minimum((t.p * 10).astype(int), 9)
        frames.append(t)
    d = pd.concat(frames, ignore_index=True)
    agg = d.groupby(['variant','event','bin'], as_index=False).agg(
        count=('p','size'), average_probability=('p','mean'), observed_frequency=('observed','mean'))
    agg['calibration_gap'] = agg['observed_frequency'] - agg['average_probability']
    agg['small_bin'] = agg['count'] < 50
    return agg


def season_phase(games: pd.DataFrame, filtered_schedule: pd.DataFrame) -> pd.DataFrame:
    d = games.merge(filtered_schedule[['season','game_id','season_type','scheduled_week']],
                    on=['season','game_id'], how='left', validate='many_to_one')
    wk = pd.to_numeric(d['scheduled_week'], errors='coerce')
    d['phase'] = np.select([d['season_type'].eq('postseason'), wk.le(4), wk.gt(4)],
                           ['postseason', 'regular_weeks_1_4', 'regular_week_5_plus'], default='unknown')
    return d.groupby(['variant','phase'], as_index=False).agg(
        games=('game_id','size'), margin_mae=('margin_abs_error','mean'),
        total_mae=('total_abs_error','mean'), margin_crps=('margin_crps','mean'),
        total_crps=('total_crps','mean'), margin_80_coverage=('margin_80_covered','mean'),
        total_80_coverage=('total_80_covered','mean'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='.')
    ap.add_argument('--input', default=str(SOURCE))
    ap.add_argument('--out-dir', default=str(DEST))
    ap.add_argument('--samples', type=int, default=1200)
    args = ap.parse_args()
    root = Path(args.root)
    out = root / args.out_dir
    outputs = ['universe_audit.csv','universe_by_season.csv','overall.csv',
               'by_season.csv','calibration_bins.csv','by_season_phase.csv','game_predictions.csv']
    if any((out / p).exists() for p in outputs):
        ap.error('Output exists. Refusing to overwrite; supply a new --out-dir.')
    d = read_and_validate(root / args.input)
    years = sorted(set(d.season.astype(int)))
    schedule = load_schedule(years, root)
    audited = attach_universe(d, schedule)
    status = audited.groupby(['season','universe_status'], as_index=False).agg(games=('game_id','size'))
    fbs = audited.loc[audited.universe_status.eq('fbs_fbs')].copy()
    other = audited.loc[audited.universe_status.ne('fbs_fbs')].copy()
    n_unknown = int((audited.universe_status == 'unknown').sum())
    print(f'Historical forecasts: {len(audited)}; confirmed FBS-FBS: {len(fbs)}; '
          f'other: {int((audited.universe_status=="other").sum())}; unknown: {n_unknown}', flush=True)
    print(status.to_string(index=False), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    audited[['game_id','season','home_team','away_team','home_classification',
             'away_classification','universe_status','season_type','scheduled_week']].to_csv(
        out/'universe_audit.csv', index=False)
    status.to_csv(out/'universe_by_season.csv', index=False)
    if n_unknown:
        raise RuntimeError('Unclassified predictions exist. Audit the saved universe_audit.csv '
                           'and repair missing schedule classifications before interpreting FBS-only metrics.')
    if len(fbs) < 500:
        raise RuntimeError('Too few confirmed FBS-FBS games; check classification fields.')
    if fbs.loc[fbs.season.eq(2019)].empty:
        raise RuntimeError('No 2019 FBS-FBS games for prior residual calibration.')
    # IMPORTANT: filter BEFORE residual resampling, not after simulation.
    minimal = fbs[d.columns]
    game, by, overall = run_experiment(minimal, start_year=2020, end_year=2024,
                                       samples=args.samples)
    cal = bin_calibration(game)
    phase = season_phase(game, fbs)
    for name, frame in [('overall.csv',overall),('by_season.csv',by),
                        ('calibration_bins.csv',cal),('by_season_phase.csv',phase),
                        ('game_predictions.csv',game)]:
        frame.to_csv(out/name, index=False)
    print('\nFBS-ONLY HISTORICAL DEVELOPMENT RESULTS (not betting evidence)')
    print(overall[['variant','games','margin_abs_error','total_abs_error','margin_crps',
                  'total_crps','margin_80_covered','total_80_covered','brier_home_win']].round(4).to_string(index=False))
    print(f'Reports saved: {out}')
    print('WARNING: model parameters were developed on these seasons; source point-in-time validity is not fully certified.')
    print('WARNING: prediction models may have trained on non-FBS opponents; filtering here applies to residual pools/evaluation.')
    print('WARNING: no sportsbook odds, EV, or profitability tests have been performed.')


if __name__ == '__main__':
    main()
