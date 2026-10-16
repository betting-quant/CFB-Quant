"""v5.3 experimental FBS-only phase-dependent scoring-residual comparison.

Compares the v5.2 pooled residual bootstrap against a predetermined 70% same-phase
plus 30% pooled residual mixture, separately for regular W1-4, regular W5+,
and postseason. Both predictions and error pools remain strictly prior-season.

No new API requests. No training changes. Development experiment, NOT betting proof.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .hybrid_probability_v51 import (read_and_validate, paired_scoring_residuals,
                                     simulate_scores, empirical_crps, LINES)
from .fbs_calibration_v52 import bin_calibration

SOURCE = Path('reports/spread_total_opponent_v45/paired_game_errors.csv')
UNIVERSE = Path('reports/spread_total_fbs_calibration_v52/universe_audit.csv')
OUT = Path('reports/spread_total_phase_volatility_v53')
PHASES = {'postseason', 'regular_weeks_1_4', 'regular_week_5_plus'}


def prepare_fbs(data: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    need = {'season', 'game_id', 'universe_status', 'season_type', 'scheduled_week'}
    if missing := need - set(universe.columns):
        raise ValueError(f'Missing v5.2 universe audit columns: {sorted(missing)}')
    d = data.copy()
    s = universe[list(need)].copy()
    for t in (d, s):
        t['game_id'] = t['game_id'].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
        t['season'] = pd.to_numeric(t['season'], errors='raise').astype(int)
        if t.duplicated(['season', 'game_id']).any():
            raise ValueError('Duplicate season/game keys; refusing ambiguous classification')
    z = d.merge(s, on=['season', 'game_id'], how='left', indicator=True, validate='one_to_one')
    if not z['_merge'].eq('both').all():
        raise ValueError('Not every prediction matched a v5.2 universe audit row')
    if z['universe_status'].eq('unknown').any() or z['universe_status'].isna().any():
        raise ValueError('Missing FBS classification in universe audit')
    z = z.loc[z.universe_status.eq('fbs_fbs')].copy()
    if z.empty:
        raise ValueError('No FBS-vs-FBS games')
    weeks = pd.to_numeric(z.scheduled_week, errors='coerce')
    post = z.season_type.astype('string').str.lower().eq('postseason')
    z['phase'] = np.select([post, ~post & weeks.le(4), ~post & weeks.gt(4)],
                           ['postseason', 'regular_weeks_1_4', 'regular_week_5_plus'], default='unknown')
    if z.phase.eq('unknown').any():
        raise ValueError('Unknown season phase; inspect original schedule')
    return z.drop(columns=['_merge']).sort_values(['season', 'game_id']).reset_index(drop=True)


def sample_indices(pool: pd.DataFrame, tests: pd.DataFrame, *, samples: int,
                   phase_share: float, seed: int, min_phase_games: int = 40
                   ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return pooled and partly phase-conditioned indices into the same past pool."""
    if not (0 <= phase_share <= 1):
        raise ValueError('phase_share must be in [0,1]')
    if samples < 200 or len(pool) < 50:
        raise ValueError('Insufficient historical data/samples')
    rng = np.random.default_rng(seed)
    n = len(tests)
    pooled = rng.integers(0, len(pool), size=(n,samples))
    mixed = pooled.copy()
    counts=[]
    for phase in sorted(PHASES):
        rows = np.flatnonzero(tests.phase.to_numpy() == phase)
        candidates = np.flatnonzero(pool.phase.to_numpy() == phase)
        if not len(rows):
            continue
        enabled = len(candidates) >= min_phase_games
        if enabled:
            chosen = rng.random((len(rows),samples)) < phase_share
            n_cho = int(chosen.sum())
            if n_cho:
                drawn = rng.choice(candidates, size=n_cho, replace=True)
                subsection = mixed[rows].copy()
                subsection[chosen] = drawn
                mixed[rows] = subsection
        counts.append({'phase':phase, 'evaluation_games':len(rows), 'prior_phase_games':len(candidates),
                       'prior_total_games':len(pool),'phase_sampling_applied':bool(enabled)})
    return pooled, mixed, pd.DataFrame(counts)


def summarize(games: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = ['margin_abs_error','total_abs_error','margin_crps','total_crps',
            'margin_80_covered','total_80_covered'] + [f'brier_{n}' for n,_,_ in LINES]
    return games.groupby(keys,as_index=False).agg(games=('game_id','size'), **{c:(c,'mean') for c in cols})


def evaluate(fbs: pd.DataFrame, *, samples: int = 1200, phase_share: float = 0.7,
             start_year: int = 2020, end_year: int = 2024, seed: int = 53
             ) -> tuple[pd.DataFrame,pd.DataFrame]:
    if not 2020 <= start_year <= end_year <= 2024:
        raise ValueError('Exploratory evaluation restricted to 2020-2024')
    outputs=[]; audits=[]
    for year in range(start_year,end_year+1):
        pool = fbs.loc[fbs.season < year].copy()
        test = fbs.loc[fbs.season == year].copy()
        if test.empty or pool.empty or pool.season.max() >= year:
            raise ValueError(f'Missing strictly prior-season FBS-only games for {year}')
        if len(pool) < 50:
            raise ValueError(f'Not enough historical FBS games for {year}')
        idx0, idx1, counts = sample_indices(pool,test,samples=samples,phase_share=phase_share,seed=seed+year)
        counts['evaluated_season']=year
        counts['max_residual_season']=int(pool.season.max())
        audits.append(counts)
        pm=test.pred_margin_v45.to_numpy(dtype=float)
        pt=test.pred_total_v3.to_numpy(dtype=float)
        am=test.actual_margin_v3.to_numpy(dtype=float)
        at=test.actual_total_v3.to_numpy(dtype=float)
        history=paired_scoring_residuals(pool,'pred_margin_v45','pred_total_v3')
        for variant, idx in [('pooled',idx0),('phase_mixed',idx1)]:
            sm,st=simulate_scores(pm,pt,history,idx)
            frame=pd.DataFrame({'game_id':test.game_id.to_numpy(), 'season':year,
                                'phase':test.phase.to_numpy(), 'variant':variant,
                                'max_residual_season':int(pool.season.max()),
                                'pred_margin':pm,'pred_total':pt,'actual_margin':am,'actual_total':at,
                                'margin_abs_error':np.abs(am-pm), 'total_abs_error':np.abs(at-pt),
                                'margin_crps':empirical_crps(sm,am),
                                'total_crps':empirical_crps(st,at),
                                'margin_80_covered':(am>=np.quantile(sm,.1,axis=1)) & (am<=np.quantile(sm,.9,axis=1)),
                                'total_80_covered':(at>=np.quantile(st,.1,axis=1)) & (at<=np.quantile(st,.9,axis=1))})
            for name,target,line in LINES:
                samples_for_line, outcomes = (sm, am) if target=='margin' else (st, at)
                p=(samples_for_line>line).mean(axis=1)
                frame[f'p_{name}']=p
                frame[f'brier_{name}']=(p-(outcomes>line).astype(float))**2
            outputs.append(frame)
        print(f'{year}: {len(test)} FBS-FBS games; prior pool={len(pool)}, max prior season={int(pool.season.max())}',flush=True)
    game=pd.concat(outputs,ignore_index=True)
    if not (game.max_residual_season < game.season).all():
        raise AssertionError('Same/future season calibration observed')
    audit=pd.concat(audits,ignore_index=True)
    return game,audit


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default=str(SOURCE))
    p.add_argument('--universe', default=str(UNIVERSE))
    p.add_argument('--out-dir', default=str(OUT))
    p.add_argument('--samples', type=int, default=1200)
    p.add_argument('--phase-share', type=float, default=0.7,
                   help='Prespecified weight on previous seasons from the same game phase')
    args=p.parse_args()
    out=Path(args.out_dir)
    files=['overall.csv','by_season.csv','by_phase.csv','phase_pool_audit.csv',
           'calibration_bins.csv','game_predictions.csv']
    if any((out/f).exists() for f in files):
        p.error(f'Existing outputs at {out}. Supply a new --out-dir rather than overwrite history.')
    for path in (Path(args.input),Path(args.universe)):
        if not path.is_file():p.error(f'Missing saved input: {path}. Do not fetch API data again.')
    d=read_and_validate(args.input)
    universe=pd.read_csv(args.universe)
    fbs=prepare_fbs(d,universe)
    game,audit=evaluate(fbs,samples=args.samples,phase_share=args.phase_share)
    out.mkdir(parents=True,exist_ok=True)
    reports={'overall.csv':summarize(game,['variant']),
             'by_season.csv':summarize(game,['variant','season']),
             'by_phase.csv':summarize(game,['variant','phase']),
             'phase_pool_audit.csv':audit,
             'calibration_bins.csv':bin_calibration(game),
             'game_predictions.csv':game}
    for fn,data in reports.items():data.to_csv(out/fn,index=False)
    print('\nFBS-ONLY PHASE-VOLATILITY RESEARCH (lower CRPS/Brier is better)')
    print(reports['overall.csv'][['variant','games','margin_crps','total_crps',
          'margin_80_covered','total_80_covered','brier_home_win']].round(4).to_string(index=False))
    print('\nBy phase:')
    print(reports['by_phase.csv'][['variant','phase','games','margin_80_covered',
          'total_80_covered','margin_crps','total_crps']].round(4).to_string(index=False))
    print(f'Reports: {out}')
    print('RESEARCH ONLY: development seasons reused; no historical market odds; upstream feature timestamps not fully audited.')
    print('FBS-only filtering applies to evaluation and past residuals, not original fitted score models.')


if __name__=='__main__':main()
