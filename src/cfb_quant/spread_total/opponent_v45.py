"""Experimental point-in-time opponent-adjusted advanced football ratings.

Reads CFB-Quant v4 cached historical CFBD observations and historical feature parquet.
Writes a *separate* v4.5 dataset and manifest. Does not touch v3/v4 outputs,
fetch an API, or include same-game observations in model features.

The sequential attack / defense-weakness decomposition is a deliberately SIMPLE
fixed-hyperparameter hypothesis, not an established optimal adjusted-efficiency model.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .advanced_v4 import load_all_years

V4_DATA = Path('data/processed/cfb_matchup_features_advanced_v4.parquet')
V3_MANIFEST = Path('reports/cfb_matchup_feature_manifest_kickoff_v3.json')
OUTPUT = Path('data/processed/cfb_matchup_features_opponent_v45.parquet')
MANIFEST = Path('reports/cfb_matchup_feature_manifest_opponent_v45.json')
COVERAGE = Path('reports/opponent_v45_coverage.csv')

# Fixed PRIOR hypotheses. Do not tune these values on the evaluation seasons.
METRICS = {
    'ppa': (0.0, 0.22, 1.50),
    'passing_ppa': (0.0, 0.22, 1.50),
    'rushing_ppa': (0.0, 0.22, 1.50),
    'success_rate': (0.45, 0.22, 0.30),
    'passing_success_rate': (0.45, 0.22, 0.30),
    'rushing_success_rate': (0.45, 0.22, 0.30),
}
SEASON_RETENTION = 0.55


def feature_names() -> list[str]:
    return [f'oa_{m}_{component}' for m in METRICS
            for component in ('attack', 'opp_def_weakness', 'expected_offense')] + [
                'oa_team_prior_games_season', 'oa_opponent_prior_games_season']


def build_adjustments(base: pd.DataFrame, observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {'game_id', 'season', 'team', 'opponent', 'home_away', 'kickoff_utc'}
    missing = required - set(base)
    if missing:
        raise ValueError(f'Base data missing required fields: {sorted(missing)}')
    rawcols = [f'raw_adv_offense_{m}' for m in METRICS]
    absent = set(['game_id', 'team', *rawcols]) - set(observations)
    if absent:
        raise ValueError(f'Raw observations missing: {sorted(absent)}')
    if base.duplicated(['game_id','team']).any() or observations.duplicated(['game_id','team']).any():
        raise ValueError('Duplicate game/team records are forbidden')
    src = base.copy()
    src['game_id'] = src.game_id.astype(str).str.replace(r'\.0$', '', regex=True)
    src['team'] = src.team.astype(str).str.strip()
    src['opponent'] = src.opponent.astype(str).str.strip()
    src['kickoff_utc'] = pd.to_datetime(src.kickoff_utc, utc=True, errors='coerce')
    if src.kickoff_utc.isna().any():
        raise ValueError('Null/invalid kickoff timestamp: cannot build pregame ratings')
    obs = observations[['game_id','team',*rawcols]].copy()
    obs['game_id'] = obs.game_id.astype(str).str.replace(r'\.0$', '', regex=True)
    obs['team'] = obs.team.astype(str).str.strip()
    src['orig_idx'] = np.arange(len(src))
    src = src.merge(obs, on=['game_id','team'], how='left', indicator='_raw_match', validate='one_to_one')
    coverage = src.groupby('season', as_index=False).agg(
        rows=('game_id','size'),
        matched=('_raw_match', lambda s: int(s.eq('both').sum())),
    )
    coverage['coverage_pct'] = (100 * coverage.matched / coverage.rows).round(2)
    if float(src['_raw_match'].eq('both').mean()) < 0.90:
        raise ValueError('Advanced observation match coverage below 90%; review source joins')
    if len(src) % 2:
        raise ValueError('Need exactly two team rows per game')
    for gid, g in src.groupby('game_id', sort=False):
        if len(g) != 2 or set(g.home_away.astype(str).str.lower()) != {'home','away'}:
            raise ValueError(f'Incomplete/invalid home-away pair: {gid}')
        a, b = g.iloc[0], g.iloc[1]
        if a.opponent != b.team or b.opponent != a.team or a.kickoff_utc != b.kickoff_utc:
            raise ValueError(f'Opponent or kickoff mismatch: {gid}')
    src = src.sort_values(['kickoff_utc','game_id','team'], kind='mergesort').reset_index(drop=True)
    attacked = {m: defaultdict(float) for m in METRICS}
    weaknesses = {m: defaultdict(float) for m in METRICS}
    games_seen = defaultdict(int)
    feature_records: list[dict] = []
    previous_season = None

    for kickoff, group in src.groupby('kickoff_utc', sort=True):
        seasons = {int(x) for x in group.season}
        if len(seasons) != 1:
            raise ValueError(f'Different seasons at identical kickoff {kickoff}')
        season = seasons.pop()
        if previous_season is not None and season < previous_season:
            raise ValueError('Season decreases in kickoff order')
        if previous_season is not None and season != previous_season:
            for m in METRICS:
                for d in (attacked[m], weaknesses[m]):
                    for k in list(d):
                        d[k] *= SEASON_RETENTION
            games_seen = defaultdict(int)
        previous_season = season
        if group.team.duplicated().any():
            raise ValueError(f'Team appears twice at same kickoff {kickoff}')

        # Freeze every feature for this kickoff BEFORE processing observations.
        for row in group.itertuples(index=False):
            f = {'orig_idx': row.orig_idx}
            for m, (baseline, _, _) in METRICS.items():
                att = attacked[m][row.team]
                weak = weaknesses[m][row.opponent]
                f[f'oa_{m}_attack'] = att
                f[f'oa_{m}_opp_def_weakness'] = weak
                f[f'oa_{m}_expected_offense'] = baseline + att + weak
            f['oa_team_prior_games_season'] = games_seen[row.team]
            f['oa_opponent_prior_games_season'] = games_seen[row.opponent]
            feature_records.append(f)

        # Simultaneous updates prevent teams at one timestamp affecting one another.
        updates_att = defaultdict(float)
        updates_def = defaultdict(float)
        for row in group.itertuples(index=False):
            for m, (baseline, learning_rate, cap) in METRICS.items():
                raw = getattr(row, f'raw_adv_offense_{m}')
                if pd.isna(raw):
                    continue
                raw = float(raw)
                if not np.isfinite(raw):
                    continue
                if 'success_rate' in m and not (0 <= raw <= 1):
                    raise ValueError(f'Invalid {m}={raw} on game {row.game_id}')
                actual_error = raw - (baseline + attacked[m][row.team] + weaknesses[m][row.opponent])
                updates_att[(m, row.team)] += learning_rate * actual_error
                updates_def[(m, row.opponent)] += learning_rate * actual_error
        for (m, team), inc in updates_att.items():
            cap = METRICS[m][2]
            attacked[m][team] = float(np.clip(attacked[m][team] + inc, -cap, cap))
        for (m, team), inc in updates_def.items():
            cap = METRICS[m][2]
            weaknesses[m][team] = float(np.clip(weaknesses[m][team] + inc, -cap, cap))
        for team in group.team:
            games_seen[team] += 1

    out = pd.DataFrame(feature_records)
    if out.orig_idx.duplicated().any() or len(out) != len(src):
        raise AssertionError('Lost or duplicated rows during sequential feature engineering')
    newcols = feature_names()
    if set(newcols) & set(base):
        raise ValueError('New feature names already exist in original data')
    out = out.set_index('orig_idx').sort_index()
    reconstructed = pd.concat([base.reset_index(drop=True), out[newcols].reset_index(drop=True)], axis=1)
    if reconstructed[newcols].isna().any().any():
        raise ValueError('Generated unexpected NaNs in pregame ratings')
    return reconstructed, coverage


def build(root: Path, *, min_year: int = 2014, max_year: int = 2025) -> None:
    data_path, manifest_path = root / V4_DATA, root / V3_MANIFEST
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError('Missing advanced v4 data or corrected v3 feature manifest')
    data = pd.read_parquet(data_path)
    data = data.loc[data.season.between(min_year,max_year)].copy()
    baseline_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    oldcols = list(baseline_manifest['feature_columns'])
    if not all(col in data for col in oldcols):
        raise ValueError('Base features absent from v4 data')
    raw = load_all_years(root, list(range(min_year,max_year + 1)))
    out, coverage = build_adjustments(data, raw)
    result_manifest = dict(baseline_manifest)
    result_manifest['feature_columns'] = oldcols + feature_names()
    result_manifest['opponent_v45'] = {
        'experimental': True, 'time_order': 'kickoff_utc',
        'base_features': len(oldcols), 'new_features': len(feature_names()),
        'note': 'Sequential pregame opponent-adjusted PPA/success-rate ratings; fixed unoptimized update parameters. Retrospective source records require separate availability audit.',
        'not_proof_of_value': True,
    }
    for p in (root / OUTPUT, root / MANIFEST, root / COVERAGE):
        if p.exists():
            raise FileExistsError(f'Output exists; refusing overwrite: {p}')
    (root / OUTPUT).parent.mkdir(parents=True,exist_ok=True)
    (root / MANIFEST).parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(root / OUTPUT, index=False)
    (root / MANIFEST).write_text(json.dumps(result_manifest,indent=2),encoding='utf-8')
    coverage.to_csv(root / COVERAGE,index=False)
    print('OPPONENT-ADJUSTED EXPERIMENTAL DATA READY')
    print(f'Team-game rows: {len(out)}; games: {out.game_id.nunique()}; added: {len(feature_names())}')
    print(f'Data: {root / OUTPUT}')
    print(f'Manifest: {root / MANIFEST}')
    print('Advanced observations matched by season:')
    print(coverage.to_string(index=False))
    print('No data downloaded. No v3/v4 files modified. No betting advantage established.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--min-year',type=int,default=2014)
    p.add_argument('--max-year',type=int,default=2025)
    args=p.parse_args()
    if args.min_year != 2014 or args.max_year > 2025 or args.max_year < args.min_year:
        p.error('Use complete historical seasons 2014 through at most 2025')
    build(Path.cwd(),min_year=args.min_year,max_year=args.max_year)

if __name__ == '__main__':
    main()
