"""Step 197: Frozen v5.7 2026 retrospective CFBD market benchmark.

Does not retrain or reforecast. Does not infer bet profit or CLV.
Requires pandas, numpy, requests, and the 2026 retrospective replay output.
Reads CFBD GET /lines?year=2026&seasonType=regular, using a local API key.
All downloaded source JSON and benchmark artifacts are saved in a new folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path.cwd().resolve()
REPLAY_ROOT = ROOT / 'reports/v58_development/retrospective_replay_2026'
OUT_ROOT = ROOT / 'reports/v58_development/market_benchmark_2026'
API_URL = 'https://api.collegefootballdata.com/lines'
SEASON = 2026


def clean_id(value):
    return str(value).strip().removesuffix('.0')


def norm_team(value):
    return ' '.join(str(value).casefold().split())


def finite(value):
    if value is None or value == '':
        return np.nan
    try:
        val = float(value)
        return val if np.isfinite(val) else np.nan
    except (TypeError, ValueError):
        return np.nan


def api_key():
    names = ('CFBD_API_KEY', 'CFBD_KEY', 'CFBD_TOKEN', 'CFBD_API_TOKEN')
    for name in names:
        if os.environ.get(name, '').strip():
            return os.environ[name].strip().removeprefix('Bearer ').strip(), f'environment:{name}'
    # Only inspect normal dotenv paths; never print a credential.
    for path in (ROOT / '.env', ROOT / 'config/.env', ROOT / '.env.local'):
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if '=' not in line or line.strip().startswith('#'):
                continue
            name, val = line.split('=', 1)
            if name.strip().removeprefix('export ').strip() in names:
                val = val.strip().strip('\"').strip("'")
                if val:
                    return val.removeprefix('Bearer ').strip(), f'local env file:{path.name}'
    raise RuntimeError(
        'No CFBD key located. Set $env:CFBD_API_KEY locally in PowerShell '\
        '(do NOT paste it into chat), then rerun. No API call was made.'
    )


def load_replay():
    if not REPLAY_ROOT.is_dir():
        raise FileNotFoundError(f'Existing Step 195 replay folder missing: {REPLAY_ROOT}')
    candidates = sorted(
        REPLAY_ROOT.glob('*/game_grades.csv'),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError('No Step 195 game_grades.csv files found')
    for file in candidates:
        audit = file.parent / 'audit.json'
        if not audit.is_file():
            continue
        metadata = json.loads(audit.read_text(encoding='utf-8-sig'))
        if not (metadata.get('archived_feature_parity') and metadata.get('archived_prediction_parity')):
            continue
        if metadata.get('no_future_data_samples_passed', 0) < 3:
            continue
        g = pd.read_csv(file, dtype={'game_id': str})
        needed = {'game_id', 'week', 'home', 'away', 'actual_margin_home',
                  'actual_total', 'fair_margin_home', 'fair_total'}
        if not needed.issubset(g):
            continue
        if g.game_id.duplicated().any() or g.empty:
            continue
        g['game_id'] = g.game_id.map(clean_id)
        if len(g) != int(metadata.get('graded_FBS_FBS_games', -1)):
            raise RuntimeError('Replay row count does not match its audit')
        if not np.isfinite(g[['actual_margin_home', 'actual_total',
                              'fair_margin_home', 'fair_total']].to_numpy(float)).all():
            raise RuntimeError('Non-finite replay forecast or result')
        return file, metadata, g
    raise RuntimeError('No audited compatible replay output exists')


def fetch_lines():
    key, source = api_key()
    print(f'Credential located: {source} (value hidden)')
    response = requests.get(
        API_URL,
        params={'year': SEASON, 'seasonType': 'regular'},
        headers={'Authorization': f'Bearer {key}', 'Accept': 'application/json'},
        timeout=90,
    )
    if response.status_code != 200:
        raise RuntimeError(f'CFBD lines request returned HTTP {response.status_code}: '
                           f'{response.text[:250]}')
    rows = response.json()
    if not isinstance(rows, list):
        raise RuntimeError('CFBD /lines response is not a list')
    if not rows:
        raise RuntimeError('CFBD returned zero 2026 historical line games')
    payload = response.content
    return rows, payload


def parse_lines(rows, g):
    expected = {str(r.game_id): (norm_team(r.home), norm_team(r.away))
                for r in g.itertuples(index=False)}
    candidates = []
    name_mismatches = set()
    source_games = set()
    for game in rows:
        if not isinstance(game, dict):
            continue
        gid = clean_id(game.get('id', game.get('gameId', '')))
        if gid not in expected:
            continue
        source_games.add(gid)
        if (norm_team(game.get('homeTeam')),
                norm_team(game.get('awayTeam'))) != expected[gid]:
            name_mismatches.add(gid)
            continue
        for line in game.get('lines') or []:
            if not isinstance(line, dict):
                continue
            provider = str(line.get('provider') or '').strip()
            if not provider:
                continue
            candidates.append({
                'game_id': gid, 'provider': provider,
                # CFBD spread convention: a negative number means home favored.
                'market_home_spread': finite(line.get('spread')),
                'market_total': finite(line.get('overUnder')),
                'market_spread_open': finite(line.get('spreadOpen')),
                'market_total_open': finite(line.get('overUnderOpen')),
            })
    if not candidates:
        raise RuntimeError('No matching 2026 line rows for replay games')
    frame = pd.DataFrame(candidates)
    # Do not silently select one of multiple conflicting records from same provider.
    duplicates = frame.duplicated(['game_id', 'provider'], keep=False)
    invalid_pairs = set(map(tuple, frame.loc[duplicates, ['game_id', 'provider']].to_numpy()))
    frame = frame.loc[~duplicates].copy()
    plausible = frame.market_home_spread.abs().le(80) & frame.market_total.between(10, 130)
    frame['has_both'] = plausible.fillna(False)
    return frame, source_games, name_mismatches, invalid_pairs


def metrics(frame, which):
    e_m = frame[which + '_margin_error'].to_numpy(float)
    e_t = frame[which + '_total_error'].to_numpy(float)
    return {'margin_mae': float(np.abs(e_m).mean()),
            'margin_rmse': float(np.sqrt(np.square(e_m).mean())),
            'margin_bias': float(e_m.mean()),
            'total_mae': float(np.abs(e_t).mean()),
            'total_rmse': float(np.sqrt(np.square(e_t).mean())),
            'total_bias': float(e_t.mean())}


def ci_of_paired_improvement(frame, metric, block=False, repeats=3500):
    model = frame['model_' + metric + '_error'].to_numpy(float)
    market = frame['market_' + metric + '_error'].to_numpy(float)
    improvements = np.abs(market) - np.abs(model)  # positive = model better
    rng = np.random.default_rng(197)
    n = len(improvements)
    if block:
        weeks = frame['week'].to_numpy()
        groups = [np.where(weeks == w)[0] for w in np.unique(weeks)]
        draws = []
        for _ in range(repeats):
            parts = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            index = np.concatenate(parts)
            draws.append(float(improvements[index].mean()))
        method = 'week-block resampling (only cached weeks; exploratory)'
    else:
        index = rng.integers(0, n, size=(repeats, n))
        draws = improvements[index].mean(axis=1)
        method = 'paired game bootstrap (independence approximation)'
    return {'mean_points': float(improvements.mean()),
            'ci_low': float(np.quantile(draws, .025)),
            'ci_high': float(np.quantile(draws, .975)),
            'method': method}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default=None,
                        help='Optional exact provider name from CFBD; otherwise maximize paired coverage')
    args = parser.parse_args()
    print('='*83)
    print('STEP 197 — CFB-QUANT V5.7 VS CFBD HISTORICAL MARKET LINES')
    print('Research replay only | no claim of live betting edge or certified closing-line value')
    print('='*83)
    replay_file, meta, g = load_replay()
    print('Validated replay:', replay_file.parent)
    print('Eligible game predictions:', len(g))
    print('Archived feature / prediction parity: PASS')
    print('Recorded independent no-future-data probes:', meta['no_future_data_samples_passed'])
    print('Fetching CFBD /lines for year=2026, seasonType=regular (single request)')
    rows, payload = fetch_lines()
    print('CFBD game records returned:', len(rows))
    market, src_ids, bad_names, duplicate_keys = parse_lines(rows, g)
    print('Replay IDs present in CFBD:', len(src_ids), '/', len(g))
    print('Rejected mismatched team identities:', len(bad_names))
    print('Rejected ambiguous same-provider duplicates:', len(duplicate_keys))
    available = market.loc[market.has_both].copy()
    coverage = (available.groupby('provider', as_index=False)
                .agg(matched_both=('game_id', 'nunique'))
                .sort_values(['matched_both', 'provider'], ascending=[False, True]))
    print('\nPROVIDER COVERAGE (both spread and total)')
    print(coverage.to_string(index=False))
    if coverage.empty:
        raise RuntimeError('No provider has valid spreads AND totals; no comparison produced')
    provider = args.provider if args.provider else str(coverage.iloc[0].provider)
    if provider not in coverage.provider.values:
        raise RuntimeError(f'Requested provider {provider!r} has no paired data')
    selected = available.loc[available.provider.eq(provider)].copy()
    matched = g.merge(selected, on='game_id', how='inner', validate='one_to_one')
    if len(matched) < 20:
        raise RuntimeError(f'Only {len(matched)} paired games; insufficient to benchmark')
    # Negative home spread means home favorite. Convert to home fair margin.
    matched['market_home_margin'] = -matched['market_home_spread']
    matched['model_margin_error'] = matched['fair_margin_home'] - matched['actual_margin_home']
    matched['market_margin_error'] = matched['market_home_margin'] - matched['actual_margin_home']
    matched['model_total_error'] = matched['fair_total'] - matched['actual_total']
    matched['market_total_error'] = matched['market_total'] - matched['actual_total']
    matched['model_minus_market_margin'] = matched['fair_margin_home'] - matched['market_home_margin']
    matched['model_minus_market_total'] = matched['fair_total'] - matched['market_total']
    print('\nSINGLE PROVIDER:', provider)
    print('Paired sample:', len(matched), 'of', len(g), f'({len(matched)/len(g):.1%})')
    model = metrics(matched, 'model')
    market_m = metrics(matched, 'market')
    print('\nMODEL VS MARKET — EXACT SAME GAMES')
    for field in ('margin_mae', 'margin_rmse', 'margin_bias', 'total_mae',
                  'total_rmse', 'total_bias'):
        print(f'{field}: model={model[field]:.3f}, market={market_m[field]:.3f}')
    boots = {}
    for item in ('margin', 'total'):
        boots[item] = {'by_game': ci_of_paired_improvement(matched, item),
                       'by_week': ci_of_paired_improvement(matched, item, block=True)}
        delta = boots[item]['by_game']
        print(f'\n{item.upper()} market MAE minus model MAE: {delta["mean_points"]:+.3f} points')
        print(f'  Paired game bootstrap 95% CI: [{delta["ci_low"]:+.3f}, {delta["ci_high"]:+.3f}]')
        delta_week = boots[item]['by_week']
        print(f'  Week-block exploratory 95% CI: [{delta_week["ci_low"]:+.3f}, {delta_week["ci_high"]:+.3f}]')
    week_rows = []
    for week, df in matched.groupby('week', sort=True):
        a, b = metrics(df, 'model'), metrics(df, 'market')
        week_rows.append({'week': week, 'games': len(df),
                          'model_margin_mae': a['margin_mae'], 'market_margin_mae': b['margin_mae'],
                          'model_total_mae': a['total_mae'], 'market_total_mae': b['total_mae']})
    weeks = pd.DataFrame(week_rows)
    print('\nBY WEEK (identical samples for each comparison)')
    print(weeks.round(3).to_string(index=False))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = OUT_ROOT / timestamp
    target.mkdir(parents=True, exist_ok=False)
    (target/'cfbd_lines_2026_response.json').write_bytes(payload)
    matched.to_csv(target/'paired_game_market_comparison.csv', index=False)
    coverage.to_csv(target/'provider_coverage.csv', index=False)
    weeks.to_csv(target/'by_week.csv', index=False)
    report = {
        'label': 'RETROSPECTIVE 2026 MARKET BENCHMARK; NO CERTIFIED HISTORICAL PRICES/TIMESTAMPS',
        'source': API_URL + '?year=2026&seasonType=regular',
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_response_sha256': hashlib.sha256(payload).hexdigest(),
        'replay_source': str(replay_file.relative_to(ROOT)),
        'selected_provider': provider,
        'selection_rule': 'max paired spread+total game coverage; never optimized on results' if args.provider is None else 'explicit provider argument',
        'paired_games': len(matched), 'replay_games': len(g),
        'provider_coverage': dict(zip(coverage.provider, coverage.matched_both.astype(int))),
        'model': model, 'market': market_m, 'bootstrap': boots,
        'historical_replay_source_publication_timestamps_certified': False,
        'historical_market_line_asof_kickoff_certified': False,
        'market_is_certified_closing_price': False,
        'actual_bets_or_returns_evaluated': False,
        'original_sources_or_models_modified': False,
    }
    (target/'audit.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print('\nRESEARCH ARTIFACTS:', target)
    print('IMPORTANT: CFBD provider line timestamps are unavailable here; no CLV/betting-edge claim.')
    print('Production models and earlier reports unchanged.')
    print('='*83)

if __name__ == '__main__':
    main()
