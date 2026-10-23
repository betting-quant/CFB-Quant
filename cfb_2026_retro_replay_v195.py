"""Research-only 2026 CFB-Quant chronological replay.

Requires local CFB-Quant repository and its installed dependencies.
No 2026 results used for training, original repo source data not overwritten.
Source publication-time PIT not certified.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from cfb_quant.live import feature_bridge_v56 as fb
from cfb_quant.live import predict_v57 as live
from cfb_quant.features.matchup_engine import build_features
from cfb_quant.spread_total.advanced_v4 import load_all_years
from cfb_quant.spread_total.opponent_v45 import build_adjustments
from cfb_quant.models.matchup_tournament import _to_numeric_frame

ROOT = Path.cwd().resolve()
ARCHIVE = ROOT / (
    'reports/v58_development/midweek_forecasts/'
    '20261020T173820706469Z/2026-10-20'
)
SCHEDULE = ROOT / 'data/raw/cfbd/games/games_2026.json'


def normalize(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paired_slice(frame, gid):
    part = frame.loc[normalize(frame['game_id']).eq(str(gid))].copy()
    if len(part) != 2 or set(part['home_away']) != {'home', 'away'}:
        raise RuntimeError(f'Expected home+away pair for game {gid}, found {len(part)} rows')
    return part.set_index(['game_id', 'team']).sort_index()


def compare_feature_sets(lhs, rhs, features, label):
    lhs = lhs.copy()
    rhs = rhs.copy()
    lhs['game_id'] = normalize(lhs['game_id'])
    rhs['game_id'] = normalize(rhs['game_id'])
    lhs = lhs.set_index(['game_id', 'team']).sort_index()
    rhs = rhs.set_index(['game_id', 'team']).sort_index()
    if not lhs.index.equals(rhs.index):
        raise RuntimeError(f'{label}: game identities differ: {lhs.index.symmetric_difference(rhs.index).tolist()[:10]}')
    a = lhs[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    b = rhs[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    match = np.isclose(a, b, atol=1e-9, rtol=1e-9, equal_nan=True)
    if not match.all():
        bad = np.argwhere(~match)
        print(f'{label}: mismatches={len(bad)} / {match.size}')
        for i, j in bad[:12]:
            print(' ', lhs.index[i], features[j], 'actual=', a[i,j], 'expected=', b[i,j])
        raise RuntimeError(f'{label}: feature parity FAILED')
    print(f'{label}: PASS ({match.size} / {match.size} matching)')


def get_schedule():
    schedule = fb.load_schedule(SCHEDULE)
    if not isinstance(schedule, list):
        raise RuntimeError('Schedule must be a list')
    return schedule


def eligible_schedule(schedule, asof):
    scored = []
    all_completed = 0
    for g in schedule:
        if not fb.done(g) or g.get('homePoints') is None or g.get('awayPoints') is None:
            continue
        if fb.kickoff(g) >= asof:
            continue
        all_completed += 1
        if not fb.fbs_both(g):
            continue
        gid = fb.gid(g)
        scored.append({
            'game_id': str(gid),
            'kickoff_utc': fb.kickoff(g),
            'week': g.get('week'),
            'actual_home': int(g['homePoints']),
            'actual_away': int(g['awayPoints']),
            'schedule_home': str(g['homeTeam']),
            'schedule_away': str(g['awayTeam']),
        })
    data = pd.DataFrame(scored)
    if data.empty or data.game_id.duplicated().any():
        raise RuntimeError('Empty or duplicated eligible 2026 FBS-FBS game IDs')
    return data, all_completed


def validate_targets(history, eligibility):
    h = history.loc[history.season.eq(2026)].copy()
    h['game_id'] = normalize(h['game_id'])
    if h.duplicated(['game_id','team']).any():
        raise RuntimeError('Duplicate 2026 team history rows')
    missing_ids = set(eligibility.game_id) - set(h.game_id)
    if missing_ids:
        raise RuntimeError(f'FBS games missing processed history: {len(missing_ids)}; examples {sorted(missing_ids)[:8]}')
    scores_ok = 0
    for g in eligibility.itertuples(index=False):
        pair = h.loc[h.game_id.eq(g.game_id)]
        hm = pair.loc[pair.home_away.eq('home')]
        aw = pair.loc[pair.home_away.eq('away')]
        if len(hm) != 1 or len(aw) != 1:
            raise RuntimeError(f'Invalid historical game pair {g.game_id}')
        if str(hm.iloc[0].team) != g.schedule_home or str(aw.iloc[0].team) != g.schedule_away:
            raise RuntimeError(f'Team name mismatch game {g.game_id}')
        if not (float(hm.iloc[0].points) == g.actual_home and float(aw.iloc[0].points) == g.actual_away):
            raise RuntimeError(f'Processed/schedule final score disagreement game {g.game_id}')
        scores_ok += 1
    print(f'Schedule/history final score alignment: PASS ({scores_ok} games)')


def make_full_features(history, archived_games, schedule, advanced):
    placeholders = fb.make_placeholders(archived_games, history)
    if len(placeholders) != 2 * len(archived_games):
        raise RuntimeError('Archived placeholders incomplete')
    all_rows = pd.concat([history, placeholders], ignore_index=True)
    if all_rows.duplicated(['game_id','team']).any():
        raise RuntimeError('Duplicate game/team rows in reconstruction')
    base, original_cols = build_features(all_rows)
    oa_frame, _ = build_adjustments(base, advanced)
    return oa_frame, list(original_cols)


def leakage_probe(history, schedule_obj, asof, full, features, advanced):
    gid = str(fb.gid(schedule_obj))
    kickoff = fb.kickoff(schedule_obj)
    # Truly pregame alternative: remove game and all later games from history;
    # make an unscored placeholder with production's own builder.
    cut = history.loc[pd.to_datetime(history.kickoff_utc, utc=True).lt(kickoff)].copy()
    future = fb.make_placeholders([schedule_obj], cut)
    if any(x in future.columns and future[x].notna().any()
           for x in ['points','opponent_points']):
        raise RuntimeError(f'Pregame placeholder has current score for {gid}')
    pre, _ = build_features(pd.concat([cut, future], ignore_index=True))
    pre_oa, _ = build_adjustments(pre, advanced)
    orig = full.loc[normalize(full['game_id']).eq(gid)]
    fresh = pre_oa.loc[normalize(pre_oa['game_id']).eq(gid)]
    compare_feature_sets(fresh, orig, features, f'No-future-data probe {gid}')


def metrics(data, prefix):
    a = data[f'{prefix}_margin_error'].to_numpy(float)
    t = data['total_error'].to_numpy(float)
    winner = data[f'{prefix}_winner_correct'].to_numpy(bool)
    return {
        'n': len(data),
        'margin_mae': float(np.abs(a).mean()),
        'margin_rmse': float(np.sqrt(np.mean(a*a))),
        'margin_bias_pred_minus_actual': float(a.mean()),
        'winner_accuracy': float(winner.mean()),
        'total_mae': float(np.abs(t).mean()),
        'total_rmse': float(np.sqrt(np.mean(t*t))),
        'total_bias_pred_minus_actual': float(t.mean()),
        'margin_over_14_abs_error': int((np.abs(a)>14).sum()),
        'total_over_20_abs_error': int((np.abs(t)>20).sum()),
    }


def main():
    print('='*86)
    print('STEP 195 — FROZEN V5.7 2026 RETROSPECTIVE CHRONOLOGICAL REPLAY')
    print('='*86)
    for p in (SCHEDULE, ARCHIVE/'audit.json', ARCHIVE/'pregame_features.parquet', ARCHIVE/'paired_forecasts.csv'):
        if not p.is_file():
            raise FileNotFoundError(p)
    archived_audit = json.loads((ARCHIVE/'audit.json').read_text(encoding='utf-8'))
    orig_file = ARCHIVE/'pregame_features.parquet'
    if sha(orig_file) != archived_audit['feature_sha256']:
        raise RuntimeError('Archived pregame snapshot hash mismatch')
    asof = pd.to_datetime(archived_audit['asof_utc'], utc=True)
    print('Archive parity cutoff UTC:', asof)
    schedule = get_schedule()
    eligible, n_completed = eligible_schedule(schedule, asof)
    print('Completed schedule games available:', n_completed)
    print('FBS-FBS games potentially gradable:', len(eligible))
    history, history_info = fb.load_histories(ROOT, schedule, asof)
    print('Historical team-game rows:', len(history))
    validate_targets(history, eligible)

    archived_forecasts = pd.read_csv(ARCHIVE/'paired_forecasts.csv', dtype={'game_id':str})
    archive_ids = set(archived_forecasts['game_id'].map(str))
    if archive_ids.intersection(set(eligible.game_id)):
        raise RuntimeError('Archive current games appeared in historical replay source: outcome contamination risk')
    archived_games = [g for g in schedule if str(fb.gid(g)) in archive_ids]
    if len(archived_games) != len(archive_ids):
        raise RuntimeError('Archived games missing schedule lookup')

    v3_cols = live.load_manifest(ROOT, live.V3_MANIFEST)
    v45_cols = live.load_manifest(ROOT, live.V45_MANIFEST)
    if len(v3_cols) != 267 or len(v45_cols) != 287 or v45_cols[:267] != v3_cols:
        raise RuntimeError('Production manifest schema inconsistent')
    advanced = load_all_years(ROOT, list(range(2014, 2027)))
    full, built_cols = make_full_features(history, archived_games, schedule, advanced)
    if built_cols != v3_cols:
        raise RuntimeError('Replayed feature builder does not reproduce 267-column production schema')
    if not set(v45_cols).issubset(full.columns):
        raise RuntimeError('Opponent-adjusted production columns missing')

    archived_original = pd.read_parquet(orig_file)
    reconstructed = full.loc[normalize(full['game_id']).isin(archive_ids)].copy()
    compare_feature_sets(reconstructed, archived_original, v45_cols,
                         'Original October 20 locked snapshot parity')
    if reconstructed[['points','opponent_points']].notna().any().any():
        raise RuntimeError('Archived current-game outcomes present in reconstructed rows')

    # Select deterministic early, middle, and late games for a complete
    # no-future-data recomputation of all 287 model inputs.
    ordered = eligible.sort_values(['kickoff_utc','game_id']).reset_index(drop=True)
    sample_indices = sorted(set([0, len(ordered)//2, len(ordered)-1]))
    lookup = {str(fb.gid(g)):g for g in schedule}
    for idx in sample_indices:
        game_id = str(ordered.iloc[idx].game_id)
        leakage_probe(history, lookup[game_id], asof, full, v45_cols, advanced)

    print('\nFITTING FROZEN 2014-2025 PRODUCTION RIDGE MODELS IN MEMORY')
    model_v3, audit_v3 = live.train_points(ROOT, v3_cols, live.V3_DATA)
    model_v45, audit_v45 = live.train_points(ROOT, v45_cols, live.V45_DATA)
    target_ids = set(eligible.game_id)
    target_frame = full.loc[normalize(full['game_id']).isin(target_ids)].copy()
    target_frame['game_id'] = normalize(target_frame['game_id'])
    if len(target_frame) != 2*len(eligible) or target_frame.duplicated(['game_id','team']).any():
        raise RuntimeError('Replay prediction row coverage mismatch')

    pred3 = np.asarray(model_v3.predict(_to_numeric_frame(target_frame, v3_cols)),dtype=float)
    pred45 = np.asarray(model_v45.predict(_to_numeric_frame(target_frame,v45_cols)),dtype=float)
    forecasts = live.game_forecasts(target_frame, pred3, pred45)
    forecasts['game_id'] = normalize(forecasts['game_id'])
    if len(forecasts) != len(eligible) or forecasts.game_id.duplicated().any():
        raise RuntimeError('Game pairing returned incomplete forecasts')

    # Separately reproduce original saved *predictions* as a final parity check.
    x3_orig = np.asarray(model_v3.predict(_to_numeric_frame(reconstructed, v3_cols)),dtype=float)
    x45_orig = np.asarray(model_v45.predict(_to_numeric_frame(reconstructed, v45_cols)),dtype=float)
    original_preds = live.game_forecasts(reconstructed, x3_orig, x45_orig)
    original_preds['game_id'] = normalize(original_preds['game_id'])
    compare_forecast = original_preds.merge(archived_forecasts, on='game_id', suffixes=('_new','_locked'),validate='one_to_one')
    if len(compare_forecast) != len(archive_ids):
        raise RuntimeError('Original forecast parity missing games')
    for c in ('fair_margin_home','fair_total'):
        if not np.allclose(compare_forecast[c+'_new'],compare_forecast[c+'_locked'],atol=.001,rtol=0):
            raise RuntimeError(f'Cannot reproduce October 20 locked forecast: {c}')
    print('Original October 20 MODEL OUTPUT parity: PASS')

    g = forecasts.merge(eligible,on='game_id',how='inner',validate='one_to_one',suffixes=('', '_schedule'))
    if len(g) != len(eligible):
        raise RuntimeError('Results join missing predictions')
    if not (g.home.eq(g.schedule_home) & g.away.eq(g.schedule_away)).all():
        raise RuntimeError('Forecast and score team names disagree')
    if not pd.to_datetime(g.kickoff_utc,utc=True).eq(pd.to_datetime(g.kickoff_utc_schedule,utc=True)).all():
        raise RuntimeError('Forecast and schedule kickoff disagree')
    g['actual_margin_home'] = g.actual_home - g.actual_away
    g['actual_total'] = g.actual_home + g.actual_away
    g['v57_margin_error'] = g.fair_margin_home-g.actual_margin_home
    g['total_error'] = g.fair_total-g.actual_total
    g['v57_margin_abs_error'] = g.v57_margin_error.abs()
    g['total_abs_error'] = g.total_error.abs()
    g['v57_winner_correct'] = np.sign(g.fair_margin_home).eq(np.sign(g.actual_margin_home))
    g = g.sort_values(['kickoff_utc','game_id']).reset_index(drop=True)

    print('\n'+'='*86)
    print('REPLAY RESULTS — V5.7 ONLY (RESEARCH RETROSPECTIVE, NOT PROSPECTIVE)')
    print('='*86)
    print(json.dumps(metrics(g,'v57'),indent=2))
    print('\nBY WEEK')
    week_data=[]
    for wk, data in g.groupby('week',dropna=False,sort=True):
        week_data.append({'week':wk,**metrics(data,'v57')})
    weekly = pd.DataFrame(week_data)
    print(weekly.to_string(index=False,float_format=lambda x:f'{x:.3f}'))
    print('\nTOP 12 TOTAL MISSES')
    print(g.nlargest(12,'total_abs_error')[['week','away','home','actual_total','fair_total','total_abs_error']].round(2).to_string(index=False))
    print('\nTOP 12 MARGIN MISSES')
    print(g.nlargest(12,'v57_margin_abs_error')[['week','away','home','actual_margin_home','fair_margin_home','v57_margin_abs_error']].round(2).to_string(index=False))

    # All safety gates must pass before writing anything to a NEW output folder.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/'reports/v58_development/retrospective_replay_2026'/stamp
    output.mkdir(parents=True,exist_ok=False)
    g.to_csv(output/'game_grades.csv',index=False)
    weekly.to_csv(output/'weekly_metrics.csv',index=False)
    report={
        'label':'RESEARCH RETROSPECTIVE CHRONOLOGICAL REPLAY; NOT SAVED PROSPECTIVE FORECASTS',
        'archive_asof_utc':asof.isoformat(),
        'graded_FBS_FBS_games':len(g),
        'last_cached_kickoff':str(g.kickoff_utc.max()),
        'source_raw_schedule_sha256':sha(SCHEDULE),
        'source_team_games_sha256':sha(ROOT/'data/processed/team_games_2026.parquet'),
        'source_advanced_2026_sha256':sha(ROOT/'data/raw/cfbd/advanced_game_v4/advanced_2026.json'),
        'original_archived_snapshot_sha256':sha(orig_file),
        'archived_feature_parity':True,
        'archived_prediction_parity':True,
        'no_future_data_samples_passed':len(sample_indices),
        'historical_CFBD_publication_times_certified':False,
        'v57_training_seasons':'2014-2025',
        'v58_graded':False,
        'metrics':metrics(g,'v57'),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
    }
    (output/'audit.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('\nRESEARCH OUTPUT SAVED:',output)
    print('No source files, training datasets, or original forecasts modified.')
    print('Prior 2026 performance is reconstructed; missing source publication timestamps remain a limitation.')
    print('='*86)

if __name__=='__main__':
    main()
