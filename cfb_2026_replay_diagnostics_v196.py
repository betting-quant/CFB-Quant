"""Read-only diagnostics of a frozen v5.7 2026 research replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path.cwd()
base = root / 'reports/v58_development/retrospective_replay_2026'
runs = sorted(p for p in base.glob('*') if (p/'game_grades.csv').is_file() and (p/'audit.json').is_file())
if not runs:
    raise FileNotFoundError(f'No completed replay at {base}')
run = runs[-1]
audit = json.loads((run/'audit.json').read_text(encoding='utf-8'))
g = pd.read_csv(run/'game_grades.csv', dtype={'game_id': str})
needed = ['game_id','week','fair_margin_home','fair_total','actual_margin_home','actual_total','actual_home','actual_away']
missing = sorted(set(needed) - set(g.columns))
if missing:
    raise RuntimeError(f'Missing replay columns: {missing}')
if g.game_id.duplicated().any() or len(g) != audit['graded_FBS_FBS_games']:
    raise RuntimeError('Replay ID/count integrity failure')
for c in ['fair_margin_home','fair_total','actual_margin_home','actual_total','actual_home','actual_away']:
    g[c] = pd.to_numeric(g[c], errors='raise')
    if not np.isfinite(g[c].to_numpy()).all():
        raise RuntimeError(f'Non-finite replay values in {c}')
if not np.allclose(g.actual_home - g.actual_away, g.actual_margin_home):
    raise RuntimeError('Graded margin disagrees with actual scores')
if not np.allclose(g.actual_home + g.actual_away, g.actual_total):
    raise RuntimeError('Graded total disagrees with actual scores')
if (g.actual_margin_home == 0).any():
    raise RuntimeError('Unexpected tied final score in CFB grading')

hist_file = root / 'data/processed/cfb_matchup_features_kickoff_v3.parquet'
hist = pd.read_parquet(hist_file, columns=['game_id','season','home_away','points','opponent_points'])
hist = hist.loc[(pd.to_numeric(hist.season,errors='coerce')<=2025)&hist.home_away.eq('home')].copy()
if hist.game_id.duplicated().any() or len(hist)<1000:
    raise RuntimeError('Unexpected historical training scores')
for c in ['points','opponent_points']:
    hist[c]=pd.to_numeric(hist[c],errors='coerce')
hist=hist.dropna(subset=['points','opponent_points'])
home_base = float((hist.points-hist.opponent_points).mean())
total_base = float((hist.points+hist.opponent_points).mean())

actual_m=g.actual_margin_home.to_numpy(float)
actual_t=g.actual_total.to_numpy(float)
pred_m=g.fair_margin_home.to_numpy(float)
pred_t=g.fair_total.to_numpy(float)
model_right=np.sign(pred_m)==np.sign(actual_m)
home_right=actual_m>0
margin_gain=np.abs(pred_m-actual_m)-np.abs(home_base-actual_m)
total_gain=np.abs(pred_t-actual_t)-np.abs(total_base-actual_t)

rng=np.random.default_rng(20261021)
idx=rng.integers(0,len(g),size=(4000,len(g)))
def interval(diff,indices=idx):
    draws=np.asarray(diff,dtype=float)[indices].mean(axis=1)
    return tuple(np.quantile(draws,[0.025,0.975]))

def wilson(correct,n):
    z=1.96
    p=correct/n
    denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return center-half,center+half

print('='*82)
print('STEP 196 — 2026 REPLAY BASELINES, SUBGROUPS & UNCERTAINTY')
print('='*82)
print('Read-only replay:', run)
print(f'Games: {len(g)} | Archive parity: {audit.get("archived_feature_parity")} | No-future probes: {audit.get("no_future_data_samples_passed")}')
print(f'Model winner accuracy: {model_right.mean():.1%} ({model_right.sum()}/{len(g)})')
print('Approx. independent-game Wilson 95% interval: {:.1%}–{:.1%}'.format(*wilson(model_right.sum(),len(g))))
print(f'Always-home winner baseline: {home_right.mean():.1%} ({home_right.sum()}/{len(g)})')
print('Winner advantage over always-home: {:+.2f} pp; approximate paired bootstrap CI [{:+.2f}, {:+.2f}] pp'.format(
    100*(model_right.astype(float)-home_right.astype(float)).mean(),
    *(100*x for x in interval(model_right.astype(float)-home_right.astype(float)))))
print(f'Frozen 2014–2025 historical games: {len(hist):,}')
print(f'Historical constant margin: home {home_base:+.3f}; historical constant total: {total_base:.3f}')
print(f'Margin MAE: model {np.abs(pred_m-actual_m).mean():.3f} vs historical constant {np.abs(home_base-actual_m).mean():.3f}')
print('Model minus baseline margin MAE: {:+.3f}; paired bootstrap 95% interval [{:+.3f}, {:+.3f}]'.format(margin_gain.mean(),*interval(margin_gain)))
print(f'Total MAE: model {np.abs(pred_t-actual_t).mean():.3f} vs historical constant {np.abs(total_base-actual_t).mean():.3f}')
print('Model minus baseline total MAE: {:+.3f}; paired bootstrap 95% interval [{:+.3f}, {:+.3f}]'.format(total_gain.mean(),*interval(total_gain)))

schedule_path=root/'data/raw/cfbd/games/games_2026.json'
schedule=json.loads(schedule_path.read_text(encoding='utf-8'))
if isinstance(schedule,dict):
    schedule=schedule.get('games',[])
lookup={str(item['id']):item for item in schedule if isinstance(item,dict) and item.get('id') is not None}
elo_model=[]
elo_base=[]
for row in g.itertuples(index=False):
    raw=lookup.get(str(row.game_id),{})
    try:
        eh=float(raw.get('homePregameElo'))
        ea=float(raw.get('awayPregameElo'))
    except (TypeError,ValueError):
        continue
    if not (np.isfinite(eh) and np.isfinite(ea) and eh>0 and ea>0 and eh != ea):
        continue
    elo_model.append(np.sign(row.fair_margin_home)==np.sign(row.actual_margin_home))
    elo_base.append(np.sign(eh-ea)==np.sign(row.actual_margin_home))
print('\nPREGAME ELO FAVORITE COMPARISON')
if elo_model:
    elo_model=np.asarray(elo_model,dtype=float)
    elo_base=np.asarray(elo_base,dtype=float)
    n=len(elo_model)
    eidx=rng.integers(0,n,size=(4000,n))
    delta=elo_model-elo_base
    print(f'Paired Elo coverage: {n}/{len(g)} games')
    print(f'Model same-games accuracy: {elo_model.mean():.1%}; higher-pregame-Elo accuracy: {elo_base.mean():.1%}')
    print('Model minus Elo winner percentage points: {:+.2f}; paired bootstrap interval [{:+.2f}, {:+.2f}]'.format(100*delta.mean(),*(100*x for x in interval(delta,eidx))))
else:
    print('No usable paired pregame Elo values; skip this baseline')

print('\nBY ABSOLUTE MODEL SPREAD (larger favorites should be easier to predict outright)')
g['spread_band']=pd.cut(g.fair_margin_home.abs(),bins=[-np.inf,3,7,14,21,np.inf],right=False,labels=['0–<3','3–<7','7–<14','14–<21','21+'])
for band, sub in g.groupby('spread_band',observed=True,sort=False):
    print(f' {band}: n={len(sub):3d}, winner={np.sign(sub.fair_margin_home).eq(np.sign(sub.actual_margin_home)).mean():.1%}, margin_MAE={(sub.fair_margin_home-sub.actual_margin_home).abs().mean():.2f}')

print('\nBY WEEK (cached schedule is incomplete in Week 7)')
for wk,sub in g.groupby('week',sort=True):
    print(f' Week {wk}: n={len(sub):3d}, margin_bias={(sub.fair_margin_home-sub.actual_margin_home).mean():+.2f}, margin_MAE={(sub.fair_margin_home-sub.actual_margin_home).abs().mean():.2f}, total_MAE={(sub.fair_total-sub.actual_total).abs().mean():.2f}')

print('\nINTERPRETATION GUARDRAILS')
print('- 2026 is retrospective, not genuine archived prospective forecasting.')
print('- Baselines are not sportsbook spreads/totals; comparisons cannot establish betting edge.')
print('- Bootstrap intervals are exploratory (games may be correlated; season cache incomplete).')
print('- Historical CFBD observation publication timestamps remain uncertified.')
print('- Original repo files modified: NONE')
print('='*82)
