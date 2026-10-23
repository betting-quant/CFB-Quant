"""Step 198: Read-only, retrospective 2026 market-residual and source sensitivity diagnostics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path('reports/v58_development/market_benchmark_2026')
files = sorted(root.glob('*/paired_game_market_comparison.csv'))
if not files:
    raise FileNotFoundError('No Step 197 market benchmark output found')
p = files[-1]
audit = json.loads((p.parent / 'audit.json').read_text(encoding='utf-8-sig'))
df = pd.read_csv(p, dtype={'game_id': str})
needed = ['game_id', 'week', 'fair_margin_home', 'actual_margin_home',
          'market_home_margin', 'fair_total', 'actual_total', 'market_total',
          'market_spread_open', 'market_total_open']
missing = sorted(set(needed).difference(df.columns))
if missing:
    raise ValueError(f'Missing required columns: {missing}')
if df.game_id.duplicated().any() or len(df) != audit['paired_games']:
    raise RuntimeError('Paired-game identity or row-count audit failure')
print('=' * 77)
print('STEP 198 — V5.7 VS MARKET: DISAGREEMENT AND PRICE-SOURCE CHECKS')
print('=' * 77)
print('Source:', p.parent)
print('Games:', len(df), '| Provider:', audit['selected_provider'])
print('Historical odds as-of-kickoff certified:', audit['historical_market_line_asof_kickoff_certified'])
print('No source files or models modified.')

for kind, actual, model, market, opener in [
    ('MARGIN', 'actual_margin_home', 'fair_margin_home', 'market_home_margin', 'market_spread_open'),
    ('TOTAL', 'actual_total', 'fair_total', 'market_total', 'market_total_open'),
]:
    frame = df.copy()
    for col in (actual, model, market, opener):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    frame['disagreement'] = frame[model] - frame[market]
    frame['actual_minus_market'] = frame[actual] - frame[market]
    frame['model_ae'] = (frame[model] - frame[actual]).abs()
    frame['market_ae'] = (frame[market] - frame[actual]).abs()
    print('\n' + kind + ': MODEL–MARKET DISAGREEMENT')
    for lo, hi, label in [(0, 3, '0–<3'), (3, 7, '3–<7'),
                          (7, 14, '7–<14'), (14, np.inf, '14+')]:
        sub = frame.loc[frame.disagreement.abs().ge(lo) & frame.disagreement.abs().lt(hi)]
        if sub.empty:
            continue
        nonzero = sub.loc[sub.disagreement.ne(0)]
        directional = (np.sign(nonzero.disagreement) ==
                       np.sign(nonzero.actual_minus_market)).mean() if len(nonzero) else np.nan
        print(f'  Gap {label:5}: n={len(sub):3} | model MAE={sub.model_ae.mean():5.2f}'
              f' | market MAE={sub.market_ae.mean():5.2f}'
              f' | model direction matches market residual={directional:.1%}')
    open_prices = -frame[opener] if kind == 'MARGIN' else frame[opener]
    valid = open_prices.notna() & np.isfinite(open_prices)
    if kind == 'MARGIN':
        valid &= open_prices.abs().le(80)
    else:
        valid &= open_prices.between(10, 130)
    opening = frame.loc[valid].copy()
    if len(opening):
        opened = open_prices.loc[valid]
        print(f'  Reported opening-line coverage: {len(opening)}/{len(frame)}')
        print(f'  Same-game MAE: model={opening.model_ae.mean():.3f},'
              f' reported line={opening.market_ae.mean():.3f},'
              f' reported opener={(opened-opening[actual]).abs().mean():.3f}')
        print(f'  Average absolute change opener-to-reported line:'
              f' {(opened-opening[market]).abs().mean():.3f}')
    else:
        print('  Opening-line comparison: unavailable (no valid reported opening prices)')

print('\nMARGIN: WEEK 1 VS LATER CACHED WEEKS')
for label, group in [('Week 1', df.loc[df.week.eq(1)]),
                     ('Weeks 2+', df.loc[df.week.ne(1)])]:
    model_ae = (group.fair_margin_home-group.actual_margin_home).abs().mean()
    market_ae = (group.market_home_margin-group.actual_margin_home).abs().mean()
    print(f'  {label}: n={len(group)} model={model_ae:.3f}'
          f' market={market_ae:.3f} model-minus-market={model_ae-market_ae:+.3f}')

raw_file = p.parent / 'cfbd_lines_2026_response.json'
if raw_file.is_file():
    rows = json.loads(raw_file.read_text(encoding='utf-8-sig'))
    by_id = {str(x.get('id')): x for x in rows if isinstance(x, dict)}
    records = []
    for g in df.itertuples(index=False):
        source = by_id.get(str(g.game_id))
        if not source:
            continue
        for line in source.get('lines') or []:
            if line.get('provider') not in ('Bovada', 'DraftKings'):
                continue
            try:
                spread = float(line['spread'])
                total = float(line['overUnder'])
            except (ValueError, TypeError, KeyError):
                continue
            if abs(spread) > 80 or not 10 <= total <= 130:
                continue
            records.append((str(g.game_id), line['provider'], -spread, total))
    prices = pd.DataFrame(records, columns=['game_id', 'provider', 'margin', 'total'])
    print('\nSOURCE SENSITIVITY (same 2026 results; no timestamp verification)')
    for provider in ('Bovada', 'DraftKings'):
        part = prices.loc[prices.provider.eq(provider)]
        part = part.loc[~part.game_id.duplicated(keep=False)]
        paired = df.merge(part, on='game_id', validate='one_to_one')
        if len(paired):
            m_mae = (paired.margin-paired.actual_margin_home).abs().mean()
            t_mae = (paired.total-paired.actual_total).abs().mean()
            print(f'  {provider:11}: n={len(paired)}, margin MAE={m_mae:.3f}, total MAE={t_mae:.3f}')
else:
    print('\nSource sensitivity: raw Step 197 CFBD JSON not found')
print('\nRESEARCH ONLY: Historical publication timestamps and closing status remain unverified.')
print('No subgroup was selected prospectively; apparent disagreement success is not betting ROI.')
print('=' * 77)
