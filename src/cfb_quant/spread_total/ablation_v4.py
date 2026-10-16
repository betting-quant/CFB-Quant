"""Exploratory v4 feature-family ablations on chronological, paired scoring forecasts.

Never fetches API data, changes training source files, or interprets MAE as betting EV.
The main v3 baseline remains the reference and the results are hypothesis-generating.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .oof_points import build_points_oof
from .core import build_game_errors

FAMILIES = ("efficiency", "explosiveness", "trenches", "tempo")
VARIANTS = ("baseline", "efficiency", "explosiveness", "trenches", "tempo", "efficiency_trenches", "all_advanced")


def family(col: str) -> str:
    if not col.startswith("adv_"):
        raise ValueError(f"Not an advanced feature: {col}")
    if "explosiveness" in col:
        return "explosiveness"
    if "line_yards" in col or "stuff_rate" in col or "power_success" in col:
        return "trenches"
    if "drives" in col or "plays" in col:
        return "tempo"
    if "ppa" in col or "success_rate" in col:
        return "efficiency"
    raise ValueError(f"Unknown advanced feature classification: {col}")


def variant_sets(base_cols: list[str], adv_cols: list[str]) -> dict[str, list[str]]:
    if len(set(base_cols)) != len(base_cols) or len(set(adv_cols)) != len(adv_cols):
        raise ValueError("Duplicate manifest feature names")
    if set(base_cols) & set(adv_cols):
        raise ValueError("Base and advanced manifest features overlap")
    if not adv_cols or any(not c.startswith('adv_') for c in adv_cols):
        raise ValueError("Expected nonempty advanced-only feature names")
    groups = {name: [c for c in adv_cols if family(c) == name] for name in FAMILIES}
    if any(not cols for cols in groups.values()):
        raise ValueError(f"Empty advanced feature family: {[n for n, cols in groups.items() if not cols]}")
    return {
        "baseline": list(base_cols),
        "efficiency": base_cols + groups["efficiency"],
        "explosiveness": base_cols + groups["explosiveness"],
        "trenches": base_cols + groups["trenches"],
        "tempo": base_cols + groups["tempo"],
        "efficiency_trenches": base_cols + groups["efficiency"] + groups["trenches"],
        "all_advanced": base_cols + adv_cols,
    }


def metrics_for_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    games = build_game_errors(oof, allow_incomplete=True)
    games = games.assign(
        margin_abs_error=(games.actual_margin - games.pred_margin).abs(),
        total_abs_error=(games.actual_total - games.pred_total).abs(),
        margin_sq_error=(games.actual_margin - games.pred_margin)**2,
        total_sq_error=(games.actual_total - games.pred_total)**2,
    )
    required = ["game_id", "season", "week", "home_team", "away_team",
                "actual_margin", "actual_total", "pred_margin", "pred_total",
                "margin_abs_error", "total_abs_error", "margin_sq_error", "total_sq_error"]
    return games[required].copy()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/processed/cfb_matchup_features_advanced_v4.parquet')
    p.add_argument('--baseline-manifest', default='reports/cfb_matchup_feature_manifest_kickoff_v3.json')
    p.add_argument('--advanced-manifest', default='reports/cfb_matchup_feature_manifest_advanced_v4.json')
    p.add_argument('--out-dir', default='reports/spread_total_v4_ablations')
    p.add_argument('--first-year', type=int, default=2019)
    p.add_argument('--last-year', type=int, default=2024)
    args = p.parse_args()
    if args.first_year < 2015 or args.last_year < args.first_year or args.last_year > 2025:
        p.error('Select a valid chronological evaluation range between 2015 and 2025')

    data = pd.read_parquet(args.data)
    bm = json.loads(Path(args.baseline_manifest).read_text(encoding='utf-8'))
    am = json.loads(Path(args.advanced_manifest).read_text(encoding='utf-8'))
    base = list(bm['feature_columns'])
    complete = list(am['feature_columns'])
    if complete[:len(base)] != base:
        raise ValueError('v4 manifest must start with EXACT v3 feature columns in the same order')
    if data.duplicated(['game_id', 'team']).any():
        raise ValueError('Duplicate team-game rows in experiment data')
    if 'kickoff_utc' not in data or pd.to_datetime(data['kickoff_utc'], errors='coerce', utc=True).isna().any():
        raise ValueError('Missing or invalid kickoff timestamps')
    if any(c.startswith('raw_adv_') for c in complete):
        raise ValueError('Current-game advanced raw stats appear in predictive manifest')
    if any(c in ('points', 'opponent_points') for c in complete):
        raise ValueError('Target leakage into manifest')
    variants = variant_sets(base, complete[len(base):])
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_games = []
    sample_ids = None
    for variant in VARIANTS:
        cols = variants[variant]
        print(f'\nRUN {variant}: {len(cols)} input features', flush=True)
        oof = build_points_oof(data, cols, first_year=args.first_year, last_year=args.last_year)
        game = metrics_for_predictions(oof)
        keys = set(game.game_id.astype(str))
        if sample_ids is not None and keys != sample_ids:
            raise ValueError(f'Different evaluation games for variant {variant}; cannot compare fairly')
        sample_ids = keys
        game.insert(0, 'variant', variant)
        all_games.append(game)
        print(f'  paired games={len(game)}; spread MAE={game.margin_abs_error.mean():.3f}; '
              f'total MAE={game.total_abs_error.mean():.3f}', flush=True)

    joined = pd.concat(all_games, ignore_index=True)
    by_year = (joined.groupby(['variant', 'season'], as_index=False)
               .agg(games=('game_id', 'size'), margin_mae=('margin_abs_error', 'mean'),
                    total_mae=('total_abs_error', 'mean'),
                    margin_rmse=('margin_sq_error', lambda x: float(np.sqrt(x.mean()))),
                    total_rmse=('total_sq_error', lambda x: float(np.sqrt(x.mean())))))
    overall = (joined.groupby('variant', as_index=False)
               .agg(games=('game_id', 'size'), margin_mae=('margin_abs_error', 'mean'),
                    total_mae=('total_abs_error', 'mean'),
                    margin_rmse=('margin_sq_error', lambda x: float(np.sqrt(x.mean()))),
                    total_rmse=('total_sq_error', lambda x: float(np.sqrt(x.mean())))))
    baseline = overall.loc[overall.variant == 'baseline'].iloc[0]
    overall['delta_margin_mae_vs_baseline'] = overall.margin_mae - baseline.margin_mae
    overall['delta_total_mae_vs_baseline'] = overall.total_mae - baseline.total_mae
    overall['feature_count'] = overall.variant.map({n: len(variants[n]) for n in variants})
    overall = overall.sort_values('variant').reset_index(drop=True)
    joined.to_csv(output / 'paired_games_all_variants.csv', index=False)
    by_year.to_csv(output / 'by_season.csv', index=False)
    overall.to_csv(output / 'overall.csv', index=False)
    print('\nEXPLORATORY CHRONOLOGICAL COMPARISON (lower MAE is better)')
    print(overall[['variant','games','feature_count','margin_mae','total_mae',
                   'delta_margin_mae_vs_baseline','delta_total_mae_vs_baseline']]
          .round(4).to_string(index=False))
    print(f'\nSaved 3 CSV reports in {output}')
    print('WARNING: Outcomes in these years have been examined during development. '
          'These experiments are NOT untouched validation or evidence of +EV. '
          'Do not promote a selected variant without a new prospective test.')


if __name__ == '__main__':
    main()
