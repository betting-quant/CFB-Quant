"""Conservative, reversible replacement of a slow pandas feature-building block.

Run from the CFB-Quant root:
  python CFB-Quant-Advanced-v4-Vectorized-Hotfix/install_hotfix.py --check
  python CFB-Quant-Advanced-v4-Vectorized-Hotfix/install_hotfix.py --apply

Does not read any .env, call APIs, or modify cached JSON or model outputs.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil
import sys

TARGET = Path('src/cfb_quant/spread_total/advanced_v4.py')
OLD_BEGIN = '    made = {}\n    for col in rawcols:\n'
OLD_END = '    colnames = list(generated.columns)\n'
NEW_MARKER = '    # V4_VECTOR_FEATURES: bulk operations preserve historical shift and missingness semantics.\n'
REPLACEMENT = '''    # V4_VECTOR_FEATURES: bulk operations preserve historical shift and missingness semantics.
    # Every value is from a prior game (or missing), never from the target game.
    raw = joined[rawcols].apply(pd.to_numeric, errors="coerce")
    teams = joined["team"]
    season_groups = [joined["team"], joined["season"]]

    # History across season boundaries, preserving the prior game's observed value.
    lagged = raw.groupby(teams, sort=False).shift(1)
    roll3 = lagged.groupby(teams, sort=False).rolling(3, min_periods=1).mean()
    roll5 = lagged.groupby(teams, sort=False).rolling(5, min_periods=1).mean()
    # Grouped rolling adds the team as an index level; restore original row IDs.
    roll3.index = roll3.index.droplevel(0)
    roll5.index = roll5.index.droplevel(0)
    roll3 = roll3.reindex(joined.index)
    roll5 = roll5.reindex(joined.index)

    # Season-to-date averages exclude the current game's result.
    # Fill missing observations with 0 ONLY for accumulation, and count only
    # real observations. A missing result does not erase earlier information.
    season_history = raw.groupby(season_groups, sort=False).shift(1)
    season_sums = season_history.fillna(0).groupby(season_groups, sort=False).cumsum()
    season_counts = season_history.notna().astype('int32').groupby(season_groups, sort=False).cumsum()
    season_avgs = season_sums.div(season_counts.where(season_counts > 0))

    suffixes = ("lag1", "avg3", "avg5", "season_avg")
    def relabel(frame, suffix):
        return frame.set_axis(
            [f"adv_{col.removeprefix('raw_adv_')}_{suffix}" for col in rawcols], axis="columns"
        )
    generated = pd.concat(
        [relabel(lagged, "lag1"), relabel(roll3, "avg3"),
         relabel(roll5, "avg5"), relabel(season_avgs, "season_avg")], axis=1
    )
    # Preserve the original per-metric feature order in the resulting manifest.
    ordered = [f"adv_{col.removeprefix('raw_adv_')}_{suffix}"
               for col in rawcols for suffix in suffixes]
    generated = generated.loc[:, ordered]
'''

def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    path = Path.cwd() / TARGET
    if not path.is_file():
        sys.exit(f'STOP: Expected file missing: {path}')
    content = path.read_text(encoding='utf-8')
    if NEW_MARKER.strip() in content:
        print('ALREADY INSTALLED: vectorized v4 feature builder detected. No files changed.')
        return
    if content.count(OLD_BEGIN) != 1 or content.count(OLD_END) != 1:
        sys.exit('STOP: source version differs from audited v4; no files changed.')
    a = content.index(OLD_BEGIN)
    b = content.index(OLD_END, a)
    replacement = content[:a] + REPLACEMENT + content[b:]
    try:
        ast.parse(replacement)
    except SyntaxError as e:
        sys.exit(f'STOP: patched source does not compile: {e}')
    if args.check:
        print('PASS: audited v4 block detected; replacement compiles; no files changed')
        return
    backup = path.with_name(path.name + '.before_vectorized_v4.bak')
    if backup.exists():
        sys.exit(f'STOP: backup already exists: {backup}; no files changed')
    shutil.copy2(path, backup)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(replacement, encoding='utf-8')
    temporary.replace(path)
    print(f'INSTALLED: {TARGET}')
    print(f'BACKUP: {backup}')
    print('Your previously downloaded CFBD cache remains intact. Do NOT fetch again.')

if __name__ == '__main__':
    main()
