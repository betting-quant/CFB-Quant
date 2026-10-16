"""Conservative 2026 CFBD team-history append from cached completed box scores.

Default is READ ONLY. --apply makes an atomic Parquet update after backing up
outside data/processed (which is globbed by the main historical feature builder).
No API calls, no model/feature builds, no predictions.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEDULE = Path('data/raw/cfbd/games/games_2026.json')
HISTORY = Path('data/processed/team_games_2026.parquet')
BOX_ROOT = Path('data/raw/cfbd/team_stats/2026')

SCHEMA = [
    'game_id', 'season', 'week', 'season_type', 'team_id', 'team',
    'conference', 'opponent', 'home_away', 'points', 'opponent_points',
    'completions', 'pass_attempts', 'passing_yards', 'passing_touchdowns',
    'rush_attempts', 'rushing_yards', 'rushing_touchdowns',
    'interceptions', 'turnovers', 'fumbles_lost', 'sacks',
    'tackles_for_loss', 'qb_hurries', 'possession_seconds', 'penalties',
    'penalty_yards', 'yards_per_rush_attempt', 'yards_per_pass'
]

# CFBD missing count stats mean 0; optional rate stats are null when missing.
COUNTS = {
    'passing_touchdowns': 'passingTDs',
    'rushing_touchdowns': 'rushingTDs',
    'interceptions': 'interceptions',  # thrown, not passesIntercepted
    'turnovers': 'turnovers',
    'fumbles_lost': 'fumblesLost',
    'sacks': 'sacks',
    'tackles_for_loss': 'tacklesForLoss',
    'qb_hurries': 'qbHurries',
}
REQUIRED_STATS = (
    'completionAttempts', 'netPassingYards', 'rushingAttempts',
    'rushingYards', 'possessionTime'
)
DECIMAL_STATS = {
    'passing_yards': 'netPassingYards',
    'rush_attempts': 'rushingAttempts',
    'rushing_yards': 'rushingYards',
    'yards_per_rush_attempt': 'yardsPerRushAttempt',
    'yards_per_pass': 'yardsPerPass',
}


def gid(value) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError('Game ID missing/invalid')
    text = str(value)
    if text.endswith('.0') and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError(f'Non-numeric CFBD game ID: {text!r}')
    return text


def parsed_number(value, *, missing=None) -> float | None:
    if value is None or str(value).strip() in {'', '--', 'N/A'}:
        return missing
    try:
        return float(str(value).replace(',', '').strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid number: {value!r}') from exc


def parsed_pair(value, key: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError(f'Invalid/missing {key}: {value!r}')
    match = re.fullmatch(r'\s*(\d+)\s*-\s*(\d+)\s*', value)
    if not match:
        raise ValueError(f'Invalid {key}: {value!r}')
    return int(match.group(1)), int(match.group(2))


def parsed_seconds(value) -> float:
    if not isinstance(value, str):
        raise ValueError(f'Invalid/missing possessionTime: {value!r}')
    fields = value.strip().split(':')
    if len(fields) not in (2, 3) or not all(p.isdigit() for p in fields):
        raise ValueError(f'Invalid possessionTime: {value!r}')
    nums = list(map(int, fields))
    if nums[-1] >= 60 or (len(nums) == 3 and nums[-2] >= 60):
        raise ValueError(f'Invalid possessionTime: {value!r}')
    return float(nums[0] * 60 + nums[1] if len(nums) == 2 else
                 nums[0] * 3600 + nums[1] * 60 + nums[2])


def is_complete(g: dict) -> bool:
    return (g.get('completed') is True and
            g.get('homePoints') is not None and
            g.get('awayPoints') is not None)


def fbs_involved(g: dict) -> bool:
    return 'fbs' in {str(g.get('homeClassification', '')).lower(),
                     str(g.get('awayClassification', '')).lower()}


def selected_schedule(games: list[dict], *, weeks: tuple[int, ...]) -> dict[str, dict]:
    by_id = {}
    for g in games:
        if int(g.get('season') or 0) != 2026:
            continue
        if g.get('seasonType') != 'regular' or int(g.get('week') or -1) not in weeks:
            continue
        if not (is_complete(g) and fbs_involved(g)):
            continue
        key = gid(g.get('id'))
        if key in by_id:
            raise ValueError(f'Duplicate scheduled game {key}')
        by_id[key] = g
    return by_id


def parse_box(g: dict, scheduled: dict) -> list[dict]:
    key = gid(g.get('id'))
    if key != gid(scheduled.get('id')):
        raise ValueError(f'Schedule/box ID mismatch: {key}')
    teams = g.get('teams')
    if not isinstance(teams, list) or len(teams) != 2:
        raise ValueError(f'Game {key}: expected exactly two team box-score entries')
    by_side = {str(t.get('homeAway', '')).lower(): t for t in teams}
    if len(by_side) != 2 or set(by_side) != {'home', 'away'}:
        raise ValueError(f'Game {key}: invalid/duplicated homeAway flags')
    result: list[dict] = []
    for side in ('home', 'away'):
        raw = by_side[side]
        other = by_side['away' if side == 'home' else 'home']
        expected_id = scheduled.get(f'{side}Id')
        if str(raw.get('teamId')) != str(expected_id):
            raise ValueError(f'Game {key} {side}: team ID mismatch ({raw.get("teamId")} != {expected_id})')
        expected_points = parsed_number(scheduled.get(f'{side}Points'))
        points = parsed_number(raw.get('points'))
        if expected_points is None or points is None or points != expected_points:
            raise ValueError(f'Game {key} {side}: points disagree with completed schedule')
        stats = raw.get('stats')
        if not isinstance(stats, list) or not stats:
            raise ValueError(f'Game {key} {side}: missing box statistics')
        if any(not isinstance(s, dict) or 'category' not in s for s in stats):
            raise ValueError(f'Game {key} {side}: malformed stat entries')
        stat = {s['category']: s.get('stat') for s in stats}
        if len(stat) != len(stats):
            raise ValueError(f'Game {key} {side}: duplicate stat keys')
        missing = [s for s in REQUIRED_STATS if s not in stat]
        if missing:
            raise ValueError(f'Game {key} {side}: incomplete required stats {missing}')
        completions, pass_attempts = parsed_pair(stat['completionAttempts'], 'completionAttempts')
        penalties, penalty_yards = (parsed_pair(stat['totalPenaltiesYards'], 'totalPenaltiesYards')
                                   if 'totalPenaltiesYards' in stat else (None, None))
        if completions > pass_attempts:
            raise ValueError(f'Game {key} {side}: completions exceed attempts')
        team_name = raw.get('team')
        other_name = other.get('team')
        if not team_name or not other_name:
            raise ValueError(f'Game {key} {side}: missing team names')
        row = {
            'game_id': int(key), 'season': 2026, 'week': int(scheduled['week']),
            'season_type': 'regular', 'team_id': int(raw['teamId']),
            'team': team_name,
            'conference': raw.get('conference') or scheduled.get(f'{side}Conference'),
            'opponent': other_name, 'home_away': side,
            'points': points, 'opponent_points': parsed_number(other.get('points')),
            'completions': float(completions), 'pass_attempts': float(pass_attempts),
            'possession_seconds': parsed_seconds(stat['possessionTime']),
            'penalties': None if penalties is None else float(penalties),
            'penalty_yards': None if penalty_yards is None else float(penalty_yards),
        }
        for column, category in COUNTS.items():
            row[column] = parsed_number(stat.get(category), missing=0.0)
        for column, category in DECIMAL_STATS.items():
            row[column] = parsed_number(stat.get(category), missing=None)
        if row['rush_attempts'] is None or row['rushing_yards'] is None or row['passing_yards'] is None:
            raise ValueError(f'Game {key} {side}: required numeric statistics unavailable')
        result.append(row)
    if result[0]['opponent_points'] != result[1]['points'] or result[1]['opponent_points'] != result[0]['points']:
        raise ValueError(f'Game {key}: inconsistent opponent scores')
    return result


def load_box_records(folder: Path, *, weeks: tuple[int, ...]) -> dict[str, dict]:
    out = {}
    for week in weeks:
        path = folder / f'team_stats_2026_regular_week_{week:02d}.json'
        if not path.is_file():
            raise FileNotFoundError(f'Week {week} file missing: {path}')
        records = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(records, list):
            raise ValueError(f'Week {week}: box-score cache must contain a list')
        for g in records:
            if not isinstance(g, dict):
                raise ValueError(f'Week {week}: invalid record')
            key = gid(g.get('id'))
            if key in out:
                raise ValueError(f'Duplicate box-score game ID {key} across weeks')
            out[key] = g
    return out


def make_candidate(old: pd.DataFrame, schedule: list[dict], raw: dict[str, dict], *,
                   weeks: tuple[int, ...]) -> tuple[pd.DataFrame, dict]:
    absent = set(SCHEMA) - set(old.columns)
    if absent:
        raise ValueError(f'Existing team history missing columns: {sorted(absent)}')
    if old.duplicated(['game_id', 'team_id']).any():
        raise ValueError('Existing team history contains duplicate game/team IDs')
    existing_groups = old.groupby('game_id', dropna=False)
    if (existing_groups.size() != 2).any() or any(set(g.home_away) != {'home','away'} for _,g in existing_groups):
        raise ValueError('Existing team history contains invalid home/away pairs')
    scheduled = selected_schedule(schedule, weeks=weeks)
    existing = {gid(x) for x in old['game_id'].unique()}
    missing_box = set(scheduled) - set(raw) - existing
    if missing_box:
        raise ValueError(f'{len(missing_box)} completed FBS-involved games missing box scores: {sorted(missing_box)[:12]}')
    rows = []
    by_week = Counter()
    all_selected_by_week = Counter()
    for game_id, game in sorted(scheduled.items()):
        all_selected_by_week[int(game['week'])] += 1
        if game_id in existing:
            continue
        rows.extend(parse_box(raw[game_id], game))
        by_week[int(game['week'])] += 1
    appended = pd.DataFrame(rows, columns=SCHEMA)
    if not appended.empty:
        if appended.duplicated(['game_id','team_id']).any():
            raise ValueError('Duplicate incoming team rows')
        for column in old.columns:
            if column not in appended:
                appended[column] = None
        appended = appended[old.columns]
        # Respect existing Parquet schema, not inferred raw data types.
        for column in old.columns:
            try:
                appended[column] = appended[column].astype(old[column].dtype)
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Cannot cast {column} to existing Parquet dtype: {exc}') from exc
        candidate = pd.concat([old, appended], ignore_index=True)
    else:
        candidate = old.copy()
    if candidate.duplicated(['game_id','team_id']).any():
        raise ValueError('Merged candidate contains duplicates')
    if len(candidate) != len(old) + len(appended):
        raise AssertionError('Unexpected candidate row count')
    counts = candidate.groupby('game_id').size()
    if (counts != 2).any():
        raise ValueError('Merged candidate includes incomplete game pairs')
    # Verify original history unchanged, including original row order.
    if not candidate.iloc[:len(old)].reset_index(drop=True).equals(old.reset_index(drop=True)):
        raise AssertionError('Original history modified')
    audit = {
        'existing_games': len(existing), 'existing_team_rows': len(old),
        'completed_fbs_involved_week4_to_6': len(scheduled),
        'completed_by_week': dict(sorted(all_selected_by_week.items())),
        'added_by_week': dict(sorted(by_week.items())),
        'added_games': len(appended) // 2,
        'added_team_rows': len(appended),
        'result_games': int(candidate['game_id'].nunique()),
        'result_team_rows': len(candidate),
        'nulls_new_rows': ({k: int(v) for k,v in appended.isna().sum().items() if v}
                           if len(appended) else {}),
    }
    return candidate, audit


def run(root: Path, *, apply: bool = False, weeks: tuple[int, ...] = (4,5,6)) -> dict:
    path = root / HISTORY
    if not path.is_file():
        raise FileNotFoundError(f'Missing original 2026 processed history: {path}')
    old = pd.read_parquet(path)
    sched = json.loads((root / SCHEDULE).read_text(encoding='utf-8-sig'))
    if not isinstance(sched, list):
        raise ValueError('Cached schedule is not a list')
    raw = load_box_records(root / BOX_ROOT, weeks=weeks)
    candidate, audit = make_candidate(old, sched, raw, weeks=weeks)
    print('2026 TEAM HISTORY AUDIT')
    print(json.dumps(audit, indent=2))
    print('Mode:', 'APPLY' if apply else 'AUDIT (read-only)')
    if not apply:
        return audit
    if not audit['added_games']:
        print('No new games; existing file unchanged.')
        return audit
    # Never save backup into data/processed: feature builder globs team_games_*.parquet.
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = root / 'data/backups/team_history_2026' / timestamp / 'team_games_2026.parquet'
    if backup.exists():
        raise FileExistsError(f'Backup already exists: {backup}')
    backup.parent.mkdir(parents=True, exist_ok=False)
    shutil.copy2(path, backup)
    staged = path.with_suffix('.pending')
    if staged.exists():
        raise FileExistsError(f'Pending file exists; investigate before continuing: {staged}')
    try:
        candidate.to_parquet(staged, index=False, engine='pyarrow')
        verified = pd.read_parquet(staged)
        if len(verified) != len(candidate) or verified['game_id'].nunique() != audit['result_games']:
            raise ValueError('Candidate Parquet read-back verification failed')
        staged.replace(path)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    report = root / 'reports' / 'team_history_refresh_2026' / f'apply_{timestamp}.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'audit': audit, 'backup': str(backup),
                                  'updated_file': str(path), 'applied_at_utc': timestamp}, indent=2),
                      encoding='utf-8')
    print('REFRESH APPLIED')
    print('Backup:', backup)
    print('Updated:', path)
    print('Report:', report)
    print('No feature tables or predictive models rebuilt. No leakage-safe inference yet.')
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit', action='store_true', help='Read-only dry run (default)')
    mode.add_argument('--apply', action='store_true', help='Back up and safely append validated games')
    args = parser.parse_args()
    run(Path.cwd(), apply=args.apply)


if __name__ == '__main__':
    main()
