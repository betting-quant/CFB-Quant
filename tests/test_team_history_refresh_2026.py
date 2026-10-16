import pandas as pd
import pytest

from cfb_quant.live.team_history_refresh_2026 import (
    SCHEMA, parsed_pair, parsed_seconds, parse_box, make_candidate,
    selected_schedule,
)


def game(gid=401856881, week=4):
    return dict(id=gid, season=2026, week=week, seasonType='regular',
                completed=True, homeClassification='fbs', awayClassification='fbs',
                homeId=277, awayId=197, homePoints=24, awayPoints=41,
                homeConference='Big 12', awayConference='Big 12')


def box(gid=401856881):
    home = dict(team='West Virginia', teamId=277, homeAway='home', conference='Big 12', points=24,
                stats=[{'category': k, 'stat': v} for k, v in {
                    'rushingTDs': '3', 'passingTDs': '0', 'interceptions': '1',
                    'turnovers': '1', 'fumblesLost': '0', 'sacks': '0',
                    'tacklesForLoss': '3', 'qbHurries': '1',
                    'possessionTime': '31:04', 'totalPenaltiesYards': '3-12',
                    'yardsPerRushAttempt': '5.1', 'rushingAttempts': '66',
                    'rushingYards': '335', 'yardsPerPass': '6.0',
                    'completionAttempts': '10-22', 'netPassingYards': '132'
                }.items()])
    away = dict(team='Oklahoma State', teamId=197, homeAway='away', conference='Big 12', points=41,
                stats=[{'category': k, 'stat': v} for k,v in {
                    'rushingTDs': '1', 'passingTDs': '3', 'interceptions': '1',
                    'turnovers': '1', 'fumblesLost': '0', 'sacks': '4',
                    'tacklesForLoss': '8', 'qbHurries': '0',
                    'possessionTime': '28:56', 'totalPenaltiesYards': '9-88',
                    'yardsPerRushAttempt': '5.4', 'rushingAttempts': '41',
                    'rushingYards': '220', 'yardsPerPass': '14.4',
                    'completionAttempts': '20-27', 'netPassingYards': '389'
                }.items()])
    return {'id': gid, 'teams': [home, away]}


def test_parse_pairs_and_possession():
    assert parsed_pair('21-24', 'completionAttempts') == (21,24)
    assert parsed_pair(' 10-90 ', 'totalPenaltiesYards') == (10,90)
    assert parsed_seconds('27:42') == 1662
    assert parsed_seconds('1:02:03') == 3723
    with pytest.raises(ValueError):
        parsed_seconds('27:65')


def test_canonical_week4_mapping():
    rows = parse_box(box(), game())
    assert len(rows) == 2
    h, a = rows
    assert h['game_id'] == 401856881
    assert h['team_id'] == 277 and a['team_id'] == 197
    assert h['completions'] == 10 and h['pass_attempts'] == 22
    assert h['rush_attempts'] == 66 and h['rushing_yards'] == 335
    assert h['passing_yards'] == 132 and h['passing_touchdowns'] == 0
    assert h['points'] == 24 and h['opponent_points'] == 41
    assert h['possession_seconds'] == 1864 and a['possession_seconds'] == 1736
    assert h['penalties'] == 3 and h['penalty_yards'] == 12
    assert a['sacks'] == 4 and a['yards_per_pass'] == 14.4
    assert set(h) == set(SCHEMA)


def test_no_confusion_with_defensive_interceptions():
    g = box()
    g['teams'][0]['stats'].append({'category': 'passesIntercepted', 'stat': '4'})
    assert parse_box(g, game())[0]['interceptions'] == 1


def test_reject_bad_boxscore_or_wrong_outcome():
    g = box()
    g['teams'][0]['points'] = 22
    with pytest.raises(ValueError, match='points disagree'):
        parse_box(g, game())
    g = box()
    g['teams'][0]['stats'] = [s for s in g['teams'][0]['stats'] if s['category'] != 'netPassingYards']
    with pytest.raises(ValueError, match='incomplete required'):
        parse_box(g, game())


def test_requires_completed_fbs_involved_games():
    eligible = game()
    cross = dict(game(12345), awayClassification='fcs')
    incomplete = dict(game(12346), completed=False)
    non_fbs = dict(game(12347), homeClassification='fcs', awayClassification='fcs')
    result = selected_schedule([eligible, cross, incomplete, non_fbs], weeks=(4,5,6))
    assert set(result) == {'401856881', '12345'}


def test_append_preserves_existing_rows_and_is_idempotent():
    old0 = parse_box(box(), game())
    for r in old0:
        r['game_id'] = 987654321
        r['week'] = 3
    old = pd.DataFrame(old0, columns=SCHEMA)
    candidate, info = make_candidate(old, [game()], {'401856881': box()}, weeks=(4,5,6))
    assert info['added_games'] == 1
    assert info['result_games'] == 2
    assert len(candidate) == 4
    assert candidate.iloc[:2].reset_index(drop=True).equals(old)
    again, info2 = make_candidate(candidate, [game()], {'401856881': box()}, weeks=(4,5,6))
    assert info2['added_games'] == 0
    assert again.equals(candidate)


def test_missing_completed_boxscore_blocks_write():
    old0 = parse_box(box(), game())
    for r in old0:
        r['game_id'] = 987654321
        r['week'] = 3
    old = pd.DataFrame(old0, columns=SCHEMA)
    with pytest.raises(ValueError, match='missing box scores'):
        make_candidate(old, [game()], {}, weeks=(4,5,6))


def test_week3_baylor_latech_original_parquet_mapping():
    g = dict(game(401867804, week=3), homeId=239, awayId=2348,
             homePoints=36, awayPoints=19)
    home_stats = {
        'rushingTDs': '2', 'passingTDs': '3', 'interceptions': '0',
        'turnovers': '2', 'fumblesLost': '2', 'sacks': '1',
        'tacklesForLoss': '5', 'qbHurries': '6',
        'possessionTime': '27:42', 'totalPenaltiesYards': '10-90',
        'yardsPerRushAttempt': '3.8', 'rushingAttempts': '44',
        'rushingYards': '165', 'yardsPerPass': '10.5',
        'completionAttempts': '21-24', 'netPassingYards': '253'
    }
    away_stats = {
        'rushingTDs': '2', 'passingTDs': '1', 'interceptions': '1',
        'turnovers': '2', 'fumblesLost': '1', 'sacks': '3',
        'tacklesForLoss': '9', 'qbHurries': '1',
        'possessionTime': '32:18', 'totalPenaltiesYards': '9-81',
        'yardsPerRushAttempt': '4.1', 'rushingAttempts': '34',
        'rushingYards': '139', 'yardsPerPass': '5.1',
        'completionAttempts': '25-44', 'netPassingYards': '223'
    }
    raw = {
        'id': 401867804,
        'teams': [
            {'team': 'Baylor', 'teamId': 239, 'homeAway': 'home',
             'points': 36, 'conference': 'Big 12',
             'stats': [{'category': k, 'stat': v} for k,v in home_stats.items()]},
            {'team': 'Louisiana Tech', 'teamId': 2348, 'homeAway': 'away',
             'points': 19, 'conference': 'Sun Belt',
             'stats': [{'category': k, 'stat': v} for k,v in away_stats.items()]},
        ]
    }
    h,a = parse_box(raw, g)
    assert (h['points'], h['opponent_points'], h['completions'],
            h['pass_attempts'], h['possession_seconds'],
            h['penalties'], h['penalty_yards']) == (36, 19, 21, 24, 1662, 10, 90)
    assert (a['points'], a['opponent_points'], a['completions'],
            a['pass_attempts'], a['possession_seconds'],
            a['penalties'], a['penalty_yards']) == (19, 36, 25, 44, 1938, 9, 81)
    assert h['passing_yards'] == 253 and a['passing_yards'] == 223
