from pathlib import Path
import json
import pandas as pd
import pytest
from audit_schedule_chronology_v2 import schedules_to_frame, audit_chronology, load_games


def fixture(reverse_id=False):
    games = [
        {"id": 102 if reverse_id else 101, "startDate": "2024-08-24T16:00:00Z", "homeTeam": "Alpha", "awayTeam": "Beta", "startTimeTBD": False},
        {"id": 101 if reverse_id else 102, "startDate": "2024-09-02T16:00:00Z", "homeTeam": "Alpha", "awayTeam": "Beta", "startTimeTBD": False},
    ]
    rows = []
    for g in games:
        for team, opp, side in [("Alpha", "Beta", "home"), ("Beta", "Alpha", "away")]:
            rows.append({"game_id": g["id"], "season": 2024, "season_type": "regular", "week": 1, "team": team, "opponent": opp, "home_away": side})
    return pd.DataFrame(rows), schedules_to_frame({2024: games})


def test_inversion_in_week1_sorts_by_time_not_game_id():
    f, s = fixture(reverse_id=True)
    _, summary, reports = audit_chronology(f, s)
    assert summary["matched_games"] == 2
    assert summary["adjacent_team_order_inversions"] == 2
    assert summary["team_game_rows_out_of_chronological_rank"] == 4
    assert len(reports["schedule_team_mismatches"]) == 0


def test_order_is_correct_when_ids_agree_with_dates():
    f, s = fixture(reverse_id=False)
    _, summary, _ = audit_chronology(f, s)
    assert summary["adjacent_team_order_inversions"] == 0
    assert summary["team_game_rows_out_of_chronological_rank"] == 0


def test_missing_schedule_reports_missing_not_zero_confidence():
    f, s = fixture()
    s = s.iloc[:1].copy()
    _, summary, reports = audit_chronology(f, s)
    assert summary["missing_games"] == 1
    assert len(reports["unmatched_team_games"]) == 2


def test_reject_duplicate_team_games():
    f, s = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        audit_chronology(pd.concat([f, f.iloc[:1]], ignore_index=True), s)


def test_reads_original_local_cache_without_key(tmp_path: Path):
    path = tmp_path / "data/raw/cfbd/games/games_2024.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"id": 100, "startDate": "2024-08-01T00:00:00Z"}]))
    games, source = load_games(tmp_path, 2024, None, offline=True)
    assert games[0]["id"] == 100
    assert "games_2024.json" in source
