import pandas as pd
from audit_spread_total_chronology import chronological_inversions, classify_missingness


def test_postseason_week_one_id_can_reverse_real_game_order():
    d = pd.DataFrame([
        {"game_id": 100, "season": 2025, "week": 1, "season_type": "regular", "team": "Alabama", "opponent": "FSU", "kickoff_utc": "2025-08-30T20:00Z"},
        {"game_id": 300, "season": 2025, "week": 1, "season_type": "postseason", "team": "Alabama", "opponent": "Oklahoma", "kickoff_utc": "2025-12-19T20:00Z"},
        {"game_id": 200, "season": 2025, "week": 1, "season_type": "postseason", "team": "Alabama", "opponent": "Indiana", "kickoff_utc": "2026-01-01T20:00Z"},
    ])
    d["kickoff_utc"] = pd.to_datetime(d.kickoff_utc, utc=True)
    inversions = chronological_inversions(d)
    assert len(inversions) == 1
    assert inversions.iloc[0].game_id == 300


def test_postseason_not_counted_as_early_regular_missingness():
    d = pd.DataFrame([
        {"season_type": "regular", "week": 1, "f": None},
        {"season_type": "regular", "week": 5, "f": 2},
        {"season_type": "postseason", "week": 1, "f": 3},
    ])
    summary = classify_missingness(d, ["f"]).set_index("phase")
    assert summary.loc["regular_weeks_1_to_4", "mean_feature_missing_pct"] == 100
    assert summary.loc["postseason", "mean_feature_missing_pct"] == 0


def test_no_false_inversion_when_sorted_times_agree():
    d = pd.DataFrame([
        {"game_id": 100, "season": 2025, "week": 1, "season_type": "regular", "team": "Alpha", "opponent": "Beta", "kickoff_utc": "2025-08-30T20:00Z"},
        {"game_id": 101, "season": 2025, "week": 2, "season_type": "regular", "team": "Alpha", "opponent": "Gamma", "kickoff_utc": "2025-09-06T20:00Z"},
    ])
    d["kickoff_utc"] = pd.to_datetime(d.kickoff_utc, utc=True)
    assert chronological_inversions(d).empty


def test_end_to_end_audit_detects_regular_week_one_ordering(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from audit_spread_total_chronology import audit
    features = []
    history = []
    for gid, date, opp in [(401635525, "2024-08-24T16:00:00Z", "Georgia Tech"),
                           (401634306, "2024-09-02T23:30:00Z", "Boston College")]:
        for team, opponent in (("Florida State", opp), (opp, "Florida State")):
            features.append({"game_id": gid, "season": 2024, "week": 1,
                             "season_type": "regular", "team": team,
                             "opponent": opponent, "stat": 3.0})
            history.append({"game_id": gid, "season": 2024, "week": 1,
                            "season_type": "regular", "team": team,
                            "opponent": opponent, "start_date": date})
    fake_features = pd.DataFrame(features)
    fake_history = pd.DataFrame(history)
    featpath = tmp_path / "data/processed/cfb_matchup_features_v0_1.parquet"
    rawpath = tmp_path / "data/processed/team_games_2024.parquet"
    featpath.parent.mkdir(parents=True)
    featpath.touch()
    rawpath.touch()
    manpath = tmp_path / "reports/cfb_matchup_feature_manifest_v0_1.json"
    manpath.parent.mkdir(parents=True)
    manpath.write_text(json.dumps({"feature_columns": ["stat"]}))
    def _fake_read_parquet(p):
        return fake_features.copy() if str(p).endswith("cfb_matchup_features_v0_1.parquet") else fake_history.copy()
    monkeypatch.setattr(pd, "read_parquet", _fake_read_parquet)
    summary = audit(tmp_path, tmp_path / "reports/audit", "data/processed/cfb_matchup_features_v0_1.parquet",
                    "reports/cfb_matchup_feature_manifest_v0_1.json", "data/processed/team_games_*.parquet")
    assert summary["chronological_inversions_in_model_order"] == 1
    assert summary["games_with_parsed_kickoff"] == 2
