import json
from pathlib import Path

import pandas as pd

from cfb_quant.models.tournament import (
    HOLDOUT_YEAR,
    TARGETS,
    VALIDATION_YEARS,
    _target_mask,
)


def test_holdout_year_is_not_a_validation_year():
    assert HOLDOUT_YEAR == 2025
    assert HOLDOUT_YEAR not in VALIDATION_YEARS
    assert max(VALIDATION_YEARS) == 2024


def test_manifest_does_not_contain_current_game_targets():
    path = Path("reports/cfb_feature_manifest_v0_1.json")
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    features = set(manifest["feature_columns"])
    assert not features.intersection(TARGETS)


def test_eligibility_uses_prior_history_not_current_role():
    frame = pd.DataFrame(
        {
            "player_games_before": [2, 2],
            "p_pass_attempts_avg3": [20.0, 0.0],
            "p_pass_attempts_career_avg": [15.0, 0.0],
            "p_dropbacks_avg3": [22.0, 0.0],
            "p_dropbacks_career_avg": [16.0, 0.0],
            "pass_attempts": [0.0, 50.0],
            "official_passer": [False, True],
        }
    )
    mask = _target_mask(frame, "pass_attempts")
    assert mask.tolist() == [True, False]