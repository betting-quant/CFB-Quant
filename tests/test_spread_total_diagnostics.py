import numpy as np
import pandas as pd
import pytest

from cfb_quant.spread_total.diagnostics import enrich, load_validation, save_reports


def sample():
    return pd.DataFrame([
        dict(game_id=1, season=2023, week=1, actual_margin=7, actual_total=43,
             original_margin=4.0, original_total=41.0, fair_home_margin_mean=5.0,
             fair_total_mean=42.0, margin_p10=-10.0, margin_p90=20.0,
             total_p10=20.0, total_p90=64.0, margin_80_covered=1.0,
             total_80_covered=1.0, prior_max_season=2022, p_home_win=0.7),
        dict(game_id=2, season=2024, week=7, actual_margin=-8, actual_total=66,
             original_margin=-2.0, original_total=49.0, fair_home_margin_mean=-3.0,
             fair_total_mean=50.0, margin_p10=-18.0, margin_p90=15.0,
             total_p10=25.0, total_p90=65.0, margin_80_covered=1.0,
             total_80_covered=0.0, prior_max_season=2023, p_home_win=0.3),
    ])


def test_runs_and_exports(tmp_path):
    data = sample()
    csv = tmp_path / 'validation.csv'
    data.to_csv(csv, index=False)
    paths = save_reports(load_validation(csv), tmp_path / 'outputs')
    assert len(paths) == 6
    assert all(p.is_file() for p in paths.values())
    df = enrich(data)
    assert np.isclose(df.total_abs_error.iloc[1], 16.0)
    assert df.total_above_p90.iloc[1] == 1


def test_rejects_duplicate_and_future_residual(tmp_path):
    bad = sample()
    bad.loc[1, 'game_id'] = bad.loc[0, 'game_id']
    path = tmp_path / 'bad.csv'
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        load_validation(path)
    bad = sample()
    bad.loc[0, 'prior_max_season'] = 2023
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Chronological'):
        load_validation(path)


def test_catches_bad_coverage():
    data = sample()
    data.loc[0, 'total_80_covered'] = 0
    with pytest.raises(ValueError, match='disagrees'):
        enrich(data)
