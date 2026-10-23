import math

import pytest

from cfb_quant.spread_total.elo_engine_v58 import (
    EloConfig,
    expected_home_win,
    update_ratings,
)


def test_equal_teams_are_even_without_home_advantage():
    assert expected_home_win(1500, 1500) == 0.5


def test_home_advantage_increases_home_expectation():
    neutral = expected_home_win(1500, 1500)
    advantaged = expected_home_win(
        1500, 1500, EloConfig(home_advantage=65)
    )
    assert advantaged > neutral


def test_expected_probabilities_are_symmetric():
    a = expected_home_win(1700, 1450)
    b = expected_home_win(1450, 1700)
    assert a + b == pytest.approx(1.0)


def test_home_win_increases_home_rating():
    result = update_ratings(1500, 1500, 31, 20)
    assert result.home_after > 1500
    assert result.away_after < 1500


def test_away_win_increases_away_rating():
    result = update_ratings(1500, 1500, 17, 24)
    assert result.home_after < 1500
    assert result.away_after > 1500


def test_rating_update_is_zero_sum():
    result = update_ratings(1700, 1400, 21, 28)
    assert result.home_after + result.away_after == pytest.approx(3100)


def test_draw_changes_nothing_for_equal_neutral_teams():
    result = update_ratings(1500, 1500, 20, 20)
    assert result.home_after == pytest.approx(1500)
    assert result.away_after == pytest.approx(1500)


def test_rating_changes_are_deterministic():
    a = update_ratings(1610, 1490, 38, 17)
    b = update_ratings(1610, 1490, 38, 17)
    assert a == b


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        EloConfig(k_factor=-5)
    with pytest.raises(ValueError):
        EloConfig(scale=0)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        expected_home_win(float("nan"), 1500)
    with pytest.raises(ValueError):
        update_ratings(1500, 1500, -3, 10)


def test_large_rating_differences_are_stable():
    high = expected_home_win(1_000_000, 0)
    low = expected_home_win(0, 1_000_000)
    assert math.isfinite(high)
    assert math.isfinite(low)
    assert high == 1.0
    assert low == 0.0
