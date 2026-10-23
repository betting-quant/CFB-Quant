"""Experimental deterministic Elo arithmetic for CFB v5.8.

No I/O, data acquisition, model promotion, or retroactive rating claims.
This implementation is independent of CFBD's Elo formula.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EloConfig:
    k_factor: float = 24.0
    home_advantage: float = 0.0
    scale: float = 400.0

    def __post_init__(self):
        for name in ("k_factor", "home_advantage", "scale"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.k_factor <= 0 or self.scale <= 0:
            raise ValueError("k_factor and scale must be positive")


@dataclass(frozen=True)
class EloUpdate:
    home_before: float
    away_before: float
    expected_home: float
    actual_home: float
    home_after: float
    away_after: float


def expected_home_win(
    home_rating: float,
    away_rating: float,
    config: EloConfig = EloConfig(),
) -> float:
    if not math.isfinite(home_rating) or not math.isfinite(away_rating):
        raise ValueError("Ratings must be finite")

    difference = (
        home_rating + config.home_advantage - away_rating
    ) / config.scale

    # Numerically stable logistic implementation.
    z = difference * math.log(10.0)

    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))

    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_points: float,
    away_points: float,
    config: EloConfig = EloConfig(),
) -> EloUpdate:
    if not all(math.isfinite(v) for v in (
        home_rating, away_rating, home_points, away_points
    )):
        raise ValueError("Inputs must be finite")

    if home_points < 0 or away_points < 0:
        raise ValueError("Scores cannot be negative")

    expected = expected_home_win(home_rating, away_rating, config)

    actual = (
        1.0 if home_points > away_points
        else 0.0 if home_points < away_points
        else 0.5
    )

    change = config.k_factor * (actual - expected)

    return EloUpdate(
        home_before=home_rating,
        away_before=away_rating,
        expected_home=expected,
        actual_home=actual,
        home_after=home_rating + change,
        away_after=away_rating - change,
    )
