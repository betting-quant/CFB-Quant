"""Deterministic Elo replay with explicit result availability.

Experimental only. Does not modify production models or forecasts.
"""

import heapq
import math
from dataclasses import dataclass
from datetime import datetime, timezone

from .elo_engine_v58 import EloConfig, expected_home_win, update_ratings


@dataclass(frozen=True)
class ReplayGame:
    game_id: str
    kickoff_utc: datetime
    home: str
    away: str
    home_points: float | None = None
    away_points: float | None = None
    result_available_utc: datetime | None = None


@dataclass(frozen=True)
class PregameSnapshot:
    game_id: str
    kickoff_utc: datetime
    home: str
    away: str
    home_elo: float
    away_elo: float
    expected_home_win: float


@dataclass(frozen=True)
class ReplayResult:
    snapshots: tuple[PregameSnapshot, ...]
    final_ratings: dict[str, float]
    updates_applied: int


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Timestamps must be timezone-aware datetimes")
    if value.utcoffset() is None:
        raise ValueError("Invalid timezone offset")
    return value.astimezone(timezone.utc)


def _validate(game):
    if not isinstance(game.game_id, str) or not game.game_id.strip():
        raise ValueError("Game ID must be a nonempty string")
    if not isinstance(game.home, str) or not game.home.strip():
        raise ValueError("Home team must be nonempty")
    if not isinstance(game.away, str) or not game.away.strip():
        raise ValueError("Away team must be nonempty")
    if game.home == game.away:
        raise ValueError("A team cannot play itself")

    kickoff = _utc(game.kickoff_utc)
    scores = (game.home_points, game.away_points)

    if game.result_available_utc is None:
        if any(x is not None for x in scores):
            raise ValueError("Scores require a result availability timestamp")
        return kickoff, None

    if any(x is None for x in scores):
        raise ValueError("Result timestamp requires both final scores")

    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Scores must be numeric")
        if not math.isfinite(score) or score < 0:
            raise ValueError("Scores must be finite and nonnegative")

    result_time = _utc(game.result_available_utc)
    if result_time <= kickoff:
        raise ValueError("Result must become available after kickoff")

    return kickoff, result_time


def replay_games(
    games,
    config: EloConfig = EloConfig(),
    initial_rating: float = 1500.0,
) -> ReplayResult:
    if not math.isfinite(initial_rating):
        raise ValueError("Initial rating must be finite")

    prepared = []
    seen_ids = set()

    for game in games:
        if not isinstance(game, ReplayGame):
            raise TypeError("All games must be ReplayGame instances")

        if game.game_id in seen_ids:
            raise ValueError("Duplicate game ID: " + game.game_id)

        seen_ids.add(game.game_id)
        kickoff, result_time = _validate(game)
        prepared.append((kickoff, game.game_id, result_time, game))

    prepared.sort(key=lambda x: (x[0], x[1]))

    ratings = {}
    snapshots = []
    pending = []
    updates = 0

    def apply_due_updates(before_time):
        nonlocal updates

        while pending and pending[0][0] < before_time:
            event_time = pending[0][0]
            batch = []

            while pending and pending[0][0] == event_time:
                _, _, game = heapq.heappop(pending)
                batch.append(game)

            # Every result in this batch uses the same pre-update
            # rating state. No result at this timestamp can affect
            # another result at this timestamp.
            frozen = dict(ratings)
            deltas = {}

            for game in batch:
                home_before = frozen[game.home]
                away_before = frozen[game.away]

                updated = update_ratings(
                    home_before,
                    away_before,
                    game.home_points,
                    game.away_points,
                    config,
                )

                home_delta = updated.home_after - home_before
                away_delta = updated.away_after - away_before

                deltas[game.home] = (
                    deltas.get(game.home, 0.0) + home_delta
                )
                deltas[game.away] = (
                    deltas.get(game.away, 0.0) + away_delta
                )

            for team in sorted(deltas):
                ratings[team] = frozen[team] + deltas[team]

            updates += len(batch)

    index = 0
    while index < len(prepared):
        kickoff = prepared[index][0]
        apply_due_updates(kickoff)

        group = []
        while index < len(prepared) and prepared[index][0] == kickoff:
            group.append(prepared[index])
            index += 1

        participating = set()
        for _, _, _, game in group:
            if game.home in participating or game.away in participating:
                raise ValueError("Team scheduled in simultaneous games")
            participating.update((game.home, game.away))

        # Capture all pregame ratings before scheduling any results.
        for _, _, _, game in group:
            home_rating = ratings.setdefault(game.home, initial_rating)
            away_rating = ratings.setdefault(game.away, initial_rating)

            snapshots.append(PregameSnapshot(
                game_id=game.game_id,
                kickoff_utc=kickoff,
                home=game.home,
                away=game.away,
                home_elo=home_rating,
                away_elo=away_rating,
                expected_home_win=expected_home_win(
                    home_rating, away_rating, config
                ),
            ))

        for _, _, result_time, game in group:
            if result_time is not None:
                heapq.heappush(
                    pending, (result_time, game.game_id, game)
                )

    # Apply remaining completed games after all pregame snapshots.
    apply_due_updates(datetime.max.replace(tzinfo=timezone.utc))

    return ReplayResult(
        snapshots=tuple(snapshots),
        final_ratings=dict(sorted(ratings.items())),
        updates_applied=updates,
    )
