"""Strict CFBD-to-Elo replay adapter.

Retrospective schedule scores are never considered sufficient evidence
of historical score availability. An explicitly timestamped completed
score capture is required before any rating update can be scheduled.

A capture timestamp proves observation no earlier than that capture;
it does not certify the actual game completion time.
"""

import math
from datetime import datetime, timezone

from .elo_replay_v58 import ReplayGame


def _timestamp(value, label):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(f"Invalid {label}") from exc
    else:
        raise ValueError(f"Missing or invalid {label}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")

    return dt.astimezone(timezone.utc)


def _score(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    if int(value) != value:
        raise ValueError(f"{label} must be an integer score")
    return int(value)


def adapt_cfbd_games(schedule_records, result_captures=()):
    """Construct replay games from source-shaped records.

    schedule_records: iterable of CFBD /games records, including id,
        startDate, homeTeam and awayTeam.

    result_captures: independently saved observations with game_id,
        captured_at_utc, completed=True, home_points, away_points.

    A capture is accepted as evidence that the result was known by
    captured_at_utc, never that it was available at kickoff or earlier.
    """
    schedule = {}

    for record in schedule_records:
        if not isinstance(record, dict):
            raise TypeError("Schedule records must be dictionaries")

        gid = record.get("id")
        if gid is None or isinstance(gid, bool):
            raise ValueError("Missing game ID")
        gid = str(gid)

        if not gid.strip() or gid in schedule:
            raise ValueError("Duplicate or empty schedule game ID")

        home = record.get("homeTeam")
        away = record.get("awayTeam")

        if not isinstance(home, str) or not home.strip():
            raise ValueError("Missing homeTeam")
        if not isinstance(away, str) or not away.strip():
            raise ValueError("Missing awayTeam")
        if home == away:
            raise ValueError("Home and away teams cannot match")

        kickoff = _timestamp(record.get("startDate"), "startDate")
        schedule[gid] = (kickoff, home, away)

    captures = {}

    for capture in result_captures:
        if not isinstance(capture, dict):
            raise TypeError("Result captures must be dictionaries")

        gid = capture.get("game_id")
        if gid is None or isinstance(gid, bool):
            raise ValueError("Missing result game_id")
        gid = str(gid)

        if gid not in schedule:
            raise ValueError("Result has no corresponding schedule game")

        if gid in captures:
            raise ValueError("Duplicate result capture game ID")

        if capture.get("completed") is not True:
            raise ValueError("Result capture must explicitly be completed")

        observed_at = _timestamp(
            capture.get("captured_at_utc"), "captured_at_utc"
        )
        home_points = _score(
            capture.get("home_points"), "home_points"
        )
        away_points = _score(
            capture.get("away_points"), "away_points"
        )

        kickoff = schedule[gid][0]
        if observed_at <= kickoff:
            raise ValueError(
                "Completed result capture must occur after kickoff"
            )

        captures[gid] = (observed_at, home_points, away_points)

    output = []

    for gid, (kickoff, home, away) in sorted(
        schedule.items(),
        key=lambda item: (item[1][0], item[0]),
    ):
        result = captures.get(gid)

        output.append(
            ReplayGame(
                game_id=gid,
                kickoff_utc=kickoff,
                home=home,
                away=away,
                home_points=result[1] if result else None,
                away_points=result[2] if result else None,
                result_available_utc=result[0] if result else None,
            )
        )

    return tuple(output)
