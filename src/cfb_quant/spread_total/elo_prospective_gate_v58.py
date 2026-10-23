"""Prospective eligibility gate for experimental Elo research.

Only snapshots after the latest archived observation can be
considered for a new forecast lock. This does not certify
historical feature availability or forecasting skill.
"""

from datetime import datetime, timezone

from .elo_replay_v58 import replay_games


def prospective_elo_candidates(merged):
    if not isinstance(merged, dict):
        raise TypeError("Merged capture must be a dictionary")

    observed = merged["latest_capture_utc"]

    if (
        not isinstance(observed, datetime)
        or observed.tzinfo is None
        or observed.utcoffset() is None
    ):
        raise ValueError("Latest capture time must be timezone-aware")

    observed = observed.astimezone(timezone.utc)
    games = tuple(merged["games"])

    if not games:
        raise ValueError("No eligible FBS games")

    if len({g.game_id for g in games}) != len(games):
        raise ValueError("Duplicate eligible game IDs")

    replay = replay_games(games)
    by_id = {g.game_id: g for g in games}

    candidates = []

    for snap in replay.snapshots:
        source = by_id[snap.game_id]

        if snap.kickoff_utc <= observed:
            continue

        if source.result_available_utc is not None:
            raise ValueError(
                "Future game already contains a completed result: "
                + snap.game_id
            )

        if source.home_points is not None or source.away_points is not None:
            raise ValueError(
                "Future game contains final scores: " + snap.game_id
            )

        if not (0 <= snap.expected_home_win <= 1):
            raise ValueError("Invalid Elo probability")

        candidates.append(snap)

    if len(replay.snapshots) != len(games):
        raise ValueError("Incomplete replay coverage")

    return {
        "capture_cutoff_utc": observed,
        "candidates": tuple(candidates),
        "candidate_count": len(candidates),
        "eligible_games": len(games),
        "updates_applied": replay.updates_applied,
        "historical_pit_certified": False,
        "production_approved": False,
    }
