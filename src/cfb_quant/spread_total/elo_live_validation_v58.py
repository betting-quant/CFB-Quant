"""Validated live Elo capture ingestion.

Invalid completed scores are quarantined, not fabricated.
Original raw source bytes and manifest remain unchanged.
"""

import math

from .elo_capture_integrity_v58 import validate_capture
from .elo_source_adapter_v58 import adapt_cfbd_games


def usable_score(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        and int(value) == value
    )


def adapt_live_capture(raw, manifest):
    records, observed, digest = validate_capture(raw, manifest)

    captures = []
    exclusions = []
    completed = 0

    for record in records:
        if record.get("completed") is not True:
            continue

        completed += 1

        home = record.get("homePoints")
        away = record.get("awayPoints")

        if not usable_score(home) or not usable_score(away):
            exclusions.append({
                "game_id": str(record["id"]),
                "home": record.get("homeTeam"),
                "away": record.get("awayTeam"),
                "reason": "completed_without_valid_final_score",
            })
            continue

        captures.append({
            "game_id": str(record["id"]),
            "captured_at_utc": observed.isoformat(),
            "completed": True,
            "home_points": int(home),
            "away_points": int(away),
        })

    games = adapt_cfbd_games(records, captures)

    assert len(games) == len(records)
    assert len(captures) + len(exclusions) == completed

    return {
        "games": games,
        "records": len(records),
        "completed": completed,
        "accepted": len(captures),
        "excluded": tuple(exclusions),
        "observed_at_utc": observed,
        "source_sha256": digest,
    }
