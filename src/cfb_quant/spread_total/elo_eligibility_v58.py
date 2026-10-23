"""Research-only FBS-FBS Elo eligibility policy.

Exclusion is based on source classification, never on outcomes,
predictions, or whether a game is convenient to replay.
"""

from .elo_live_validation_v58 import adapt_live_capture


def adapt_fbs_capture(raw, manifest):
    data = adapt_live_capture(raw, manifest)

    # Revalidate the original bytes and preserve the source record order.
    from .elo_capture_integrity_v58 import validate_capture
    records, _, _ = validate_capture(raw, manifest)

    eligible_ids = {
        str(r["id"])
        for r in records
        if r.get("homeClassification") == "fbs"
        and r.get("awayClassification") == "fbs"
    }

    games = tuple(
        game for game in data["games"]
        if game.game_id in eligible_ids
    )

    excluded_games = len(data["games"]) - len(games)
    accepted = sum(g.home_points is not None for g in games)

    return {
        **data,
        "games": games,
        "eligible_games": len(games),
        "eligible_completed_results": accepted,
        "ineligible_games": excluded_games,
        "eligibility_policy": "both_teams_classified_fbs",
    }
