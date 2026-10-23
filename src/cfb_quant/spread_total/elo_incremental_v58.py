"""Incremental, first-observed CFBD Elo research ingestion.

Observations are ordered by their archived retrieval timestamps.
The earliest valid observed score is frozen; later revisions are
reported, not silently substituted. No historical backdating.
"""

from .elo_capture_integrity_v58 import validate_capture
from .elo_live_validation_v58 import usable_score
from .elo_source_adapter_v58 import adapt_cfbd_games


def merge_fbs_captures(captures):
    """Merge iterable of (raw_bytes, manifest) capture pairs.

    Returns source-supported FBS-FBS replay games plus audit details.
    All captures must belong to one coherent schedule universe.
    """
    verified = []

    for raw, manifest in captures:
        records, observed, digest = validate_capture(raw, manifest)
        verified.append((observed, digest, records))

    if not verified:
        raise ValueError("At least one verified capture is required")

    verified.sort(key=lambda item: (item[0], item[1]))

    if len({item[0] for item in verified}) != len(verified):
        raise ValueError("Ambiguous duplicate capture timestamps")

    identities = {}
    first_results = {}
    revisions = []
    bad_scores = []
    latest_records = None

    for observed, digest, records in verified:
        current_ids = set()

        for record in records:
            gid = str(record["id"])
            current_ids.add(gid)

            identity = (
                record.get("homeTeam"),
                record.get("awayTeam"),
                record.get("startDate"),
            )

            if gid in identities and identities[gid] != identity:
                raise ValueError(
                    "Conflicting team/kickoff identity for game " + gid
                )

            identities[gid] = identity

            if (
                record.get("homeClassification") != "fbs"
                or record.get("awayClassification") != "fbs"
            ):
                continue

            if record.get("completed") is not True:
                continue

            home = record.get("homePoints")
            away = record.get("awayPoints")

            if not usable_score(home) or not usable_score(away):
                bad_scores.append((gid, observed.isoformat()))
                continue

            score = (int(home), int(away))

            if gid not in first_results:
                first_results[gid] = (
                    observed, score, digest
                )
            elif score != first_results[gid][1]:
                revisions.append({
                    "game_id": gid,
                    "first_score": first_results[gid][1],
                    "later_score": score,
                    "later_observed_utc": observed.isoformat(),
                    "source_sha256": digest,
                })

        latest_records = records

    latest_ids = {
        str(r["id"]) for r in latest_records
    }

    for gid in first_results:
        if gid not in latest_ids:
            raise ValueError(
                "Previously observed result missing from latest schedule: "
                + gid
            )

    eligible_records = [
        r for r in latest_records
        if r.get("homeClassification") == "fbs"
        and r.get("awayClassification") == "fbs"
    ]

    eligible_ids = {str(r["id"]) for r in eligible_records}
    result_captures = []

    for gid, (observed, score, digest) in first_results.items():
        if gid not in eligible_ids:
            raise ValueError(
                "Previously eligible game changed classification: " + gid
            )

        result_captures.append({
            "game_id": gid,
            "captured_at_utc": observed.isoformat(),
            "completed": True,
            "home_points": score[0],
            "away_points": score[1],
        })

    games = adapt_cfbd_games(eligible_records, result_captures)

    return {
        "games": games,
        "capture_count": len(verified),
        "eligible_games": len(games),
        "accepted_results": len(result_captures),
        "first_results": {
            gid: {
                "observed_at_utc": observed.isoformat(),
                "home_points": score[0],
                "away_points": score[1],
                "source_sha256": digest,
            }
            for gid, (observed, score, digest)
            in sorted(first_results.items())
        },
        "revisions": tuple(revisions),
        "invalid_score_observations": tuple(bad_scores),
        "latest_capture_utc": verified[-1][0],
    }
