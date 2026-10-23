"""Research-only prospective Elo grading.

Scores must come from hash-verified captures observed after kickoff.
Forecasts remain immutable; missing results remain pending.
"""

import hashlib
import json
import math
from datetime import datetime, timezone

from .elo_capture_integrity_v58 import validate_capture
from .elo_live_validation_v58 import usable_score


def utc(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)


def grade_lock(forecast_bytes, lock_manifest_bytes, captures):
    payload = json.loads(forecast_bytes)
    manifest = json.loads(lock_manifest_bytes)

    digest = hashlib.sha256(forecast_bytes).hexdigest()
    if digest != manifest["forecast_payload_sha256"]:
        raise ValueError("Forecast SHA-256 mismatch")

    lock_time = utc(payload["locked_at_utc"])
    rows = payload["forecasts"]

    if (
        payload["forecast_count"] != len(rows)
        or manifest["forecast_count"] != len(rows)
    ):
        raise ValueError("Forecast count mismatch")

    if manifest["lock_created_at_utc"] != payload["locked_at_utc"]:
        raise ValueError("Lock timestamp mismatch")

    if payload["production_approved"] is not False:
        raise ValueError("Only experimental forecasts can be graded")

    by_id = {}

    for row in rows:
        gid = str(row["game_id"])
        kickoff = utc(row["kickoff_utc"])

        if gid in by_id:
            raise ValueError("Duplicate game ID")
        if kickoff <= lock_time:
            raise ValueError("Forecast was not locked before kickoff")

        p = row["home_win_probability"]
        q = row["away_win_probability"]

        if not (
            isinstance(p, (int, float))
            and not isinstance(p, bool)
            and math.isfinite(p)
            and 0 <= p <= 1
            and isinstance(q, (int, float))
            and not isinstance(q, bool)
            and math.isfinite(q)
            and 0 <= q <= 1
            and math.isclose(p + q, 1.0, abs_tol=1e-12)
        ):
            raise ValueError("Invalid forecast probability")

        by_id[gid] = row

    observations = {gid: [] for gid in by_id}
    seen_times = set()
    count = 0

    for raw, capture_manifest in captures:
        records, observed, source_hash = validate_capture(
            raw, capture_manifest
        )

        if observed in seen_times:
            raise ValueError("Duplicate capture timestamps")
        seen_times.add(observed)
        count += 1

        if observed <= lock_time:
            continue

        for record in records:
            gid = str(record["id"])

            if gid not in by_id:
                continue

            row = by_id[gid]
            kickoff = utc(row["kickoff_utc"])

            if observed <= kickoff:
                continue

            if record.get("completed") is not True:
                continue

            if (
                record.get("homeTeam") != row["home"]
                or record.get("awayTeam") != row["away"]
                or utc(record["startDate"]) != kickoff
            ):
                observations[gid].append(
                    ("identity_conflict", observed, source_hash)
                )
                continue

            hp = record.get("homePoints")
            ap = record.get("awayPoints")

            if not usable_score(hp) or not usable_score(ap):
                continue

            observations[gid].append(
                ((int(hp), int(ap)), observed, source_hash)
            )

    graded = []
    pending = []
    quarantined = []

    for gid, row in by_id.items():
        found = sorted(
            observations[gid],
            key=lambda x: (x[1], x[2]),
        )

        if any(x[0] == "identity_conflict" for x in found):
            quarantined.append({
                "game_id": gid,
                "reason": "source_identity_conflict",
            })
            continue

        scores = {x[0] for x in found}

        if len(scores) > 1:
            quarantined.append({
                "game_id": gid,
                "reason": "conflicting_final_scores",
            })
            continue

        if not found:
            pending.append(gid)
            continue

        (hp, ap), observed, source_hash = found[0]

        if hp == ap:
            quarantined.append({
                "game_id": gid,
                "reason": "tied_final_score",
            })
            continue

        outcome = int(hp > ap)
        p = row["home_win_probability"]
        clipped = min(max(p, 1e-15), 1 - 1e-15)

        graded.append({
            "game_id": gid,
            "home": row["home"],
            "away": row["away"],
            "home_points": hp,
            "away_points": ap,
            "home_win_probability": p,
            "home_won": outcome,
            "brier": (p - outcome) ** 2,
            "log_loss": -(
                outcome * math.log(clipped)
                + (1 - outcome) * math.log(1 - clipped)
            ),
            "winner_correct": (
                (p > 0.5 and outcome == 1)
                or (p < 0.5 and outcome == 0)
            ),
            "probability_tied": p == 0.5,
            "first_result_observed_utc": observed.isoformat(),
            "result_source_sha256": source_hash,
        })

    graded.sort(key=lambda x: x["game_id"])
    pending.sort()
    quarantined.sort(key=lambda x: x["game_id"])

    n = len(graded)
    decisive = [g for g in graded if not g["probability_tied"]]

    assert n + len(pending) + len(quarantined) == len(rows)

    return {
        "capture_count": count,
        "forecast_count": len(rows),
        "graded_count": n,
        "pending_count": len(pending),
        "quarantined_count": len(quarantined),
        "mean_brier": (
            sum(g["brier"] for g in graded) / n if n else None
        ),
        "mean_log_loss": (
            sum(g["log_loss"] for g in graded) / n if n else None
        ),
        "decisive_accuracy": (
            sum(g["winner_correct"] for g in decisive) / len(decisive)
            if decisive else None
        ),
        "decisive_count": len(decisive),
        "graded": graded,
        "pending": pending,
        "quarantined": quarantined,
    }
