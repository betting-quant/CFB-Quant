import hashlib
import json

import pytest

from cfb_quant.spread_total.elo_capture_integrity_v58 import (
    validate_capture,
    verified_schedule_and_results,
)
from cfb_quant.spread_total.elo_replay_v58 import replay_games


ENDPOINT = "https://api.collegefootballdata.com/games"
T1 = "2025-09-01T18:00:00Z"
T2 = "2025-09-01T23:00:00Z"
T3 = "2025-09-02T18:00:00Z"


def pack(records, observed=T2):
    raw = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    manifest = {
        "source_endpoint": ENDPOINT,
        "retrieved_at_utc": observed,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }

    return raw, manifest


def game(gid, start=T1, home="A", away="B",
         completed=False, hp=None, ap=None):
    return {
        "id": gid,
        "startDate": start,
        "homeTeam": home,
        "awayTeam": away,
        "completed": completed,
        "homePoints": hp,
        "awayPoints": ap,
    }


def test_valid_capture_digest_and_count():
    raw, meta = pack([game(1)])
    rows, timestamp, digest = validate_capture(raw, meta)
    assert len(rows) == 1
    assert digest == meta["payload_sha256"]
    assert timestamp.isoformat() == "2025-09-01T23:00:00+00:00"


def test_mutated_payload_rejected():
    raw, meta = pack([game(1)])
    altered = raw.replace(b'"id":1', b'"id":2')
    with pytest.raises(ValueError, match="SHA-256"):
        validate_capture(altered, meta)


def test_forged_wrong_digest_rejected():
    raw, meta = pack([game(1)])
    meta["payload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        validate_capture(raw, meta)


def test_unexpected_endpoint_rejected():
    raw, meta = pack([game(1)])
    meta["source_endpoint"] = "https://example.com/games"
    with pytest.raises(ValueError, match="endpoint"):
        validate_capture(raw, meta)


def test_naive_retrieval_time_rejected():
    raw, meta = pack([game(1)])
    meta["retrieved_at_utc"] = "2025-09-01T23:00:00"
    with pytest.raises(ValueError, match="timezone"):
        validate_capture(raw, meta)


def test_duplicate_game_id_rejected():
    raw, meta = pack([game(1), game(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_capture(raw, meta)


def test_schedule_capture_does_not_import_results():
    raw, meta = pack([game(1, completed=True, hp=28, ap=14)])
    output = verified_schedule_and_results(raw, meta)
    assert output[0].home_points is None


def test_verified_result_capture_updates_later_game():
    schedule_raw, schedule_meta = pack([
        game(1),
        game(2, T3, "A", "C"),
    ])

    results_raw, results_meta = pack([
        game(1, completed=True, hp=28, ap=14),
    ])

    adapted = verified_schedule_and_results(
        schedule_raw,
        schedule_meta,
        results_raw,
        results_meta,
    )

    result = replay_games(adapted)
    assert result.snapshots[1].home_elo > 1500
    assert result.updates_applied == 1


def test_capture_after_next_kickoff_cannot_leak():
    schedule_raw, schedule_meta = pack([
        game(1),
        game(2, T3, "A", "C"),
    ])

    results_raw, results_meta = pack(
        [game(1, completed=True, hp=28, ap=14)],
        "2025-09-03T00:00:00Z",
    )

    result = replay_games(
        verified_schedule_and_results(
            schedule_raw,
            schedule_meta,
            results_raw,
            results_meta,
        )
    )

    assert result.snapshots[1].home_elo == 1500


def test_incomplete_result_not_used():
    raw, meta = pack([game(1)])
    adapted = verified_schedule_and_results(
        raw, meta, raw, meta
    )
    assert adapted[0].home_points is None


def test_malformed_completed_score_rejected():
    schedule_raw, schedule_meta = pack([game(1)])
    results_raw, results_meta = pack([
        game(1, completed=True, hp=-1, ap=14),
    ])

    with pytest.raises(ValueError, match="home_points"):
        verified_schedule_and_results(
            schedule_raw,
            schedule_meta,
            results_raw,
            results_meta,
        )


def test_unpaired_capture_arguments_rejected():
    raw, meta = pack([game(1)])
    with pytest.raises(ValueError, match="paired"):
        verified_schedule_and_results(raw, meta, raw, None)


def test_unrelated_result_is_not_imported():
    schedule_raw, schedule_meta = pack([game(1)])
    results_raw, results_meta = pack([
        game(99, completed=True, hp=28, ap=14),
    ])

    output = verified_schedule_and_results(
        schedule_raw,
        schedule_meta,
        results_raw,
        results_meta,
    )

    assert len(output) == 1
    assert output[0].home_points is None
