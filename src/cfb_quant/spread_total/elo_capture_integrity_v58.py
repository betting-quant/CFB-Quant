"""Validate archived CFBD capture payloads and manifests.

SHA-256 detects disagreement between saved bytes and metadata.
It does not authenticate the capture time or original publisher.
"""

import hashlib
import json
import re
from datetime import datetime, timezone

from .elo_source_adapter_v58 import adapt_cfbd_games

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _utc(value):
    if not isinstance(value, str):
        raise ValueError("retrieved_at_utc must be a string")

    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid retrieved_at_utc") from exc

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("retrieved_at_utc must have a timezone")

    return dt.astimezone(timezone.utc)


def validate_capture(payload_bytes, manifest):
    """Return validated payload records, retrieval time, and digest.

    The raw JSON must be a list of records with unique game IDs.
    Do not silently normalize or rewrite payload bytes.
    """
    if not isinstance(payload_bytes, bytes):
        raise TypeError("Payload must be raw bytes")
    if not isinstance(manifest, dict):
        raise TypeError("Manifest must be a dictionary")

    if manifest.get("source_endpoint") != (
        "https://api.collegefootballdata.com/games"
    ):
        raise ValueError("Unexpected source endpoint")

    digest = hashlib.sha256(payload_bytes).hexdigest()
    expected = manifest.get("payload_sha256")

    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise ValueError("Invalid payload_sha256")

    if digest != expected:
        raise ValueError("Payload SHA-256 mismatch")

    retrieved = _utc(manifest.get("retrieved_at_utc"))

    try:
        records = json.loads(payload_bytes.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSON payload") from exc

    if not isinstance(records, list):
        raise ValueError("Payload must contain a list of games")

    seen = set()

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Game record must be an object")

        gid = record.get("id")

        if gid is None or isinstance(gid, bool) or not str(gid).strip():
            raise ValueError("Missing game ID")

        gid = str(gid)

        if gid in seen:
            raise ValueError("Duplicate game ID")

        seen.add(gid)

    return tuple(records), retrieved, digest


def verified_schedule_and_results(
    schedule_bytes,
    schedule_manifest,
    results_bytes=None,
    results_manifest=None,
):
    """Adapt only source-captured scores from verified archives.

    The schedule may be retrospective; its embedded scores are ignored.
    The result capture's retrieval time is used as the conservative
    observation time, never an inferred historical final-whistle time.
    """
    schedules, _, _ = validate_capture(
        schedule_bytes, schedule_manifest
    )

    if (results_bytes is None) != (results_manifest is None):
        raise ValueError("Result payload and manifest must be paired")

    if results_bytes is None:
        return adapt_cfbd_games(schedules)

    results, retrieved, _ = validate_capture(
        results_bytes, results_manifest
    )

    schedule_ids = {str(x["id"]) for x in schedules}
    captures = []

    for record in results:
        gid = str(record["id"])

        if gid not in schedule_ids:
            # A season-wide response can contain unrelated games.
            continue

        if record.get("completed") is not True:
            continue

        captures.append({
            "game_id": gid,
            "captured_at_utc": retrieved.isoformat(),
            "completed": True,
            "home_points": record.get("homePoints"),
            "away_points": record.get("awayPoints"),
        })

    return adapt_cfbd_games(schedules, captures)
