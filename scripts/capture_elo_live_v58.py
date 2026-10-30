"""One-shot prospective CFBD capture for experimental Elo v5.8.

No production changes, no historical timestamp backdating.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from cfb_quant.spread_total.elo_capture_integrity_v58 import (
    validate_capture,
)
from cfb_quant.spread_total.elo_live_validation_v58 import (
    adapt_live_capture,
)
from cfb_quant.spread_total.elo_incremental_v58 import merge_fbs_captures
from cfb_quant.spread_total.elo_replay_v58 import replay_games

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/v58_development/elo_live_captures_v58"
ENDPOINT = "https://api.collegefootballdata.com/games"


def api_key():
    for name in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    env = ROOT / ".env"

    if env.is_file():
        for line in env.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue

            name, value = line.split("=", 1)

            if name.strip() in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
                return value.strip().strip('"').strip("'")

    raise RuntimeError("CFBD API credential not found")


def main():
    print("=" * 72)
    print("STEP 33 - LIVE CFBD ELO CAPTURE")
    print("=" * 72)

    url = ENDPOINT + "?" + urlencode({
        "year": 2026,
        "seasonType": "regular",
    })

    request = Request(
        url,
        headers={
            "Authorization": "Bearer " + api_key(),
            "Accept": "application/json",
        },
    )

    print("Requesting CFBD 2026 regular-season games")

    with urlopen(request, timeout=60) as response:
        raw = response.read()
        response_status = response.status

    captured_at = datetime.now(timezone.utc)
    digest = hashlib.sha256(raw).hexdigest()

    manifest = {
        "source_endpoint": ENDPOINT,
        "request_parameters": {
            "year": 2026,
            "seasonType": "regular",
        },
        "retrieved_at_utc": captured_at.isoformat(),
        "payload_sha256": digest,
        "response_status": response_status,
        "capture_type": "live_observation",
        "historical_point_in_time_certified": False,
        "production_approved": False,
        "observation_policy": (
            "Scores first observed in this response cannot "
            "be used before its retrieval timestamp."
        ),
    }

    records, retrieved, verified_digest = validate_capture(
        raw, manifest
    )

    if not records:
        raise RuntimeError("CFBD returned zero games")

    if response_status != 200:
        raise RuntimeError("Unexpected CFBD HTTP status")

    stamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
    destination = BASE / stamp

    destination.mkdir(parents=True, exist_ok=False)

    payload_path = destination / "games_2026.raw.json"
    manifest_path = destination / "manifest.json"

    with payload_path.open("xb") as f:
        f.write(raw)

    with manifest_path.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    adapted = adapt_live_capture(raw, manifest)
    merged = merge_fbs_captures([(raw, manifest)])
    games = tuple(merged["games"])
    exclusions = list(adapted["excluded"])

    exclusions_path = destination / "excluded_completed_games.json"
    with exclusions_path.open("x", encoding="utf-8") as f:
        json.dump(exclusions, f, indent=2)
        f.write("\n")

    replay = replay_games(games)

    completed = adapted["completed"]
    accepted = adapted["accepted"]

    assert accepted + len(exclusions) == completed
    assert 0 < len(games) <= len(records)
    assert all(
        g.home_points is not None
        for g in games
        if g.result_available_utc is not None
    )
    assert verified_digest == digest
    assert adapted["source_sha256"] == digest

    assert all(
        g.result_available_utc == retrieved
        for g in games
        if g.home_points is not None
    )

    validate_capture(
        payload_path.read_bytes(),
        json.loads(
            manifest_path.read_text(encoding="utf-8")
        ),
    )

    future = [
        s for s in replay.snapshots
        if s.kickoff_utc > retrieved
    ]

    print("\nCAPTURE SAVED")
    print("Directory:", destination.relative_to(ROOT))
    print("Observed UTC:", retrieved.isoformat())
    print("SHA256:", digest)
    print("Source records:", len(records))
    print("Completed results observed:", completed)
    print("Completed results accepted:", accepted)
    print("Completed results quarantined:", len(exclusions))
    for item in exclusions[:10]:
        print("QUARANTINED:", item)
    print("Eligible FBS replay games:", len(games))
    print("Replay snapshots:", len(replay.snapshots))
    print("Replay updates applied:", replay.updates_applied)
    print("Future kickoffs in capture:", len(future))
    print("Teams represented:", len(replay.final_ratings))

    print("\nFIRST FIVE FUTURE PREGAME SNAPSHOTS")

    for s in future[:5]:
        print(
            s.kickoff_utc.isoformat(),
            "|", s.away, "@", s.home,
            "| Home Elo:", round(s.home_elo, 2),
            "| Away Elo:", round(s.away_elo, 2),
            "| Home win probability:",
            round(s.expected_home_win, 4),
        )

    print("\nLIMITATIONS")
    print("Capture time: recorded locally, not externally attested")
    print("Earlier games: NOT point-in-time certified")
    print("Historical score publication times: NOT reconstructed")
    print("Initial ratings and parameters: EXPERIMENTAL")
    print("Production model: UNCHANGED")

    print("\nSTEP 33 COMPLETE")


if __name__ == "__main__":
    main()
