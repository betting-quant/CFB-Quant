import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path.cwd()
OUT = ROOT / "reports/independent_matchup_research/2026-10-22"
SNAPSHOT = ROOT / (
    "reports/live_feature_bridge_v56/"
    "2026-10-22_20261022T210901Z"
)

GAMES = [
    {
        "filename": "james_madison_at_app_state_cfbd_research_20261022.json",
        "game_id": "401869845",
        "away": "James Madison",
        "home": "Appalachian State",
        "kickoff_utc": "2026-10-22T23:00:00+00:00",
    },
    {
        "filename": "liberty_at_kennesaw_state_cfbd_research_20261022.json",
        "game_id": "401871056",
        "away": "Liberty",
        "home": "Kennesaw State",
        "kickoff_utc": "2026-10-22T23:00:00+00:00",
    },
    {
        "filename": "east_carolina_at_memphis_cfbd_research_20261022.json",
        "game_id": "401862807",
        "away": "East Carolina",
        "home": "Memphis",
        "kickoff_utc": "2026-10-22T23:30:00+00:00",
    },
]

def get_key():
    for name in ("CFBD_API_KEY", "CFBD_API_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value

    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip().removeprefix("export ") in (
                "CFBD_API_KEY", "CFBD_API_TOKEN"
            ):
                return value.strip().strip('"').strip("'")
    return None

def fetch_advanced(team, key):
    aliases = {"Appalachian State": "App State"}
    params = urlencode({
        "year": 2026,
        "team": aliases.get(team, team),
    })
    url = (
        "https://api.collegefootballdata.com/"
        "stats/season/advanced?" + params
    )
    req = Request(
        url,
        headers={
            "Authorization": "Bearer " + key,
            "Accept": "application/json",
            "User-Agent": "CFB-Quant-Independent-Research/1.0",
        },
    )
    with urlopen(req, timeout=35) as response:
        result = json.load(response)
    if not isinstance(result, list):
        raise ValueError("Unexpected CFBD response for " + team)
    if not result:
        raise ValueError("No advanced statistics returned for " + team)
    aliases = {
        "Appalachian State": "App State",
    }
    api_team = aliases.get(team, team)
    matches = [
        entry for entry in result
        if str(entry.get("team", "")).casefold()
        == api_team.casefold()
    ]
    if not matches:
        raise ValueError("CFBD returned a different team for " + team)
    return matches

def main():
    key = get_key()
    if not key:
        print("ERROR: CFBD_API_KEY not found.")
        print("Configure your CFBD API key in the environment")
        print("or the project's local .env file.")
        return 1

    if not (SNAPSHOT / "audit.json").is_file():
        print("ERROR: Saved October 22 snapshot not found.")
        return 1

    audit = json.loads(
        (SNAPSHOT / "audit.json").read_text(encoding="utf-8")
    )
    if audit.get("slate_date_eastern") != "2026-10-22":
        print("ERROR: Snapshot date mismatch.")
        return 1
    if audit.get("no_current_game_outcomes") is not True:
        print("ERROR: Snapshot outcome exclusion not confirmed.")
        return 1

    started = datetime.now(timezone.utc)
    print("=" * 70)
    print("CFBD MATCHUP RESEARCH EXPORT")
    print("Retrieval start:", started.isoformat())
    print("Historical snapshot:", SNAPSHOT.name)
    print("=" * 70)

    # Never retrieve post-kickoff statistics for this pregame export.
    for game in GAMES:
        kickoff = datetime.fromisoformat(game["kickoff_utc"])
        if started >= kickoff:
            print("BLOCKED: One or more games have begun.")
            print("Post-kickoff API data cannot be certified pregame.")
            return 2

    teams = sorted(set(
        name
        for game in GAMES
        for name in (game["away"], game["home"])
    ))

    stats = {}
    for team in teams:
        print("Requesting 2026 CFBD advanced stats:", team, flush=True)
        try:
            stats[team] = fetch_advanced(team, key)
            print("  Records:", len(stats[team]), flush=True)
        except (HTTPError, URLError, ValueError) as exc:
            print("ERROR retrieving", team, type(exc).__name__)
            if isinstance(exc, HTTPError):
                print("HTTP status:", exc.code)
            else:
                print(str(exc)[:300])
            return 3

    retrieved = datetime.now(timezone.utc)
    for game in GAMES:
        if retrieved >= datetime.fromisoformat(game["kickoff_utc"]):
            print("BLOCKED: Retrieval completed after kickoff.")
            return 2

    OUT.mkdir(parents=True, exist_ok=True)
    for game in GAMES:
        destination = OUT / game["filename"]
        if destination.exists():
            print("STOP: Existing research file not overwritten:", destination)
            return 4

    for game in GAMES:
        document = {
            "schema": "cfbd_matchup_research_v1",
            "game": {
                "game_id": game["game_id"],
                "date": "2026-10-22",
                "away": game["away"],
                "home": game["home"],
                "kickoff_utc": game["kickoff_utc"],
            },
            "provenance": {
                "source": "CollegeFootballData API",
                "endpoint": "/stats/season/advanced",
                "season": 2026,
                "retrieved_at_utc": retrieved.isoformat(),
                "snapshot_asof_utc": audit.get("asof_utc"),
                "snapshot_path": str(SNAPSHOT),
                "source_point_in_time_certified": False,
                "retrieved_before_kickoff": True,
                "cfbd_ppa_is_not_epa": True,
                "epa_available": False,
                "warning": (
                    "These are current API season-to-date records. "
                    "Historical source publication timing is not "
                    "independently certified. Do not use as proof "
                    "of point-in-time data availability."
                ),
            },
            "team_advanced_stats": {
                game["away"]: stats[game["away"]],
                game["home"]: stats[game["home"]],
            },
            "missing_research": [
                "Verified quarterback and injury reports",
                "Weather and game environment",
                "True EPA, unless independently sourced",
                "Opponent-adjusted efficiency, unless supplied",
                "Personnel-level pass-block and pressure metrics",
                "Complete 2026 game-level advanced-stat history",
            ],
            "independence": {
                "sportsbook_data_used": False,
                "cfb_quant_predictions_used": False,
                "current_game_outcomes_used": False,
            },
        }
        target = OUT / game["filename"]
        with target.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
        print("\nCREATED:", target)
        print("Size:", target.stat().st_size, "bytes")

    print("\nALL THREE JSON FILES CREATED")
    print("Directory:", OUT)
    print("CFBD records: API retrieved, not invented")
    print("PPA kept separate from EPA")
    print("Historical PIT availability: NOT CERTIFIED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
