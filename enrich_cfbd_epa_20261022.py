import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / "reports/independent_matchup_research/2026-10-22"
URL = "https://collegefootballdata.com/metrics/season"

GAMES = {
    "james_madison_at_app_state_cfbd_research_20261022.json": (
        "James Madison", "App State"
    ),
    "liberty_at_kennesaw_state_cfbd_research_20261022.json": (
        "Liberty", "Kennesaw State"
    ),
    "east_carolina_at_memphis_cfbd_research_20261022.json": (
        "East Carolina", "Memphis"
    ),
}

ALIASES = {"Appalachian State": "App State"}

def clean(value):
    return re.sub(r"\s+", " ", str(value)).strip()

def fail(message):
    raise RuntimeError(message)

def main():
    now = datetime.now(timezone.utc)

    originals = {}
    for filename, expected in GAMES.items():
        path = BASE / filename
        if not path.is_file():
            fail("Original JSON missing: " + str(path))

        data = json.loads(path.read_text(encoding="utf-8"))
        game = data["game"]
        kickoff = datetime.fromisoformat(game["kickoff_utc"])

        if now >= kickoff:
            fail("Kickoff reached. Refusing new pregame EPA retrieval.")

        actual = (
            ALIASES.get(game["away"], game["away"]),
            ALIASES.get(game["home"], game["home"]),
        )
        if actual != expected:
            fail("Matchup mismatch in " + filename)

        originals[filename] = data

    print("=" * 72)
    print("2026 CFBD EPA — PUBLIC SOURCE EXTRACTION")
    print("=" * 72)

    request = Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html",
        },
    )

    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")

    retrieved = datetime.now(timezone.utc)

    print("Retrieved UTC:", retrieved.isoformat())
    print("HTML bytes:", len(html.encode("utf-8")))
    print("Source SHA256:",
          hashlib.sha256(html.encode("utf-8")).hexdigest())

    for data in originals.values():
        if retrieved >= datetime.fromisoformat(
            data["game"]["kickoff_utc"]
        ):
            fail("Kickoff reached during retrieval.")

    # Parse actual HTML tables, never generate assumed EPA values.
    try:
        tables = pd.read_html(StringIO(html))
    except (ValueError, ImportError) as exc:
        fail(
            "Published table was not available to the HTML parser: "
            + str(exc)
        )

    print("HTML tables found:", len(tables))

    selected = None
    for table in tables:
        columns = [clean(c).lower() for c in table.columns]
        if (
            any("team" in c for c in columns)
            and sum("epa" in c for c in columns) >= 2
        ):
            selected = table
            break

    if selected is None:
        for index, table in enumerate(tables):
            print("Table", index, "columns:", list(table.columns))
        fail(
            "Could not identify a table with team and EPA columns. "
            "No files were modified."
        )

    columns = list(selected.columns)
    team_col = next(
        c for c in columns if "team" in clean(c).lower()
    )

    epa_cols = [
        c for c in columns if "epa" in clean(c).lower()
    ]

    print("Selected table columns:", columns)
    print("Rows:", len(selected))

    if len(epa_cols) != 2:
        fail("Expected exactly two EPA columns.")

    # This page's default view is:
    # X = offensive EPA/play; Y = defensive EPA/play.
    if not (
        "offens" in clean(epa_cols[0]).lower()
        and "defens" in clean(epa_cols[1]).lower()
    ):
        fail(
            "EPA column orientation not independently verified. "
            "No files were modified."
        )

    wanted = {
        "James Madison", "App State", "Liberty",
        "Kennesaw State", "East Carolina", "Memphis"
    }
    result = {}

    for _, row in selected.iterrows():
        label = clean(row[team_col])

        matches = [
            name for name in wanted
            if label == name
            or label.startswith(name + " ")
        ]

        if len(matches) != 1:
            continue

        name = matches[0]
        if name in result:
            fail("Duplicate EPA team: " + name)

        offense = float(row[epa_cols[0]])
        defense = float(row[epa_cols[1]])

        if not (-5 < offense < 5 and -5 < defense < 5):
            fail("Implausible EPA values for " + name)

        result[name] = {
            "offensive_epa_per_play": offense,
            "defensive_epa_allowed_per_play": defense,
        }

    if set(result) != wanted:
        fail(
            "Missing EPA teams: "
            + repr(sorted(wanted - set(result)))
            + ". No files were modified."
        )

    print("\nVERIFIED TABLE EXTRACTION")
    for team, values in sorted(result.items()):
        print(team, values)

    planned = []

    for filename, data in originals.items():
        output = BASE / filename.replace(
            ".json", "_with_epa.json"
        )

        if output.exists():
            fail("Enriched file already exists: " + str(output))

        game = data["game"]
        away = ALIASES.get(game["away"], game["away"])
        home = ALIASES.get(game["home"], game["home"])

        data["epa_research"] = {
            "source": URL,
            "retrieved_at_utc": retrieved.isoformat(),
            "metric": "Expected Points Added per play",
            "view": "2026 full season, garbage time excluded",
            "offense_column": str(epa_cols[0]),
            "defense_column": str(epa_cols[1]),
            "point_in_time_certified": False,
            "source_data_cutoff_verified": False,
            "teams": {
                game["away"]: result[away],
                game["home"]: result[home],
            },
            "limitations": [
                "Published page values are dynamic",
                "Underlying source-game cutoff not certified",
                "Passing and rushing EPA splits not included",
                "Do not confuse EPA with CFBD PPA",
                "No opponent adjustment independently verified",
            ],
        }

        # The original missing_research list may mention
        # true EPA; retain historical notes instead of
        # silently removing research limitations.
        planned.append((output, data))

    for output, data in planned:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
        print("\nCREATED:", output)

    print("\nEPA ENRICHMENT COMPLETE — PASS")
    print("Files created:", len(planned))
    print("Original files: UNCHANGED")
    print("Python forecast models: UNCHANGED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
