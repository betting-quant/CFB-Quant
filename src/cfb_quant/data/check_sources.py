from __future__ import annotations

import os

import requests
import sportsdataverse as sdv


CFBD_BASE_URL = "https://api.collegefootballdata.com"


def check_sportsdataverse() -> None:
    print()
    print("=" * 80)
    print("SPORTSDATAVERSE CFB CHECK")
    print("=" * 80)

    pbp = sdv.cfb.load_cfb_pbp(
        seasons=[2025]
    )

    print("Rows:", pbp.height)
    print("Columns:", len(pbp.columns))

    wanted = [
        column
        for column in [
            "game_id",
            "season",
            "week",
            "home",
            "away",
            "pos_team",
            "def_pos_team",
            "play_type",
            "yards_gained",
            "EPA",
            "pass",
            "rush",
        ]
        if column in pbp.columns
    ]

    print()
    print("Useful columns found:")
    for column in wanted:
        print(f"  {column}")

    print()
    print(
        pbp.select(
            wanted
        ).head(3)
    )


def check_cfbd() -> None:
    print()
    print("=" * 80)
    print("COLLEGEFOOTBALLDATA CHECK")
    print("=" * 80)

    api_key = os.environ.get(
        "CFBD_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "CFBD_API_KEY is not set."
        )

    response = requests.get(
        f"{CFBD_BASE_URL}/games",
        params={
            "year": 2025,
            "classification": "fbs",
        },
        headers={
            "Authorization": (
                f"Bearer {api_key}"
            )
        },
        timeout=30,
    )

    response.raise_for_status()

    games = response.json()

    print("2025 FBS games:", len(games))

    if games:
        game = games[0]

        print()
        print("Example game:")
        for key in [
            "id",
            "season",
            "week",
            "seasonType",
            "startDate",
            "homeTeam",
            "awayTeam",
            "homePoints",
            "awayPoints",
        ]:
            print(
                f"  {key}: "
                f"{game.get(key)}"
            )


def main() -> None:
    check_sportsdataverse()
    check_cfbd()

    print()
    print("=" * 80)
    print("CFB DATA SOURCES VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
