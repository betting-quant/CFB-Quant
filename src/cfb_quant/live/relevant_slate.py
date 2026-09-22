from __future__ import annotations

import argparse
from zoneinfo import ZoneInfo

import pandas as pd

import cfb_quant.live.slate as slate


TOP_25 = {
    "Texas",
    "Georgia",
    "Notre Dame",
    "Indiana",
    "Miami",
    "Ohio State",
    "LSU",
    "Ole Miss",
    "Texas A&M",
    "Alabama",
    "BYU",
    "USC",
    "Texas Tech",
    "Penn State",
    "Tennessee",
    "SMU",
    "Utah",
    "Iowa",
    "Michigan",
    "Missouri",
    "Oregon",
    "Houston",
    "Louisville",
    "Oklahoma",
    "Virginia",
}

RECEIVING_VOTES = {
    "Washington",
    "Florida",
    "Boise State",
    "Oklahoma State",
    "Western Michigan",
    "Tulsa",
    "Mississippi State",
    "South Carolina",
    "Virginia Tech",
    "UCLA",
    "James Madison",
    "Arizona",
    "Pittsburgh",
    "Kansas State",
}

POWER_CONFERENCES = {
    "ACC",
    "Big 12",
    "Big Ten",
    "SEC",
}


def _normalize_team(name: str) -> str:
    aliases = {
        "Miami (FL)": "Miami",
        "Southern Cal": "USC",
    }

    name = str(name).strip()
    return aliases.get(name, name)


def select_relevant_games(
    season: int,
    week: int,
    date: str,
    timezone: str,
) -> list[dict]:

    tz = ZoneInfo(timezone)
    games = slate._load_schedule(season)

    selected = []

    for game in games:

        if int(game.get("season", -1)) != season:
            continue

        if int(game.get("week", -1)) != week:
            continue

        if bool(game.get("completed", False)):
            continue

        start = pd.to_datetime(
            game["startDate"],
            utc=True,
        ).tz_convert(tz)

        if start.date().isoformat() != date:
            continue

        home_class = str(
            game.get("homeClassification", "")
        ).lower()

        away_class = str(
            game.get("awayClassification", "")
        ).lower()

        # HARD RULE:
        # absolutely no FCS games.
        if home_class != "fbs" or away_class != "fbs":
            continue

        home = _normalize_team(
            game.get("homeTeam", "")
        )

        away = _normalize_team(
            game.get("awayTeam", "")
        )

        home_conf = str(
            game.get("homeConference", "")
        )

        away_conf = str(
            game.get("awayConference", "")
        )

        ranked_game = (
            home in TOP_25
            or away in TOP_25
        )

        vote_game = (
            home in RECEIVING_VOTES
            or away in RECEIVING_VOTES
        )

        power_home = (
            home_conf in POWER_CONFERENCES
            or home == "Notre Dame"
        )

        power_away = (
            away_conf in POWER_CONFERENCES
            or away == "Notre Dame"
        )

        power_matchup = (
            power_home
            and power_away
        )

        if not (
            ranked_game
            or vote_game
            or power_matchup
        ):
            continue

        game = dict(game)

        game["_local_start"] = start
        game["_selection_reason"] = (
            "TOP 25"
            if ranked_game
            else "RECEIVING VOTES"
            if vote_game
            else "POWER MATCHUP"
        )

        selected.append(game)

    selected = sorted(
        selected,
        key=lambda g: g["_local_start"],
    )

    print()
    print("=" * 95)
    print("FILTERED SATURDAY GAMES")
    print("=" * 95)

    for game in selected:

        print(
            f"{game['_local_start'].strftime('%I:%M %p')}  "
            f"{game['awayTeam']} @ {game['homeTeam']}  "
            f"[{game['_selection_reason']}]"
        )

    print()
    print(
        f"QUALIFYING GAMES: {len(selected)}"
    )

    return selected


# Replace only the game-selection function.
# Everything else uses the existing tested slate workflow:
# batches, feature engine, trained models and role filtering.
slate.select_date_games = select_relevant_games


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--season",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--week",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--date",
        required=True,
    )

    parser.add_argument(
        "--timezone",
        default="America/Edmonton",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    slate.run_date_slate(
        season=args.season,
        week=args.week,
        date=args.date,
        timezone=args.timezone,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
