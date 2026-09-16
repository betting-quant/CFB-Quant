from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import sportsdataverse as sdv


CFBD_BASE_URL = "https://api.collegefootballdata.com"

RAW_ROOT = Path("data/raw")

PBP_DIR = RAW_ROOT / "sportsdataverse" / "pbp"
CFBD_GAMES_DIR = RAW_ROOT / "cfbd" / "games"
CFBD_CALENDAR_DIR = RAW_ROOT / "cfbd" / "calendar"
CFBD_TEAM_STATS_DIR = RAW_ROOT / "cfbd" / "team_stats"
CFBD_PLAYER_STATS_DIR = RAW_ROOT / "cfbd" / "player_stats"


def ensure_directories() -> None:
    for path in [
        PBP_DIR,
        CFBD_GAMES_DIR,
        CFBD_CALENDAR_DIR,
        CFBD_TEAM_STATS_DIR,
        CFBD_PLAYER_STATS_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_json(
    payload: Any,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def cfbd_get(
    endpoint: str,
    *,
    params: dict[str, Any],
    attempts: int = 4,
) -> Any:
    api_key = os.environ.get(
        "CFBD_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "CFBD_API_KEY is not set."
        )

    url = (
        f"{CFBD_BASE_URL}"
        f"{endpoint}"
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            response = requests.get(
                url,
                params=params,
                headers={
                    "Authorization": (
                        f"Bearer {api_key}"
                    )
                },
                timeout=90,
            )

            if response.status_code == 429:
                wait_seconds = min(
                    5 * attempt,
                    30,
                )

                print(
                    "    rate limited; "
                    f"waiting {wait_seconds}s"
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

            return response.json()

        except Exception as exc:
            last_error = exc

            if attempt == attempts:
                break

            wait_seconds = min(
                3 * attempt,
                15,
            )

            print(
                "    request failed; "
                f"retrying in {wait_seconds}s"
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"CFBD request failed: {endpoint} "
        f"{params}"
    ) from last_error


def download_pbp(
    season: int,
    *,
    force: bool,
) -> Path:
    output = (
        PBP_DIR
        / f"cfb_pbp_{season}.parquet"
    )

    if (
        output.exists()
        and not force
    ):
        print(
            f"  PBP: cached -> {output}"
        )

        return output

    print(
        f"  PBP: downloading {season}"
    )

    frame = sdv.cfb.load_cfb_pbp(
        seasons=[
            season
        ]
    )

    if frame.height == 0:
        raise RuntimeError(
            f"No PBP rows returned for {season}."
        )

    temporary = output.with_suffix(
        ".parquet.tmp"
    )

    frame.write_parquet(
        temporary
    )

    temporary.replace(
        output
    )

    print(
        f"  PBP: {frame.height:,} rows"
    )

    return output


def download_games(
    season: int,
    *,
    force: bool,
) -> Path:
    output = (
        CFBD_GAMES_DIR
        / f"games_{season}.json"
    )

    if (
        output.exists()
        and not force
    ):
        print(
            f"  Games: cached -> {output}"
        )

        return output

    payload = cfbd_get(
        "/games",
        params={
            "year": season,
            "seasonType": "both",
            "classification": "fbs",
        },
    )

    save_json(
        payload,
        output,
    )

    print(
        f"  Games: {len(payload):,}"
    )

    return output


def download_calendar(
    season: int,
    *,
    force: bool,
) -> list[dict[str, Any]]:
    output = (
        CFBD_CALENDAR_DIR
        / f"calendar_{season}.json"
    )

    if (
        output.exists()
        and not force
    ):
        print(
            f"  Calendar: cached -> {output}"
        )

        return json.loads(
            output.read_text(
                encoding="utf-8"
            )
        )

    payload = cfbd_get(
        "/calendar",
        params={
            "year": season,
        },
    )

    save_json(
        payload,
        output,
    )

    print(
        f"  Calendar periods: "
        f"{len(payload)}"
    )

    return payload


def season_week_keys(
    calendar: list[
        dict[str, Any]
    ],
) -> list[
    tuple[
        str,
        int,
    ]
]:
    keys: set[
        tuple[
            str,
            int,
        ]
    ] = set()

    for item in calendar:
        season_type = str(
            item.get(
                "seasonType",
                ""
            )
        ).strip()

        week = item.get(
            "week"
        )

        if (
            not season_type
            or week is None
        ):
            continue

        keys.add(
            (
                season_type,
                int(
                    week
                ),
            )
        )

    return sorted(
        keys,
        key=lambda value: (
            0
            if value[0] == "regular"
            else 1,
            value[1],
            value[0],
        ),
    )


def download_weekly_endpoint(
    *,
    season: int,
    season_type: str,
    week: int,
    endpoint: str,
    output_dir: Path,
    prefix: str,
    force: bool,
) -> Path:
    output = (
        output_dir
        / str(
            season
        )
        / (
            f"{prefix}_{season}_"
            f"{season_type}_"
            f"week_{week:02d}.json"
        )
    )

    if (
        output.exists()
        and not force
    ):
        print(
            "    cached "
            f"{season_type} week {week}"
        )

        return output

    payload = cfbd_get(
        endpoint,
        params={
            "year": season,
            "week": week,
            "seasonType": season_type,
            "classification": "fbs",
        },
    )

    save_json(
        payload,
        output,
    )

    print(
        "    "
        f"{season_type} week {week}: "
        f"{len(payload):,} games"
    )

    return output


def download_box_scores(
    season: int,
    *,
    calendar: list[
        dict[str, Any]
    ],
    force: bool,
) -> None:
    keys = season_week_keys(
        calendar
    )

    print(
        f"  Weekly periods: {len(keys)}"
    )

    print(
        "  Team box scores:"
    )

    for season_type, week in keys:
        download_weekly_endpoint(
            season=season,
            season_type=season_type,
            week=week,
            endpoint="/games/teams",
            output_dir=(
                CFBD_TEAM_STATS_DIR
            ),
            prefix="team_stats",
            force=force,
        )

    print(
        "  Player box scores:"
    )

    for season_type, week in keys:
        download_weekly_endpoint(
            season=season,
            season_type=season_type,
            week=week,
            endpoint="/games/players",
            output_dir=(
                CFBD_PLAYER_STATS_DIR
            ),
            prefix="player_stats",
            force=force,
        )


def validate_season(
    season: int,
) -> None:
    pbp_path = (
        PBP_DIR
        / f"cfb_pbp_{season}.parquet"
    )

    games_path = (
        CFBD_GAMES_DIR
        / f"games_{season}.json"
    )

    calendar_path = (
        CFBD_CALENDAR_DIR
        / f"calendar_{season}.json"
    )

    if not pbp_path.exists():
        raise RuntimeError(
            f"Missing PBP file for {season}."
        )

    if not games_path.exists():
        raise RuntimeError(
            f"Missing games file for {season}."
        )

    if not calendar_path.exists():
        raise RuntimeError(
            f"Missing calendar file for {season}."
        )

    games = json.loads(
        games_path.read_text(
            encoding="utf-8"
        )
    )

    calendar = json.loads(
        calendar_path.read_text(
            encoding="utf-8"
        )
    )

    keys = season_week_keys(
        calendar
    )

    expected_team = [
        (
            CFBD_TEAM_STATS_DIR
            / str(
                season
            )
            / (
                f"team_stats_{season}_"
                f"{season_type}_"
                f"week_{week:02d}.json"
            )
        )
        for season_type, week in keys
    ]

    expected_player = [
        (
            CFBD_PLAYER_STATS_DIR
            / str(
                season
            )
            / (
                f"player_stats_{season}_"
                f"{season_type}_"
                f"week_{week:02d}.json"
            )
        )
        for season_type, week in keys
    ]

    missing_team = [
        path
        for path in expected_team
        if not path.exists()
    ]

    missing_player = [
        path
        for path in expected_player
        if not path.exists()
    ]

    if missing_team:
        raise RuntimeError(
            f"{season}: missing "
            f"{len(missing_team)} "
            "team-stat files."
        )

    if missing_player:
        raise RuntimeError(
            f"{season}: missing "
            f"{len(missing_player)} "
            "player-stat files."
        )

    pbp = pd.read_parquet(
        pbp_path,
        columns=[
            "game_id",
        ],
    )

    print()
    print(
        f"  VERIFIED {season}"
    )

    print(
        f"    PBP rows: "
        f"{len(pbp):,}"
    )

    print(
        f"    PBP games: "
        f"{pbp['game_id'].nunique():,}"
    )

    print(
        f"    CFBD FBS games: "
        f"{len(games):,}"
    )

    print(
        f"    weekly periods: "
        f"{len(keys)}"
    )


def download_season(
    season: int,
    *,
    force: bool,
) -> None:
    print()
    print("=" * 80)
    print(
        f"CFB HISTORICAL DOWNLOAD — {season}"
    )
    print("=" * 80)

    download_pbp(
        season,
        force=force,
    )

    download_games(
        season,
        force=force,
    )

    calendar = (
        download_calendar(
            season,
            force=force,
        )
    )

    download_box_scores(
        season,
        calendar=calendar,
        force=force,
    )

    validate_season(
        season
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw historical CFB data "
            "for game and player-prop modeling."
        )
    )

    parser.add_argument(
        "--start-season",
        type=int,
        default=2014,
    )

    parser.add_argument(
        "--end-season",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redownload files that already exist."
        ),
    )

    args = parser.parse_args()

    if (
        args.start_season
        > args.end_season
    ):
        raise ValueError(
            "--start-season cannot exceed "
            "--end-season."
        )

    ensure_directories()

    for season in range(
        args.start_season,
        args.end_season + 1,
    ):
        download_season(
            season,
            force=args.force,
        )

    print()
    print("=" * 80)
    print("CFB HISTORICAL DOWNLOAD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
