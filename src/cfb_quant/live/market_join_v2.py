from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROJECTIONS = Path(
    "reports/live_player_production_2026_week_4_v0_1.csv"
)

MARKET = Path(
    "reports/sportsbook_props_2026_week_4.csv"
)

OUTPUT = Path(
    "reports/market_join_2026_week_4_v0_2.csv"
)

UNMATCHED_OUTPUT = Path(
    "reports/market_unmatched_2026_week_4_v0_2.csv"
)


VALID_TARGETS = {
    "pass_attempts",
    "completions",
    "passing_yards",
    "rush_attempts",
    "rushing_yards",
    "receptions",
    "receiving_yards",
}


TEAM_ALIASES = {
    "california": "cal",
    "cal": "cal",
    "usc": "usc",
    "southern california": "usc",

    "miami florida": "miami",
    "miami fl": "miami",
    "miami": "miami",

    "miami ohio": "miami ohio",

    "ul monroe": "ulm",
    "ulm": "ulm",

    "ul lafayette": "louisiana",
    "louisiana": "louisiana",

    "florida intl": "florida international",
    "florida international": "florida international",

    "nc state": "nc state",
    "north carolina state": "nc state",

    "umass": "massachusetts",
    "massachusetts": "massachusetts",

    "uconn": "connecticut",
    "connecticut": "connecticut",

    "utep": "utep",
    "texas el paso": "utep",

    "smu": "smu",
    "southern methodist": "smu",
}


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().casefold()

    text = (
        text.replace("&", " and ")
        .replace("’", "'")
        .replace("‘", "'")
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def normalize_team(value) -> str:
    text = clean_text(value)

    return TEAM_ALIASES.get(
        text,
        text,
    )


def normalize_player(value) -> str:
    text = clean_text(value)

    suffixes = {
        "jr",
        "sr",
        "ii",
        "iii",
        "iv",
        "v",
    }

    parts = text.split()

    while (
        parts
        and parts[-1] in suffixes
    ):
        parts = parts[:-1]

    return "".join(parts)


def main():

    if not PROJECTIONS.exists():
        raise RuntimeError(
            f"Missing projections: {PROJECTIONS}"
        )

    if not MARKET.exists():
        raise RuntimeError(
            f"Missing market CSV: {MARKET}"
        )

    projections = pd.read_csv(
        PROJECTIONS
    )

    market = pd.read_csv(
        MARKET
    )

    print("=" * 100)
    print("CFB MARKET JOIN v0.2")
    print("=" * 100)

    print(
        f"Projection rows: {len(projections):,}"
    )

    print(
        f"Market rows: {len(market):,}"
    )

    required_market = {
        "away_team",
        "home_team",
        "player_name",
        "target",
        "line",
        "over_odds",
        "under_odds",
        "sportsbook",
    }

    missing = (
        required_market
        - set(market.columns)
    )

    if missing:
        raise RuntimeError(
            "Market CSV missing columns: "
            + ", ".join(sorted(missing))
        )

    invalid_targets = sorted(
        set(
            market[
                "target"
            ].dropna()
        )
        - VALID_TARGETS
    )

    if invalid_targets:
        raise RuntimeError(
            f"Invalid targets: "
            f"{invalid_targets}"
        )

    projections[
        "_team_key"
    ] = projections[
        "team"
    ].map(
        normalize_team
    )

    projections[
        "_opponent_key"
    ] = projections[
        "opponent"
    ].map(
        normalize_team
    )

    projections[
        "player_key_norm"
    ] = projections[
        "player_name"
    ].map(
        normalize_player
    )

    market[
        "away_key_norm"
    ] = market[
        "away_team"
    ].map(
        normalize_team
    )

    market[
        "home_key_norm"
    ] = market[
        "home_team"
    ].map(
        normalize_team
    )

    market[
        "player_key_norm"
    ] = market[
        "player_name"
    ].map(
        normalize_player
    )

    game_map = (
        projections[
            [
                "game_id",
                "team",
                "opponent",
                "_team_key",
                "_opponent_key",
            ]
        ]
        .drop_duplicates()
    )

    resolved_rows = []

    for row in market.itertuples(
        index=False
    ):

        away = row.away_key_norm
        home = row.home_key_norm

        possible = game_map.loc[
            (
                (
                    game_map["_team_key"].eq(
                        away
                    )
                    &
                    game_map["_opponent_key"].eq(
                        home
                    )
                )
                |
                (
                    game_map["_team_key"].eq(
                        home
                    )
                    &
                    game_map["_opponent_key"].eq(
                        away
                    )
                )
            )
        ]

        game_ids = (
            possible[
                "game_id"
            ]
            .drop_duplicates()
            .tolist()
        )

        record = row._asdict()

        if len(game_ids) == 1:
            record["game_id"] = (
                game_ids[0]
            )
            record[
                "game_match_status"
            ] = "matched"

        elif len(game_ids) == 0:
            record["game_id"] = pd.NA
            record[
                "game_match_status"
            ] = "no_game_match"

        else:
            record["game_id"] = pd.NA
            record[
                "game_match_status"
            ] = "ambiguous_game"

        resolved_rows.append(
            record
        )

    resolved = pd.DataFrame(
        resolved_rows
    )

    matched_games = resolved.loc[
        resolved[
            "game_match_status"
        ].eq(
            "matched"
        )
    ].copy()

    projection_keep = [
        "game_id",
        "team",
        "opponent",
        "player_name",
        "player_key_norm",
        "target",
        "projection_original",
        "projection_matchup",
        "projection_blended",
        "blend_weight_matchup",
        "player_games_before",
        "player_season_games_before",
    ]

    joined = matched_games.merge(
        projections[
            projection_keep
        ],
        on=[
            "game_id",
            "player_key_norm",
            "target",
        ],
        how="left",
        suffixes=(
            "_market",
            "_model",
        ),
    )

    joined[
        "player_match_status"
    ] = "matched"

    missing_player = joined[
        "projection_blended"
    ].isna()

    joined.loc[
        missing_player,
        "player_match_status",
    ] = "no_player_target_match"

    joined[
        "line"
    ] = pd.to_numeric(
        joined["line"],
        errors="coerce",
    )

    joined[
        "over_odds"
    ] = pd.to_numeric(
        joined["over_odds"],
        errors="coerce",
    )

    joined[
        "under_odds"
    ] = pd.to_numeric(
        joined["under_odds"],
        errors="coerce",
    )

    joined[
        "projection_edge"
    ] = (
        joined[
            "projection_blended"
        ]
        - joined[
            "line"
        ]
    )

    joined[
        "abs_projection_edge"
    ] = joined[
        "projection_edge"
    ].abs()

    joined[
        "model_disagreement"
    ] = (
        joined[
            "projection_matchup"
        ]
        - joined[
            "projection_original"
        ]
    )

    joined[
        "lean_side"
    ] = joined[
        "projection_edge"
    ].apply(
        lambda value:
            "OVER"
            if pd.notna(value)
            and value > 0
            else "UNDER"
            if pd.notna(value)
            and value < 0
            else "EVEN"
            if pd.notna(value)
            else ""
    )

    no_game = resolved.loc[
        ~resolved[
            "game_match_status"
        ].eq(
            "matched"
        )
    ].copy()

    no_player = joined.loc[
        joined[
            "player_match_status"
        ].ne(
            "matched"
        )
    ].copy()

    unmatched = pd.concat(
        [
            no_game,
            no_player,
        ],
        ignore_index=True,
        sort=False,
    )

    matched = joined.loc[
        joined[
            "player_match_status"
        ].eq(
            "matched"
        )
    ].copy()

    matched = matched.sort_values(
        "abs_projection_edge",
        ascending=False,
    )

    matched.to_csv(
        OUTPUT,
        index=False,
    )

    unmatched.to_csv(
        UNMATCHED_OUTPUT,
        index=False,
    )

    print()
    print("=" * 100)
    print("MATCH RESULTS")
    print("=" * 100)

    print(
        "Market rows:",
        len(market),
    )

    print(
        "Game matched:",
        int(
            resolved[
                "game_match_status"
            ]
            .eq("matched")
            .sum()
        ),
    )

    print(
        "Fully matched:",
        len(matched),
    )

    print(
        "Unmatched:",
        len(unmatched),
    )

    if not matched.empty:

        print()
        print(
            "TOP RAW PROJECTION DIFFERENCES"
        )
        print("-" * 100)

        columns = [
            "away_team",
            "home_team",
            "player_name_model",
            "target",
            "line",
            "projection_blended",
            "projection_edge",
            "lean_side",
            "model_disagreement",
            "over_odds",
            "under_odds",
        ]

        print(
            matched[
                columns
            ]
            .head(40)
            .to_string(
                index=False
            )
        )

    if not unmatched.empty:

        print()
        print(
            "UNMATCHED ROWS"
        )
        print("-" * 100)

        display = [
            col
            for col in [
                "away_team",
                "home_team",
                "player_name",
                "player_name_market",
                "target",
                "line",
                "game_match_status",
                "player_match_status",
            ]
            if col in unmatched.columns
        ]

        print(
            unmatched[
                display
            ]
            .head(100)
            .to_string(
                index=False
            )
        )

    print()
    print(
        f"Matched output: "
        f"{OUTPUT}"
    )

    print(
        f"Unmatched output: "
        f"{UNMATCHED_OUTPUT}"
    )


if __name__ == "__main__":
    main()
