"""Regression tests for the isolated 2026 source recovery."""

from pathlib import Path
import json

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

RECOVERY = ROOT / (
    "reports/research/full_2026_source_recovery_v1"
)
RAW = RECOVERY / "team_games_2026_complete.parquet"
TIMES = RECOVERY / "team_game_kickoff_map.csv"
MANIFEST = RECOVERY / "manifest.json"
ARCHIVE = ROOT / "data/raw/cfbd/games/games_2026.json"

KEYS = ["game_id", "season", "team", "home_away"]


def game_id(value):
    return str(value).strip().removesuffix(".0")


@pytest.fixture(scope="module")
def sources():
    paths = [RAW, TIMES, MANIFEST, ARCHIVE]
    missing = [str(p) for p in paths if not p.is_file()]

    if missing:
        pytest.skip(
            "Local research artifacts unavailable: "
            + ", ".join(missing)
        )

    raw = pd.read_parquet(RAW)
    times = pd.read_csv(TIMES)

    with MANIFEST.open(encoding="utf-8") as f:
        manifest = json.load(f)

    with ARCHIVE.open(encoding="utf-8") as f:
        archived = json.load(f)

    raw["game_id"] = raw.game_id.map(game_id)
    times["game_id"] = times.game_id.map(game_id)

    times["kickoff_utc"] = pd.to_datetime(
        times["kickoff_utc"],
        format="ISO8601",
        utc=True,
        errors="raise",
    )

    return raw, times, manifest, archived


def test_recovery_counts_and_identities(sources):
    raw, times, manifest, _ = sources

    assert len(raw) == 1140
    assert raw.game_id.nunique() == 570

    assert len(times) == 1140
    assert not raw.duplicated(KEYS).any()
    assert not times.duplicated(KEYS).any()
    assert times.kickoff_utc.notna().all()

    assert set(map(tuple, raw[KEYS].to_numpy())) == set(
        map(tuple, times[KEYS].to_numpy())
    )

    assert manifest["research_only"] is True
    assert manifest["production_approved"] is False
    assert (
        manifest["historical_source_availability_certified"]
        is False
    )


def test_recovered_week8_9_kickoffs_match_archive(sources):
    raw, times, _, archived = sources

    ids = set(
        raw.loc[raw.week.isin([8, 9]), "game_id"]
    )
    assert len(ids) == 60

    primary = (
        times.loc[times.game_id.isin(ids)]
        .groupby("game_id")["kickoff_utc"]
        .first()
    )
    assert len(primary) == 60

    matches = {}

    for game in archived:
        if not isinstance(game, dict):
            continue

        gid = game_id(game.get("id"))

        if gid not in ids:
            continue

        value = game.get("startDate")
        assert value, f"Missing archived kickoff: {gid}"

        kickoff = pd.to_datetime(
            value,
            utc=True,
            format="ISO8601",
            errors="raise",
        )

        if gid in matches:
            assert matches[gid] == kickoff
        else:
            matches[gid] = kickoff

    assert set(matches) == ids

    for gid, archived_time in matches.items():
        assert primary[gid] == archived_time, (
            f"Kickoff disagreement: {gid}"
        )
