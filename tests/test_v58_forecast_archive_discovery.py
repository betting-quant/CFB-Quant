"""Offline archive discovery tests."""

import csv
from datetime import datetime, timezone

import pytest

from cfb_quant.live.forecast_archive_discovery_v58 import (
    REQUIRED, discover,
)

COLUMNS = [
    "game_id", "kickoff_utc", "away", "home",
    "fair_margin_home", "fair_home_spread", "fair_total",
    "boosting_margin_home", "blend_margin_home",
    "blend_home_spread", "blend_total",
]


def write_archive(root, name, game_ids, day="2026-10-24"):
    directory = root / name
    directory.mkdir(parents=True)
    target = directory / "paired_forecasts.csv"

    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for gid in game_ids:
            row = {
                "game_id": gid,
                "kickoff_utc": f"{day}T18:00:00+00:00",
                "away": "Away",
                "home": "Home",
            }
            for column in set(COLUMNS) - set(row):
                row[column] = "1.5"
            writer.writerow(row)

    return target


def test_discovers_two_disjoint_archives(tmp_path):
    write_archive(tmp_path, "first", ["101", "102"])
    write_archive(tmp_path, "second", ["103"])

    result = discover(
        [tmp_path], "2026-10-20", "2026-10-24"
    )
    assert result["archive_count"] == 2
    assert result["unique_game_count"] == 3
    assert result["overlap_count"] == 0
    assert result["authenticity_verified"] is False


def test_overlapping_games_rejected(tmp_path):
    write_archive(tmp_path, "first", ["101", "102"])
    write_archive(tmp_path, "second", ["102", "103"])

    with pytest.raises(ValueError, match="Overlapping"):
        discover([tmp_path], "2026-10-20", "2026-10-24")


def test_duplicate_within_file_rejected(tmp_path):
    write_archive(tmp_path, "first", ["101", "101"])

    with pytest.raises(ValueError, match="duplicate game ID"):
        discover([tmp_path], "2026-10-20", "2026-10-24")


def test_nonfinite_projection_rejected(tmp_path):
    path = write_archive(tmp_path, "first", ["101"])
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("1.5", "nan", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="Nonfinite"):
        discover([tmp_path], "2026-10-20", "2026-10-24")


def test_out_of_window_not_selected(tmp_path):
    write_archive(tmp_path, "first", ["101"], "2026-10-24")
    write_archive(tmp_path, "old", ["102"], "2026-10-17")

    result = discover(
        [tmp_path], "2026-10-20", "2026-10-24"
    )
    assert result["archive_count"] == 1
    assert result["unique_game_count"] == 1
