"""Offline tests for automatic research readiness."""

from pathlib import Path
import hashlib

import pytest

from cfb_quant.live import automatic_readiness_v58 as automatic


def discovery_for(tmp_path, ids=("101",)):
    path = tmp_path / "paired_forecasts.csv"
    path.write_text("synthetic archive", encoding="utf-8")

    return {
        "archive_count": 1,
        "unique_game_count": len(ids),
        "overlap_count": 0,
        "archives": [{
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "game_ids": list(ids),
            "rows": len(ids),
        }],
    }


def test_discovered_archive_integrity(tmp_path):
    data = discovery_for(tmp_path)
    paths = automatic.verify_discovered(data)
    assert len(paths) == 1


def test_changed_archive_rejected(tmp_path):
    data = discovery_for(tmp_path)
    path = Path(data["archives"][0]["path"])
    path.write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed"):
        automatic.verify_discovered(data)


def test_duplicate_archive_games_rejected(tmp_path):
    data = discovery_for(tmp_path, ids=("101", "101"))

    with pytest.raises(ValueError, match="Duplicate"):
        automatic.verify_discovered(data)


def test_approval_fails_before_discovery(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not run")

    monkeypatch.setattr(automatic, "discover", forbidden)

    with pytest.raises(FileNotFoundError):
        automatic.run(
            2026, "2026-10-20", "2026-10-24",
            tmp_path / "missing_approval.json",
            [tmp_path],
            tmp_path / "output",
        )


def test_readiness_failure_propagates(tmp_path, monkeypatch):
    from cfb_quant.live import approved_baseline_v58

    data = discovery_for(tmp_path)
    monkeypatch.setattr(
        approved_baseline_v58, "verify_any",
        lambda *args: (tmp_path / "schedule.json", {}),
    )
    monkeypatch.setattr(
        automatic, "discover",
        lambda *args: data,
    )
    monkeypatch.setattr(
        automatic, "readiness_run",
        lambda *args: {
            "ready": False,
            "statuses": {"ARCHIVED": 1},
        },
    )

    with pytest.raises(RuntimeError, match="ATTENTION"):
        automatic.run(
            2026, "2026-10-20", "2026-10-24",
            tmp_path / "approval.json",
            [tmp_path],
            tmp_path / "output",
        )
