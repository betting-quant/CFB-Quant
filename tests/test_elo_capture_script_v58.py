import hashlib
import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/capture_elo_live_v58.py"
)


@pytest.mark.parametrize("replay_fails", [False, True])
def test_capture_preserves_evidence_and_filters_fbs(
    tmp_path, monkeypatch, replay_fails
):
    main = runpy.run_path(str(SCRIPT))["main"]
    scope = main.__globals__

    raw = b'[{"id": 1}, {"id": 2}]'
    digest = hashlib.sha256(raw).hexdigest()
    calls = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return raw

    def fake_validate(payload, manifest):
        assert payload == raw
        assert manifest["payload_sha256"] == digest
        observed = datetime.fromisoformat(
            manifest["retrieved_at_utc"]
        )
        return [{"id": 1}, {"id": 2}], observed, digest

    def fake_adapt(payload, manifest):
        calls.append("adapt")
        return {
            "games": (),
            "records": 2,
            "completed": 2,
            "accepted": 1,
            "excluded": (
                {
                    "game_id": "2",
                    "reason": "completed_without_valid_final_score",
                },
            ),
            "source_sha256": digest,
        }

    def fake_merge(captures):
        calls.append("merge")
        assert len(captures) == 1
        payload, manifest = captures[0]
        assert payload == raw

        observed = datetime.fromisoformat(
            manifest["retrieved_at_utc"]
        )

        games = (
            SimpleNamespace(
                home_points=21,
                result_available_utc=observed,
            ),
            SimpleNamespace(
                home_points=None,
                result_available_utc=None,
            ),
        )

        return {
            "games": games,
            "latest_capture_utc": observed,
        }

    def fake_replay(games):
        calls.append("replay")
        assert len(games) == 2

        if replay_fails:
            raise ValueError("synthetic replay failure")

        observed = datetime.now(timezone.utc)

        snap = SimpleNamespace(
            kickoff_utc=observed + timedelta(days=1),
            home="Test Home",
            away="Test Away",
            home_elo=1500.0,
            away_elo=1500.0,
            expected_home_win=0.5,
        )

        return SimpleNamespace(
            snapshots=(snap,),
            final_ratings={
                "Test Home": 1500.0,
                "Test Away": 1500.0,
            },
            updates_applied=1,
        )

    destination = tmp_path / "captures"

    monkeypatch.setitem(scope, "ROOT", tmp_path)
    monkeypatch.setitem(scope, "BASE", destination)
    monkeypatch.setitem(scope, "api_key", lambda: "test-key")
    monkeypatch.setitem(
        scope, "urlopen", lambda *args, **kwargs: FakeResponse()
    )
    monkeypatch.setitem(scope, "validate_capture", fake_validate)
    monkeypatch.setitem(scope, "adapt_live_capture", fake_adapt)
    monkeypatch.setitem(scope, "merge_fbs_captures", fake_merge)
    monkeypatch.setitem(scope, "replay_games", fake_replay)

    if replay_fails:
        with pytest.raises(
            ValueError, match="synthetic replay failure"
        ):
            main()
    else:
        main()

    folders = list(destination.iterdir())
    assert len(folders) == 1
    folder = folders[0]

    saved_raw = folder / "games_2026.raw.json"
    saved_manifest = folder / "manifest.json"

    assert saved_raw.read_bytes() == raw

    manifest = json.loads(
        saved_manifest.read_text(encoding="utf-8")
    )
    assert manifest["payload_sha256"] == digest
    assert manifest["production_approved"] is False

    assert calls == ["adapt", "merge", "replay"]

    exclusions_path = folder / "excluded_completed_games.json"

    if replay_fails:
        # The original capture survives downstream failure.
        assert saved_raw.exists()
    else:
        exclusions = json.loads(
            exclusions_path.read_text(encoding="utf-8")
        )
        assert len(exclusions) == 1
        assert exclusions[0]["game_id"] == "2"
