"""Windows atomic forecast promotion regressions."""

import pytest

from cfb_quant.live import forecast_publication_v58 as pub
from tests.test_verified_forecast_publication_v58 import inputs


CLOCK = "2026-10-21T22:02:00Z"


def access_denied(code=5):
    error = PermissionError("simulated access denied")
    error.winerror = code
    return error


def test_transient_access_denied_succeeds(tmp_path, monkeypatch):
    frame, audit = inputs(tmp_path)
    original = pub.Path.rename
    attempts = []

    def flaky(self, target):
        attempts.append(1)
        if len(attempts) <= 2:
            raise access_denied()
        return original(self, target)

    monkeypatch.setattr(pub.Path, "rename", flaky)
    monkeypatch.setattr("time.sleep", lambda _: None)

    output = pub.publish(
        tmp_path / "out",
        frame,
        audit,
        clock=lambda: CLOCK,
    )

    assert len(attempts) == 3
    assert output.is_dir()
    assert (output / "slate_forecasts.csv").is_file()
    assert (output / "model_audit.json").is_file()


def test_permanent_access_denied_cleans_staging(
    tmp_path, monkeypatch
):
    frame, audit = inputs(tmp_path)
    attempts = []

    def blocked(self, target):
        attempts.append(1)
        raise access_denied()

    monkeypatch.setattr(pub.Path, "rename", blocked)
    monkeypatch.setattr("time.sleep", lambda _: None)

    parent = tmp_path / "out"

    with pytest.raises(PermissionError):
        pub.publish(
            parent, frame, audit, clock=lambda: CLOCK
        )

    assert len(attempts) == 6
    assert list(parent.iterdir()) == []


def test_nonretryable_access_error_fails_immediately(
    tmp_path, monkeypatch
):
    frame, audit = inputs(tmp_path)
    attempts = []

    def blocked(self, target):
        attempts.append(1)
        raise access_denied(123)

    monkeypatch.setattr(pub.Path, "rename", blocked)
    monkeypatch.setattr("time.sleep", lambda _: None)

    parent = tmp_path / "out"

    with pytest.raises(PermissionError):
        pub.publish(
            parent, frame, audit, clock=lambda: CLOCK
        )

    assert len(attempts) == 1
    assert list(parent.iterdir()) == []
