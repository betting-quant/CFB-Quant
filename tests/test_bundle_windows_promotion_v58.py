"""Regression tests for bounded Windows directory promotion."""

import pytest

from cfb_quant.live import forecast_input_bundle_v58 as mod
from tests.test_forecast_input_freeze_v58 import setup_inputs


def simulated_access_denied():
    exc = PermissionError("Simulated Windows access denied")
    exc.winerror = 5
    return exc


def setup(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)
    parent = tmp_path / "bundles"
    return root, snapshot, parent


def patch_windows(monkeypatch):
    monkeypatch.setattr(mod.os, "name", "nt", raising=False)


def test_transient_access_denied_eventually_succeeds(
    tmp_path, monkeypatch
):
    root, snapshot, parent = setup(tmp_path)

    original_rename = mod.Path.rename
    calls = []

    def flaky_rename(self, target):
        if self.name == "payload":
            calls.append(1)
            if len(calls) <= 2:
                raise simulated_access_denied()
        return original_rename(self, target)

    # Pretend we are on Windows without modifying os.name,
    # which can interfere with pytest on other platforms.
    monkeypatch.setattr(mod.Path, "rename", flaky_rename)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    bundle = mod.create(root, snapshot, parent)

    assert len(calls) == 3
    assert mod.verify(bundle)["status"] == "VERIFIED"


def test_permanent_access_denied_never_publishes(
    tmp_path, monkeypatch
):
    root, snapshot, parent = setup(tmp_path)
    calls = []

    def blocked_rename(self, target):
        if self.name == "payload":
            calls.append(1)
            raise simulated_access_denied()
        raise AssertionError("Unexpected rename target")

    monkeypatch.setattr(mod.Path, "rename", blocked_rename)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError, match="access denied"):
        mod.create(root, snapshot, parent)

    assert len(calls) == 6
    assert not list(parent.iterdir())
