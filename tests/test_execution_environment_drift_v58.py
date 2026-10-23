"""Environment provenance drift regression tests."""

import copy

import pytest

from cfb_quant.live import execution_provenance_v58 as prov


def manifest():
    runtime = prov.environment()
    return {
        "schema": prov.SCHEMA,
        "environment": runtime,
        "environment_identity": prov.identity(runtime),
    }


def test_unchanged_environment_passes():
    result = prov.verify_environment(manifest())
    assert result["status"] == "UNCHANGED"


def test_changed_package_version_is_rejected(monkeypatch):
    recorded = manifest()

    changed = copy.deepcopy(recorded["environment"])
    changed["packages"]["numpy"] = "999.0.0"

    monkeypatch.setattr(
        prov, "environment", lambda: changed
    )

    with pytest.raises(
        ValueError, match="Execution environment changed"
    ):
        prov.verify_environment(recorded)


def test_tampered_environment_identity_is_rejected():
    recorded = manifest()
    recorded["environment_identity"] = "0" * 64

    with pytest.raises(
        ValueError, match="Environment manifest identity mismatch"
    ):
        prov.verify_environment(recorded)
