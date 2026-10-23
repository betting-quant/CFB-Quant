"""Research bundle replay planning regression tests."""

import pytest

from cfb_quant.live.bundle_replay_plan_v58 import prepare
from cfb_quant.live.forecast_input_bundle_v58 import create
from tests.test_forecast_input_freeze_v58 import setup_inputs


def fixture(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    snapshot = setup_inputs(root)
    bundle = create(root, snapshot, tmp_path / "bundles")
    return root, bundle


def test_replay_uses_frozen_inputs_and_separate_outputs(tmp_path):
    root, bundle = fixture(tmp_path)
    plan = prepare(bundle, root)

    assert plan["input_root"] == bundle.resolve()
    assert plan["snapshot"].is_relative_to(bundle.resolve())
    assert plan["output_root"].is_relative_to(root.resolve())
    assert not plan["output_root"].is_relative_to(bundle.resolve())
    assert plan["allow_started_archived"] is True


def test_replay_is_never_prospective(tmp_path):
    root, bundle = fixture(tmp_path)
    plan = prepare(bundle, root)

    assert plan["execution_mode"] == (
        "BUNDLE_BACKED_RETROSPECTIVE_RESEARCH"
    )
    assert plan["research_only"] is True
    assert plan["prospective_performance_eligible"] is False
    assert plan["source_availability_certified"] is False
    assert plan["production_forecast_authorized"] is False


def test_replay_plan_does_not_write_outputs(tmp_path):
    root, bundle = fixture(tmp_path)
    plan = prepare(bundle, root)

    assert not plan["output_root"].exists()


def test_rejects_output_directory_containing_bundle(tmp_path):
    root, bundle = fixture(tmp_path)

    # Deliberately use the bundles parent as the output repository.
    with pytest.raises(ValueError, match="Frozen bundle is inside"):
        prepare(bundle, bundle.parent)


def test_tampered_bundle_cannot_be_planned(tmp_path):
    root, bundle = fixture(tmp_path)

    from cfb_quant.live.forecast_input_freeze_v58 import DEPENDENCIES

    target = bundle / DEPENDENCIES["v3_training"]
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="Bundle integrity mismatch"):
        prepare(bundle, root)
