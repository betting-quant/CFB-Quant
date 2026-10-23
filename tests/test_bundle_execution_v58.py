"""Bundle-backed execution resolution tests."""

import shutil

import pytest

from cfb_quant.live.bundle_execution_v58 import resolve
from cfb_quant.live.forecast_input_bundle_v58 import create
from cfb_quant.live.forecast_input_freeze_v58 import DEPENDENCIES
from tests.test_forecast_input_freeze_v58 import setup_inputs


def bundle_fixture(tmp_path):
    root = tmp_path / "original"
    root.mkdir()

    snapshot = setup_inputs(root)
    bundle = create(root, snapshot, tmp_path / "bundles")

    return root, snapshot, bundle


def test_resolves_only_bundled_inputs(tmp_path):
    root, snapshot, bundle = bundle_fixture(tmp_path)
    context = resolve(bundle)

    assert context["data_root"] == bundle.resolve()
    assert context["snapshot"].is_relative_to(bundle.resolve())
    assert context["snapshot"] != snapshot
    assert context["dependency_count"] == 10
    assert context["prospective_performance_eligible"] is False


def test_original_mutation_does_not_break_replay(tmp_path):
    root, _, bundle = bundle_fixture(tmp_path)

    (root / DEPENDENCIES["v3_training"]).write_bytes(
        b"mutated-original"
    )

    context = resolve(bundle)
    assert context["data_root"] == bundle.resolve()


def test_missing_bundled_input_never_falls_back(tmp_path):
    root, _, bundle = bundle_fixture(tmp_path)

    (bundle / DEPENDENCIES["v3_training"]).unlink()

    with pytest.raises(FileNotFoundError):
        resolve(bundle)

    # Even though the original file still exists, it is not used.
    assert (root / DEPENDENCIES["v3_training"]).is_file()


def test_modified_bundled_input_rejected(tmp_path):
    _, _, bundle = bundle_fixture(tmp_path)

    (bundle / DEPENDENCIES["v45_training"]).write_bytes(
        b"modified-bundle"
    )

    with pytest.raises(
        ValueError, match="Bundle integrity mismatch"
    ):
        resolve(bundle)


def test_changed_bundle_directory_identity_rejected(tmp_path):
    _, _, bundle = bundle_fixture(tmp_path)

    renamed = bundle.parent / "unverified_bundle_name"
    shutil.move(str(bundle), str(renamed))

    with pytest.raises(
        ValueError, match="Bundle identity mismatch"
    ):
        resolve(renamed)
