"""Offline tracked derivation regression tests."""

import shutil

import pytest

from cfb_quant.live.tracked_derivation_v58 import run_tracked
from cfb_quant.live.derivation_receipt_v58 import verify_derivation


def workspace(root):
    (root / "raw").mkdir()
    (root / "processed").mkdir()
    (root / "reports").mkdir()
    (root / "raw" / "schedule.json").write_bytes(b'[{"id":1}]')
    (root / "raw" / "box.json").write_bytes(b'[{"gameId":1}]')
    output = root / "processed" / "history.parquet"
    output.write_bytes(b"existing-output")
    return output


def execute(root, operation):
    return run_tracked(
        root,
        input_paths=["raw/schedule.json", "raw/box.json"],
        output_path="processed/history.parquet",
        receipt_path="reports/derivation.json",
        operation=operation,
    )


def test_successful_tracked_derivation(tmp_path):
    output = workspace(tmp_path)

    result = execute(
        tmp_path,
        lambda: output.write_bytes(b"updated-output"),
    )

    assert result["status"] == "PASS"
    assert result["output_changed"] is True
    assert result["verification"]["input_count"] == 2


def test_source_mutation_blocks_receipt(tmp_path):
    output = workspace(tmp_path)

    def modify_source_and_output():
        (tmp_path / "raw" / "box.json").write_bytes(b"changed")
        output.write_bytes(b"updated")

    with pytest.raises(ValueError, match="Input changed"):
        execute(tmp_path, modify_source_and_output)

    assert not (tmp_path / "reports" / "derivation.json").exists()


def test_operation_failure_blocks_receipt(tmp_path):
    workspace(tmp_path)

    def failure():
        raise RuntimeError("Synthetic operation failure")

    with pytest.raises(RuntimeError, match="Synthetic"):
        execute(tmp_path, failure)

    assert not (tmp_path / "reports" / "derivation.json").exists()


def test_existing_receipt_blocks_execution(tmp_path):
    workspace(tmp_path)
    receipt = tmp_path / "reports" / "derivation.json"
    receipt.write_text("reserved", encoding="utf-8")

    called = []

    with pytest.raises(FileExistsError):
        execute(tmp_path, lambda: called.append(True))

    assert not called


def test_relocated_tracked_workspace(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    output = workspace(original)

    execute(
        original,
        lambda: output.write_bytes(b"research-output"),
    )

    relocated = tmp_path / "different_pc"
    shutil.copytree(original, relocated)

    assert verify_derivation(
        relocated, "reports/derivation.json"
    )["status"] == "PASS"


def test_output_input_collision_rejected(tmp_path):
    workspace(tmp_path)
    called = []

    with pytest.raises(ValueError, match="Output cannot"):
        run_tracked(
            tmp_path,
            input_paths=["processed/history.parquet"],
            output_path="processed/history.parquet",
            receipt_path="reports/other.json",
            operation=lambda: called.append(True),
        )

    assert not called
