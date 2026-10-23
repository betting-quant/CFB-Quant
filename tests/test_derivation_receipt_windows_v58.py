"""Regression for nested derivation receipt paths on Windows."""

from cfb_quant.live.derivation_receipt_v58 import (
    make_receipt,
    verify_derivation,
)


def test_nested_posix_relative_receipt_path(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "reports" / "v58").mkdir(parents=True)

    (tmp_path / "raw" / "source.json").write_text(
        "[]", encoding="utf-8"
    )
    (tmp_path / "processed" / "output.parquet").write_bytes(
        b"synthetic-output"
    )

    relative = "reports/v58/nested/receipt.json"
    (tmp_path / "reports" / "v58" / "nested").mkdir()

    receipt = make_receipt(
        tmp_path,
        input_paths=["raw/source.json"],
        output_path="processed/output.parquet",
        started_utc="2026-10-22T15:00:00Z",
        completed_utc="2026-10-22T15:00:01Z",
        receipt_path=relative,
    )

    assert receipt.is_file()
    assert verify_derivation(tmp_path, relative)["status"] == "PASS"
