"""Research-only source-to-artifact derivation receipts."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from cfb_quant.live.acquisition_receipt_v58 import parse_time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Missing relative path")
    if (
        "\\" in relative
        or relative.startswith("/")
        or PureWindowsPath(relative).drive
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise ValueError("Unsafe relative path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Path escapes research workspace")
    if not path.is_file():
        raise ValueError("Expected regular file")
    return path


def make_receipt(
    root,
    *,
    input_paths,
    output_path,
    started_utc,
    completed_utc,
    receipt_path,
):
    """Describe an existing derivation; never perform the derivation."""
    root = Path(root).resolve(strict=True)
    start = parse_time(started_utc)
    end = parse_time(completed_utc)
    if end < start:
        raise ValueError("Derivation completed before it started")

    if not isinstance(input_paths, (list, tuple)) or not input_paths:
        raise ValueError("Nonempty input list required")
    if len(set(input_paths)) != len(input_paths):
        raise ValueError("Duplicate derivation inputs")

    inputs = [
        {"path": relative, "sha256": digest(checked_path(root, relative))}
        for relative in input_paths
    ]
    output = checked_path(root, output_path)

    if output_path in input_paths:
        raise ValueError("Output cannot also be an input")

    receipt = {
        "schema": "cfb_v58_derivation_receipt_v1",
        "research_only": True,
        "inputs": inputs,
        "output": {
            "path": output_path,
            "sha256": digest(output),
        },
        "derivation_started_utc": start.isoformat(),
        "derivation_completed_utc": end.isoformat(),
        "input_acquisition_times_certified": False,
        "historical_point_in_time_certified": False,
        "production_forecast_authorized": False,
    }

    if not isinstance(receipt_path, str):
        raise ValueError("Receipt path must be a workspace-relative string")
    if (
        "\\" in receipt_path
        or receipt_path.startswith("/")
        or PureWindowsPath(receipt_path).drive
        or any(
            component in ("", ".", "..")
            for component in receipt_path.split("/")
        )
    ):
        raise ValueError("Unsafe receipt path")

    destination = root.joinpath(*receipt_path.split("/"))
    if destination.exists():
        raise FileExistsError(destination)

    # Prevent a receipt from pointing outside the workspace.
    checked_parent = destination.parent.resolve(strict=True)
    if not checked_parent.is_relative_to(root):
        raise ValueError("Receipt directory escapes workspace")
    if destination == output or any(
        destination == checked_path(root, path)
        for path in input_paths
    ):
        raise ValueError("Receipt cannot overwrite a source or output")

    # Ensure the verified workspace-contained parent still exists
    # at the moment the receipt is written.
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)

    return destination


def verify_derivation(root, receipt_path):
    root = Path(root).resolve(strict=True)
    path = checked_path(root, receipt_path)
    receipt = json.loads(path.read_text(encoding="utf-8"))

    if receipt.get("schema") != "cfb_v58_derivation_receipt_v1":
        raise ValueError("Unrecognized derivation receipt")
    if (
        receipt.get("research_only") is not True
        or receipt.get("production_forecast_authorized") is not False
        or receipt.get("historical_point_in_time_certified") is not False
        or receipt.get("input_acquisition_times_certified") is not False
    ):
        raise ValueError("Invalid derivation authorization claims")

    inputs = receipt.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Missing derivation inputs")

    seen = set()
    for item in inputs:
        relative = item["path"]
        if relative in seen:
            raise ValueError("Duplicate derivation input")
        seen.add(relative)
        if digest(checked_path(root, relative)) != item["sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {relative}")

    output = receipt["output"]
    if output["path"] in seen:
        raise ValueError("Output is also an input")
    if digest(checked_path(root, output["path"])) != output["sha256"]:
        raise ValueError("Derived artifact fingerprint mismatch")

    start = parse_time(receipt["derivation_started_utc"])
    end = parse_time(receipt["derivation_completed_utc"])
    if end < start:
        raise ValueError("Invalid derivation chronology")

    return {
        "status": "PASS",
        "research_only": True,
        "input_count": len(inputs),
        "output": output["path"],
        "output_sha256": output["sha256"],
        "historical_point_in_time_certified": False,
        "production_forecast_authorized": False,
    }
