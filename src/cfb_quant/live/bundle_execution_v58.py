"""Resolve verified research inputs from a frozen bundle only."""

import hashlib
import json
from pathlib import Path

from cfb_quant.live.forecast_input_bundle_v58 import (
    MANIFEST,
    verify,
)
from cfb_quant.live.forecast_input_freeze_v58 import (
    take,
)


def content_identity(records):
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve(bundle):
    """Return a verified execution root and bundled snapshot path."""

    bundle = Path(bundle).resolve()

    # Verify the complete saved file set against its manifest.
    result = verify(bundle)

    if result["status"] != "VERIFIED":
        raise ValueError("Input bundle verification failed")

    manifest = json.loads(
        (bundle / MANIFEST).read_text(encoding="utf-8")
    )
    records = manifest["dependencies"]

    expected_name = "bundle_" + content_identity(records)

    if bundle.name != expected_name:
        raise ValueError("Bundle identity mismatch")

    snapshot_record = records["snapshot_parquet"]
    audit_record = records["snapshot_audit"]

    snapshot_parquet = (
        bundle / snapshot_record["path"]
    ).resolve()
    snapshot_audit = (
        bundle / audit_record["path"]
    ).resolve()

    if (
        snapshot_parquet.name != "pregame_features.parquet"
        or snapshot_audit.name != "audit.json"
        or snapshot_parquet.parent != snapshot_audit.parent
    ):
        raise ValueError("Invalid bundled snapshot structure")

    snapshot = snapshot_parquet.parent

    # Reapply the original input fingerprint validation, this time
    # exclusively against paths under the bundle.
    current = take(bundle, snapshot)

    if current["dependencies"] != records:
        raise ValueError("Bundle fingerprint disagreement")

    return {
        "schema": "cfb_v58_bundle_execution_resolution_v1",
        "research_only": True,
        "data_root": bundle,
        "snapshot": snapshot,
        "bundle_identity": content_identity(records),
        "bundle_manifest_sha256":
            result["bundle_manifest_sha256"],
        "dependency_count": 10,
        "source_availability_certified": False,
        "historical_point_in_time_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
    }
