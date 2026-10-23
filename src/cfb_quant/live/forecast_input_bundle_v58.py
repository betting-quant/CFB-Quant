"""Verified research input bundles. No PIT certification."""

import hashlib
import json
import shutil
import tempfile
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from cfb_quant.live.forecast_input_freeze_v58 import (
    take,
    verify_unchanged,
)


MANIFEST = "bundle_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(bundle):
    bundle = Path(bundle).resolve()

    if not bundle.is_dir():
        raise FileNotFoundError(bundle)

    manifest_path = bundle / MANIFEST
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("schema") != "cfb_v58_input_bundle_v1":
        raise ValueError("Invalid input bundle schema")

    records = manifest.get("dependencies")

    if not isinstance(records, dict) or len(records) != 10:
        raise ValueError("Incomplete input bundle")

    from cfb_quant.live.forecast_input_freeze_v58 import (
        DEPENDENCIES,
    )

    required = set(DEPENDENCIES) | {
        "snapshot_parquet", "snapshot_audit"
    }

    if set(records) != required:
        raise ValueError("Unexpected bundle dependency set")

    for name, item in records.items():
        relative = item.get("path")

        if not isinstance(relative, str):
            raise ValueError(f"Invalid bundle path: {name}")

        candidate = (bundle / relative).resolve()

        try:
            candidate.relative_to(bundle)
        except ValueError as exc:
            raise ValueError(
                f"Bundle path escapes root: {name}"
            ) from exc

        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        if (
            digest(candidate) != item.get("sha256")
            or candidate.stat().st_size != item.get("bytes")
        ):
            raise ValueError(
                f"Bundle integrity mismatch: {name}"
            )

    return {
        "status": "VERIFIED",
        "dependency_count": 10,
        "bundle_manifest_sha256": digest(manifest_path),
        "source_availability_certified": False,
        "prospective_performance_eligible": False,
    }


def create(root, snapshot, destination_parent):
    root = Path(root).resolve()
    parent = Path(destination_parent).resolve()

    # Existing fingerprint module verifies all required input paths.
    original = take(root, snapshot)

    records = original["dependencies"]

    # All copied files retain their original repository-relative paths.
    # The separate destination must not be nested in an input file path.
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(parent)

    parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".pending_input_bundle_", dir=parent
    ) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()

        for name, item in records.items():
            relative = item["path"]
            source = (root / relative).resolve()
            target = stage / relative

            target.parent.mkdir(
                parents=True, exist_ok=True
            )

            # Detect mutation during copying, not just afterward.
            if digest(source) != item["sha256"]:
                raise ValueError(
                    f"Source changed before copy: {name}"
                )

            shutil.copyfile(source, target)

            if (
                digest(target) != item["sha256"]
                or target.stat().st_size != item["bytes"]
            ):
                raise ValueError(
                    f"Copied input mismatch: {name}"
                )

        # Ensure the original inputs did not change during copying.
        verify_unchanged(root, original)

        manifest = {
            "schema": "cfb_v58_input_bundle_v1",
            "research_only": True,
            "created_at_utc":
                datetime.now(timezone.utc).isoformat(),
            "dependency_count": 10,
            "dependencies": records,
            "source_availability_certified": False,
            "historical_point_in_time_certified": False,
            "prospective_performance_eligible": False,
            "production_forecast_authorized": False,
        }

        (stage / MANIFEST).write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        verification = verify(stage)

        if verification["status"] != "VERIFIED":
            raise ValueError("Bundle staging verification failed")

        # Content-derived identity for the dependency set.
        identity = hashlib.sha256(
            json.dumps(
                records,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        destination = parent / ("bundle_" + identity)

        if destination.exists():
            raise FileExistsError(
                f"Identical bundle already exists: {destination}"
            )

        # Final source check before making bundle visible.
        verify_unchanged(root, original)

        # Windows scanners and filesystem filters can temporarily
        # deny a directory rename after large file copies.
        delays = (0, 0.25, 0.5, 1, 2, 4)

        for attempt, delay in enumerate(delays):
            if delay:
                time.sleep(delay)

            # Revalidate source identity after every wait.
            verify_unchanged(root, original)

            if destination.exists():
                raise FileExistsError(destination)

            try:
                stage.rename(destination)
                break
            except PermissionError as exc:
                if (
                    os.name != "nt"
                    or getattr(exc, "winerror", None) != 5
                    or attempt == len(delays) - 1
                ):
                    raise

        # A successful promotion must still be independently valid.
        if verify(destination)["status"] != "VERIFIED":
            raise ValueError("Promoted bundle verification failed")

    return destination
