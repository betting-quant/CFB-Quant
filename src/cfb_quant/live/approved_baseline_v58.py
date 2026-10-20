"""Explicit, hash-pinned research schedule baseline approvals."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cfb_quant.live.daily_schedule_sync_v58 import (
    index_games, validate,
)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approve(snapshot, registry, season, reason):
    snapshot = Path(snapshot).resolve()
    registry = Path(registry).resolve()

    if not reason or not reason.strip():
        raise ValueError("An explicit approval reason is required")
    if registry.exists():
        raise FileExistsError(
            "Approval already exists. Refusing automatic replacement."
        )
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    raw = json.loads(snapshot.read_text(encoding="utf-8"))
    games = validate(raw, season, len(raw))
    if not games:
        raise ValueError("Empty schedule cannot be approved")

    digest = fingerprint(snapshot)

    record = {
        "schema": "approved_baseline_v58_v1",
        "research_only": True,
        "season": season,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_reason": reason.strip(),
        "snapshot_path": str(snapshot),
        "snapshot_sha256": digest,
        "season_games": len(games),
        "independent_schedule_completeness_certified": False,
        "auto_rotation_allowed": False,
    }

    registry.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(registry), flags, 0o600)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    except BaseException:
        registry.unlink(missing_ok=True)
        raise

    return record


def verify(registry, expected_season=None):
    registry = Path(registry).resolve()
    record = json.loads(registry.read_text(encoding="utf-8"))

    if record.get("schema") != "approved_baseline_v58_v1":
        raise ValueError("Unsupported approval schema")
    if record.get("auto_rotation_allowed") is not False:
        raise ValueError("Unsafe automatic baseline rotation setting")
    if record.get("research_only") is not True:
        raise ValueError("Unexpected approval scope")

    season = int(record["season"])
    if expected_season is not None and season != expected_season:
        raise ValueError("Approved baseline season mismatch")

    snapshot = Path(record["snapshot_path"]).resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    actual = fingerprint(snapshot)
    if actual != record["snapshot_sha256"]:
        raise RuntimeError("Approved schedule SHA256 mismatch")

    games = validate(
        json.loads(snapshot.read_text(encoding="utf-8")),
        season,
        int(record["season_games"]),
    )
    if len(games) != int(record["season_games"]):
        raise RuntimeError("Approved schedule game count mismatch")

    return snapshot, record



def verify_portable(registry, expected_season=None):
    """Verify a portable, hash-pinned research schedule."""
    registry = Path(registry).resolve()
    record = json.loads(registry.read_text(encoding="utf-8"))

    if not isinstance(record, dict):
        raise ValueError("Portable approval must be an object")
    if record.get("schema") != "portable_baseline_v58_v1":
        raise ValueError("Unsupported portable approval schema")
    if record.get("research_only") is not True:
        raise ValueError("Unexpected approval scope")
    if record.get("auto_rotation_allowed") is not False:
        raise ValueError("Unsafe automatic baseline rotation")
    if record.get("derived_from_original_approval") is not True:
        raise ValueError("Missing original approval provenance")
    if record.get("schedule_file") != "schedule.json":
        raise ValueError("Unsafe portable schedule path")

    season = int(record["season"])
    if expected_season is not None and season != expected_season:
        raise ValueError("Approved baseline season mismatch")

    snapshot = (registry.parent / "schedule.json").resolve()
    if snapshot.parent != registry.parent:
        raise ValueError("Portable schedule escapes approval directory")
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    digest = record.get("snapshot_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest.lower())
    ):
        raise ValueError("Invalid approved SHA256")

    if fingerprint(snapshot) != digest.lower():
        raise RuntimeError("Approved schedule SHA256 mismatch")

    count = int(record["season_games"])
    games = validate(
        json.loads(snapshot.read_text(encoding="utf-8")),
        season,
        count,
    )
    if len(games) != count:
        raise RuntimeError("Approved schedule game count mismatch")

    return snapshot, record


def verify_any(registry, expected_season=None):
    """Verify either legacy or portable schedule approval."""
    registry = Path(registry).resolve()
    record = json.loads(registry.read_text(encoding="utf-8"))

    if not isinstance(record, dict):
        raise ValueError("Approval record must be an object")

    schema = record.get("schema")
    if schema == "approved_baseline_v58_v1":
        return verify(registry, expected_season)
    if schema == "portable_baseline_v58_v1":
        return verify_portable(registry, expected_season)

    raise ValueError("Unsupported approval schema")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--approve", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--reason")
    args = parser.parse_args()

    if args.approve:
        if args.snapshot is None:
            parser.error("--approve requires --snapshot")
        if not args.reason:
            parser.error("--approve requires --reason")
        approve(args.snapshot, args.registry, args.season, args.reason)

    snapshot, record = verify_any(args.registry, args.season)
    print("APPROVED BASELINE: VERIFIED")
    print("Season:", record["season"])
    print("Games:", record["season_games"])
    print("Snapshot:", snapshot)
    print("SHA256:", record["snapshot_sha256"])
    if record["schema"] == "portable_baseline_v58_v1":
        print("Original approval:", record["original_approval_utc"])
        print("Approval type: PORTABLE")
    else:
        print("Approval:", record["approved_at_utc"])
        print("Approval type: LEGACY")
    print("Automatic rotation: DISABLED")


if __name__ == "__main__":
    main()
