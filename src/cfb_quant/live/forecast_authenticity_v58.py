"""Research-only checks for archived forecast authenticity."""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args, check=True):
    p = subprocess.run(
        ["git", *args], cwd=root,
        capture_output=True, text=True,
        encoding="utf-8", errors="replace"
    )
    if check and p.returncode:
        raise RuntimeError(
            f"Git {args} failed: {p.stderr.strip()}"
        )
    return p.returncode, p.stdout.strip()


def verify(root, csv, remote_head):
    root = Path(root).resolve()
    csv = Path(csv).resolve()
    relative = csv.relative_to(root).as_posix()

    frame = pd.read_csv(csv, dtype={"game_id": str})
    required = {
        "game_id", "kickoff_utc", "away", "home",
        "fair_home_spread", "fair_total",
        "blend_home_spread"
    }
    if not required <= set(frame):
        raise ValueError(f"Missing forecast columns: {relative}")
    if frame.empty or frame.game_id.duplicated().any():
        raise ValueError(f"Invalid forecast game IDs: {relative}")

    kickoffs = pd.to_datetime(
        frame.kickoff_utc, utc=True, errors="raise"
    )
    if kickoffs.isna().any():
        raise ValueError("Null forecast kickoff")

    earliest = kickoffs.min()
    errors = []
    warnings = []

    working_blob = git(root, "hash-object", "--", relative)[1]
    commits = git(
        root, "log", "--all", "--format=%H", "--", relative
    )[1].splitlines()

    matches = []
    for commit in commits:
        code, committed_blob = git(
            root, "rev-parse", f"{commit}:{relative}",
            check=False
        )
        if code != 0 or committed_blob != working_blob:
            continue

        code, _ = git(
            root, "merge-base", "--is-ancestor",
            commit, remote_head, check=False
        )
        if code != 0:
            continue

        commit_time = pd.to_datetime(
            git(root, "show", "-s", "--format=%cI", commit)[1],
            utc=True
        )
        matches.append({
            "commit": commit,
            "committer_utc": commit_time.isoformat(),
            "before_first_kickoff": bool(commit_time < earliest)
        })

    if not matches:
        errors.append("No exact matching remote-reachable Git blob")
    elif not any(x["before_first_kickoff"] for x in matches):
        errors.append("No matching pre-kickoff Git committer timestamp")

    manifest = None
    manifest_path = None
    for name in ("forecast_audit.json", "audit.json", "manifest.json"):
        candidate = csv.parent / name
        if candidate.exists():
            if manifest is not None:
                errors.append("Ambiguous: multiple audit manifests")
                break
            manifest_path = candidate
            manifest = json.loads(candidate.read_text(encoding="utf-8"))

    if manifest is None:
        errors.append("Missing audit manifest")
        manifest = {}

    if not isinstance(manifest, dict):
        errors.append("Audit manifest is not an object")
        manifest = {}

    actual_hash = sha256(csv)
    expected_hashes = []

    for field in ("forecast_sha256", "predictions_sha256"):
        if field in manifest:
            expected_hashes.append((field, manifest[field]))

    files = manifest.get("files")
    if isinstance(files, dict) and "paired_forecasts.csv" in files:
        expected_hashes.append((
            "files.paired_forecasts.csv",
            files["paired_forecasts.csv"]
        ))

    if not expected_hashes:
        errors.append("Manifest lacks forecast-content hash")
    else:
        for field, expected in expected_hashes:
            if str(expected).lower() != actual_hash.lower():
                errors.append(f"Manifest hash mismatch: {field}")

    claimed_times = []
    for field in (
        "snapshot_asof_utc", "asof_utc",
        "created_utc", "created_at_utc",
        "generated_utc", "prediction_created_utc"
    ):
        if field in manifest:
            ts = pd.to_datetime(
                manifest[field], utc=True, errors="coerce"
            )
            if pd.isna(ts):
                errors.append(f"Invalid audit timestamp: {field}")
            elif ts >= earliest:
                errors.append(
                    f"Audit timestamp not before kickoff: {field}"
                )
            else:
                claimed_times.append(field)

    if not claimed_times:
        errors.append("No valid pre-kickoff audit timestamp")

    if manifest.get("research_only") is not True:
        warnings.append("Research-only status not explicitly true")

    if manifest.get("historical_pit_provenance_certified") is not True:
        warnings.append("Historical PIT provenance not certified")

    # Remote ancestry is verifiable now, but historical first-push
    # time is not established by Git metadata.
    status = (
        "GIT_PREKICKOFF_SUPPORTED"
        if not errors else "AUTHENTICITY_UNVERIFIED"
    )

    return {
        "archive": relative,
        "games": len(frame),
        "csv_sha256": actual_hash,
        "first_kickoff_utc": earliest.isoformat(),
        "manifest": (
            manifest_path.relative_to(root).as_posix()
            if manifest_path else None
        ),
        "matching_commits": matches,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "externally_timestamp_verified": False,
        "historical_source_pit_certified": False
    }


def verify_all(root, archives):
    root = Path(root).resolve()
    remote = git(
        root, "ls-remote", "origin", "refs/heads/main"
    )[1]
    if not remote:
        raise RuntimeError("Remote main unavailable")

    remote_head = remote.split()[0]
    # Ensure local object database can resolve the remote hash.
    git(root, "cat-file", "-e", f"{remote_head}^{{commit}}")

    return {
        "remote_head": remote_head,
        "results": [
            verify(root, csv, remote_head)
            for csv in archives
        ]
    }
