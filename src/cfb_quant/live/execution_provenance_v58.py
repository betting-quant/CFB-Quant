"""Research execution source and environment provenance."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


SCHEMA = "cfb_v58_execution_provenance_v1"

PACKAGES = (
    "numpy", "pandas", "scipy",
    "scikit-learn", "pyarrow", "joblib",
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def identity(obj):
    encoded = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_files(root):
    root = Path(root).resolve()
    source = root / "src/cfb_quant"

    if not source.is_dir():
        raise FileNotFoundError(source)

    paths = list(source.rglob("*.py"))

    for name in (
        "pyproject.toml",
        "uv.lock",
        "poetry.lock",
        "requirements.txt",
        "requirements.lock",
    ):
        path = root / name
        if path.is_file():
            paths.append(path)

    if not paths:
        raise ValueError("No Python source files found")

    records = {}

    for path in sorted(set(paths)):
        if path.is_symlink():
            raise ValueError(f"Symlink in source tree: {path}")

        resolved = path.resolve()
        relative = resolved.relative_to(root).as_posix()

        records[relative] = {
            "sha256": sha256(resolved),
            "bytes": resolved.stat().st_size,
        }

    return records


def environment():
    packages = {}

    for package in PACKAGES:
        try:
            packages[package] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    return {
        "python_version": sys.version,
        "python_implementation":
            platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def git_state(root):
    root = Path(root).resolve()

    def run(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: {' '.join(args)}: "
                f"{result.stderr.strip()}"
            )
        return result.stdout

    commit = run("rev-parse", "HEAD").strip()
    status = run(
        "status", "--porcelain=v1", "-uno"
    )
    untracked = run(
        "ls-files", "--others", "--exclude-standard"
    ).splitlines()

    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(status.strip()),
        "untracked_file_count": len(untracked),
    }


def capture(root, *, bundle_identity=None, samples=None, seed=None):
    """Capture source identity, runtime and execution settings."""
    root = Path(root).resolve()

    if bundle_identity is not None:
        if (
            not isinstance(bundle_identity, str)
            or len(bundle_identity) != 64
            or any(c not in "0123456789abcdef"
                   for c in bundle_identity)
        ):
            raise ValueError("Invalid bundle identity")

    records = source_files(root)
    runtime = environment()

    configuration = {
        "bundle_identity": bundle_identity,
        "samples": samples,
        "seed": seed,
    }

    result = {
        "schema": SCHEMA,
        "research_only": True,
        "source_files": records,
        "source_file_count": len(records),
        "code_identity": identity(records),
        "environment": runtime,
        "environment_identity": identity(runtime),
        "git": git_state(root),
        "configuration": configuration,
        "execution_identity": identity({
            "code_identity": identity(records),
            "environment_identity": identity(runtime),
            "configuration": configuration,
        }),
        "historical_point_in_time_certified": False,
        "source_availability_certified": False,
        "prospective_performance_eligible": False,
        "production_forecast_authorized": False,
    }

    return result


def verify_sources(root, manifest):
    """Reject altered, missing or newly added Python source files."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Invalid provenance schema")

    expected = manifest.get("source_files")

    if not isinstance(expected, dict):
        raise ValueError("Invalid provenance source records")

    current = source_files(root)

    if set(expected) != set(current):
        raise ValueError("Source file set changed")

    if current != expected:
        changes = [
            name for name in sorted(current)
            if current[name] != expected[name]
        ]
        raise ValueError(
            "Source identity changed: " + ", ".join(changes[:5])
        )

    if manifest.get("source_file_count") != len(current):
        raise ValueError("Source count mismatch")

    if manifest.get("code_identity") != identity(current):
        raise ValueError("Code identity mismatch")

    return {
        "status": "UNCHANGED",
        "source_file_count": len(current),
        "code_identity": identity(current),
    }



def verify_environment(manifest):
    """Reject changes to the recorded execution environment."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Invalid provenance schema")

    recorded = manifest.get("environment")
    recorded_identity = manifest.get("environment_identity")

    if not isinstance(recorded, dict):
        raise ValueError("Missing execution environment")

    if recorded_identity != identity(recorded):
        raise ValueError("Environment manifest identity mismatch")

    current = environment()

    if current != recorded:
        raise ValueError("Execution environment changed")

    return {
        "status": "UNCHANGED",
        "environment_identity": identity(current),
    }
