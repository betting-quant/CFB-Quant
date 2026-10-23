"""Tests for execution code provenance."""

import pytest

from cfb_quant.live import execution_provenance_v58 as prov


def repository(tmp_path):
    root = tmp_path / "repo"
    source = root / "src/cfb_quant/models"
    source.mkdir(parents=True)

    (source / "model.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )

    (root / "pyproject.toml").write_text(
        "[project]\nname='example'\nversion='0.1'\n",
        encoding="utf-8",
    )

    return root


def test_code_identity_is_deterministic(tmp_path):
    root = repository(tmp_path)

    first = prov.source_files(root)
    second = prov.source_files(root)

    assert first == second
    assert prov.identity(first) == prov.identity(second)


def test_modified_model_changes_identity(tmp_path):
    root = repository(tmp_path)
    original = prov.source_files(root)

    model = root / "src/cfb_quant/models/model.py"
    model.write_text("VALUE = 2\n", encoding="utf-8")

    changed = prov.source_files(root)

    assert prov.identity(original) != prov.identity(changed)


def test_verification_detects_mutation(tmp_path):
    root = repository(tmp_path)
    files = prov.source_files(root)

    manifest = {
        "schema": prov.SCHEMA,
        "source_files": files,
        "source_file_count": len(files),
        "code_identity": prov.identity(files),
    }

    assert prov.verify_sources(
        root, manifest
    )["status"] == "UNCHANGED"

    (root / "src/cfb_quant/models/model.py").write_text(
        "VALUE = 3\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="Source identity changed"
    ):
        prov.verify_sources(root, manifest)


def test_new_source_file_is_detected(tmp_path):
    root = repository(tmp_path)
    files = prov.source_files(root)

    manifest = {
        "schema": prov.SCHEMA,
        "source_files": files,
        "source_file_count": len(files),
        "code_identity": prov.identity(files),
    }

    (root / "src/cfb_quant/models/added.py").write_text(
        "NEW = True\n", encoding="utf-8"
    )

    with pytest.raises(
        ValueError, match="Source file set changed"
    ):
        prov.verify_sources(root, manifest)
