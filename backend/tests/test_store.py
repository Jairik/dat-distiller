"""Domain store: immutability, the version tree, Provenance survival."""

from __future__ import annotations

import pytest

from dat_distiller.store import DatasetStore, Provenance
from dat_distiller.store.paths import AppPaths
from dat_distiller.store.provenance import PROVENANCE_COLUMN
from dat_distiller.store.store import (
    DatasetVersionNotFoundError,
    DuplicateProjectError,
    ParentNotFoundError,
    ProjectNotFoundError,
)
import pandas as pd


@pytest.fixture
def store(isolated_data_dir) -> DatasetStore:
    return DatasetStore(AppPaths(root=isolated_data_dir))


def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})


def test_project_crud_and_duplicates(store: DatasetStore) -> None:
    project = store.create_project("Churn")
    assert store.list_projects() == [project]
    assert store.get_project(project.id).name == "Churn"
    with pytest.raises(DuplicateProjectError):
        store.create_project("Churn")
    store.delete_project(project.id)
    assert store.list_projects() == []
    with pytest.raises(ProjectNotFoundError):
        store.get_project(project.id)


def test_versions_form_a_tree_with_branching(store: DatasetStore) -> None:
    project = store.create_project("Tree")
    root = store.create_version(project.id, sample_df())
    left = store.create_version(project.id, sample_df(), parent_id=root.id)
    right = store.create_version(project.id, sample_df(), parent_id=root.id)
    versions = store.list_versions(project.id)
    assert [v.parent_id for v in versions] == [None, root.id, root.id]
    assert [v.id for v in versions] == [root.id, left.id, right.id]
    assert [v.number for v in versions] == [1, 2, 3]  # monotonic per Project
    with pytest.raises(ParentNotFoundError):
        store.create_version(project.id, sample_df(), parent_id="nope")


def test_versions_are_immutable_files(store: DatasetStore) -> None:
    project = store.create_project("Immutable")
    version = store.create_version(project.id, sample_df())
    path = store.paths.version_file(version.id)
    before = path.read_bytes()
    # Reading through the API surface never rewrites the file.
    store.load_dataframe(version.id)
    store.get_version(version.id)
    assert path.read_bytes() == before
    # A new version is a *new file*, never an overwrite.
    child = store.create_version(project.id, sample_df(), parent_id=version.id)
    assert child.id != version.id
    assert path.read_bytes() == before


def test_provenance_survives_derivation_and_queries_as_breakdown(
    store: DatasetStore,
) -> None:
    project = store.create_project("Prov")
    root = store.create_version(
        project.id,
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        provenance=[
            Provenance.row("uploaded"),
            Provenance.row("synthetic", provider="openrouter", model="m", mode="hybrid", seed=7),
        ],
    )
    assert root.provenance_summary == {
        "uploaded": 1,
        "synthetic": 1,
        "jev": 0,
        "human_reviewed": 0,
    }
    assert root.seed is None  # seed was per-row Provenance here

    # Labeling derivation: rows keep their row Provenance, labels get `jev`.
    labeled = store.load_dataframe(root.id, include_provenance=True)
    labeled["spam"] = [True, False]
    child = store.create_version(
        project.id,
        labeled,
        parent_id=root.id,
        origin="labeled",
        provenance=[
            {"label_origins": {"spam": {"origin": "jev", "confidence": 0.9}}},
            {"label_origins": {"spam": {"origin": "jev", "confidence": 0.4}}},
        ],
    )
    # 1 jev (low confidence), 1 still synthetic (its jev label was... also jev)
    assert child.provenance_summary["jev"] == 2
    frame = store.load_dataframe(child.id, include_provenance=True)
    first = Provenance.parse(frame[PROVENANCE_COLUMN].iloc[0])
    assert first["row_origin"] == "uploaded"  # survived the derivation
    assert first["label_origins"]["spam"]["origin"] == "jev"

    # A human override moves the row to the human_reviewed bucket.
    frame2 = store.load_dataframe(child.id, include_provenance=True)
    reviewed = store.create_version(
        project.id,
        frame2,
        parent_id=child.id,
        origin="reviewed",
        provenance=[
            {"label_origins": {"spam": {"origin": "human_reviewed", "confidence": 1.0}}},
            {},
        ],
    )
    assert reviewed.provenance_summary == {
        "uploaded": 0,
        "synthetic": 0,
        "jev": 1,
        "human_reviewed": 1,
    }


def test_load_dataframe_hides_provenance_by_default(store: DatasetStore) -> None:
    project = store.create_project("Hide")
    version = store.create_version(project.id, sample_df())
    df = store.load_dataframe(version.id)
    assert list(df.columns) == ["a", "b"]
    assert PROVENANCE_COLUMN not in store.get_version(version.id).__dict__
    with pytest.raises(DatasetVersionNotFoundError):
        store.load_dataframe("nope")


def test_delete_project_removes_data_files(store: DatasetStore) -> None:
    project = store.create_project("Delete me")
    version = store.create_version(project.id, sample_df())
    path = store.paths.version_file(version.id)
    assert path.exists()
    store.delete_project(project.id)
    assert not path.exists()
    with pytest.raises(DatasetVersionNotFoundError):
        store.get_version(version.id)
