from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from community_graphs.config import DatasetConfig
from community_graphs.csr import validate_csr_matrix
from community_graphs.generation import generate_dataset
from community_graphs.hashing import csr_structure_sha256, file_sha256
from community_graphs.sigl_loader import (
    densify_graph,
    load_community_graphs_for_sigl,
    load_community_graphs_sparse,
)
from community_graphs.validation import ValidationError, validate_dataset_dir


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, **overrides) -> DatasetConfig:
    data = dict(
        family="erdos-renyi",
        instances=12,
        nodes=14,
        params={"p": 0.25},
        train_ratio=0.75,
        seed=123,
        output_dir=tmp_path,
        overwrite=True,
    )
    data.update(overrides)
    return DatasetConfig(**data)


def _generate(tmp_path: Path, **overrides) -> Path:
    generate_dataset(_config(tmp_path, **overrides), command=["test"])
    return tmp_path


def _load(path: Path, name: str):
    with (path / name).open("rb") as file:
        return pickle.load(file)


def test_pickles_contain_direct_lists_of_csr(tmp_path):
    dataset = _generate(tmp_path)
    train = _load(dataset, "CommunityGraphs_train.pkl")
    test = _load(dataset, "CommunityGraphs_test.pkl")
    assert isinstance(train, list)
    assert isinstance(test, list)
    assert all(type(graph) is sparse.csr_matrix for graph in train + test)


def test_matrices_are_square_binary_symmetric_without_self_loops_and_nnz_invariant(tmp_path):
    dataset = _generate(tmp_path)
    manifest = list(__import__("csv").DictReader((dataset / "manifest.csv").open()))
    graphs = _load(dataset, "CommunityGraphs_train.pkl") + _load(dataset, "CommunityGraphs_test.pkl")
    for graph in graphs:
        assert graph.shape[0] == graph.shape[1]
        assert set(graph.data.tolist()) <= {1}
        assert (graph != graph.T).nnz == 0
        assert not bool(np.any(graph.diagonal() != 0))
    for row in manifest:
        assert int(row["nnz"]) == 2 * int(row["num_edges"])


def test_train_test_counts_and_no_cross_split_duplicates(tmp_path):
    dataset = _generate(tmp_path, instances=10, train_ratio=0.8)
    train = _load(dataset, "CommunityGraphs_train.pkl")
    test = _load(dataset, "CommunityGraphs_test.pkl")
    assert len(train) == 8
    assert len(test) == 2
    assert not ({csr_structure_sha256(g) for g in train} & {csr_structure_sha256(g) for g in test})


def test_same_configuration_produces_same_hashes(tmp_path):
    a = _generate(tmp_path / "a", seed=77)
    b = _generate(tmp_path / "b", seed=77)
    assert [csr_structure_sha256(g) for g in _load(a, "CommunityGraphs_train.pkl")] == [
        csr_structure_sha256(g) for g in _load(b, "CommunityGraphs_train.pkl")
    ]


def test_worker_count_does_not_change_logical_pickles(tmp_path):
    a = _generate(tmp_path / "a", seed=77, workers=1)
    b = _generate(tmp_path / "b", seed=77, workers=4)
    assert [csr_structure_sha256(g) for g in _load(a, "CommunityGraphs_test.pkl")] == [
        csr_structure_sha256(g) for g in _load(b, "CommunityGraphs_test.pkl")
    ]


def test_different_seed_changes_at_least_one_structure(tmp_path):
    a = _generate(tmp_path / "a", seed=77)
    b = _generate(tmp_path / "b", seed=78)
    assert [csr_structure_sha256(g) for g in _load(a, "CommunityGraphs_train.pkl")] != [
        csr_structure_sha256(g) for g in _load(b, "CommunityGraphs_train.pkl")
    ]


def test_dtype_survives_pickle_roundtrip_and_indices_are_valid(tmp_path):
    dataset = _generate(tmp_path, dtype="uint8")
    for graph in _load(dataset, "CommunityGraphs_train.pkl"):
        assert graph.dtype == np.dtype("uint8")
        assert graph.has_sorted_indices
        assert graph.indptr[0] == 0
        assert graph.indptr[-1] == len(graph.indices)
        assert graph.indices.size == 0 or graph.indices.max() < graph.shape[1]


def test_sparse_loader_does_not_densify(tmp_path, monkeypatch):
    dataset = _generate(tmp_path)

    def fail_toarray(self, *args, **kwargs):
        raise AssertionError("toarray should not be called by sparse loader")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", fail_toarray)
    graphs = load_community_graphs_sparse(dataset / "CommunityGraphs_train.pkl")
    assert all(isinstance(graph, sparse.csr_matrix) for graph in graphs)


def test_sigl_loader_returns_float32_square_c_contiguous_arrays(tmp_path):
    dataset = _generate(tmp_path)
    arrays = load_community_graphs_for_sigl(dataset / "CommunityGraphs_train.pkl")
    assert isinstance(arrays, list)
    assert all(isinstance(array, np.ndarray) for array in arrays)
    assert all(array.dtype == np.float32 for array in arrays)
    assert all(array.shape[0] == array.shape[1] for array in arrays)
    assert all(array.flags["C_CONTIGUOUS"] for array in arrays)


def test_dense_memory_limit_blocks_large_conversion():
    graph = sparse.eye(10, format="csr", dtype=np.uint8)
    with pytest.raises(MemoryError):
        densify_graph(graph, dtype=np.float32, max_dense_bytes=10)


def test_explicit_zero_can_be_normalized_but_loader_rejects_it(tmp_path):
    graph = sparse.csr_matrix(([1, 0], [1, 2], [0, 2, 2, 2]), shape=(3, 3), dtype=np.uint8)
    normalized = validate_csr_matrix(graph, normalize_explicit_zeros=True)
    assert normalized.nnz == 1
    path = tmp_path / "bad.pkl"
    with path.open("wb") as file:
        pickle.dump([graph], file)
    with pytest.raises(ValueError):
        load_community_graphs_sparse(path)


def test_corrupt_csr_is_rejected():
    graph = sparse.csr_matrix(([1, 1], [2, 1], [0, 2, 2, 2]), shape=(3, 3), dtype=np.uint8)
    assert not graph.has_sorted_indices
    with pytest.raises(ValueError):
        validate_csr_matrix(graph)


def test_validator_detects_metadata_count_mismatch(tmp_path):
    dataset = _generate(tmp_path)
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["splits"]["train_instances"] += 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError):
        validate_dataset_dir(dataset, quiet=True)


def test_validator_detects_cross_split_duplicates(tmp_path):
    dataset = _generate(tmp_path)
    train = _load(dataset, "CommunityGraphs_train.pkl")
    test = _load(dataset, "CommunityGraphs_test.pkl")
    test[0] = train[0].copy()
    with (dataset / "CommunityGraphs_test.pkl").open("wb") as file:
        pickle.dump(test, file, protocol=pickle.HIGHEST_PROTOCOL)
    metadata_path = dataset / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["hashes"]["CommunityGraphs_test.pkl"] = file_sha256(dataset / "CommunityGraphs_test.pkl")
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError, match="Cross-split duplicates"):
        validate_dataset_dir(dataset, quiet=True)


def test_variable_size_mode_preserves_different_shapes(tmp_path):
    dataset = _generate(
        tmp_path,
        instances=8,
        train_ratio=0.5,
        size_mode="linspace",
        nodes=None,
        nodes_min=8,
        nodes_max=16,
    )
    shapes = {graph.shape for graph in _load(dataset, "CommunityGraphs_train.pkl") + _load(dataset, "CommunityGraphs_test.pkl")}
    assert len(shapes) > 1


def test_dry_run_does_not_create_pickles(tmp_path):
    result = generate_dataset(_config(tmp_path, dry_run=True, overwrite=False), command=["test"])
    assert result["dry_run"] is True
    assert not (tmp_path / "CommunityGraphs_train.pkl").exists()


def test_overwrite_required_for_non_empty_output(tmp_path):
    _generate(tmp_path)
    with pytest.raises(FileExistsError):
        generate_dataset(_config(tmp_path, overwrite=False), command=["test"])


def test_explicit_split_mode(tmp_path):
    dataset = _generate(tmp_path, instances=None, train_ratio=None, train_instances=5, test_instances=4)
    assert len(_load(dataset, "CommunityGraphs_train.pkl")) == 5
    assert len(_load(dataset, "CommunityGraphs_test.pkl")) == 4


def test_prepare_sigl_input_script(tmp_path):
    dataset = _generate(tmp_path / "dataset")
    output = tmp_path / "dense.pkl"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "prepare_sigl_input.py"),
            "--input",
            str(dataset / "CommunityGraphs_train.pkl"),
            "--output",
            str(output),
            "--max-dense-mib",
            "1",
        ],
        check=True,
    )
    with output.open("rb") as file:
        arrays = pickle.load(file)
    assert isinstance(arrays, list)
    assert all(isinstance(array, np.ndarray) for array in arrays)


def test_stochastic_block_model_generation(tmp_path):
    dataset = _generate(
        tmp_path,
        family="stochastic-block-model",
        nodes=12,
        params={"sizes": [4, 4, 4], "probabilities": [[0.5, 0.05, 0.05], [0.05, 0.5, 0.05], [0.05, 0.05, 0.5]]},
    )
    assert validate_dataset_dir(dataset, quiet=True)["train_graphs"] == 9
