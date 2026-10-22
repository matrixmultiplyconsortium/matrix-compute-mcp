"""Unit tests for `matrix_compute_mcp.ai.similarity`."""

import numpy as np
import pytest

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig
from matrix_compute_mcp.ai import (
    SimilaritySearchResult,
    distributed_cosine_similarity,
    l2_normalize_rows,
)


@pytest.fixture(scope="module")
def shared_client():
    cfg = ClientConfig.load(offline_mock_mode=True, default_tile_size=16)
    with MatrixComputeClient(config=cfg) as client:
        yield client


def _reference_cosine_similarity(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    q_norm = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    c_norm = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return q_norm @ c_norm.T


def test_full_cosine_similarity_matrix(shared_client):
    rng = np.random.RandomState(101)
    queries = rng.randn(12, 32).astype(np.float32)
    corpus = rng.randn(40, 32).astype(np.float32)

    sim = distributed_cosine_similarity(queries, corpus, client=shared_client)
    expected = _reference_cosine_similarity(queries, corpus)

    assert sim.shape == (12, 40)
    np.testing.assert_allclose(sim, expected, rtol=1e-4, atol=1e-5)


def test_single_1d_query_vector(shared_client):
    rng = np.random.RandomState(102)
    query_1d = rng.randn(24).astype(np.float32)
    corpus = rng.randn(18, 24).astype(np.float32)

    sim = distributed_cosine_similarity(query_1d, corpus, client=shared_client)
    expected = _reference_cosine_similarity(query_1d.reshape(1, -1), corpus)[0]

    assert sim.shape == (18,)
    np.testing.assert_allclose(sim, expected, rtol=1e-4, atol=1e-5)


def test_top_k_similarity_search(shared_client):
    rng = np.random.RandomState(103)
    queries = rng.randn(8, 16).astype(np.float32)
    corpus = rng.randn(25, 16).astype(np.float32)

    scores, indices = distributed_cosine_similarity(
        queries, corpus, top_k=5, client=shared_client
    )
    expected_full = _reference_cosine_similarity(queries, corpus)
    expected_indices = np.argsort(-expected_full, axis=1)[:, :5]
    expected_scores = np.take_along_axis(expected_full, expected_indices, axis=1)

    assert scores.shape == (8, 5)
    assert indices.shape == (8, 5)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-4, atol=1e-5)


def test_top_k_result_object(shared_client):
    rng = np.random.RandomState(104)
    queries = rng.randn(4, 16).astype(np.float32)
    corpus = rng.randn(10, 16).astype(np.float32)

    res = distributed_cosine_similarity(
        queries,
        corpus,
        top_k=3,
        client=shared_client,
        return_result_object=True,
    )
    assert isinstance(res, SimilaritySearchResult)
    d = res.to_dict()
    assert d["num_queries"] == 4
    assert d["top_k"] == 3
    assert len(d["scores"]) == 4
    assert len(d["indices"][0]) == 3


def test_zero_norm_vector_handling(shared_client):
    queries = np.zeros((3, 8), dtype=np.float32)
    queries[1] = np.ones(8, dtype=np.float32)
    corpus = np.ones((5, 8), dtype=np.float32)
    corpus[2] = 0.0

    norm_q = l2_normalize_rows(queries)
    assert not np.any(np.isnan(norm_q))
    np.testing.assert_allclose(norm_q[0], 0.0)

    sim = distributed_cosine_similarity(queries, corpus, client=shared_client)
    assert not np.any(np.isnan(sim))
    np.testing.assert_allclose(sim[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(sim[1, 0], 1.0, atol=1e-5)
    np.testing.assert_allclose(sim[1, 2], 0.0, atol=1e-6)


def test_similarity_dimension_validation(shared_client):
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        distributed_cosine_similarity(
            np.ones((4, 8), dtype=np.float32),
            np.ones((6, 12), dtype=np.float32),
            client=shared_client,
        )

    with pytest.raises(ValueError, match="top_k"):
        distributed_cosine_similarity(
            np.ones((4, 8), dtype=np.float32),
            np.ones((6, 8), dtype=np.float32),
            top_k=10,
            client=shared_client,
        )
