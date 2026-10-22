"""Unit tests for `distributed_pca_svd` (`matrix_compute_mcp.ai.spectral`)."""

import numpy as np
import pytest

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig
from matrix_compute_mcp.ai import PCASVDResult, distributed_pca_svd


@pytest.fixture(scope="module")
def shared_client():
    cfg = ClientConfig.load(offline_mock_mode=True, default_tile_size=16)
    with MatrixComputeClient(config=cfg) as client:
        yield client


def test_randomized_pca_svd_low_rank_recovery(shared_client):
    rng = np.random.RandomState(301)
    # Construct synthetic rank-3 matrix in R^{30 x 20} with distinct singular values
    u_true, _ = np.linalg.qr(rng.randn(30, 3).astype(np.float32))
    v_true, _ = np.linalg.qr(rng.randn(20, 3).astype(np.float32))
    s_true = np.array([25.0, 12.0, 5.0], dtype=np.float32)
    mat = (u_true * s_true) @ v_true.T

    res = distributed_pca_svd(
        mat,
        k=3,
        center=False,
        method="randomized",
        oversampling=6,
        power_iters=2,
        client=shared_client,
    )
    assert isinstance(res, PCASVDResult)

    # Verify tuple unpacking
    u, s, vt = res
    assert u.shape == (30, 3)
    assert s.shape == (3,)
    assert vt.shape == (3, 20)

    np.testing.assert_allclose(s, s_true, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res.reconstruct(), mat, rtol=1e-3, atol=1e-3)


def test_centered_pca_explained_variance_ratio(shared_client):
    rng = np.random.RandomState(302)
    # Create dataset where first 2 components capture >99% variance
    latent = rng.randn(40, 2).astype(np.float32) * np.array([10.0, 4.0], dtype=np.float32)
    proj = rng.randn(2, 16).astype(np.float32)
    offset = np.linspace(-5.0, 5.0, 16, dtype=np.float32)
    data = latent @ proj + offset

    res = distributed_pca_svd(
        data,
        k=2,
        center=True,
        method="randomized",
        client=shared_client,
    )
    assert res.mean is not None
    assert res.explained_variance_ratio.shape == (2,)
    assert float(np.sum(res.explained_variance_ratio)) > 0.99
    np.testing.assert_allclose(res.reconstruct(), data, rtol=1e-3, atol=1e-3)


def test_stateful_power_iteration_pca_svd(shared_client):
    rng = np.random.RandomState(303)
    u_true, _ = np.linalg.qr(rng.randn(24, 2).astype(np.float32))
    v_true, _ = np.linalg.qr(rng.randn(16, 2).astype(np.float32))
    s_true = np.array([20.0, 6.0], dtype=np.float32)
    mat = (u_true * s_true) @ v_true.T

    res = distributed_pca_svd(
        mat,
        k=2,
        center=False,
        method="power_iteration",
        iterations=35,
        spot_check_probability=0.5,
        client=shared_client,
    )
    np.testing.assert_allclose(res.S, s_true, rtol=1e-2, atol=5e-2)
    np.testing.assert_allclose(res.reconstruct(), mat, rtol=2e-2, atol=5e-2)


def test_pca_svd_validation_errors(shared_client):
    mat = np.ones((8, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="k must satisfy"):
        distributed_pca_svd(mat, k=0, client=shared_client)
    with pytest.raises(ValueError, match="k must satisfy"):
        distributed_pca_svd(mat, k=7, client=shared_client)
    with pytest.raises(ValueError, match="Unsupported method"):
        distributed_pca_svd(mat, k=2, method="invalid_method", client=shared_client)
