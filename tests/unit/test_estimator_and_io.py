"""Unit tests for `estimator.py` and `io_utils.py`."""

from pathlib import Path
import struct

import numpy as np
import pytest

from matrix_compute_mcp.estimator import estimate_job_cost
from matrix_compute_mcp.io_utils import (
    load_matrix_input,
    load_safetensors_bytes,
    load_safetensors_file,
    save_matrix_output,
    save_safetensors_bytes,
    save_safetensors_file,
)


def test_estimate_job_cost_strategies():
    # 1. OUTPUT_GRID_2D: download optimal (zero download redundancy)
    est_2d = estimate_job_cost(
        m=2048,
        k=2048,
        n=2048,
        dtype="fp32",
        partitioning_strategy="output_grid_2d",
        grid_rows=2,
        grid_cols=2,
        sats_per_gflop=0.5,
    )
    assert est_2d.num_tasks == 4
    assert est_2d.is_download_optimal is True
    assert est_2d.download_redundancy_ratio == 0.0
    assert est_2d.total_flops == 2 * 2048 * 2048 * 2048
    assert est_2d.estimated_total_sats == (
        est_2d.estimated_worker_bounty_sats + est_2d.estimated_platform_fee_sats
    )

    # 2. OUTER_PRODUCT_1D: upload optimal (zero upload redundancy)
    est_1d = estimate_job_cost(
        m=2048,
        k=2048,
        n=2048,
        dtype="fp16",
        partitioning_strategy="outer_product_1d",
        num_strips=4,
        sats_per_gflop=0.5,
    )
    assert est_1d.num_tasks == 4
    assert est_1d.is_upload_optimal is True
    assert est_1d.upload_redundancy_ratio == 0.0
    assert est_1d.bytes_per_element == 2

    # 3. BLOCK_3D
    est_3d = estimate_job_cost(
        m=2048,
        k=2048,
        n=2048,
        dtype="bf16",
        partitioning_strategy="block_3d",
        tile_size=1024,
        sats_per_gflop=0.5,
    )
    assert est_3d.num_tasks == 8
    assert est_3d.upload_redundancy_ratio > 0.0
    assert est_3d.download_redundancy_ratio > 0.0


def test_safetensors_roundtrip_and_io(tmp_path: Path):
    rng = np.random.RandomState(401)
    mat_a = rng.randn(10, 12).astype(np.float32)
    mat_b = rng.randn(12, 8).astype(np.float16)

    st_path = tmp_path / "weights.safetensors"
    save_safetensors_file({"a": mat_a, "b": mat_b}, st_path, metadata={"source": "unit_test"})

    loaded_dict = load_safetensors_file(st_path)
    np.testing.assert_allclose(loaded_dict["a"], mat_a)
    np.testing.assert_allclose(loaded_dict["b"], mat_b)

    # Test load_matrix_input with tensor_key
    a_loaded = load_matrix_input(st_path, tensor_key="a")
    b_loaded = load_matrix_input(st_path, tensor_key="b", dtype=np.float32)
    np.testing.assert_allclose(a_loaded, mat_a)
    np.testing.assert_allclose(b_loaded, mat_b.astype(np.float32))

    # Test .npy and .npz roundtrip
    npy_path = tmp_path / "out.npy"
    save_matrix_output(mat_a, npy_path)
    np.testing.assert_allclose(load_matrix_input(npy_path), mat_a)

    npz_path = tmp_path / "out.npz"
    save_matrix_output(mat_a, npz_path, tensor_key="custom_key")
    np.testing.assert_allclose(load_matrix_input(npz_path, tensor_key="custom_key"), mat_a)


def test_adversarial_safetensors_parsing():
    with pytest.raises(ValueError, match="fewer than 8 bytes"):
        load_safetensors_bytes(b"short")

    # Header length larger than payload
    bad_len = struct.pack("<Q", 5000) + b"{}"
    with pytest.raises(ValueError, match="truncated JSON header"):
        load_safetensors_bytes(bad_len)
