"""
Distributed PCA & Truncated SVD (`matrix_compute_mcp.ai.spectral`).

Executes verified randomized SVD (Halko-Martinsson-Tropp subspace iteration) and
stateful power-iteration eigen/singular decomposition on remote worker VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig, PartitioningStrategy


@dataclass
class PCASVDResult:
    """
    Structured output of `distributed_pca_svd`.

    Supports both attribute access (`res.U`, `res.S`, `res.Vt`, `res.components`,
    `res.explained_variance`, `res.explained_variance_ratio`, `res.transformed`)
    and 3-tuple unpacking (`U, S, Vt = distributed_pca_svd(A, k=4)`).
    """

    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: Optional[np.ndarray] = None

    @property
    def components(self) -> np.ndarray:
        """Principal axes in feature space, shape (k, N)."""
        return self.Vt

    @property
    def singular_values(self) -> np.ndarray:
        return self.S

    @property
    def transformed(self) -> np.ndarray:
        """Projected low-dimensional representation U * S of shape (M, k)."""
        return self.U * self.S.reshape(1, -1)

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.U
        yield self.S
        yield self.Vt

    def __getitem__(self, idx: int) -> np.ndarray:
        return (self.U, self.S, self.Vt)[idx]

    def reconstruct(self) -> np.ndarray:
        """Reconstructs the rank-k approximation in original space."""
        approx = (self.U * self.S.reshape(1, -1)) @ self.Vt
        if self.mean is not None:
            approx = approx + self.mean.reshape(1, -1)
        return approx

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": int(self.S.shape[0]),
            "shape_u": list(self.U.shape),
            "shape_s": list(self.S.shape),
            "shape_vt": list(self.Vt.shape),
            "singular_values": self.S.tolist(),
            "explained_variance": self.explained_variance.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
        }


def _stabilize_svd_signs(u: np.ndarray, vt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enforces deterministic sign convention on singular vectors: the largest-magnitude
    entry in each column of U is made non-negative.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_rows, range(u.shape[1])])
    signs[signs == 0] = 1.0
    u_fixed = u * signs.reshape(1, -1)
    vt_fixed = vt * signs.reshape(-1, 1)
    return u_fixed, vt_fixed


def distributed_pca_svd(
    matrix: Any,
    k: int,
    center: bool = True,
    method: str = "randomized",
    oversampling: int = 8,
    power_iters: int = 2,
    iterations: int = 30,
    spot_check_probability: float = 0.25,
    dtype: str = "fp32",
    client: Optional[MatrixComputeClient] = None,
    config: Optional[ClientConfig] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> PCASVDResult:
    r"""
    Computes the top-k Singular Value Decomposition / Principal Component Analysis
    of a 2D matrix $A \in \mathbb{R}^{M \times N}$ using verified distributed GPU execution.

    Supported methods:
    1. `method="randomized"` (Default):
       Halko-Martinsson-Tropp randomized SVD with subspace power iterations:
       - $Y = A \Omega$ via `client.matmul()`
       - For $j = 1 \dots q$: QR-stabilized $Z = A^T Q$, $Y = A Q_Z$ via `client.matmul()`
       - $B = Q^T A$ via `client.matmul()`
       - Exact small SVD of $B = \tilde{U} \Sigma V^T$ locally.
    2. `method="power_iteration"`:
       Stateful remote VRAM session leasing via `client.power_iteration()` on the
       Gram matrix $G = A^T A$ (or $A A^T$) with Hotelling deflation across $k$ components.

    Args:
        matrix: 2D array or tensor of shape (M, N).
        k: Number of top singular components to extract ($1 \le k \le \min(M, N)$).
        center: If True, mean-centers columns before decomposition (standard PCA).
        method: `'randomized'` or `'power_iteration'`.
        oversampling: Extra sketch dimensions $p$ for randomized SVD (default 8).
        power_iters: Number of subspace power iterations for randomized SVD (default 2).
        iterations: Number of stateful power steps per component when `method="power_iteration"`.
        spot_check_probability: Freivalds spot-check probability for stateful power iteration.
        dtype: Target precision ('fp32', 'fp16', 'bf16').
        client: Optional pre-initialized `MatrixComputeClient`.
        config: Optional `ClientConfig` instance.
        random_state: Deterministic PRNG seed.

    Returns:
        `PCASVDResult` (supports `U, S, Vt = distributed_pca_svd(matrix, k)`).
    """
    if hasattr(matrix, "detach") and hasattr(matrix, "cpu"):
        arr = matrix.detach().cpu().numpy()
    elif hasattr(matrix, "data") and isinstance(getattr(matrix, "data", None), np.ndarray):
        arr = matrix.data
    else:
        arr = np.asarray(matrix, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for PCA/SVD, got shape {arr.shape}")

    m, n = arr.shape
    min_dim = min(m, n)
    if m == 0 or n == 0:
        raise ValueError(f"Matrix dimensions must be non-zero, got shape {arr.shape}")
    if k <= 0 or k > min_dim:
        raise ValueError(f"k must satisfy 1 <= k <= min(M, N)={min_dim}, got k={k}")

    norm_method = method.strip().lower()
    if norm_method not in ("randomized", "power_iteration"):
        raise ValueError(
            f"Unsupported method '{method}'. Supported: 'randomized', 'power_iteration'"
        )

    a_work = np.ascontiguousarray(arr, dtype=np.float32)
    col_mean: Optional[np.ndarray] = None
    if center:
        col_mean = np.mean(a_work, axis=0, dtype=np.float32)
        a_work = a_work - col_mean.reshape(1, -1)

    total_var_sum = float(np.sum(a_work * a_work))
    dof = max(1, m - 1) if center else max(1, m)
    total_variance = total_var_sum / float(dof)

    def _run_decomposition(active_client: MatrixComputeClient) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(random_state)

        if norm_method == "randomized":
            sketch_dim = min(min_dim, k + max(0, int(oversampling)))
            omega = rng.randn(n, sketch_dim).astype(np.float32)

            # Range finder: Y = A @ Omega
            y_mat = active_client.matmul(
                A=a_work,
                B=omega,
                dtype=dtype,
                partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
            )
            q_mat, _ = np.linalg.qr(y_mat, mode="reduced")

            # Subspace power iterations for rapid singular spectrum decay
            for _ in range(max(0, int(power_iters))):
                z_mat = active_client.matmul(
                    A=np.ascontiguousarray(a_work.T),
                    B=np.ascontiguousarray(q_mat),
                    dtype=dtype,
                    partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
                )
                q_z, _ = np.linalg.qr(z_mat, mode="reduced")
                y_mat = active_client.matmul(
                    A=a_work,
                    B=np.ascontiguousarray(q_z),
                    dtype=dtype,
                    partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
                )
                q_mat, _ = np.linalg.qr(y_mat, mode="reduced")

            # Project onto low-dimensional subspace: B = Q^T @ A
            b_proj = active_client.matmul(
                A=np.ascontiguousarray(q_mat.T),
                B=a_work,
                dtype=dtype,
                partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
            )

            u_tilde, s_vals, vt_all = np.linalg.svd(b_proj, full_matrices=False)
            u_full = q_mat @ u_tilde

            u_k = np.ascontiguousarray(u_full[:, :k], dtype=np.float32)
            s_k = np.ascontiguousarray(s_vals[:k], dtype=np.float32)
            vt_k = np.ascontiguousarray(vt_all[:k, :], dtype=np.float32)
            u_k, vt_k = _stabilize_svd_signs(u_k, vt_k)
            return u_k, s_k, vt_k

        # Stateful power_iteration method using remote VRAM session leasing
        # Form Gram matrix G = A^T @ A (shape N x N) via distributed GEMM
        gram = active_client.matmul(
            A=np.ascontiguousarray(a_work.T),
            B=a_work,
            dtype=dtype,
            partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
        )
        gram_deflated = gram.astype(np.float32).copy()

        v_list = []
        s_list = []
        u_list = []

        for comp_idx in range(k):
            v0 = rng.randn(n).astype(np.float32)
            # Orthogonalize v0 against previously found right singular vectors
            for prev_v in v_list:
                v0 = v0 - np.dot(v0, prev_v) * prev_v
            v0_norm = float(np.linalg.norm(v0))
            if v0_norm > 1e-12:
                v0 = v0 / v0_norm

            v_final = active_client.power_iteration(
                A=gram_deflated,
                v0=v0,
                iterations=max(1, int(iterations)),
                dtype=dtype,
                spot_check_probability=spot_check_probability,
                normalize=True,
            )
            v_vec = np.asarray(v_final, dtype=np.float32).reshape(-1)
            # Re-orthogonalize against previous vectors for numerical cleanliness
            for prev_v in v_list:
                v_vec = v_vec - np.dot(v_vec, prev_v) * prev_v
            v_norm = float(np.linalg.norm(v_vec))
            if v_norm > 1e-12:
                v_vec = v_vec / v_norm

            # Compute u_i * sigma_i = A @ v_i via verified distributed GEMM
            av_col = active_client.matmul(
                A=a_work,
                B=v_vec.reshape(n, 1),
                dtype=dtype,
                partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
            ).reshape(-1)
            sigma = float(np.linalg.norm(av_col))
            if sigma > 1e-12:
                u_vec = av_col / sigma
            else:
                u_vec = np.zeros(m, dtype=np.float32)
                u_vec[comp_idx % m] = 1.0

            v_list.append(v_vec)
            s_list.append(sigma)
            u_list.append(u_vec)

            # Hotelling deflation: G <- G - sigma^2 * v_i v_i^T
            gram_deflated = gram_deflated - (sigma * sigma) * np.outer(v_vec, v_vec)

        u_k = np.column_stack(u_list).astype(np.float32)
        s_k = np.array(s_list, dtype=np.float32)
        vt_k = np.vstack(v_list).astype(np.float32)
        u_k, vt_k = _stabilize_svd_signs(u_k, vt_k)
        return u_k, s_k, vt_k

    if client is not None:
        u_out, s_out, vt_out = _run_decomposition(client)
    else:
        client_kwargs = dict(kwargs)
        if config is None and "offline_mock_mode" not in client_kwargs:
            client_kwargs["offline_mock_mode"] = True
        with MatrixComputeClient(config=config, **client_kwargs) as temp_client:
            u_out, s_out, vt_out = _run_decomposition(temp_client)

    explained_var = (s_out ** 2) / float(dof)
    if total_variance > 1e-12:
        explained_ratio = np.clip(explained_var / total_variance, 0.0, 1.0)
    else:
        explained_ratio = np.zeros_like(explained_var)

    return PCASVDResult(
        U=u_out,
        S=s_out,
        Vt=vt_out,
        explained_variance=explained_var.astype(np.float32),
        explained_variance_ratio=explained_ratio.astype(np.float32),
        mean=col_mean,
    )
