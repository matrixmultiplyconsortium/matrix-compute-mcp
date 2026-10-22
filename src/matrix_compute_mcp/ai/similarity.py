"""
Distributed Cosine Similarity & Exact K-NN Vector Search (`matrix_compute_mcp.ai.similarity`).

Uses `OUTPUT_GRID_2D` zero-reduction tiling to compute massive exact cosine similarity
matrices across millions of embedding vectors on distributed GPU workers with O(n^2)
Freivalds verification and zero download redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig, PartitioningStrategy

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SimilaritySearchResult:
    """Result container for top-K distributed cosine similarity search."""

    scores: np.ndarray
    indices: np.ndarray
    similarity_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": self.scores.tolist(),
            "indices": self.indices.tolist(),
            "num_queries": int(self.scores.shape[0]),
            "top_k": int(self.scores.shape[1]),
        }


def l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalizes each row of a 2D matrix to unit L2 norm in float32.
    Zero-norm rows remain all zeros without producing NaNs or Infs.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Embedding matrix dimensions must be non-zero, got shape {arr.shape}")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    zero_mask = norms <= eps
    safe_norms = np.where(zero_mask, 1.0, norms)
    normalized = arr / safe_norms
    if np.any(zero_mask):
        normalized = np.where(zero_mask, 0.0, normalized)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def distributed_cosine_similarity(
    queries: Any,
    corpus_embeddings: Any,
    top_k: Optional[int] = None,
    dtype: str = "fp32",
    client: Optional[MatrixComputeClient] = None,
    config: Optional[ClientConfig] = None,
    tile_size: Optional[int] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    grid_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-12,
    return_result_object: bool = False,
    **kwargs: Any,
) -> Union[ Any, Tuple[Any, Any], SimilaritySearchResult ]:
    """
    Computes exact cosine similarity between `queries` (M, D) and `corpus_embeddings` (N, D)
    using distributed GPU workers and `OUTPUT_GRID_2D` zero-reduction tiling.

    Mathematical formulation:
        1. Local O(M*D + N*D) row-wise L2 normalization:
           Q_hat = queries / ||queries||_2, C_hat = corpus_embeddings / ||corpus_embeddings||_2
        2. Distributed verified GEMM with OUTPUT_GRID_2D partitioning:
           S = Q_hat @ C_hat.T  in R^{M x N}
        3. Optional exact top-K extraction along each query row.

    Args:
        queries: 2D array/tensor of shape (M, D) or 1D query vector of shape (D,).
        corpus_embeddings: 2D array/tensor of shape (N, D).
        top_k: Optional number of top matches per query to return. If provided, returns
            `(top_k_scores, top_k_indices)` (or `SimilaritySearchResult` if `return_result_object=True`).
        dtype: Target GEMM precision ('fp32', 'fp16', 'bf16'). Defaults to 'fp32'.
        client: Optional pre-initialized `MatrixComputeClient`.
        config: Optional `ClientConfig` used when creating an ephemeral client.
        tile_size: Optional tile size override.
        grid_rows: Optional row partition count P_r for `OUTPUT_GRID_2D`.
        grid_cols: Optional column partition count P_c for `OUTPUT_GRID_2D`.
        grid_shape: Optional (P_r, P_c) tuple for `OUTPUT_GRID_2D`.
        eps: Numerical stability epsilon for zero-norm vector handling.
        return_result_object: When `top_k` is set, return a `SimilaritySearchResult` object.
        **kwargs: Additional keyword arguments forwarded to `MatrixComputeClient`.

    Returns:
        - If `top_k is None`: Similarity matrix of shape (M, N) (or (N,) if 1D query was passed),
          preserving PyTorch tensor type if inputs were PyTorch tensors.
        - If `top_k is not None`: Tuple `(top_k_scores, top_k_indices)` each of shape (M, top_k),
          sorted in descending order of cosine similarity.
    """
    is_torch_input = _TORCH_AVAILABLE and isinstance(queries, torch.Tensor)
    torch_device = queries.device if is_torch_input else None

    if hasattr(queries, "detach") and hasattr(queries, "cpu"):
        q_np = queries.detach().cpu().numpy()
    elif hasattr(queries, "data") and isinstance(getattr(queries, "data", None), np.ndarray):
        q_np = queries.data
    else:
        q_np = np.asarray(queries)

    if hasattr(corpus_embeddings, "detach") and hasattr(corpus_embeddings, "cpu"):
        c_np = corpus_embeddings.detach().cpu().numpy()
    elif hasattr(corpus_embeddings, "data") and isinstance(getattr(corpus_embeddings, "data", None), np.ndarray):
        c_np = corpus_embeddings.data
    else:
        c_np = np.asarray(corpus_embeddings)

    squeeze_query = False
    if q_np.ndim == 1:
        q_np = q_np.reshape(1, -1)
        squeeze_query = True

    if q_np.ndim != 2 or c_np.ndim != 2:
        raise ValueError(
            f"queries and corpus_embeddings must be 2D matrices. Got shapes {q_np.shape} and {c_np.shape}"
        )

    m, d_q = q_np.shape
    n, d_c = c_np.shape
    if m == 0 or n == 0 or d_q == 0:
        raise ValueError(
            f"queries and corpus_embeddings must be non-empty. Got shapes {q_np.shape} and {c_np.shape}"
        )
    if d_q != d_c:
        raise ValueError(
            f"Embedding dimension mismatch: queries has D={d_q}, corpus_embeddings has D={d_c}"
        )

    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if top_k > n:
            raise ValueError(f"top_k ({top_k}) cannot exceed corpus size N ({n})")

    # 1. Row-wise L2 normalization
    q_norm = l2_normalize_rows(q_np, eps=eps)
    c_norm = l2_normalize_rows(c_np, eps=eps)
    b_transposed = np.ascontiguousarray(c_norm.T)  # Shape (D, N)

    # 2. Execute verified distributed GEMM using OUTPUT_GRID_2D (zero download redundancy)
    def _run_gemm(active_client: MatrixComputeClient) -> np.ndarray:
        return active_client.matmul(
            A=q_norm,
            B=b_transposed,
            dtype=dtype,
            tile_size=tile_size,
            partitioning_strategy=PartitioningStrategy.OUTPUT_GRID_2D,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_shape=grid_shape,
        )

    if client is not None:
        sim_matrix = _run_gemm(client)
    else:
        client_kwargs = dict(kwargs)
        if config is None and "offline_mock_mode" not in client_kwargs:
            client_kwargs["offline_mock_mode"] = True
        with MatrixComputeClient(config=config, **client_kwargs) as temp_client:
            sim_matrix = _run_gemm(temp_client)

    # Clamp numerical noise to [-1.0, 1.0]
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    if top_k is None:
        out_sim = sim_matrix[0] if squeeze_query else sim_matrix
        if is_torch_input:
            return torch.from_numpy(out_sim).to(device=torch_device)
        return out_sim

    # 3. Extract exact top-K indices and scores sorted descending per query row
    if top_k == n:
        sorted_indices = np.argsort(-sim_matrix, axis=1)
    else:
        part_indices = np.argpartition(-sim_matrix, kth=top_k - 1, axis=1)[:, :top_k]
        row_idx = np.arange(m)[:, None]
        part_scores = sim_matrix[row_idx, part_indices]
        order = np.argsort(-part_scores, axis=1)
        sorted_indices = part_indices[row_idx, order]

    row_idx = np.arange(m)[:, None]
    top_scores = sim_matrix[row_idx, sorted_indices]
    top_indices = sorted_indices.astype(np.int64)

    if return_result_object:
        return SimilaritySearchResult(
            scores=top_scores,
            indices=top_indices,
            similarity_matrix=sim_matrix,
        )

    if is_torch_input:
        return (
            torch.from_numpy(top_scores).to(device=torch_device),
            torch.from_numpy(top_indices).to(device=torch_device, dtype=torch.long),
        )

    return top_scores, top_indices
