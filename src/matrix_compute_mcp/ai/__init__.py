"""
High-Level Verifiable AI Primitives (`matrix_compute_mcp.ai`).

Exposes:
- `distributed_cosine_similarity` & `SimilaritySearchResult`
- `DistributedLinear` & `DistributedLoRALinear`
- `distributed_pca_svd` & `PCASVDResult`
"""

from .similarity import (
    SimilaritySearchResult,
    distributed_cosine_similarity,
    l2_normalize_rows,
)
from .spectral import (
    PCASVDResult,
    distributed_pca_svd,
)
from .torch_layers import (
    DistributedLinear,
    DistributedLoRALinear,
)

__all__ = [
    "distributed_cosine_similarity",
    "l2_normalize_rows",
    "SimilaritySearchResult",
    "DistributedLinear",
    "DistributedLoRALinear",
    "distributed_pca_svd",
    "PCASVDResult",
]
