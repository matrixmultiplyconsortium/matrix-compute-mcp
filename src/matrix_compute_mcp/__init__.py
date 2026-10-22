"""
`matrix-compute-mcp` (`matrix_compute_mcp`): Official Model Context Protocol (MCP)
server and high-level verifiable AI primitives (`matrix_compute_mcp.ai`) built on top
of `matrix-compute-sdk`.
"""

from . import ai
from .ai import (
    DistributedLinear,
    DistributedLoRALinear,
    PCASVDResult,
    SimilaritySearchResult,
    distributed_cosine_similarity,
    distributed_pca_svd,
    l2_normalize_rows,
)
from .estimator import (
    JobCostEstimate,
    estimate_job_cost,
)
from .io_utils import (
    load_matrix_input,
    load_safetensors_bytes,
    load_safetensors_file,
    save_matrix_output,
    save_safetensors_bytes,
    save_safetensors_file,
)
from .server import (
    MCPServerConfig,
    MatrixComputeMCPServer,
    TOOL_DEFINITIONS,
    main,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ai",
    # AI Primitives
    "distributed_cosine_similarity",
    "l2_normalize_rows",
    "SimilaritySearchResult",
    "DistributedLinear",
    "DistributedLoRALinear",
    "distributed_pca_svd",
    "PCASVDResult",
    # Cost & Bandwidth Estimator
    "JobCostEstimate",
    "estimate_job_cost",
    # Safe Tensor I/O
    "load_matrix_input",
    "save_matrix_output",
    "load_safetensors_file",
    "save_safetensors_file",
    "load_safetensors_bytes",
    "save_safetensors_bytes",
    # MCP Server
    "MCPServerConfig",
    "MatrixComputeMCPServer",
    "TOOL_DEFINITIONS",
    "main",
]
