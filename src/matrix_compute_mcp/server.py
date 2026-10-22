"""
Official Model Context Protocol (MCP) Server for Distributed Matrix Multiplication (`matrix-compute-mcp`).

Exposes verified decentralized GPU supercomputing tools to AI agents (Claude Desktop, Cursor,
Antigravity, LangChain, OpenClaw) over stdio and JSON-RPC 2.0:
- `get_network_status`
- `estimate_job_cost`
- `execute_distributed_matmul`
- `batch_similarity_search`
- `compute_pca_svd`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig, PartitioningStrategy

from matrix_compute_mcp.ai.similarity import distributed_cosine_similarity
from matrix_compute_mcp.ai.spectral import distributed_pca_svd
from matrix_compute_mcp.estimator import DEFAULT_SATS_PER_GFLOP, estimate_job_cost
from matrix_compute_mcp.io_utils import load_matrix_input, save_matrix_output

logger = logging.getLogger(__name__)

_MCP_SDK_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP  # type: ignore

    _MCP_SDK_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore


MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "matrix-compute-mcp"
SERVER_VERSION = "0.1.0"
INLINE_RESULT_ELEMENT_LIMIT = 4096


@dataclass
class MCPServerConfig:
    """Configuration for the `matrix-compute-mcp` server."""

    nostr_relays: List[str] = field(default_factory=lambda: ["ws://localhost:8080"])
    settlement_api_url: str = "http://localhost:8000"
    l402_macaroon: Optional[str] = None
    l402_preimage: Optional[str] = None
    trial_api_key: Optional[str] = None
    offline_mock_mode: bool = True
    default_tile_size: int = 64
    dtype_preference: str = "fp32"
    max_price_per_gflop: float = 10.0
    min_worker_vram_mb: int = 1024

    @classmethod
    def from_env(cls, **overrides: Any) -> "MCPServerConfig":
        relays_env = os.environ.get("MATRIX_NOSTR_RELAYS", "")
        relays = (
            [r.strip() for r in relays_env.split(",") if r.strip()]
            if relays_env.strip()
            else ["ws://localhost:8080"]
        )
        settlement_url = os.environ.get("MATRIX_SETTLEMENT_API_URL", "http://localhost:8000")
        macaroon = os.environ.get("MATRIX_L402_MACAROON") or None
        preimage = os.environ.get("MATRIX_L402_PREIMAGE") or None
        api_key = os.environ.get("MATRIX_TRIAL_API_KEY") or None

        mock_env = os.environ.get("MATRIX_OFFLINE_MOCK_MODE", "true").strip().lower()
        offline_mock = mock_env in ("1", "true", "yes", "on")

        cfg = cls(
            nostr_relays=relays,
            settlement_api_url=settlement_url,
            l402_macaroon=macaroon,
            l402_preimage=preimage,
            trial_api_key=api_key,
            offline_mock_mode=offline_mock,
        )
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_client_config(self) -> ClientConfig:
        return ClientConfig.load(
            nostr_relays=self.nostr_relays,
            settlement_api_url=self.settlement_api_url,
            offline_mock_mode=self.offline_mock_mode,
            default_tile_size=self.default_tile_size,
            dtype_preference=self.dtype_preference,
            max_price_per_gflop=self.max_price_per_gflop,
            min_worker_vram_mb=self.min_worker_vram_mb,
        )


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "get_network_status",
        "description": (
            "Queries the Matrix Compute discovery relay for live GPU worker availability, "
            "total VRAM capacity, confidential computing (TEE) enclaves, and sats_per_gflop pricing."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_vram_mb": {
                    "type": "integer",
                    "description": "Minimum worker VRAM filter in MB (optional).",
                },
                "max_price_per_gflop": {
                    "type": "number",
                    "description": "Maximum worker price filter in satoshis per GFLOP (optional).",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "estimate_job_cost",
        "description": (
            "Calculates exact upload/download network bandwidth (MB), task count, GFLOPs, "
            "and satoshi cost estimates for a distributed matrix multiplication C (m x n) = A (m x k) x B (k x n)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "m": {"type": "integer", "description": "Rows of matrix A (M > 0)."},
                "k": {"type": "integer", "description": "Inner dimension K (columns of A / rows of B)."},
                "n": {"type": "integer", "description": "Columns of matrix B (N > 0)."},
                "dtype": {
                    "type": "string",
                    "enum": ["fp32", "fp16", "bf16"],
                    "default": "fp32",
                    "description": "Precision format ('fp32', 'fp16', 'bf16').",
                },
                "partitioning_strategy": {
                    "type": "string",
                    "enum": ["output_grid_2d", "outer_product_1d", "block_3d"],
                    "default": "output_grid_2d",
                    "description": "Matrix partitioning strategy.",
                },
                "tile_size": {
                    "type": "integer",
                    "default": 1024,
                    "description": "Tile block size dimension.",
                },
                "num_strips": {
                    "type": "integer",
                    "default": 4,
                    "description": "Strip count when using outer_product_1d.",
                },
                "grid_rows": {
                    "type": "integer",
                    "description": "Row partitions P_r when using output_grid_2d.",
                },
                "grid_cols": {
                    "type": "integer",
                    "description": "Column partitions P_c when using output_grid_2d.",
                },
                "sats_per_gflop": {
                    "type": "number",
                    "description": "Override worker price in satoshis per GFLOP.",
                },
            },
            "required": ["m", "k", "n"],
            "additionalProperties": False,
        },
    },
    {
        "name": "execute_distributed_matmul",
        "description": (
            "Executes verified distributed matrix multiplication C = A x B across decentralized "
            "GPU workers with dynamic Freivalds verification and L402 settlement. Accepts file paths "
            "(.npy, .npz, .safetensors) or inline 2D JSON arrays."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "matrix_a": {
                    "description": "File path (.npy/.npz/.safetensors) or inline 2D array for matrix A (M x K).",
                },
                "matrix_b": {
                    "description": "File path (.npy/.npz/.safetensors) or inline 2D array for matrix B (K x N).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output file path (.npy, .npz, or .safetensors) to write C.",
                },
                "tensor_key_a": {
                    "type": "string",
                    "description": "Optional tensor key when loading matrix_a from .safetensors or .npz.",
                },
                "tensor_key_b": {
                    "type": "string",
                    "description": "Optional tensor key when loading matrix_b from .safetensors or .npz.",
                },
                "dtype": {
                    "type": "string",
                    "enum": ["fp32", "fp16", "bf16"],
                    "default": "fp32",
                    "description": "Computation precision ('fp32', 'fp16', 'bf16').",
                },
                "partitioning_strategy": {
                    "type": "string",
                    "enum": ["output_grid_2d", "outer_product_1d", "block_3d"],
                    "default": "output_grid_2d",
                    "description": "Matrix partitioning topology.",
                },
                "tile_size": {
                    "type": "integer",
                    "description": "Optional tile block size override.",
                },
            },
            "required": ["matrix_a", "matrix_b"],
            "additionalProperties": False,
        },
    },
    {
        "name": "batch_similarity_search",
        "description": (
            "Runs exact cosine similarity search between query embeddings (M x D) and corpus "
            "embeddings (N x D) on distributed GPUs using OUTPUT_GRID_2D zero-reduction tiling."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "queries": {
                    "description": "File path (.npy/.npz/.safetensors) or inline 2D array of shape (M, D).",
                },
                "corpus_embeddings": {
                    "description": "File path (.npy/.npz/.safetensors) or inline 2D array of shape (N, D).",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top matching corpus indices and scores to return per query.",
                },
                "dtype": {
                    "type": "string",
                    "enum": ["fp32", "fp16", "bf16"],
                    "default": "fp32",
                    "description": "Target GEMM precision.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional file path (.npy/.safetensors) to save the full similarity matrix.",
                },
            },
            "required": ["queries", "corpus_embeddings"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compute_pca_svd",
        "description": (
            "Computes verified top-k Principal Component Analysis (PCA) / Singular Value Decomposition (SVD) "
            "on distributed GPU workers via randomized subspace iteration or stateful power iteration."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "matrix": {
                    "description": "File path (.npy/.npz/.safetensors) or inline 2D array of shape (M, N).",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top singular/principal components to extract.",
                },
                "center": {
                    "type": "boolean",
                    "default": True,
                    "description": "Mean-center columns before decomposition (standard PCA).",
                },
                "method": {
                    "type": "string",
                    "enum": ["randomized", "power_iteration"],
                    "default": "randomized",
                    "description": "Decomposition algorithm ('randomized' or 'power_iteration').",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional output file path (.npz or .safetensors) to save U, S, Vt.",
                },
            },
            "required": ["matrix", "k"],
            "additionalProperties": False,
        },
    },
]


class MatrixComputeMCPServer:
    """
    Official Model Context Protocol (MCP) Server for `matrix-compute-mcp`.

    Manages an underlying `MatrixComputeClient` session, authenticates via L402 macaroon
    or trial API key, and exposes high-level distributed GPU tools over MCP JSON-RPC 2.0
    and FastMCP (when installed).
    """

    def __init__(
        self,
        config: Optional[MCPServerConfig] = None,
        client: Optional[MatrixComputeClient] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config if config is not None else MCPServerConfig.from_env(**kwargs)
        self._external_client = client is not None
        self._client: Optional[MatrixComputeClient] = client
        self._tool_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "get_network_status": self._handle_get_network_status,
            "estimate_job_cost": self._handle_estimate_job_cost,
            "execute_distributed_matmul": self._handle_execute_distributed_matmul,
            "batch_similarity_search": self._handle_batch_similarity_search,
            "compute_pca_svd": self._handle_compute_pca_svd,
        }
        self.fastmcp = self._build_fastmcp() if _MCP_SDK_AVAILABLE else None

    @property
    def client(self) -> MatrixComputeClient:
        if self._client is None:
            self._client = MatrixComputeClient(config=self.config.to_client_config())
        return self._client

    def close(self) -> None:
        if self._client is not None and not self._external_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "MatrixComputeMCPServer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _auth_mode(self) -> str:
        if self.config.l402_macaroon:
            return "l402_macaroon"
        if self.config.trial_api_key:
            return "trial_api_key"
        if self.config.offline_mock_mode:
            return "offline_mock_trial"
        return "anonymous_l402"

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def get_network_status(
        self,
        min_vram_mb: Optional[int] = None,
        max_price_per_gflop: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Queries `RelayClient` for live GPU worker availability and pricing."""
        return self._handle_get_network_status(
            {"min_vram_mb": min_vram_mb, "max_price_per_gflop": max_price_per_gflop}
        )

    def _handle_get_network_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        min_vram = args.get("min_vram_mb")
        max_price = args.get("max_price_per_gflop")

        workers = self.client.relay_client.query_workers(
            min_vram_mb=int(min_vram) if min_vram is not None else self.config.min_worker_vram_mb,
            max_price_per_gflop=(
                float(max_price) if max_price is not None else self.config.max_price_per_gflop
            ),
        )

        worker_summaries: List[Dict[str, Any]] = []
        prices: List[float] = []
        total_vram_mb = 0

        for w in workers:
            vram = int(getattr(w, "vram_mb", 0))
            price = float(getattr(w, "sats_per_gflop", DEFAULT_SATS_PER_GFLOP))
            total_vram_mb += vram
            prices.append(price)
            worker_summaries.append(
                {
                    "pubkey": getattr(w, "pubkey", ""),
                    "endpoint": getattr(w, "endpoint", ""),
                    "vram_mb": vram,
                    "sats_per_gflop": price,
                    "supported_dtypes": list(getattr(w, "supported_dtypes", ["fp32"])),
                    "tee_capable": bool(getattr(w, "tee_capable", False)),
                    "nat_mode": getattr(w, "nat_mode", "direct"),
                }
            )

        min_p = min(prices) if prices else DEFAULT_SATS_PER_GFLOP
        max_p = max(prices) if prices else DEFAULT_SATS_PER_GFLOP
        median_p = float(np.median(prices)) if prices else DEFAULT_SATS_PER_GFLOP

        return {
            "status": "online",
            "auth_mode": self._auth_mode(),
            "offline_mock_mode": self.config.offline_mock_mode,
            "active_workers": len(worker_summaries),
            "total_available_vram_mb": total_vram_mb,
            "pricing_sats_per_gflop": {
                "min": min_p,
                "median": median_p,
                "max": max_p,
            },
            "workers": worker_summaries,
        }

    def _handle_estimate_job_cost(self, args: Dict[str, Any]) -> Dict[str, Any]:
        m = int(args["m"])
        k = int(args["k"])
        n = int(args["n"])
        dtype = str(args.get("dtype", self.config.dtype_preference))
        strategy = str(args.get("partitioning_strategy", "output_grid_2d"))
        tile_size = int(args.get("tile_size", 1024))
        num_strips = int(args.get("num_strips", 4))
        grid_rows = args.get("grid_rows")
        grid_cols = args.get("grid_cols")
        sats_per_gflop = args.get("sats_per_gflop")

        if sats_per_gflop is None:
            status = self._handle_get_network_status({})
            sats_per_gflop = status["pricing_sats_per_gflop"]["median"]

        est = estimate_job_cost(
            m=m,
            k=k,
            n=n,
            dtype=dtype,
            partitioning_strategy=strategy,
            tile_size=tile_size,
            num_strips=num_strips,
            grid_rows=int(grid_rows) if grid_rows is not None else None,
            grid_cols=int(grid_cols) if grid_cols is not None else None,
            sats_per_gflop=float(sats_per_gflop),
        )
        return est.to_dict()

    def _handle_execute_distributed_matmul(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mat_a_src = args["matrix_a"]
        mat_b_src = args["matrix_b"]
        output_path = args.get("output_path")
        tensor_key_a = args.get("tensor_key_a")
        tensor_key_b = args.get("tensor_key_b")
        dtype = str(args.get("dtype", self.config.dtype_preference))
        strategy = str(args.get("partitioning_strategy", "output_grid_2d"))
        tile_size = args.get("tile_size")

        a = load_matrix_input(mat_a_src, tensor_key=tensor_key_a)
        b = load_matrix_input(mat_b_src, tensor_key=tensor_key_b)

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"Inner dimension mismatch: matrix_a shape {a.shape} vs matrix_b shape {b.shape}"
            )

        t0 = time.perf_counter()
        c = self.client.matmul(
            A=a,
            B=b,
            dtype=dtype,
            tile_size=int(tile_size) if tile_size is not None else None,
            partitioning_strategy=strategy,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        est = estimate_job_cost(
            m=a.shape[0],
            k=a.shape[1],
            n=b.shape[1],
            dtype=dtype,
            partitioning_strategy=strategy,
            tile_size=int(tile_size) if tile_size is not None else self.config.default_tile_size,
        )

        response: Dict[str, Any] = {
            "status": "verified",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "shape_c": list(c.shape),
            "dtype": dtype,
            "partitioning_strategy": strategy,
            "freivalds_verified": True,
            "elapsed_ms": elapsed_ms,
            "estimated_total_sats": est.estimated_total_sats,
            "total_transfer_mb": est.total_transfer_mb,
        }

        if output_path:
            saved = save_matrix_output(c, output_path, tensor_key="result")
            response["output_path"] = str(saved)

        if c.size <= INLINE_RESULT_ELEMENT_LIMIT:
            response["result"] = c.tolist()
        else:
            response["result_preview_top_left"] = c[: min(4, c.shape[0]), : min(4, c.shape[1])].tolist()

        return response

    def _handle_batch_similarity_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        q_src = args["queries"]
        c_src = args["corpus_embeddings"]
        top_k = int(args.get("top_k", 5))
        dtype = str(args.get("dtype", self.config.dtype_preference))
        output_path = args.get("output_path")

        q_mat = load_matrix_input(q_src, tensor_key=args.get("tensor_key_queries"))
        c_mat = load_matrix_input(c_src, tensor_key=args.get("tensor_key_corpus"))

        eff_top_k = min(top_k, c_mat.shape[0])
        t0 = time.perf_counter()
        search_res = distributed_cosine_similarity(
            queries=q_mat,
            corpus_embeddings=c_mat,
            top_k=eff_top_k,
            dtype=dtype,
            client=self.client,
            return_result_object=True,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        result_dict = search_res.to_dict()
        result_dict.update(
            {
                "status": "verified",
                "num_corpus": int(c_mat.shape[0]),
                "embedding_dim": int(q_mat.shape[1]),
                "partitioning_strategy": "output_grid_2d",
                "elapsed_ms": elapsed_ms,
            }
        )

        if output_path and search_res.similarity_matrix is not None:
            saved = save_matrix_output(
                search_res.similarity_matrix, output_path, tensor_key="similarity_matrix"
            )
            result_dict["output_path"] = str(saved)

        return result_dict

    def _handle_compute_pca_svd(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mat_src = args["matrix"]
        k = int(args["k"])
        center = bool(args.get("center", True))
        method = str(args.get("method", "randomized"))
        output_path = args.get("output_path")

        mat = load_matrix_input(mat_src, tensor_key=args.get("tensor_key"))
        t0 = time.perf_counter()
        pca_res = distributed_pca_svd(
            matrix=mat,
            k=k,
            center=center,
            method=method,
            client=self.client,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        out = pca_res.to_dict()
        out["status"] = "verified"
        out["method"] = method
        out["center"] = center
        out["elapsed_ms"] = elapsed_ms

        if pca_res.Vt.size <= INLINE_RESULT_ELEMENT_LIMIT:
            out["components"] = pca_res.Vt.tolist()
        if pca_res.transformed.size <= INLINE_RESULT_ELEMENT_LIMIT:
            out["transformed"] = pca_res.transformed.tolist()

        if output_path:
            out_p = Path(output_path).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.suffix.lower() == ".safetensors":
                from matrix_compute_mcp.io_utils import save_safetensors_file

                save_safetensors_file(
                    {"U": pca_res.U, "S": pca_res.S, "Vt": pca_res.Vt},
                    out_p,
                )
            else:
                np.savez(str(out_p), U=pca_res.U, S=pca_res.S, Vt=pca_res.Vt)
            out["output_path"] = str(out_p)

        return out

    # -------------------------------------------------------------------------
    # FastMCP Integration (when `mcp` package is installed)
    # -------------------------------------------------------------------------

    def _build_fastmcp(self) -> Any:
        mcp_app = FastMCP(SERVER_NAME)

        @mcp_app.tool(name="get_network_status")
        def _tool_get_network_status(
            min_vram_mb: Optional[int] = None,
            max_price_per_gflop: Optional[float] = None,
        ) -> Dict[str, Any]:
            return self._handle_get_network_status(
                {"min_vram_mb": min_vram_mb, "max_price_per_gflop": max_price_per_gflop}
            )

        @mcp_app.tool(name="estimate_job_cost")
        def _tool_estimate_job_cost(
            m: int,
            k: int,
            n: int,
            dtype: str = "fp32",
            partitioning_strategy: str = "output_grid_2d",
            tile_size: int = 1024,
            num_strips: int = 4,
            grid_rows: Optional[int] = None,
            grid_cols: Optional[int] = None,
            sats_per_gflop: Optional[float] = None,
        ) -> Dict[str, Any]:
            return self._handle_estimate_job_cost(
                {
                    "m": m,
                    "k": k,
                    "n": n,
                    "dtype": dtype,
                    "partitioning_strategy": partitioning_strategy,
                    "tile_size": tile_size,
                    "num_strips": num_strips,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "sats_per_gflop": sats_per_gflop,
                }
            )

        @mcp_app.tool(name="execute_distributed_matmul")
        def _tool_execute_distributed_matmul(
            matrix_a: Any,
            matrix_b: Any,
            output_path: Optional[str] = None,
            dtype: str = "fp32",
            partitioning_strategy: str = "output_grid_2d",
            tile_size: Optional[int] = None,
        ) -> Dict[str, Any]:
            return self._handle_execute_distributed_matmul(
                {
                    "matrix_a": matrix_a,
                    "matrix_b": matrix_b,
                    "output_path": output_path,
                    "dtype": dtype,
                    "partitioning_strategy": partitioning_strategy,
                    "tile_size": tile_size,
                }
            )

        @mcp_app.tool(name="batch_similarity_search")
        def _tool_batch_similarity_search(
            queries: Any,
            corpus_embeddings: Any,
            top_k: int = 5,
            dtype: str = "fp32",
            output_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            return self._handle_batch_similarity_search(
                {
                    "queries": queries,
                    "corpus_embeddings": corpus_embeddings,
                    "top_k": top_k,
                    "dtype": dtype,
                    "output_path": output_path,
                }
            )

        @mcp_app.tool(name="compute_pca_svd")
        def _tool_compute_pca_svd(
            matrix: Any,
            k: int,
            center: bool = True,
            method: str = "randomized",
            output_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            return self._handle_compute_pca_svd(
                {
                    "matrix": matrix,
                    "k": k,
                    "center": center,
                    "method": method,
                    "output_path": output_path,
                }
            )

        return mcp_app

    # -------------------------------------------------------------------------
    # Native MCP JSON-RPC 2.0 Protocol Handler & Stdio Transport
    # -------------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns the list of MCP tool definitions with JSON Schema input schemas."""
        return list(TOOL_DEFINITIONS)

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Invokes an MCP tool by name and formats the response per the Model Context Protocol
        specification (`content: [{"type": "text", "text": ...}]`, `structuredContent`, `isError`).
        """
        if name not in self._tool_handlers:
            raise KeyError(f"Unknown MCP tool: '{name}'")

        args = arguments or {}
        try:
            payload = self._tool_handlers[name](args)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload),
                    }
                ],
                "structuredContent": payload,
                "isError": False,
            }
        except Exception as exc:
            err_payload = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "tool": name,
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(err_payload),
                    }
                ],
                "structuredContent": err_payload,
                "isError": True,
            }

    def handle_jsonrpc_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single MCP JSON-RPC 2.0 request or notification dictionary.
        Returns a JSON-RPC 2.0 response dictionary (or `None` for notifications).
        """
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id") if isinstance(message, dict) else None,
                "error": {"code": -32600, "message": "Invalid Request: expected jsonrpc 2.0"},
            }

        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # Notifications have no 'id' field and produce no response
        if "id" not in message:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
            }

        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {},
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": self.list_tools(),
                },
            }

        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"resources": []},
            }

        if method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"prompts": []},
            }

        if method == "tools/call":
            if not isinstance(params, dict) or "name" not in params:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32602, "message": "Invalid params: missing tool 'name'"},
                }
            tool_name = params["name"]
            tool_args = params.get("arguments") or {}
            if tool_name not in self._tool_handlers:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: '{tool_name}'"},
                }
            result = self.call_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: '{method}'"},
        }

    def serve_stdio(
        self,
        input_stream: Optional[io.TextIOBase] = None,
        output_stream: Optional[io.TextIOBase] = None,
    ) -> None:
        """
        Runs the MCP JSON-RPC 2.0 newline-delimited stdio server loop.
        Compatible with Claude Desktop, Cursor, Antigravity, and MCP Inspector.
        """
        in_stream = input_stream if input_stream is not None else sys.stdin
        out_stream = output_stream if output_stream is not None else sys.stdout

        for raw_line in in_stream:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                out_stream.write(json.dumps(err_resp) + "\n")
                out_stream.flush()
                continue

            resp = self.handle_jsonrpc_message(msg)
            if resp is not None:
                out_stream.write(json.dumps(resp) + "\n")
                out_stream.flush()


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint for `uvx matrix-compute-mcp` / `matrix-compute-mcp`."""
    parser = argparse.ArgumentParser(
        prog="matrix-compute-mcp",
        description="Official Model Context Protocol (MCP) Server for Distributed Matrix Multiplication",
    )
    parser.add_argument("--version", action="version", version=f"{SERVER_NAME} {SERVER_VERSION}")
    parser.add_argument(
        "--relays",
        type=str,
        default=None,
        help="Comma-separated Nostr relay URLs (or MATRIX_NOSTR_RELAYS env var)",
    )
    parser.add_argument(
        "--settlement-url",
        type=str,
        default=None,
        help="Settlement service URL (or MATRIX_SETTLEMENT_API_URL env var)",
    )
    parser.add_argument(
        "--l402-macaroon",
        type=str,
        default=None,
        help="L402 Macaroon token for authenticated compute settlement",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Trial API key for zero-setup agent onboarding",
    )
    parser.add_argument(
        "--mock-mode",
        dest="offline_mock_mode",
        action="store_true",
        default=None,
        help="Run with hermetic in-process mock relay, settlement, and GPU workers",
    )
    parser.add_argument(
        "--live-mode",
        dest="offline_mock_mode",
        action="store_false",
        help="Disable offline mock mode and connect to live Nostr relays and workers",
    )

    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.relays:
        overrides["nostr_relays"] = [r.strip() for r in args.relays.split(",") if r.strip()]
    if args.settlement_url:
        overrides["settlement_api_url"] = args.settlement_url
    if args.l402_macaroon:
        overrides["l402_macaroon"] = args.l402_macaroon
    if args.api_key:
        overrides["trial_api_key"] = args.api_key
    if args.offline_mock_mode is not None:
        overrides["offline_mock_mode"] = args.offline_mock_mode

    cfg = MCPServerConfig.from_env(**overrides)
    with MatrixComputeMCPServer(config=cfg) as server:
        server.serve_stdio()


if __name__ == "__main__":
    main()
