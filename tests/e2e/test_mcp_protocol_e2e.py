"""
End-to-End MCP JSON-RPC 2.0 Protocol & Tool Execution Tests (`tests/e2e/test_mcp_protocol_e2e.py`).

Verifies that `matrix-compute-mcp` functions end-to-end with MCP clients (Claude Desktop,
Cursor, Antigravity) over JSON-RPC 2.0 stdio transport using L402 macaroons or trial API keys.
"""

import io
import json
from pathlib import Path

import numpy as np
import pytest

from matrix_compute_mcp.io_utils import load_matrix_input, save_safetensors_file
from matrix_compute_mcp.server import MCPServerConfig, MatrixComputeMCPServer


def test_mcp_stdio_full_agent_session_with_l402_macaroon(tmp_path: Path):
    """
    Simulates a complete MCP client session over stdio (`initialize`, `notifications/initialized`,
    `tools/list`, and `tools/call` across all 5 exposed tools).
    """
    rng = np.random.RandomState(501)
    mat_a = rng.randn(16, 20).astype(np.float32)
    mat_b = rng.randn(20, 12).astype(np.float32)
    queries = rng.randn(6, 16).astype(np.float32)
    corpus = rng.randn(24, 16).astype(np.float32)

    inputs_st = tmp_path / "inputs.safetensors"
    save_safetensors_file(
        {
            "A": mat_a,
            "B": mat_b,
            "queries": queries,
            "corpus": corpus,
        },
        inputs_st,
    )
    out_c_st = tmp_path / "output_c.safetensors"

    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "antigravity-mcp-client", "version": "1.0.0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_network_status",
                "arguments": {},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "estimate_job_cost",
                "arguments": {
                    "m": 4096,
                    "k": 4096,
                    "n": 4096,
                    "dtype": "fp16",
                    "partitioning_strategy": "output_grid_2d",
                    "grid_rows": 2,
                    "grid_cols": 2,
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "execute_distributed_matmul",
                "arguments": {
                    "matrix_a": str(inputs_st),
                    "matrix_b": str(inputs_st),
                    "tensor_key_a": "A",
                    "tensor_key_b": "B",
                    "output_path": str(out_c_st),
                    "dtype": "fp32",
                    "partitioning_strategy": "output_grid_2d",
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "batch_similarity_search",
                "arguments": {
                    "queries": queries.tolist(),
                    "corpus_embeddings": corpus.tolist(),
                    "top_k": 4,
                },
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "compute_pca_svd",
                "arguments": {
                    "matrix": mat_a.tolist(),
                    "k": 3,
                    "center": True,
                    "method": "randomized",
                },
            },
        },
    ]

    in_buffer = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    out_buffer = io.StringIO()

    cfg = MCPServerConfig(
        offline_mock_mode=True,
        l402_macaroon="mda0_mock_macaroon_token_hex",
        default_tile_size=8,
    )

    with MatrixComputeMCPServer(config=cfg) as server:
        server.serve_stdio(input_stream=in_buffer, output_stream=out_buffer)

    response_lines = [line for line in out_buffer.getvalue().splitlines() if line.strip()]
    # 7 requests had 'id'; 1 was a notification (no response)
    assert len(response_lines) == 7
    responses = {resp["id"]: resp for resp in (json.loads(line) for line in response_lines)}

    # 1. Initialize response
    assert responses[1]["result"]["serverInfo"]["name"] == "matrix-compute-mcp"

    # 2. tools/list response
    tool_names = {t["name"] for t in responses[2]["result"]["tools"]}
    assert {
        "get_network_status",
        "estimate_job_cost",
        "execute_distributed_matmul",
        "batch_similarity_search",
        "compute_pca_svd",
    }.issubset(tool_names)

    # 3. get_network_status response
    status_data = responses[3]["result"]["structuredContent"]
    assert status_data["status"] == "online"
    assert status_data["auth_mode"] == "l402_macaroon"
    assert status_data["active_workers"] >= 2
    assert status_data["total_available_vram_mb"] >= 49152

    # 4. estimate_job_cost response
    cost_data = responses[4]["result"]["structuredContent"]
    assert cost_data["is_download_optimal"] is True
    assert cost_data["estimated_total_sats"] > 0
    assert cost_data["total_transfer_mb"] > 0

    # 5. execute_distributed_matmul response & saved SafeTensors file verification
    matmul_data = responses[5]["result"]["structuredContent"]
    assert matmul_data["status"] == "verified"
    assert matmul_data["freivalds_verified"] is True
    assert matmul_data["shape_c"] == [16, 12]
    c_saved = load_matrix_input(out_c_st)
    np.testing.assert_allclose(c_saved, mat_a @ mat_b, rtol=1e-4, atol=1e-5)

    # 6. batch_similarity_search response
    sim_data = responses[6]["result"]["structuredContent"]
    assert sim_data["status"] == "verified"
    assert sim_data["num_queries"] == 6
    assert sim_data["top_k"] == 4
    assert len(sim_data["indices"]) == 6
    assert len(sim_data["indices"][0]) == 4

    # 7. compute_pca_svd response
    pca_data = responses[7]["result"]["structuredContent"]
    assert pca_data["status"] == "verified"
    assert pca_data["k"] == 3
    assert len(pca_data["singular_values"]) == 3


def test_mcp_trial_api_key_and_error_handling():
    cfg = MCPServerConfig(
        offline_mock_mode=True,
        trial_api_key="trial_free_500_sats_key",
    )
    with MatrixComputeMCPServer(config=cfg) as server:
        status = server.get_network_status()
        assert status["auth_mode"] == "trial_api_key"

        # Invalid inner dimension should return MCP tool error with isError=True
        err_res = server.call_tool(
            "execute_distributed_matmul",
            {
                "matrix_a": [[1.0, 2.0], [3.0, 4.0]],
                "matrix_b": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            },
        )
        assert err_res["isError"] is True
        assert "Inner dimension mismatch" in err_res["structuredContent"]["error"]
