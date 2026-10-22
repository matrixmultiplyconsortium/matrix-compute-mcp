# `matrix-compute-mcp`: Official MCP Server & High-Level Verifiable AI Primitives

`matrix-compute-mcp` is an independent extension package built on top of [`matrix-compute-sdk`](../matrix-compute-sdk) that provides:
1. **High-Level Verifiable AI Primitives (`matrix_compute_mcp.ai`):** Drop-in PyTorch layers (`DistributedLinear`, `DistributedLoRALinear`), massive exact cosine similarity search (`distributed_cosine_similarity`), and distributed spectral PCA/SVD (`distributed_pca_svd`) — all verified in $O(n^2)$ via Freivalds' algorithm.
2. **Official Model Context Protocol Server (`matrix-compute-mcp`):** Exposes decentralized GPU TFLOPS to autonomous AI agents (Claude Desktop, Cursor, Antigravity, LangChain, OpenClaw) over stdio/JSON-RPC 2.0 with L402 Macaroon or trial API key authentication.

---

## Quick Start

### 1. MCP Client Configuration (Claude Desktop / Cursor / Antigravity)

```json
{
  "mcpServers": {
    "matrix-compute": {
      "command": "uvx",
      "args": ["matrix-compute-mcp"],
      "env": {
        "MATRIX_NOSTR_RELAYS": "wss://relay.matrixcompute.io",
        "MATRIX_SETTLEMENT_API_URL": "https://settlement.matrixcompute.io",
        "MATRIX_L402_MACAROON": "<your-l402-macaroon-or-leave-unset-for-trial>"
      }
    }
  }
}
```

### 2. Exposed MCP Tools
| Tool Name | Description |
|---|---|
| `get_network_status` | Queries the discovery relay for live GPU worker count, total VRAM, TEE enclaves, and `sats_per_gflop` pricing. |
| `estimate_job_cost` | Computes exact upload/download bandwidth (MB), task count, GFLOPs, and satoshi cost estimates via `BandwidthProfiler`. |
| `execute_distributed_matmul` | Executes verified distributed matrix multiplication $C = A \times B$ from `.npy`, `.npz`, `.safetensors`, or inline tensors. |
| `batch_similarity_search` | Runs massive exact $K$-NN cosine similarity search using `OUTPUT_GRID_2D` zero-reduction tiling. |
| `compute_pca_svd` | Runs verified top-$k$ randomized SVD or stateful remote-VRAM power-iteration PCA/SVD. |

### 3. High-Level Verifiable AI Primitives (`matrix_compute_mcp.ai`)

```python
import numpy as np
from matrix_compute_mcp.ai import (
    DistributedLinear,
    DistributedLoRALinear,
    distributed_cosine_similarity,
    distributed_pca_svd,
)

# 1. Exact K-NN Cosine Similarity with OUTPUT_GRID_2D zero download redundancy
scores, indices = distributed_cosine_similarity(queries, corpus_embeddings, top_k=10)

# 2. Drop-in PyTorch Linear & LoRA Linear layers with verified forward & backward passes
layer = DistributedLinear(in_features=4096, out_features=4096)
lora_layer = DistributedLoRALinear(in_features=4096, out_features=4096, r=16, lora_alpha=32.0)

# 3. Distributed Randomized SVD / Stateful Power-Iteration PCA
U, S, Vt = distributed_pca_svd(embeddings, k=32, method="randomized")
```
