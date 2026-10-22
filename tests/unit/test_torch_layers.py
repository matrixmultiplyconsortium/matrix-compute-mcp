"""Unit tests for `DistributedLinear` and `DistributedLoRALinear` (`matrix_compute_mcp.ai.torch_layers`)."""

import numpy as np
import pytest

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig
from matrix_compute_mcp.ai.torch_layers import (
    DistributedLinear,
    DistributedLoRALinear,
    _TORCH_AVAILABLE,
)

if _TORCH_AVAILABLE:
    import torch
else:
    from matrix_compute_mcp.ai.torch_layers import Tensor


@pytest.fixture(scope="module")
def shared_client():
    cfg = ClientConfig.load(offline_mock_mode=True, default_tile_size=16)
    with MatrixComputeClient(config=cfg) as client:
        yield client


def _make_tensor(arr: np.ndarray, requires_grad: bool = False):
    if _TORCH_AVAILABLE:
        return torch.tensor(arr, dtype=torch.float32, requires_grad=requires_grad)
    return Tensor(arr, requires_grad=requires_grad)


def _to_numpy(t) -> np.ndarray:
    if hasattr(t, "detach") and hasattr(t, "cpu"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def test_distributed_linear_forward_2d_and_3d(shared_client):
    rng = np.random.RandomState(201)
    layer = DistributedLinear(16, 10, bias=True, client=shared_client)

    w_np = _to_numpy(layer.weight)
    b_np = _to_numpy(layer.bias)

    # 2D input: (batch=6, in_features=16)
    x_2d_np = rng.randn(6, 16).astype(np.float32)
    out_2d = layer(_make_tensor(x_2d_np))
    expected_2d = x_2d_np @ w_np.T + b_np
    np.testing.assert_allclose(_to_numpy(out_2d), expected_2d, rtol=1e-4, atol=1e-5)

    # 3D sequence input: (batch=2, seq_len=5, in_features=16)
    x_3d_np = rng.randn(2, 5, 16).astype(np.float32)
    out_3d = layer(_make_tensor(x_3d_np))
    expected_3d = (x_3d_np.reshape(-1, 16) @ w_np.T + b_np).reshape(2, 5, 10)
    assert out_3d.shape == (2, 5, 10)
    np.testing.assert_allclose(_to_numpy(out_3d), expected_3d, rtol=1e-4, atol=1e-5)


def test_distributed_linear_backward_gradients(shared_client):
    rng = np.random.RandomState(202)
    layer = DistributedLinear(12, 8, bias=True, client=shared_client)
    layer.zero_grad()

    x_np = rng.randn(5, 12).astype(np.float32)
    x = _make_tensor(x_np, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    w_np = _to_numpy(layer.weight)
    dy = np.ones((5, 8), dtype=np.float32)

    expected_dx = dy @ w_np
    expected_dw = dy.T @ x_np
    expected_db = dy.sum(axis=0)

    np.testing.assert_allclose(_to_numpy(x.grad), expected_dx, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(layer.weight.grad), expected_dw, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(layer.bias.grad), expected_db, rtol=1e-4, atol=1e-5)


def test_distributed_lora_linear_zero_init_and_adapter_backward(shared_client):
    rng = np.random.RandomState(203)
    lora_layer = DistributedLoRALinear(
        in_features=16,
        out_features=12,
        r=4,
        lora_alpha=8.0,
        bias=True,
        freeze_base=True,
        client=shared_client,
    )

    x_np = rng.randn(6, 16).astype(np.float32)
    w_np = _to_numpy(lora_layer.weight)
    b_np = _to_numpy(lora_layer.bias)

    # At initialization lora_B == 0, so output must match exact base projection
    y_init = lora_layer(_make_tensor(x_np))
    expected_base = x_np @ w_np.T + b_np
    np.testing.assert_allclose(_to_numpy(y_init), expected_base, rtol=1e-4, atol=1e-5)

    # Set non-zero lora_B and verify forward + backward
    b_adapter = rng.randn(12, 4).astype(np.float32) * 0.1
    a_adapter = _to_numpy(lora_layer.lora_A).copy()
    lora_layer.load_state_dict(
        {
            "weight": w_np,
            "bias": b_np,
            "lora_A": a_adapter,
            "lora_B": b_adapter,
        }
    )
    lora_layer.zero_grad()

    x = _make_tensor(x_np, requires_grad=True)
    y = lora_layer(x)

    scaling = 8.0 / 4.0
    expected_lora = expected_base + (x_np @ a_adapter.T @ b_adapter.T) * scaling
    np.testing.assert_allclose(_to_numpy(y), expected_lora, rtol=1e-4, atol=1e-5)

    y.sum().backward()

    # Frozen base weight should not accumulate gradients
    assert lora_layer.weight.grad is None
    assert lora_layer.lora_A.grad is not None
    assert lora_layer.lora_B.grad is not None

    dy = np.ones((6, 12), dtype=np.float32)
    expected_db_lora = (dy.T @ (x_np @ a_adapter.T)) * scaling
    expected_da_lora = ((dy @ b_adapter).T @ x_np) * scaling
    expected_dx = (dy @ w_np) + ((dy @ b_adapter) @ a_adapter) * scaling

    np.testing.assert_allclose(_to_numpy(lora_layer.lora_B.grad), expected_db_lora, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(lora_layer.lora_A.grad), expected_da_lora, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(_to_numpy(x.grad), expected_dx, rtol=1e-4, atol=1e-5)
