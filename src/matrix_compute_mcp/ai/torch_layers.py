"""
Drop-in PyTorch Distributed Linear & LoRA Linear Layers (`matrix_compute_mcp.ai.torch_layers`).

Provides `DistributedLinear` and `DistributedLoRALinear` that offload large batched
forward and backward linear projections to decentralized GPU workers while verifying
every GEMM pass mathematically via Freivalds' algorithm.

Supports native `torch.nn.Module` + `torch.autograd.Function` when PyTorch is installed,
and includes a full NumPy-backed autograd `Tensor` / `Parameter` / `Module` engine when
running in lightweight environments without PyTorch binaries.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from matrix_compute.client import MatrixComputeClient
from matrix_compute.config import ClientConfig, PartitioningStrategy

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _execute_verified_matmul(
    a: np.ndarray,
    b: np.ndarray,
    client: Optional[MatrixComputeClient] = None,
    config: Optional[ClientConfig] = None,
    dtype: str = "fp32",
    tile_size: Optional[int] = None,
    partitioning_strategy: Union[str, PartitioningStrategy] = PartitioningStrategy.OUTPUT_GRID_2D,
) -> np.ndarray:
    """Executes verified C = A @ B on the distributed compute network."""
    a_contig = np.ascontiguousarray(a, dtype=np.float32)
    b_contig = np.ascontiguousarray(b, dtype=np.float32)

    if client is not None:
        return client.matmul(
            A=a_contig,
            B=b_contig,
            dtype=dtype,
            tile_size=tile_size,
            partitioning_strategy=partitioning_strategy,
        )

    eff_config = config if config is not None else ClientConfig.load(offline_mock_mode=True)
    with MatrixComputeClient(config=eff_config) as temp_client:
        return temp_client.matmul(
            A=a_contig,
            B=b_contig,
            dtype=dtype,
            tile_size=tile_size,
            partitioning_strategy=partitioning_strategy,
        )


if _TORCH_AVAILABLE:

    class _DistributedLinearAutograd(torch.autograd.Function):
        """
        Custom PyTorch Autograd Function offloading forward (Y = X @ W^T + b)
        and backward (dX = dY @ W, dW = dY^T @ X) to verified distributed workers.
        """

        @staticmethod
        def forward(
            ctx: Any,
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            client: Optional[MatrixComputeClient],
            config: Optional[ClientConfig],
            dtype: str,
            tile_size: Optional[int],
            partitioning_strategy: Union[str, PartitioningStrategy],
        ) -> torch.Tensor:
            ctx.save_for_backward(input_tensor, weight, bias)
            ctx.client = client
            ctx.config = config
            ctx.dtype = dtype
            ctx.tile_size = tile_size
            ctx.partitioning_strategy = partitioning_strategy

            orig_shape = input_tensor.shape
            in_features = orig_shape[-1]
            x_2d = input_tensor.detach().cpu().numpy().reshape(-1, in_features)
            w_np = weight.detach().cpu().numpy()  # (out_features, in_features)

            y_2d = _execute_verified_matmul(
                a=x_2d,
                b=w_np.T,
                client=client,
                config=config,
                dtype=dtype,
                tile_size=tile_size,
                partitioning_strategy=partitioning_strategy,
            )

            if bias is not None:
                b_np = bias.detach().cpu().numpy()
                y_2d = y_2d + b_np.reshape(1, -1)

            out_shape = (*orig_shape[:-1], w_np.shape[0])
            out_np = y_2d.reshape(out_shape)
            return torch.from_numpy(out_np).to(device=input_tensor.device, dtype=input_tensor.dtype)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Any, ...]:
            input_tensor, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None

            orig_shape = input_tensor.shape
            in_features = orig_shape[-1]
            out_features = weight.shape[0]
            dy_2d = grad_output.detach().cpu().numpy().reshape(-1, out_features)

            if ctx.needs_input_grad[0]:
                w_np = weight.detach().cpu().numpy()  # (out_features, in_features)
                dx_2d = _execute_verified_matmul(
                    a=dy_2d,
                    b=w_np,
                    client=ctx.client,
                    config=ctx.config,
                    dtype=ctx.dtype,
                    tile_size=ctx.tile_size,
                    partitioning_strategy=ctx.partitioning_strategy,
                )
                grad_input = torch.from_numpy(dx_2d.reshape(orig_shape)).to(
                    device=input_tensor.device, dtype=input_tensor.dtype
                )

            if ctx.needs_input_grad[1]:
                x_2d = input_tensor.detach().cpu().numpy().reshape(-1, in_features)
                # dW = dY^T @ X  (out_features x batch) @ (batch x in_features)
                dw_2d = _execute_verified_matmul(
                    a=dy_2d.T,
                    b=x_2d,
                    client=ctx.client,
                    config=ctx.config,
                    dtype=ctx.dtype,
                    tile_size=ctx.tile_size,
                    partitioning_strategy=ctx.partitioning_strategy,
                )
                grad_weight = torch.from_numpy(dw_2d).to(device=weight.device, dtype=weight.dtype)

            if bias is not None and ctx.needs_input_grad[2]:
                db_np = dy_2d.sum(axis=0)
                grad_bias = torch.from_numpy(db_np).to(device=bias.device, dtype=bias.dtype)

            return grad_input, grad_weight, grad_bias, None, None, None, None, None

    class DistributedLinear(nn.Module):
        """
        Drop-in replacement for `torch.nn.Linear` that offloads batched forward and backward
        linear projections ($Y = X W^T + b$) to the distributed GPU compute network with
        Freivalds mathematical verification.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            client: Optional[MatrixComputeClient] = None,
            config: Optional[ClientConfig] = None,
            dtype_preference: str = "fp32",
            tile_size: Optional[int] = None,
            partitioning_strategy: Union[str, PartitioningStrategy] = PartitioningStrategy.OUTPUT_GRID_2D,
            device: Any = None,
            dtype: Any = None,
        ) -> None:
            super().__init__()
            if in_features <= 0 or out_features <= 0:
                raise ValueError(
                    f"in_features and out_features must be > 0, got ({in_features}, {out_features})"
                )
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.client = client
            self.config = config
            self.dtype_preference = dtype_preference
            self.tile_size = tile_size
            self.partitioning_strategy = partitioning_strategy

            factory_kwargs = {}
            if device is not None:
                factory_kwargs["device"] = device
            if dtype is not None:
                factory_kwargs["dtype"] = dtype

            self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if input.shape[-1] != self.in_features:
                raise ValueError(
                    f"Input feature dimension mismatch: expected {self.in_features}, got {input.shape[-1]}"
                )
            return _DistributedLinearAutograd.apply(
                input,
                self.weight,
                self.bias,
                self.client,
                self.config,
                self.dtype_preference,
                self.tile_size,
                self.partitioning_strategy,
            )

    class DistributedLoRALinear(nn.Module):
        """
        Drop-in LoRA Linear layer (`torch.nn.Module`) that offloads the frozen base projection
        $X W_0^T$ to remote verified GPU workers while computing the low-rank adapter
        $\\frac{\\alpha}{r} (X A^T) B^T$ locally.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: float = 16.0,
            lora_dropout: float = 0.0,
            bias: bool = True,
            freeze_base: bool = True,
            client: Optional[MatrixComputeClient] = None,
            config: Optional[ClientConfig] = None,
            dtype_preference: str = "fp32",
            tile_size: Optional[int] = None,
            partitioning_strategy: Union[str, PartitioningStrategy] = PartitioningStrategy.OUTPUT_GRID_2D,
        ) -> None:
            super().__init__()
            if in_features <= 0 or out_features <= 0:
                raise ValueError(
                    f"in_features and out_features must be > 0, got ({in_features}, {out_features})"
                )
            if r <= 0:
                raise ValueError(f"LoRA rank r must be > 0, got {r}")

            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.r = int(r)
            self.lora_alpha = float(lora_alpha)
            self.scaling = self.lora_alpha / float(self.r)
            self.freeze_base = freeze_base
            self.client = client
            self.config = config
            self.dtype_preference = dtype_preference
            self.tile_size = tile_size
            self.partitioning_strategy = partitioning_strategy

            self.weight = nn.Parameter(
                torch.empty((out_features, in_features)), requires_grad=not freeze_base
            )
            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if input.shape[-1] != self.in_features:
                raise ValueError(
                    f"Input feature dimension mismatch: expected {self.in_features}, got {input.shape[-1]}"
                )
            base_out = _DistributedLinearAutograd.apply(
                input,
                self.weight,
                self.bias,
                self.client,
                self.config,
                self.dtype_preference,
                self.tile_size,
                self.partitioning_strategy,
            )
            lora_out = (self.dropout(input) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return base_out + lora_out

else:
    # -------------------------------------------------------------------------
    # Built-in Autograd Tensor / Parameter / Module Engine for PyTorch-less environments
    # -------------------------------------------------------------------------

    class Tensor:
        """
        Lightweight PyTorch-compatible autograd Tensor backed by NumPy `ndarray`.
        Supports `.backward()`, `.grad`, `.detach()`, `.numpy()`, and shape operations.
        """

        def __init__(
            self,
            data: Union[np.ndarray, List[Any], float],
            requires_grad: bool = False,
            _children: Tuple["Tensor", ...] = (),
            _backward_fn: Optional[Callable[[], None]] = None,
        ) -> None:
            self.data: np.ndarray = np.ascontiguousarray(data, dtype=np.float32)
            self.requires_grad: bool = bool(requires_grad)
            self.grad: Optional[np.ndarray] = None
            self._prev: Tuple["Tensor", ...] = _children
            self._backward_fn: Optional[Callable[[], None]] = _backward_fn

        @property
        def shape(self) -> Tuple[int, ...]:
            return self.data.shape

        @property
        def ndim(self) -> int:
            return self.data.ndim

        @property
        def dtype(self) -> np.dtype:
            return self.data.dtype

        @property
        def T(self) -> "Tensor":
            return Tensor(self.data.T, requires_grad=self.requires_grad)

        def __array__(self, dtype: Any = None) -> np.ndarray:
            if dtype is not None:
                return self.data.astype(dtype)
            return self.data

        def numpy(self) -> np.ndarray:
            return self.data

        def detach(self) -> "Tensor":
            return Tensor(self.data.copy(), requires_grad=False)

        def cpu(self) -> "Tensor":
            return self

        def tolist(self) -> Any:
            return self.data.tolist()

        def item(self) -> float:
            return float(self.data.item())

        def zero_grad(self) -> None:
            self.grad = None

        def sum(self) -> "Tensor":
            out = Tensor(
                np.array(self.data.sum(), dtype=np.float32),
                requires_grad=self.requires_grad,
                _children=(self,) if self.requires_grad else (),
            )

            def _backward() -> None:
                if self.requires_grad and out.grad is not None:
                    g = np.ones_like(self.data, dtype=np.float32) * float(out.grad)
                    self.grad = g if self.grad is None else (self.grad + g)

            out._backward_fn = _backward
            return out

        def mean(self) -> "Tensor":
            n = float(self.data.size)
            out = Tensor(
                np.array(self.data.mean(), dtype=np.float32),
                requires_grad=self.requires_grad,
                _children=(self,) if self.requires_grad else (),
            )

            def _backward() -> None:
                if self.requires_grad and out.grad is not None:
                    g = (np.ones_like(self.data, dtype=np.float32) / n) * float(out.grad)
                    self.grad = g if self.grad is None else (self.grad + g)

            out._backward_fn = _backward
            return out

        def __add__(self, other: Union["Tensor", np.ndarray, float]) -> "Tensor":
            other_t = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
            req_grad = self.requires_grad or other_t.requires_grad
            out = Tensor(
                self.data + other_t.data,
                requires_grad=req_grad,
                _children=(self, other_t) if req_grad else (),
            )

            def _backward() -> None:
                if out.grad is None:
                    return
                if self.requires_grad:
                    g_self = out.grad
                    while g_self.ndim > self.data.ndim:
                        g_self = g_self.sum(axis=0)
                    for ax, dim in enumerate(self.data.shape):
                        if dim == 1 and g_self.shape[ax] > 1:
                            g_self = g_self.sum(axis=ax, keepdims=True)
                    self.grad = g_self if self.grad is None else (self.grad + g_self)
                if other_t.requires_grad:
                    g_other = out.grad
                    while g_other.ndim > other_t.data.ndim:
                        g_other = g_other.sum(axis=0)
                    for ax, dim in enumerate(other_t.data.shape):
                        if dim == 1 and g_other.shape[ax] > 1:
                            g_other = g_other.sum(axis=ax, keepdims=True)
                    other_t.grad = g_other if other_t.grad is None else (other_t.grad + g_other)

            out._backward_fn = _backward
            return out

        def __mul__(self, scalar: float) -> "Tensor":
            s = float(scalar)
            out = Tensor(
                self.data * s,
                requires_grad=self.requires_grad,
                _children=(self,) if self.requires_grad else (),
            )

            def _backward() -> None:
                if self.requires_grad and out.grad is not None:
                    g = out.grad * s
                    self.grad = g if self.grad is None else (self.grad + g)

            out._backward_fn = _backward
            return out

        def backward(self, grad: Optional[Union[np.ndarray, "Tensor"]] = None) -> None:
            if grad is None:
                if self.data.size == 1:
                    self.grad = np.ones_like(self.data, dtype=np.float32)
                else:
                    self.grad = np.ones_like(self.data, dtype=np.float32)
            elif isinstance(grad, Tensor):
                self.grad = grad.data.copy()
            else:
                self.grad = np.asarray(grad, dtype=np.float32).copy()

            topo: List[Tensor] = []
            visited = set()

            def _build_topo(v: Tensor) -> None:
                if id(v) not in visited:
                    visited.add(id(v))
                    for child in v._prev:
                        _build_topo(child)
                    topo.append(v)

            _build_topo(self)
            for v in reversed(topo):
                if v._backward_fn is not None:
                    v._backward_fn()

    class Parameter(Tensor):
        """Trainable parameter tensor."""

        def __init__(self, data: Union[np.ndarray, List[Any]], requires_grad: bool = True) -> None:
            super().__init__(data=data, requires_grad=requires_grad)

    class Module:
        """Base neural network module compatible with `torch.nn.Module`."""

        def __init__(self) -> None:
            self._parameters: Dict[str, Optional[Parameter]] = {}
            self.training: bool = True

        def __setattr__(self, name: str, value: Any) -> None:
            if isinstance(value, Parameter):
                if "_parameters" in self.__dict__:
                    self._parameters[name] = value
            super().__setattr__(name, value)

        def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
            self._parameters[name] = param
            super().__setattr__(name, param)

        def parameters(self) -> Iterator[Parameter]:
            for p in self._parameters.values():
                if p is not None and p.requires_grad:
                    yield p

        def named_parameters(self) -> Iterator[Tuple[str, Parameter]]:
            for name, p in self._parameters.items():
                if p is not None:
                    yield name, p

        def zero_grad(self) -> None:
            for p in self._parameters.values():
                if p is not None:
                    p.zero_grad()

        def state_dict(self) -> Dict[str, np.ndarray]:
            return {name: p.data.copy() for name, p in self._parameters.items() if p is not None}

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            for name, val in state_dict.items():
                if name in self._parameters and self._parameters[name] is not None:
                    arr = val.data if hasattr(val, "data") else np.asarray(val, dtype=np.float32)
                    if arr.shape != self._parameters[name].shape:
                        raise ValueError(
                            f"Shape mismatch for parameter '{name}': expected {self._parameters[name].shape}, got {arr.shape}"
                        )
                    self._parameters[name].data = np.ascontiguousarray(arr, dtype=np.float32)

        def train(self, mode: bool = True) -> "Module":
            self.training = bool(mode)
            return self

        def eval(self) -> "Module":
            return self.train(False)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    class DistributedLinear(Module):
        r"""
        Drop-in `torch.nn.Linear` replacement offloading batched forward ($Y = X W^T + b$)
        and backward ($\nabla X = \nabla Y W$, $\nabla W = \nabla Y^T X$) projections to
        distributed GPU workers with Freivalds verification.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            client: Optional[MatrixComputeClient] = None,
            config: Optional[ClientConfig] = None,
            dtype_preference: str = "fp32",
            tile_size: Optional[int] = None,
            partitioning_strategy: Union[str, PartitioningStrategy] = PartitioningStrategy.OUTPUT_GRID_2D,
            device: Any = None,
            dtype: Any = None,
        ) -> None:
            super().__init__()
            if in_features <= 0 or out_features <= 0:
                raise ValueError(
                    f"in_features and out_features must be > 0, got ({in_features}, {out_features})"
                )
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.client = client
            self.config = config
            self.dtype_preference = dtype_preference
            self.tile_size = tile_size
            self.partitioning_strategy = partitioning_strategy

            rng = np.random.RandomState(42)
            bound = 1.0 / math.sqrt(self.in_features)
            w_init = rng.uniform(-bound, bound, size=(self.out_features, self.in_features)).astype(
                np.float32
            )
            self.weight = Parameter(w_init, requires_grad=True)
            if bias:
                b_init = rng.uniform(-bound, bound, size=(self.out_features,)).astype(np.float32)
                self.bias: Optional[Parameter] = Parameter(b_init, requires_grad=True)
            else:
                self.register_parameter("bias", None)

        def reset_parameters(self) -> None:
            rng = np.random.RandomState(42)
            bound = 1.0 / math.sqrt(self.in_features)
            self.weight.data = rng.uniform(
                -bound, bound, size=(self.out_features, self.in_features)
            ).astype(np.float32)
            if self.bias is not None:
                self.bias.data = rng.uniform(-bound, bound, size=(self.out_features,)).astype(
                    np.float32
                )

        def forward(self, input: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
            is_raw_ndarray = isinstance(input, np.ndarray) and not isinstance(input, Tensor)
            x_tensor = input if isinstance(input, Tensor) else Tensor(input, requires_grad=False)

            if x_tensor.ndim < 2 or x_tensor.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input with last dimension {self.in_features} (ndim >= 2), got shape {x_tensor.shape}"
                )

            orig_shape = x_tensor.shape
            x_2d = x_tensor.data.reshape(-1, self.in_features)
            w_np = self.weight.data  # (out_features, in_features)

            y_2d = _execute_verified_matmul(
                a=x_2d,
                b=w_np.T,
                client=self.client,
                config=self.config,
                dtype=self.dtype_preference,
                tile_size=self.tile_size,
                partitioning_strategy=self.partitioning_strategy,
            )

            if self.bias is not None:
                y_2d = y_2d + self.bias.data.reshape(1, -1)

            out_shape = (*orig_shape[:-1], self.out_features)
            out_data = y_2d.reshape(out_shape)

            children: List[Tensor] = [x_tensor, self.weight]
            if self.bias is not None:
                children.append(self.bias)
            req_grad = any(c.requires_grad for c in children)

            out_tensor = Tensor(
                out_data,
                requires_grad=req_grad,
                _children=tuple(children) if req_grad else (),
            )

            def _backward() -> None:
                if out_tensor.grad is None:
                    return
                dy_2d = out_tensor.grad.reshape(-1, self.out_features)

                if x_tensor.requires_grad:
                    dx_2d = _execute_verified_matmul(
                        a=dy_2d,
                        b=self.weight.data,
                        client=self.client,
                        config=self.config,
                        dtype=self.dtype_preference,
                        tile_size=self.tile_size,
                        partitioning_strategy=self.partitioning_strategy,
                    )
                    dx = dx_2d.reshape(orig_shape)
                    x_tensor.grad = dx if x_tensor.grad is None else (x_tensor.grad + dx)

                if self.weight.requires_grad:
                    dw_2d = _execute_verified_matmul(
                        a=dy_2d.T,
                        b=x_2d,
                        client=self.client,
                        config=self.config,
                        dtype=self.dtype_preference,
                        tile_size=self.tile_size,
                        partitioning_strategy=self.partitioning_strategy,
                    )
                    self.weight.grad = (
                        dw_2d if self.weight.grad is None else (self.weight.grad + dw_2d)
                    )

                if self.bias is not None and self.bias.requires_grad:
                    db = dy_2d.sum(axis=0)
                    self.bias.grad = db if self.bias.grad is None else (self.bias.grad + db)

            out_tensor._backward_fn = _backward
            if is_raw_ndarray and not x_tensor.requires_grad:
                return out_tensor
            return out_tensor

    class DistributedLoRALinear(Module):
        """
        Drop-in LoRA Linear layer that offloads the frozen base projection $X W_0^T$
        to remote verified GPU workers while computing the low-rank adapter
        $\\frac{\\alpha}{r} (X A^T) B^T$ locally.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 8,
            lora_alpha: float = 16.0,
            lora_dropout: float = 0.0,
            bias: bool = True,
            freeze_base: bool = True,
            client: Optional[MatrixComputeClient] = None,
            config: Optional[ClientConfig] = None,
            dtype_preference: str = "fp32",
            tile_size: Optional[int] = None,
            partitioning_strategy: Union[str, PartitioningStrategy] = PartitioningStrategy.OUTPUT_GRID_2D,
        ) -> None:
            super().__init__()
            if in_features <= 0 or out_features <= 0:
                raise ValueError(
                    f"in_features and out_features must be > 0, got ({in_features}, {out_features})"
                )
            if r <= 0:
                raise ValueError(f"LoRA rank r must be > 0, got {r}")

            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.r = int(r)
            self.lora_alpha = float(lora_alpha)
            self.scaling = self.lora_alpha / float(self.r)
            self.lora_dropout = float(lora_dropout)
            self.freeze_base = bool(freeze_base)
            self.client = client
            self.config = config
            self.dtype_preference = dtype_preference
            self.tile_size = tile_size
            self.partitioning_strategy = partitioning_strategy

            rng = np.random.RandomState(42)
            bound = 1.0 / math.sqrt(self.in_features)
            w_init = rng.uniform(-bound, bound, size=(self.out_features, self.in_features)).astype(
                np.float32
            )
            self.weight = Parameter(w_init, requires_grad=not self.freeze_base)
            if bias:
                self.bias: Optional[Parameter] = Parameter(
                    np.zeros(self.out_features, dtype=np.float32), requires_grad=True
                )
            else:
                self.register_parameter("bias", None)

            a_init = rng.uniform(-bound, bound, size=(self.r, self.in_features)).astype(np.float32)
            b_init = np.zeros((self.out_features, self.r), dtype=np.float32)
            self.lora_A = Parameter(a_init, requires_grad=True)
            self.lora_B = Parameter(b_init, requires_grad=True)

        def forward(self, input: Union[Tensor, np.ndarray]) -> Tensor:
            x_tensor = input if isinstance(input, Tensor) else Tensor(input, requires_grad=False)
            if x_tensor.ndim < 2 or x_tensor.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input with last dimension {self.in_features} (ndim >= 2), got shape {x_tensor.shape}"
                )

            orig_shape = x_tensor.shape
            x_2d = x_tensor.data.reshape(-1, self.in_features)

            # 1. Offload heavy base matrix multiplication X @ W_0^T to distributed network
            base_2d = _execute_verified_matmul(
                a=x_2d,
                b=self.weight.data.T,
                client=self.client,
                config=self.config,
                dtype=self.dtype_preference,
                tile_size=self.tile_size,
                partitioning_strategy=self.partitioning_strategy,
            )
            if self.bias is not None:
                base_2d = base_2d + self.bias.data.reshape(1, -1)

            # 2. Local low-rank LoRA projection: (X @ A^T) @ B^T * (alpha / r)
            hidden_r = x_2d @ self.lora_A.data.T  # (batch, r)
            lora_2d = (hidden_r @ self.lora_B.data.T) * self.scaling  # (batch, out_features)

            total_2d = base_2d + lora_2d
            out_shape = (*orig_shape[:-1], self.out_features)
            out_data = total_2d.reshape(out_shape)

            children: List[Tensor] = [x_tensor, self.weight, self.lora_A, self.lora_B]
            if self.bias is not None:
                children.append(self.bias)
            req_grad = any(c.requires_grad for c in children)

            out_tensor = Tensor(
                out_data,
                requires_grad=req_grad,
                _children=tuple(children) if req_grad else (),
            )

            def _backward() -> None:
                if out_tensor.grad is None:
                    return
                dy_2d = out_tensor.grad.reshape(-1, self.out_features)

                # LoRA adapter gradients (computed locally in O((d_in + d_out) * r * batch))
                if self.lora_B.requires_grad:
                    # dL/dB = scale * dY^T @ (X @ A^T)
                    db_lora = (dy_2d.T @ hidden_r) * self.scaling
                    self.lora_B.grad = (
                        db_lora if self.lora_B.grad is None else (self.lora_B.grad + db_lora)
                    )

                if self.lora_A.requires_grad:
                    # dL/dA = scale * (dY @ B)^T @ X
                    dy_b = dy_2d @ self.lora_B.data  # (batch, r)
                    da_lora = (dy_b.T @ x_2d) * self.scaling
                    self.lora_A.grad = (
                        da_lora if self.lora_A.grad is None else (self.lora_A.grad + da_lora)
                    )

                if self.bias is not None and self.bias.requires_grad:
                    db = dy_2d.sum(axis=0)
                    self.bias.grad = db if self.bias.grad is None else (self.bias.grad + db)

                if self.weight.requires_grad:
                    dw_2d = _execute_verified_matmul(
                        a=dy_2d.T,
                        b=x_2d,
                        client=self.client,
                        config=self.config,
                        dtype=self.dtype_preference,
                        tile_size=self.tile_size,
                        partitioning_strategy=self.partitioning_strategy,
                    )
                    self.weight.grad = (
                        dw_2d if self.weight.grad is None else (self.weight.grad + dw_2d)
                    )

                if x_tensor.requires_grad:
                    dx_base = _execute_verified_matmul(
                        a=dy_2d,
                        b=self.weight.data,
                        client=self.client,
                        config=self.config,
                        dtype=self.dtype_preference,
                        tile_size=self.tile_size,
                        partitioning_strategy=self.partitioning_strategy,
                    )
                    dx_lora = ((dy_2d @ self.lora_B.data) @ self.lora_A.data) * self.scaling
                    dx = (dx_base + dx_lora).reshape(orig_shape)
                    x_tensor.grad = dx if x_tensor.grad is None else (x_tensor.grad + dx)

            out_tensor._backward_fn = _backward
            return out_tensor
