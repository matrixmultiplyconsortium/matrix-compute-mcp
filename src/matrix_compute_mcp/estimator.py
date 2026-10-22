"""
Bandwidth and Satoshi Cost Estimator for matrix-compute-mcp.

Wraps `matrix_compute.tiling.profiler.BandwidthProfiler` to compute exact network
upload/download bandwidth (MB), task counts, GFLOPs, and satoshi cost estimates
across `output_grid_2d`, `outer_product_1d`, and `block_3d` partitioning topologies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional

from matrix_compute.config import PartitioningStrategy
from matrix_compute.tiling.precision import get_bytes_per_element, normalize_dtype
from matrix_compute.tiling.profiler import BandwidthProfile, BandwidthProfiler

DEFAULT_SATS_PER_GFLOP = 0.0005
DEFAULT_PLATFORM_FEE_RATE = 0.05
MIN_TASK_BOUNTY_SATS = 1


@dataclass
class JobCostEstimate:
    """Exact bandwidth and satoshi cost estimate for a distributed matrix job."""

    m: int
    k: int
    n: int
    dtype: str
    strategy: str
    num_tasks: int
    total_flops: int
    total_gflops: float
    bytes_per_element: int
    upload_bytes: int
    upload_mb: float
    download_bytes: int
    download_mb: float
    total_transfer_bytes: int
    total_transfer_mb: float
    upload_redundancy_ratio: float
    download_redundancy_ratio: float
    is_upload_optimal: bool
    is_download_optimal: bool
    sats_per_gflop: float
    estimated_worker_bounty_sats: int
    estimated_platform_fee_sats: int
    estimated_total_sats: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_strategy_name(strategy: Optional[str]) -> str:
    if strategy is None:
        return "output_grid_2d"
    if isinstance(strategy, PartitioningStrategy):
        return strategy.value.lower()
    s = str(strategy).strip().lower()
    if s in ("output_grid_2d", "2d", "grid_2d"):
        return "output_grid_2d"
    if s in ("outer_product_1d", "1d", "outer_product"):
        return "outer_product_1d"
    if s in ("block_3d", "3d", "block"):
        return "block_3d"
    raise ValueError(
        f"Unsupported partitioning_strategy '{strategy}'. "
        "Supported: 'output_grid_2d', 'outer_product_1d', 'block_3d'"
    )


def estimate_job_cost(
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
    platform_fee_rate: float = DEFAULT_PLATFORM_FEE_RATE,
) -> JobCostEstimate:
    """
    Calculates exact bandwidth (MB) and satoshi cost estimates for C (m x n) = A (m x k) x B (k x n)
    using `matrix_compute.tiling.profiler.BandwidthProfiler` without allocating full matrices in RAM.

    Args:
        m: Rows of matrix A.
        k: Inner dimension (columns of A / rows of B).
        n: Columns of matrix B.
        dtype: Precision format ('fp32', 'fp16', 'bf16').
        partitioning_strategy: 'output_grid_2d', 'outer_product_1d', or 'block_3d'.
        tile_size: Block dimension for 3D tiling or default 2D grid sizing.
        num_strips: Number of K-strips for 1D outer product partitioning.
        grid_rows: Row partitions P_r for 2D output grid (inferred from tile_size if None).
        grid_cols: Column partitions P_c for 2D output grid (inferred from tile_size if None).
        sats_per_gflop: Worker price in satoshis per GFLOP.
        platform_fee_rate: Platform routing fee fraction (default 5% = 0.05).

    Returns:
        `JobCostEstimate` instance with exact bandwidth and satoshi projections.
    """
    if m <= 0 or k <= 0 or n <= 0:
        raise ValueError(f"Matrix dimensions (m, k, n) must all be > 0, got ({m}, {k}, {n})")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if num_strips <= 0:
        raise ValueError(f"num_strips must be > 0, got {num_strips}")

    norm_dtype = normalize_dtype(dtype)
    bytes_per_elem = get_bytes_per_element(norm_dtype)
    strat = _normalize_strategy_name(partitioning_strategy)

    if strat == "output_grid_2d":
        p_r = grid_rows if grid_rows is not None else max(1, math.ceil(m / tile_size))
        p_c = grid_cols if grid_cols is not None else max(1, math.ceil(n / tile_size))
        if p_r <= 0 or p_c <= 0:
            raise ValueError(f"grid_rows and grid_cols must be > 0, got ({p_r}, {p_c})")
        profile = BandwidthProfiler.theoretical_2d_output_grid_bandwidth(
            m=m,
            n=n,
            k=k,
            p_r=p_r,
            p_c=p_c,
            dtype=norm_dtype,
        )
    elif strat == "outer_product_1d":
        eff_strips = min(num_strips, k)
        # Construct synthetic lightweight task descriptors for BandwidthProfiler.profile_tasks
        tasks: List[Dict[str, Any]] = []
        base_width = k // eff_strips
        remainder = k % eff_strips
        k_cursor = 0
        for idx in range(eff_strips):
            w = base_width + (1 if idx < remainder else 0)
            tasks.append(
                {
                    "m_i": m,
                    "k_j": w,
                    "n_k": n,
                    "row_slice": (0, m),
                    "col_slice": (0, n),
                    "k_slice": (k_cursor, k_cursor + w),
                }
            )
            k_cursor += w
        profile = BandwidthProfiler.profile_tasks(
            tasks=tasks,
            m=m,
            k=k,
            n=n,
            dtype=norm_dtype,
            strategy="outer_product_1d",
        )
    else:
        # block_3d
        info_3d = BandwidthProfiler.theoretical_3d_block_upload(m=m, k=k, n=n, tile_size=tile_size)
        l_blocks = info_3d["l_blocks"]
        optimal_download_elems = m * n
        download_elems = l_blocks * m * n
        download_redundancy_ratio = float(l_blocks - 1)
        profile = BandwidthProfile(
            m=m,
            k=k,
            n=n,
            num_tasks=info_3d["total_tasks"],
            strategy="block_3d",
            dtype=norm_dtype,
            total_a_elements=info_3d["j_blocks"] * m * k,
            total_b_elements=info_3d["i_blocks"] * k * n,
            total_uploaded_elements=info_3d["total_uploaded_elements"],
            optimal_elements=info_3d["optimal_elements"],
            redundancy_elements=info_3d["redundancy_elements"],
            redundancy_ratio=info_3d["redundancy_ratio"],
            bytes_per_element=bytes_per_elem,
            total_uploaded_bytes=info_3d["total_uploaded_elements"] * bytes_per_elem,
            optimal_bytes=info_3d["optimal_elements"] * bytes_per_elem,
            redundancy_bytes=info_3d["redundancy_elements"] * bytes_per_elem,
            is_upload_optimal=(info_3d["redundancy_elements"] == 0),
            download_elements=download_elems,
            optimal_download_elements=optimal_download_elems,
            download_redundancy_ratio=download_redundancy_ratio,
            is_download_optimal=(l_blocks == 1),
        )

    upload_bytes = int(profile.total_uploaded_bytes)
    download_bytes = int(profile.download_elements * bytes_per_elem)
    total_transfer_bytes = upload_bytes + download_bytes

    mb_divisor = 1024.0 * 1024.0
    upload_mb = round(upload_bytes / mb_divisor, 6)
    download_mb = round(download_bytes / mb_divisor, 6)
    total_transfer_mb = round(total_transfer_bytes / mb_divisor, 6)

    total_flops = 2 * int(m) * int(k) * int(n)
    total_gflops = float(total_flops) / 1e9

    eff_price = float(sats_per_gflop) if sats_per_gflop is not None else DEFAULT_SATS_PER_GFLOP
    if eff_price < 0:
        raise ValueError(f"sats_per_gflop must be >= 0, got {eff_price}")

    raw_bounty = math.ceil(total_gflops * eff_price)
    worker_bounty_sats = max(MIN_TASK_BOUNTY_SATS * profile.num_tasks, raw_bounty) if eff_price > 0 else 0
    platform_fee_sats = int(round(worker_bounty_sats * max(0.0, float(platform_fee_rate))))
    total_sats = worker_bounty_sats + platform_fee_sats

    return JobCostEstimate(
        m=m,
        k=k,
        n=n,
        dtype=norm_dtype,
        strategy=profile.strategy,
        num_tasks=profile.num_tasks,
        total_flops=total_flops,
        total_gflops=round(total_gflops, 6),
        bytes_per_element=bytes_per_elem,
        upload_bytes=upload_bytes,
        upload_mb=upload_mb,
        download_bytes=download_bytes,
        download_mb=download_mb,
        total_transfer_bytes=total_transfer_bytes,
        total_transfer_mb=total_transfer_mb,
        upload_redundancy_ratio=round(profile.redundancy_ratio, 6),
        download_redundancy_ratio=round(profile.download_redundancy_ratio, 6),
        is_upload_optimal=profile.is_upload_optimal,
        is_download_optimal=profile.is_download_optimal,
        sats_per_gflop=eff_price,
        estimated_worker_bounty_sats=worker_bounty_sats,
        estimated_platform_fee_sats=platform_fee_sats,
        estimated_total_sats=total_sats,
    )
