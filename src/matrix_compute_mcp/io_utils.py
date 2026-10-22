"""
Safe Tensor I/O utilities for matrix-compute-mcp.

Supports reading and writing 2D matrices from:
- NumPy `.npy` and `.npz` files
- HuggingFace `.safetensors` files (via `safetensors.numpy` or built-in spec-compliant binary parser/writer)
- Inline 2D JSON lists or NumPy arrays
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_SAFETENSORS_AVAILABLE = False
try:
    from safetensors.numpy import load_file as _st_load_file
    from safetensors.numpy import save_file as _st_save_file

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    pass

# SafeTensors dtype mapping per official specification
_NP_TO_ST_DTYPE: Dict[np.dtype, str] = {
    np.dtype("float32"): "F32",
    np.dtype("float16"): "F16",
    np.dtype("float64"): "F64",
    np.dtype("int32"): "I32",
    np.dtype("int64"): "I64",
    np.dtype("int16"): "I16",
    np.dtype("int8"): "I8",
    np.dtype("uint8"): "U8",
    np.dtype("bool"): "BOOL",
}

_ST_TO_NP_DTYPE: Dict[str, np.dtype] = {v: k for k, v in _NP_TO_ST_DTYPE.items()}

MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024  # 100 MB spec limit


def save_safetensors_bytes(tensors: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Serializes a dictionary of NumPy arrays into the official `.safetensors` binary format:
    - 8 bytes: little-endian unsigned 64-bit integer N (size of JSON header)
    - N bytes: UTF-8 JSON header mapping tensor names to {dtype, shape, data_offsets}
    - Raw contiguous little-endian tensor byte buffers
    """
    if not tensors:
        raise ValueError("Cannot save empty tensor dictionary to .safetensors")

    header: Dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    data_buffers: List[bytes] = []
    current_offset = 0

    for name, arr in tensors.items():
        if name == "__metadata__":
            raise ValueError("Tensor name '__metadata__' is reserved in .safetensors")
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        # Ensure little-endian contiguous array
        if arr.dtype.byteorder == ">":
            arr = arr.byteswap().newbyteorder("<")
        arr_contig = np.ascontiguousarray(arr)

        st_dtype = _NP_TO_ST_DTYPE.get(arr_contig.dtype)
        if st_dtype is None:
            arr_contig = np.ascontiguousarray(arr_contig, dtype=np.float32)
            st_dtype = "F32"

        raw = arr_contig.tobytes(order="C")
        start = current_offset
        end = start + len(raw)
        current_offset = end

        header[name] = {
            "dtype": st_dtype,
            "shape": list(arr_contig.shape),
            "data_offsets": [start, end],
        }
        data_buffers.append(raw)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment with spaces per SafeTensors convention
    remainder = len(header_bytes) % 8
    if remainder != 0:
        header_bytes += b" " * (8 - remainder)

    header_len_bytes = struct.pack("<Q", len(header_bytes))
    return header_len_bytes + header_bytes + b"".join(data_buffers)


def load_safetensors_bytes(data: bytes) -> Dict[str, np.ndarray]:
    """
    Parses a `.safetensors` binary payload into a dictionary of NumPy arrays.
    """
    if len(data) < 8:
        raise ValueError("Invalid .safetensors file: fewer than 8 bytes")

    (header_len,) = struct.unpack("<Q", data[:8])
    if header_len == 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"Invalid .safetensors header length: {header_len}")

    header_end = 8 + header_len
    if len(data) < header_end:
        raise ValueError("Invalid .safetensors file: truncated JSON header")

    try:
        header_str = data[8:header_end].decode("utf-8")
        header = json.loads(header_str)
    except Exception as exc:
        raise ValueError(f"Invalid .safetensors JSON header: {exc}") from exc

    if not isinstance(header, dict):
        raise ValueError("Invalid .safetensors header: root must be a JSON object")

    buffer_view = memoryview(data)[header_end:]
    result: Dict[str, np.ndarray] = {}

    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"Invalid tensor entry for '{key}'")

        st_dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")

        if st_dtype not in _ST_TO_NP_DTYPE and st_dtype != "BF16":
            raise ValueError(f"Unsupported .safetensors dtype '{st_dtype}' for tensor '{key}'")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ValueError(f"Invalid shape for tensor '{key}': {shape}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) and o >= 0 for o in offsets)
        ):
            raise ValueError(f"Invalid data_offsets for tensor '{key}': {offsets}")

        start, end = offsets
        if start > end or end > len(buffer_view):
            raise ValueError(f"Out-of-bounds data_offsets [{start}, {end}] for tensor '{key}'")

        raw_slice = buffer_view[start:end]
        if st_dtype == "BF16":
            # Convert bfloat16 raw uint16 high words to float32
            u16 = np.frombuffer(raw_slice, dtype="<u2")
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape).copy()
        else:
            np_dtype = _ST_TO_NP_DTYPE[st_dtype].newbyteorder("<")
            arr = np.frombuffer(raw_slice, dtype=np_dtype).reshape(shape).copy()

        result[key] = arr

    return result


def save_safetensors_file(
    tensors: Dict[str, np.ndarray],
    path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Saves a dictionary of NumPy arrays to a `.safetensors` file."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _SAFETENSORS_AVAILABLE:
        _st_save_file(tensors, str(out_path), metadata=metadata)
    else:
        payload = save_safetensors_bytes(tensors, metadata=metadata)
        out_path.write_bytes(payload)
    return out_path


def load_safetensors_file(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Loads a dictionary of NumPy arrays from a `.safetensors` file."""
    in_path = Path(path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"SafeTensors file not found: {in_path}")
    if _SAFETENSORS_AVAILABLE:
        return _st_load_file(str(in_path))
    return load_safetensors_bytes(in_path.read_bytes())


def load_matrix_input(
    source: Union[str, Path, List[List[float]], np.ndarray],
    tensor_key: Optional[str] = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """
    Loads a 2D matrix from a file path (`.npy`, `.npz`, `.safetensors`), an inline
    2D list, or a NumPy ndarray.

    Args:
        source: File path string/Path, nested Python list, or np.ndarray.
        tensor_key: Optional key name when loading from `.safetensors` or `.npz`.
            If omitted, uses the first available tensor key.
        dtype: Target NumPy dtype (default `np.float32`).

    Returns:
        Contiguous 2D `np.ndarray`.
    """
    if isinstance(source, np.ndarray):
        arr = source
    elif isinstance(source, (list, tuple)):
        arr = np.asarray(source, dtype=dtype)
    elif isinstance(source, (str, Path)):
        path_str = str(source).strip()
        # Check if inline JSON array string was passed
        if path_str.startswith("["):
            try:
                parsed = json.loads(path_str)
                arr = np.asarray(parsed, dtype=dtype)
            except Exception as exc:
                raise ValueError(f"Failed to parse inline JSON array: {exc}") from exc
        else:
            file_path = Path(path_str).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Matrix file not found: {file_path}")

            suffix = file_path.suffix.lower()
            if suffix == ".npy":
                arr = np.load(str(file_path), allow_pickle=False)
            elif suffix == ".npz":
                with np.load(str(file_path), allow_pickle=False) as npz:
                    keys = list(npz.keys())
                    if not keys:
                        raise ValueError(f"Empty .npz archive: {file_path}")
                    key = tensor_key if tensor_key is not None else keys[0]
                    if key not in npz:
                        raise KeyError(f"Key '{key}' not found in {file_path}. Available: {keys}")
                    arr = npz[key]
            elif suffix == ".safetensors":
                tensors = load_safetensors_file(file_path)
                keys = list(tensors.keys())
                if not keys:
                    raise ValueError(f"No tensors found in .safetensors file: {file_path}")
                key = tensor_key if tensor_key is not None else keys[0]
                if key not in tensors:
                    raise KeyError(f"Key '{key}' not found in {file_path}. Available: {keys}")
                arr = tensors[key]
            else:
                raise ValueError(
                    f"Unsupported matrix file format '{suffix}'. Supported: .npy, .npz, .safetensors"
                )
    else:
        raise TypeError(f"Unsupported matrix input type: {type(source).__name__}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {arr.ndim}D array with shape {arr.shape}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Matrix dimensions must be non-zero, got shape {arr.shape}")

    if dtype is not None and arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype)
    return np.ascontiguousarray(arr)


def save_matrix_output(
    matrix: np.ndarray,
    output_path: Union[str, Path],
    tensor_key: str = "result",
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Saves a 2D NumPy matrix to `.npy`, `.npz`, or `.safetensors`.
    """
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    if suffix == ".npy":
        np.save(str(out_path), matrix, allow_pickle=False)
    elif suffix == ".npz":
        np.savez(str(out_path), **{tensor_key: matrix})
    elif suffix == ".safetensors":
        save_safetensors_file({tensor_key: matrix}, out_path, metadata=metadata)
    else:
        raise ValueError(
            f"Unsupported output file format '{suffix}'. Supported: .npy, .npz, .safetensors"
        )
    return out_path
