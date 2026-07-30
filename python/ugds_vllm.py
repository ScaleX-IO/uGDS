"""Minimal uGDS model-weight iterator shaped for ``model.load_weights``."""

from __future__ import annotations

import ctypes as C
import os
from dataclasses import dataclass
from typing import Any, Iterator

_ABI_VERSION = 1
_SUCCESS = 0
_SINGLE_FILE = 1
_HF_INDEX = 2
_OPAQUE_FD = 1
_HIP_DMABUF = 1
_ALIGNMENT = 64 * 1024


class UGDSError(RuntimeError):
    pass


@dataclass(frozen=True)
class UGDSWeightConfig:
    checkpoint: str
    manifest: str
    controller_path: str
    capacity_lbas: int
    controller_page_size: int
    lba_size: int = 4096
    torch_device: str = "cuda"
    library: str = "libugds.so"


class _Status(C.Structure):
    _fields_ = [("err", C.c_int), ("cu_err", C.c_int)]


class _HandleValue(C.Union):
    _fields_ = [("fd", C.c_int), ("handle", C.c_void_p)]


class _HandleDescr(C.Structure):
    _fields_ = [("type", C.c_int), ("handle", _HandleValue)]


class _MapDescr(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("type", C.c_int),
        ("path", C.c_char_p),
    ]


class _Mapping(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("name", C.c_void_p),
        ("name_length", C.c_size_t),
        ("shard_index", C.c_size_t),
        ("shard_name", C.c_char_p),
        ("shard_path", C.c_char_p),
        ("dtype", C.c_char_p),
        ("shape", C.POINTER(C.c_uint64)),
        ("rank", C.c_size_t),
        ("file_offset", C.c_uint64),
        ("nbytes", C.c_uint64),
    ]


class _Geometry(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("namespace_id", C.c_uint32),
        ("lba_size", C.c_uint32),
        ("controller_page_size", C.c_uint32),
        ("capacity_lbas", C.c_uint64),
    ]


class _LbaMapDescr(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("manifest_path", C.c_char_p),
        ("tensor_map", C.c_void_p),
        ("geometry", C.POINTER(_Geometry)),
    ]


class _Plan(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("io_begin_lba", C.c_uint64),
        ("io_lba_count", C.c_uint64),
        ("io_offset", C.c_uint64),
        ("io_size", C.c_uint64),
        ("payload_skip", C.c_uint64),
        ("payload_size", C.c_uint64),
    ]


class _ReadDescr(C.Structure):
    _fields_ = [
        ("struct_size", C.c_size_t),
        ("abi_version", C.c_uint32),
        ("io_handle", C.c_void_p),
        ("destination", C.c_void_p),
        ("destination_size", C.c_size_t),
        ("staging", C.c_void_p),
        ("staging_size", C.c_size_t),
        ("staging_buffer_flags", C.c_int),
    ]


def _new(struct_type: type[C.Structure]) -> C.Structure:
    value = struct_type()
    value.struct_size = C.sizeof(struct_type)
    value.abi_version = _ABI_VERSION
    return value


def _check(status: _Status, operation: str) -> None:
    if status.err != _SUCCESS:
        raise UGDSError(
            f"{operation} failed: status={status.err}, gpu={status.cu_err}"
        )


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("ugds_vllm requires PyTorch") from error
    return torch


def _load_library(path: str) -> C.CDLL:
    lib = C.CDLL(path)
    lib.uGDSDriverOpen.argtypes = []
    lib.uGDSDriverOpen.restype = _Status
    lib.uGDSDriverClose.argtypes = []
    lib.uGDSDriverClose.restype = _Status
    lib.uGDSHandleRegister.argtypes = [C.POINTER(C.c_void_p), C.POINTER(_HandleDescr)]
    lib.uGDSHandleRegister.restype = _Status
    lib.uGDSHandleDeregister.argtypes = [C.c_void_p]
    lib.uGDSHandleDeregister.restype = None
    lib.uGDSBufRegister.argtypes = [C.c_void_p, C.c_size_t, C.c_int]
    lib.uGDSBufRegister.restype = _Status
    lib.uGDSBufDeregister.argtypes = [C.c_void_p]
    lib.uGDSBufDeregister.restype = _Status
    lib.uGDSTensorMapOpen.argtypes = [C.POINTER(C.c_void_p), C.POINTER(_MapDescr)]
    lib.uGDSTensorMapOpen.restype = _Status
    lib.uGDSTensorMapClose.argtypes = [C.c_void_p]
    lib.uGDSTensorMapClose.restype = None
    lib.uGDSTensorMapGetCount.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
    lib.uGDSTensorMapGetCount.restype = _Status
    lib.uGDSTensorMapGetByIndex.argtypes = [
        C.c_void_p,
        C.c_size_t,
        C.POINTER(_Mapping),
    ]
    lib.uGDSTensorMapGetByIndex.restype = _Status
    lib.uGDSTensorLbaMapOpen.argtypes = [
        C.POINTER(C.c_void_p),
        C.POINTER(_LbaMapDescr),
    ]
    lib.uGDSTensorLbaMapOpen.restype = _Status
    lib.uGDSTensorLbaMapClose.argtypes = [C.c_void_p]
    lib.uGDSTensorLbaMapClose.restype = None
    lib.uGDSTensorLbaMapPlan.argtypes = [
        C.c_void_p,
        C.POINTER(_Mapping),
        C.POINTER(_Plan),
    ]
    lib.uGDSTensorLbaMapPlan.restype = _Status
    lib.uGDSTensorReadInto.argtypes = [
        C.c_void_p,
        C.POINTER(_Mapping),
        C.POINTER(_ReadDescr),
    ]
    lib.uGDSTensorReadInto.restype = _Status
    return lib


_DTYPE_ATTR = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "U16": "uint16",
    "I16": "int16",
    "U32": "uint32",
    "I32": "int32",
    "U64": "uint64",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
    "C64": "complex64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "F8_E8M0": "float8_e8m0fnu",
    "F8_E4M3FNUZ": "float8_e4m3fnuz",
    "F8_E5M2FNUZ": "float8_e5m2fnuz",
}


def _torch_dtype(torch: Any, name: str) -> Any:
    attribute = _DTYPE_ATTR.get(name)
    dtype = getattr(torch, attribute, None) if attribute else None
    if dtype is None:
        raise UGDSError(f"PyTorch does not support safetensors dtype {name}")
    return dtype


def ugds_weights_iterator(config: UGDSWeightConfig) -> Iterator[tuple[str, Any]]:
    """Yield vLLM weight pairs; the MVP owns uGDS exclusively per process."""
    torch = _load_torch()
    device = torch.device(config.torch_device)
    if device.type != "cuda":
        raise ValueError("uGDS weights must target a CUDA/ROCm device")

    lib = _load_library(config.library)
    driver_open = False
    fd = -1
    handle = C.c_void_p()
    tensor_map = C.c_void_p()
    lba_map = C.c_void_p()
    staging_registered = False
    staging_address = 0
    staging_tensor = None

    try:
        _check(lib.uGDSDriverOpen(), "uGDSDriverOpen")
        driver_open = True
        fd = os.open(config.controller_path, os.O_RDWR | os.O_CLOEXEC)
        handle_descr = _HandleDescr(type=_OPAQUE_FD)
        handle_descr.handle.fd = fd
        _check(
            lib.uGDSHandleRegister(C.byref(handle), C.byref(handle_descr)),
            "uGDSHandleRegister",
        )

        checkpoint = os.fsencode(config.checkpoint)
        map_descr = _new(_MapDescr)
        map_descr.type = (
            _HF_INDEX if config.checkpoint.endswith(".index.json") else _SINGLE_FILE
        )
        map_descr.path = checkpoint
        _check(
            lib.uGDSTensorMapOpen(C.byref(tensor_map), C.byref(map_descr)),
            "uGDSTensorMapOpen",
        )

        geometry = _new(_Geometry)
        geometry.namespace_id = 1
        geometry.lba_size = config.lba_size
        geometry.controller_page_size = config.controller_page_size
        geometry.capacity_lbas = config.capacity_lbas
        lba_descr = _new(_LbaMapDescr)
        lba_descr.manifest_path = os.fsencode(config.manifest)
        lba_descr.tensor_map = tensor_map
        lba_descr.geometry = C.pointer(geometry)
        _check(
            lib.uGDSTensorLbaMapOpen(C.byref(lba_map), C.byref(lba_descr)),
            "uGDSTensorLbaMapOpen",
        )

        count = C.c_size_t()
        _check(
            lib.uGDSTensorMapGetCount(tensor_map, C.byref(count)),
            "uGDSTensorMapGetCount",
        )
        max_io_size = 0
        for index in range(count.value):
            mapping = _new(_Mapping)
            plan = _new(_Plan)
            _check(
                lib.uGDSTensorMapGetByIndex(tensor_map, index, C.byref(mapping)),
                "uGDSTensorMapGetByIndex",
            )
            _check(
                lib.uGDSTensorLbaMapPlan(lba_map, C.byref(mapping), C.byref(plan)),
                "uGDSTensorLbaMapPlan",
            )
            max_io_size = max(max_io_size, plan.io_size)

        buffer_flags = _HIP_DMABUF if getattr(torch.version, "hip", None) else 0
        if max_io_size:
            with torch.cuda.device(device):
                staging_tensor = torch.empty(
                    max_io_size + _ALIGNMENT - 1,
                    dtype=torch.uint8,
                    device=device,
                )
            staging_address = (
                staging_tensor.data_ptr() + _ALIGNMENT - 1
            ) & -_ALIGNMENT
            with torch.cuda.device(device):
                _check(
                    lib.uGDSBufRegister(
                        staging_address, max_io_size, buffer_flags
                    ),
                    "uGDSBufRegister(staging)",
                )
            staging_registered = True

        for index in range(count.value):
            mapping = _new(_Mapping)
            _check(
                lib.uGDSTensorMapGetByIndex(tensor_map, index, C.byref(mapping)),
                "uGDSTensorMapGetByIndex",
            )
            name = C.string_at(mapping.name, mapping.name_length).decode("utf-8")
            dtype_name = mapping.dtype.decode("ascii")
            shape = tuple(mapping.shape[i] for i in range(mapping.rank))
            with torch.cuda.device(device):
                weight = torch.empty(
                    shape,
                    dtype=_torch_dtype(torch, dtype_name),
                    device=device,
                )
            weight_bytes = weight.numel() * weight.element_size()
            if weight_bytes != mapping.nbytes:
                raise UGDSError(f"{name}: dtype/shape byte length mismatch")

            read = _new(_ReadDescr)
            read.io_handle = handle
            read.destination = weight.data_ptr()
            read.destination_size = weight_bytes
            read.staging = staging_address or None
            read.staging_size = max_io_size
            read.staging_buffer_flags = buffer_flags
            with torch.cuda.device(device):
                _check(
                    lib.uGDSTensorReadInto(
                        lba_map, C.byref(mapping), C.byref(read)
                    ),
                    f"uGDSTensorReadInto({name})",
                )
            yield name, weight
    finally:
        if staging_registered:
            with torch.cuda.device(device):
                lib.uGDSBufDeregister(staging_address)
        staging_tensor = None
        if lba_map.value:
            lib.uGDSTensorLbaMapClose(lba_map)
        if tensor_map.value:
            lib.uGDSTensorMapClose(tensor_map)
        if handle.value:
            lib.uGDSHandleDeregister(handle)
        if fd >= 0:
            os.close(fd)
        if driver_open:
            lib.uGDSDriverClose()


__all__ = ["UGDSError", "UGDSWeightConfig", "ugds_weights_iterator"]
