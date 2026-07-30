#!/usr/bin/env python3
"""Compare GPU weight-loading paths with one measurement contract."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator


WeightIterator = Iterator[tuple[str, Any]]
Loader = Callable[[], WeightIterator]


def _safetensors_loader(path: str, device: str | int) -> Loader:
    from safetensors import safe_open

    def load() -> WeightIterator:
        with safe_open(path, framework="pt", device=device) as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)

    return load


def _instanttensor_loader(
    path: str, device: Any, backend_name: str
) -> tuple[Loader, dict[str, Any]]:
    from instanttensor import Backend, safe_open

    selected = None if backend_name == "auto" else Backend[backend_name.upper()]
    details: dict[str, Any] = {"requested_backend": backend_name}

    def load() -> WeightIterator:
        with safe_open(
            path,
            framework="pt",
            device=device.index,
            backend=selected,
            copy=True,
        ) as handle:
            details["selected_backend"] = str(handle.backend)
            details["instanttensor_io"] = {
                "buffer_size": handle.buffer_size,
                "chunk_size": handle.chunk_size,
                "concurrency": handle.concurrency,
                "io_depth": handle.io_depth,
            }
            yield from handle.tensors()

    return load, details


def _ugds_loader(args: argparse.Namespace) -> Loader:
    if not all((args.manifest, args.controller, args.capacity_lbas)):
        raise ValueError(
            "uGDS requires --manifest, --controller, and --capacity-lbas"
        )
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from ugds_vllm import UGDSWeightConfig, ugds_weights_iterator

    config = UGDSWeightConfig(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        controller_path=args.controller,
        capacity_lbas=args.capacity_lbas,
        controller_page_size=args.controller_page_size,
        lba_size=args.lba_size,
        torch_device=str(args.device),
        library=args.library,
    )
    return lambda: ugds_weights_iterator(config)


def _drop_file_cache(path: str) -> None:
    if not hasattr(os, "posix_fadvise"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


_DTYPES = {
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
}


def _allocate_parameters(path: str, device: Any) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    parameters = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor_slice = handle.get_slice(name)
            dtype = getattr(torch, _DTYPES.get(tensor_slice.get_dtype(), ""), None)
            if dtype is None:
                raise ValueError(f"unsupported dtype for {name}")
            parameters[name] = torch.empty(
                tuple(tensor_slice.get_shape()), dtype=dtype, device=device
            )
    return parameters


def _verify(load: Loader, path: str, device: Any) -> None:
    import torch

    reference = dict(_safetensors_loader(path, device.index)())
    candidate = dict(load())
    torch.cuda.synchronize(device)
    if reference.keys() != candidate.keys():
        raise AssertionError("tensor names differ from safetensors baseline")
    for name, expected in reference.items():
        actual = candidate[name]
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            raise AssertionError(f"{name}: dtype or shape mismatch")
        if not torch.equal(expected, actual):
            raise AssertionError(f"{name}: tensor bytes differ")
    del reference, candidate
    torch.cuda.empty_cache()


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _summarize(samples: list[float], payload_bytes: int) -> dict[str, float]:
    median = statistics.median(samples)
    return {
        "median_seconds": median,
        "p95_seconds": _percentile(samples, 0.95),
        "p99_seconds": _percentile(samples, 0.99),
        "mean_seconds": statistics.mean(samples),
        "stdev_seconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median_effective_gbps": payload_bytes / median / 1e9,
    }


def _measure(
    load: Loader,
    parameters: dict[str, Any],
    checkpoint: str,
    device: Any,
    cache: str,
    warmups: int,
    samples: int,
) -> tuple[list[float], list[float]]:
    import torch

    timings: list[float] = []
    warmup_timings: list[float] = []
    for index in range(warmups + samples):
        if cache == "cold":
            _drop_file_cache(checkpoint)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loaded = 0
        for name, tensor in load():
            destination = parameters.get(name)
            if destination is None:
                raise AssertionError(f"unexpected tensor {name}")
            if destination.dtype != tensor.dtype or destination.shape != tensor.shape:
                raise AssertionError(f"{name}: dtype or shape changed")
            destination.copy_(tensor)
            loaded += 1
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if loaded != len(parameters):
            raise AssertionError("loader did not produce every parameter")
        (warmup_timings if index < warmups else timings).append(elapsed)
    return warmup_timings, timings


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("loader", choices=("safetensors", "instanttensor", "ugds"))
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache", choices=("cold", "warm"), default="cold")
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--instanttensor-backend", default="auto")
    parser.add_argument("--manifest")
    parser.add_argument("--controller")
    parser.add_argument("--capacity-lbas", type=int)
    parser.add_argument("--controller-page-size", type=int, default=4096)
    parser.add_argument("--lba-size", type=int, default=4096)
    parser.add_argument("--library", default="libugds.so")
    args = parser.parse_args()
    if args.warmups < 0 or args.samples < 1:
        parser.error("--warmups must be non-negative and --samples must be positive")
    return args


def main() -> None:
    import torch

    args = _arguments()
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.device = torch.device(args.device)
    if args.device.type != "cuda":
        raise ValueError("benchmark device must be CUDA/ROCm")
    torch.cuda.set_device(args.device)
    torch.empty(1, device=args.device)
    torch.cuda.synchronize(args.device)

    details: dict[str, Any] = {}
    if args.loader == "safetensors":
        load = _safetensors_loader(args.checkpoint, "cpu")
    elif args.loader == "instanttensor":
        load, details = _instanttensor_loader(
            args.checkpoint, args.device, args.instanttensor_backend
        )
    else:
        load = _ugds_loader(args)

    parameters = _allocate_parameters(args.checkpoint, args.device)
    payload_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in parameters.values()
    )
    warmups, samples = _measure(
        load,
        parameters,
        args.checkpoint,
        args.device,
        args.cache,
        args.warmups,
        args.samples,
    )
    if args.verify and args.loader != "safetensors":
        _verify(load, args.checkpoint, args.device)
    result = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "hostname": socket.gethostname(),
        "loader": args.loader,
        "cache": args.cache,
        "checkpoint": args.checkpoint,
        "file_bytes": os.path.getsize(args.checkpoint),
        "payload_bytes": payload_bytes,
        "tensor_count": len(parameters),
        "contract": "iterator_to_preallocated_gpu_parameters",
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "safetensors": _version("safetensors"),
        "instanttensor": _version("instanttensor"),
        "warmup_seconds": warmups,
        "samples_seconds": samples,
        "summary": _summarize(samples, payload_bytes),
        **details,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
