# vLLM + LMCache + uGDS End-to-End Benchmark

This benchmark validates the following Qwen3-0.6B inference path:

```text
vLLM -> LMCache MP Server -> uGDS -> NVMe SSD
```

It sends one cold request followed by a warm request with the same prefix, then
uses vLLM metrics to verify that the warm request hits KV cache stored by
LMCache through uGDS.

## Benchmark Files

| File | Purpose |
|---|---|
| `quickstart.sh` | Main entry point: prepares the environment, builds uGDS, binds the SSD, downloads the model, and runs the benchmark |
| `setup_env.sh` | Creates `.venv`, installs the current vLLM and LMCache sources, builds `libugds.so`, and builds `ugds_drv.ko` when `BUILD_UGDS_DRIVER=1` |
| `run_e2e_bench.sh` | Starts LMCache and vLLM, invokes the benchmark client, collects logs, and stops the services |
| `bench_client.py` | Sends the cold and warm requests, reads vLLM metrics, and writes `bench-result.json` |
| `tests/test_bench_client.py` | Unit tests for `bench_client.py`; uses mocks to test metrics and streaming-response parsing and is not part of the real benchmark |

`bench_client.py` connects to a real vLLM service and produces benchmark data.
`tests/test_bench_client.py` only validates the client-side parsing logic and
does not require a GPU, LMCache, or uGDS.

## Recommended Test Environment

| Component | Recommendation |
|---|---|
| GPU | NVIDIA CUDA GPU with at least 16 GB of memory |
| GPU driver | NVIDIA open kernel driver 550 or newer |
| CUDA | CUDA Toolkit 12.x with `nvcc` available |
| Linux | Kernel headers matching the running kernel |
| SSD | A dedicated NVMe SSD larger than the configured uGDS slab |
| Build tools | `cmake`, `make`, a C/C++ compiler, `curl`, `git`, `pciutils`, and `util-linux` |
| Network | Access to a Python package index and Hugging Face |

The three repositories must be siblings in the same workspace:

```text
workspace/
├── uGDS/
├── LMCache/
└── vllm/
```

The default model is `Qwen/Qwen3-0.6B`, and the default uGDS slab size is
1 GiB. This benchmark has been validated on an NVIDIA A100 40 GB, CUDA 12.4,
and a Samsung 990 PRO.

> **Data loss warning:** uGDS bypasses the filesystem and overwrites raw blocks
> on the target SSD. Use a dedicated, disposable SSD. Never use a system disk,
> a mounted disk, or a disk containing important data.

## Run the Benchmark

First identify the dedicated SSD's PCI address and verify that it is not
mounted:

```bash
lspci -Dnn | grep -i 'non-volatile memory'
lsblk -o NAME,PATH,SIZE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS
```

Enter the benchmark directory and replace the example PCI address with the
address of the target SSD:

```bash
cd uGDS/examples/vllm_lmcache_bench

UGDS_PCI_SLOT=0000:b8:00.0 \
I_UNDERSTAND_UGDS_ERASES_DEVICE=1 \
./quickstart.sh
```

`quickstart.sh` creates an isolated `.venv`, builds the uGDS driver and library,
binds the selected SSD to `ugds_drv`, downloads the model, and starts LMCache
and vLLM to run the benchmark.

If the system already has multiple `/dev/ugds_drv*` devices, explicitly specify
the character device that corresponds to the PCI address:

```bash
UGDS_PCI_SLOT=0000:b8:00.0 \
UGDS_DEVICE=/dev/ugds_drv0 \
I_UNDERSTAND_UGDS_ERASES_DEVICE=1 \
./quickstart.sh
```

Common overrides:

```bash
GPU_ID=1 \
MODEL=/models/Qwen3-0.6B \
L1_SIZE_GB=2 \
UGDS_PCI_SLOT=0000:b8:00.0 \
I_UNDERSTAND_UGDS_ERASES_DEVICE=1 \
./quickstart.sh
```

## Expected Results

The terminal output follows this format; exact values depend on the hardware
and system load:

```text
Running cold request...
Running warm request...
Cold TTFT: 0.1408 s
Warm TTFT: 0.1121 s
TTFT speedup: 1.26x
External query tokens: 4764
External hit tokens: 2304
Local prefix-cache hit tokens: 0
PASS: LMCache external cache hit increased by 2304 tokens

Artifacts: .../artifacts/20260823-001746-573777
```

The complete run exits with status 0 when:

- the LMCache log contains `GDSContext: uGDS raw-device slab opened`;
- `external_cache_query_tokens > 0`; and
- `external_cache_hit_tokens > 0`.

The benchmark disables vLLM's local prefix cache, so
`local_prefix_cache_hit_tokens` is expected to be `0`. TTFT speedup is an
observed performance result, not a pass condition.

Each run creates a separate artifact directory:

```text
artifacts/<run-id>/
├── bench-result.json
├── lmcache.log
├── lmcache-status.json
├── versions.txt
└── vllm.log
```

`bench-result.json` uses the following format:

```json
{
  "cold": {
    "completion_tokens": 8,
    "latency_seconds": 0.15999239403754473,
    "prompt_tokens": 2382,
    "ttft_seconds": 0.14077616506256163
  },
  "external_cache_hit_tokens": 2304.0,
  "external_cache_query_tokens": 4764.0,
  "local_prefix_cache_hit_tokens": 0.0,
  "model": "/path/to/uGDS/examples/vllm_lmcache_bench/models/Qwen3-0.6B",
  "passed": true,
  "prompt_repeats": 64,
  "ttft_speedup": 1.2555576759465554,
  "warm": {
    "completion_tokens": 8,
    "latency_seconds": 0.1462629099842161,
    "prompt_tokens": 2382,
    "ttft_seconds": 0.11212241998873651
  }
}
```

| Field | Meaning |
|---|---|
| `cold` / `warm` | Latency, TTFT, and token counts for the cold and warm requests |
| `ttft_speedup` | `cold.ttft_seconds / warm.ttft_seconds` |
| `external_cache_query_tokens` | Total tokens queried from LMCache by both requests |
| `external_cache_hit_tokens` | Tokens retrieved from LMCache/uGDS |
| `local_prefix_cache_hit_tokens` | Tokens served by vLLM's local prefix cache; expected to be 0 |
| `passed` | `true` when both external queries and external hits are greater than 0 |

`lmcache.log` should also contain entries in this format:

```text
GDSContext: uGDS raw-device slab opened at /dev/ugds_drv0 (1.0 GiB)
Stored 2048 tokens in 0.009 seconds
Stored 256 tokens in 0.001 seconds
Retrieved 2304 tokens in 0.006 seconds
```
