# RFC: Minimal safetensors read path over raw LBA storage

- Status: Draft
- Author: CyberSecurityErial
- Tracking issue: [#28](https://github.com/ScaleX-IO/uGDS/issues/28)
- Intended scope: inference-time, read-only model weights

## Summary

This RFC proposes a small optional safetensors layer above the existing uGDS
raw I/O API. It maps a tensor name to its logical byte range in a canonical
safetensors shard, maps that logical range to a read-only contiguous object in
an NVMe namespace, reads an aligned envelope through uGDS, and exposes the
payload as a GPU tensor.

The design does not add a filesystem to uGDS. Standard `.safetensors` files and
Hugging Face shard indexes remain the interchange format. The raw object and a
sidecar manifest are deployment artifacts for uGDS.

## Motivation

uGDS accepts namespace byte offsets that are converted directly to LBAs. It
does not resolve paths or file extents. Safetensors, by contrast, describes a
tensor with a name, dtype, shape, and logical offsets inside a file.

Inference checkpoints are normally immutable. We can therefore bridge the two
models with stable metadata and contiguous raw objects instead of implementing
directory operations, allocation, journaling, permissions, or other general
filesystem features.

## Goals

- Load a complete, contiguous tensor from a standard safetensors checkpoint.
- Preserve each canonical safetensors shard byte-for-byte.
- Keep the existing uGDS raw read/write API behavior unchanged.
- Use a small sidecar manifest to bind a shard to a raw namespace range.
- Make `read_into(tensor_name, destination)` the main data operation.
- Validate metadata and physical ranges before submitting NVMe commands.
- Keep the implementation and test matrix deliberately small.
- Target only the vLLM model-weight loading path.

## Non-goals

- POSIX path support or a general-purpose filesystem.
- FIEMAP or fragmented filesystem extent support.
- In-place mutation of an imported checkpoint.
- Per-tensor repacking or a new checkpoint file format.
- Arbitrary strided slices or sub-byte dtypes.
- Tensor-parallel-aware packing, quantization transforms, or QKV/MoE merging.
- Transparent integration with every framework in the first implementation.
- A replacement for the current uGDS completion engine.
- Multiple controllers or namespaces in one process.

## Terminology

- **Canonical shard**: an unchanged standard `.safetensors` file.
- **Raw object**: one canonical shard copied byte-for-byte into a contiguous,
  aligned namespace range, followed by zero padding.
- **Primary mapping**: tensor name to shard and logical file byte range.
- **Secondary mapping**: shard logical file byte range to namespace byte range
  and LBA range.
- **Tensor mapping**: valid bytes in an aligned GPU allocation to a tensor with
  the parsed dtype and shape.
- **Sidecar manifest**: host-resident deployment metadata. It is not part of
  the safetensors or Hugging Face formats.

## Design overview

```text
model.safetensors.index.json + canonical shards
                       |
                       | offline validation/import
                       v
             read-only raw objects  <---- sidecar manifest
                       |
tensor name -- primary mapping -- logical file range
                       |
                 secondary mapping
                       |
                aligned LBA ranges
                       |
                 uGDS batch/read
                       |
               registered GPU arena
                       |
                  tensor mapping
                       v
                  framework tensor
```

The control plane resolves and validates ranges. The existing uGDS data plane
only sees aligned physical byte offsets, sizes, and registered GPU buffers.

## 1. Primary mapping

The primary mapping consumes either a single canonical shard or the standard
Hugging Face `model.safetensors.index.json` convention.

The canonical header and shard index are parsed on the CPU when the primary map
is opened. The sidecar stores only the raw-object placement needed by the
secondary map.

For a tensor entry:

```json
{
  "dtype": "F16",
  "shape": [4096, 4096],
  "data_offsets": [1048576, 34603008]
}
```

the logical file range is:

```text
data_base  = 8 + header_length
file_begin = data_base + BEGIN
file_end   = data_base + END
nbytes     = END - BEGIN
```

For sharded checkpoints, `weight_map[tensor_name]` selects the canonical shard
before applying the formula above.

The importer must reject:

- an oversized, truncated, or invalid UTF-8 header;
- duplicate tensor keys;
- negative, reversed, overlapping, or out-of-bounds ranges;
- holes or trailing unindexed payload in the safetensors data buffer;
- integer overflow while calculating shape size or absolute offsets;
- a byte count that does not match the dtype and shape;
- unsupported or sub-byte dtypes in the MVP.

Zero-sized tensors are valid metadata entries and require no I/O.

## 2. Secondary mapping

Each canonical shard is stored as one raw object:

```text
object_base                                             object_end
    |                                                        |
    v                                                        v
    [ canonical safetensors bytes ][ zero alignment padding ]
```

At import time, the target geometry defines:

```text
io_alignment     = lcm(namespace_lba_size, controller_page_size)
object_alignment = lcm(64 KiB, io_alignment)
```

`object_base` and the padded object end are aligned to `object_alignment`.
Objects do not overlap and must be fully contained by an explicitly reserved
namespace region. All addition, multiplication, LCM, and alignment operations
use checked arithmetic.

For a logical tensor byte range:

```text
object_base   = object_base_lba * namespace_lba_size
physical_begin = object_base + file_begin
physical_end   = object_base + file_end
```

Manifest `base_lba` values are absolute within the NVMe namespace, not relative
to the reserved region.

The I/O envelope is expanded outwards:

```text
io_begin     = align_down(physical_begin, io_alignment)
io_end       = align_up(physical_end, io_alignment)
payload_skip = physical_begin - io_begin
io_size      = io_end - io_begin
```

The zero padding after each raw object makes an outward-aligned read of the
last tensor safe. The planner still validates that the envelope stays inside
the declared padded object and reserved namespace region.

The MVP requires one contiguous raw object per canonical shard. Consequently,
the secondary mapping is constant-time addition rather than an extent lookup.

## Sidecar manifest

The sidecar is versioned and written on a normal host filesystem. A conceptual
representation is:

```json
{
  "format": "ugds-safetensors-manifest",
  "version": 1,
  "device": {
    "namespace_id": 1,
    "lba_size": 4096,
    "capacity_lbas": 1000000000
  },
  "region": {
    "base_lba": 65536,
    "length_lbas": 500000000
  },
  "objects": {
    "model-00001-of-00002.safetensors": {
      "base_lba": 65536,
      "canonical_size": 4294967296,
      "padded_size": 4295032832,
      "header_length": 16384
    }
  }
}
```

The MVP binds each object to its canonical shard name, size, and header length,
and validates the namespace geometry and reserved-region bounds. Generation
and content hashes are post-MVP work.

The standard Hugging Face index is not modified with uGDS-specific fields.

## 3. Tensor mapping

The C adapter does not depend on vLLM, but it is intentionally limited to the
vLLM model-weight loading path. It exposes validated weight metadata and reads
one weight into an already allocated contiguous GPU buffer:

```text
TensorInfo        = {name, dtype, rank, shape, nbytes}
TensorDestination = {device pointer, capacity bytes}
```

The public data operation is one synchronous `TensorReadInto`. The caller owns
one registered staging buffer and reuses it across all weights.

### vLLM-facing shape

A thin Python binding matches vLLM's existing weight iterator without changing
vLLM model code:

```python
config = UGDSWeightConfig(...)
model.load_weights(ugds_weights_iterator(config))
```

The binding only allocates CUDA weights, calls `TensorReadInto`, and yields
`(checkpoint_name, torch.Tensor)`. It does not import or modify vLLM and does
not implement a general-purpose raw-file API.

### Staging

The MVP always reads the outward-aligned envelope into the caller's reusable
64 KiB-aligned staging base registered at that exact address, then
synchronously copies only `payload_size` bytes from `payload_skip` into the
destination. The same staging buffer is not shared by concurrent calls. Direct
DMA into a framework tensor is a later optimization.

### Completion

The MVP may use the existing synchronous or batch path. A tensor becomes ready
only after every underlying command succeeds and any staging copy completes.
Partial reads and timed-out requests are errors and never publish a tensor.

The current uGDS timeout paths do not prove that an outstanding command has
been drained before returning.

TODO(post-MVP): poison a timed-out queue and keep its handle, PRPs, and
registered buffers alive until the command is drained or the controller is
safely reset. Until this is implemented, timeout recovery is not
production-safe and a timeout must not be interpreted as command cancellation.

## 4. MVP read-only rules

### Runtime validation

The performance-validation build keeps only checks that prevent an invalid
read or memory overwrite:

- tensor and aligned LBA ranges stay within the declared object, region, and
  namespace capacity;
- checked arithmetic is used for offsets and sizes;
- destination capacity is at least the tensor byte length;
- the copied payload stays within the staging envelope.

The adapter exposes no write operation. During a benchmark, the caller must not
modify the manifest/raw objects or close maps, handles, and buffers while a
reader call is active.

Generation switching, hashes, concurrent-close protection, and in-flight
reference tracking are post-MVP TODOs, consistent with the current uGDS core.

## 5. Focused test matrix

The test matrix intentionally covers boundaries rather than every combination.

| Case | Purpose |
|---|---|
| Single shard, a few common dtypes | Primary, secondary, and tensor mapping happy path |
| Two shards with a standard index | Tensor-to-shard selection |
| Unaligned tensor begin and length | Outward-aligned envelope and guard-byte safety |
| Invalid header or out-of-bounds range | Parser and checked-arithmetic rejection |
| Device mismatch or short destination | Bounds enforcement |
| Differential comparison | Bytes, dtype, and shape match the official loader |

Hardware tests should use a dedicated test region and never assume that LBA 0
is safe. Pure metadata and planning tests do not require an NVMe device.

The weight-load benchmark uses one fresh process per sample. It preallocates
GPU parameters outside the timed region, consumes the loader's `(name, tensor)`
pairs into those parameters, and stops only after GPU synchronization. This
matches the intended vLLM loading boundary without importing vLLM itself.

### Current validation status

Metadata mapping, LBA planning, the tensor-read boundary, and the benchmark
contract have been validated without storage hardware. The current development
environment has no usable uGDS device or dedicated writable NVMe namespace, so
real uGDS reads and uGDS performance remain unverified. This RFC makes no uGDS
performance claim until that hardware validation is completed.

## Proposed implementation shape

The implementation should remain optional and avoid reorganizing the existing
uGDS sources. The expected additions are limited to:

- one small public safetensors adapter interface;
- one compact implementation unit for metadata, manifest, and planning;
- one offline import/index utility;
- one focused correctness/performance case;
- one thin vLLM-shaped Python weight iterator.

A small read-only device query may be added to expose controller serial/model,
NSID, LBA size, and namespace capacity to the adapter. This does not change the
semantics of existing I/O entry points. During the single-controller MVP, the
I/O handle, query, and buffer registration refer to the same controller.

The exact parser dependency and public symbol names will be reviewed with the
first implementation step. A permissive handwritten JSON parser is not
acceptable; the selected parser must reject duplicate keys and preserve the
safetensors validation rules.

## Incremental implementation plan

The draft implementation PR proceeds in separately reviewable steps:

1. Primary mapping and metadata-only tests.
2. Secondary contiguous-object mapping and manifest validation.
3. GPU weight mapping with aligned envelope and vLLM-oriented `read_into`.
4. Minimal read-only and bounds rules for performance validation.
5. The focused end-to-end case matrix.

No later step starts until the preceding step has been reviewed.

The timeout/drain prerequisite is deliberately deferred to the post-MVP TODO
in the Completion section.

## Alternatives considered

### FIEMAP snapshot

FIEMAP can translate a regular file to physical extents before the filesystem
is unmounted. It adds filesystem-specific flags, stacked block-device mapping,
fragmentation, and invalidation rules. It is intentionally deferred until the
contiguous raw-object path is proven.

### General-purpose filesystem

A filesystem would provide paths and allocation but also requires metadata
updates, crash consistency, ownership, and controller coordination. These
features are unnecessary for immutable inference weights.

### Per-tensor aligned format

Repacking each tensor can improve random access, but the result is not a
standard safetensors file and needs cache invalidation and small-tensor
bundling. It may be added later as a derived cache; it is not the canonical
storage contract.

## Acceptance criteria

The MVP is complete when:

1. A valid standard safetensors checkpoint can be imported as immutable,
   contiguous raw objects with a committed sidecar manifest.
2. A named tensor resolves to validated aligned LBA requests.
3. uGDS reads the tensor into GPU memory without using a filesystem data path.
4. The resulting dtype, shape, and bytes match the official safetensors loader.
5. Invalid metadata, stale identity, out-of-bounds ranges, partial I/O, and
   unsafe destinations fail without publishing a tensor.
6. Existing uGDS APIs and tests continue to behave as before.
