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

The canonical header and shard index are parsed on the CPU during import. The
validated tensor entries are persisted in the sidecar, so the runtime does not
need to recover metadata through a GPU data read.

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
  "generation": 7,
  "device": {
    "namespace_id": 1,
    "namespace_identity": "device-specific-stable-id",
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
      "canonical_sha256": "...",
      "header_sha256": "...",
      "header_length": 16384,
      "tensors": {
        "model.layers.0.input_layernorm.weight": {
          "dtype": "F16",
          "shape": [4096],
          "data_offsets": [0, 8192]
        }
      }
    }
  }
}
```

The exact serialization is an implementation detail, but the version,
generation, device identity, geometry, bounds, sizes, hashes, and validated
tensor metadata are mandatory. Tensor metadata must remain derivable from and
bound to the canonical header hash. Runtime geometry must match the geometry
used to calculate the imported object alignment.

The standard Hugging Face index is not modified with uGDS-specific fields.

## 3. Tensor mapping

The core adapter remains framework-neutral. It exposes validated tensor
metadata and accepts a destination descriptor conceptually equivalent to:

```text
TensorInfo       = {dtype, rank, shape, nbytes}
TensorDestination = {base pointer, allocation bytes, valid offset,
                     backend, device}
```

The core operation is `read_into(name, destination)`. Constructing a PyTorch,
JAX, or other framework object is the responsibility of a later framework
adapter.

### Framework allocation convenience

An optional framework adapter may implement `get_tensor(name)` by allocating a
GPU arena with alignment headroom, registering the aligned base once, and
reading `[io_begin, io_end)` into that base. The returned framework tensor is a
view beginning at `payload_skip`, with the dtype and shape from the validated
primary mapping.

The tensor view retains shared ownership of the backing arena and buffer
registration. Releasing the archive or request must not invalidate a live
tensor.

If `payload_skip` does not satisfy the dtype alignment required to create a
typed view, the adapter uses staging plus a device-to-device copy.

### Existing destination

For `read_into(name, destination)`, direct DMA is allowed only when all of the
following hold:

- the destination is contiguous and on a supported GPU device;
- its dtype, shape, and byte length match the tensor metadata;
- its registered range includes any required aligned guard bytes;
- the source and destination layout can be represented by the current uGDS
  page-based PRP path.

Otherwise, the adapter reads into a registered staging arena and copies only
the valid payload into the destination. An aligned raw read must never
overwrite memory before or after an existing framework tensor.

### Completion

The MVP may use the existing synchronous or batch path. A tensor becomes ready
only after every underlying command succeeds and any staging copy completes.
Partial reads and timed-out requests are errors and never publish a tensor.

The current uGDS timeout paths do not prove that an outstanding command has
been drained before returning. Before hardware tensor mapping is enabled, a
narrow core prerequisite must ensure that a timed-out queue is poisoned and
that its handle, PRPs, and registered buffers remain alive until the command is
drained or the controller is safely reset. No new asynchronous engine is
required, but the adapter alone cannot provide this guarantee.

## 4. Read-only and safety rules

### Immutability

- An object is immutable after it is committed.
- A model update writes new objects under a new manifest generation.
- Readers open one immutable manifest snapshot for their lifetime.
- The active generation changes only after every new object is complete and
  validated.
- Old generations can be reclaimed only when no reader references them.

### Import and commit

The importer performs these steps in order:

1. Validate the canonical shard and compute its identity hashes.
2. Reserve a non-overlapping aligned raw range.
3. Write the exact canonical bytes and zero padding.
4. Complete the storage durability operation supported by the import path.
5. Verify object bounds and, in strict mode, its content hash.
6. Publish the new sidecar generation last.

A partially imported object is unreachable because no committed manifest
references it.

### Runtime validation

Before reading data, the runtime validates:

- manifest magic, version, and supported flags;
- namespace identity, NSID, LBA size, and capacity;
- reserved region and object bounds without integer overflow;
- canonical and padded sizes;
- header identity and tensor metadata bounds;
- destination pointer, allocation length, and registration ownership.

The runtime treats raw storage as read-only even though the lower-level uGDS
API also exposes writes.

The MVP supports one controller and NSID 1 per process, matching the current
uGDS handle and global buffer-registration model. The archive owns that handle
for its lifetime. Supporting multiple controllers requires a later
controller-scoped buffer-registration API.

### Lifetime

The ownership chain is:

```text
archive -> plan/request -> GPU arena -> uGDS buffer registration
```

Each in-flight request retains every object it needs. Buffer deregistration and
handle close must fail or wait while a request is in flight. Cancellation may
stop unsubmitted work, but submitted NVMe commands must still be drained before
their buffers are released.

The initial implementation should enforce these rules in the safetensors
adapter without requiring a broad uGDS core refactor. The timeout/drain rule
described above is a separately reviewed prerequisite because it cannot be
contained safely in the adapter.

## 5. Focused test matrix

The test matrix intentionally covers boundaries rather than every combination.

| Case | Purpose |
|---|---|
| Single shard, a few common dtypes | Primary, secondary, and tensor mapping happy path |
| Two shards with a standard index | Tensor-to-shard selection |
| Unaligned tensor begin and length | Outward-aligned envelope and guard-byte safety |
| Invalid header or out-of-bounds range | Parser and checked-arithmetic rejection |
| Stale manifest or device mismatch | Read-only identity enforcement |
| Differential comparison | Bytes, dtype, and shape match the official loader |

Hardware tests should use a dedicated test region and never assume that LBA 0
is safe. Pure metadata and planning tests do not require an NVMe device.

## Proposed implementation shape

The implementation should remain optional and avoid reorganizing the existing
uGDS sources. The expected additions are limited to:

- one small public safetensors adapter interface;
- one compact implementation unit for metadata, manifest, and planning;
- one offline import/index utility;
- one focused functional test and metadata-only unit tests;
- an optional Python/framework adapter after the raw read path works.

A small read-only device query may be added to expose controller serial/model,
NSID, LBA size, and namespace capacity to the adapter. This does not change the
semantics of existing I/O entry points. During the single-controller MVP, the
query and buffer registration refer to the same archive-owned controller.

The exact parser dependency and public symbol names will be reviewed with the
first implementation step. A permissive handwritten JSON parser is not
acceptable; the selected parser must reject duplicate keys and preserve the
safetensors validation rules.

## Incremental implementation plan

The draft implementation PR proceeds in separately reviewable steps:

1. Primary mapping and metadata-only tests.
2. Secondary contiguous-object mapping and manifest validation.
3. The narrow timeout/drain prerequisite, then GPU tensor mapping with aligned
   envelope and framework-neutral `read_into`.
4. Read-only generation, identity, bounds, and lifetime rules.
5. The focused end-to-end case matrix.

No later step starts until the preceding step has been reviewed.

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
