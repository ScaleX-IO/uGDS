# GPUDirect RDMA 구현 — Devlog

**일시**: 2026-06-27
**워크트리**: `~/workspace/ScaleX-IO/uGDS-rdma` (branch: `rdma-support`)
**기반**: upstream/main (63ea9c9) + hip-backend-pr 병합 (bfcf5b8)

## 설계 승인 이력

8라운드 Codex 대립 리뷰 수렴 (R1: 14건 → R8: 0건 APPROVED).

| 라운드 | C | M | m | 총 |
|--------|---|---|---|----|
| R1-R7  | – | – | – | 14→6→6→7→5→3→2 |
| R8     | 0 | 0 | 0 | **APPROVED** ✅ |

## 구현 진행 (Phase별)

### Phase 1: fd lifecycle (서브에이전트 진행 중)
- `map.h`: `MAP_TYPE_DMABUF_CUDA` (0x10), `dmabuf_close_fn` typedef, `retain_fd`, `close_fn` 필드 추가 ✅
- `dma.h`: `_nvm_dma_set_dmabuf_info()` 선언 (서브에이전트)
- `nvm_dma.h`: `NVM_MAP_RDMA` 플래그, `nvm_dma_get_dmabuf_info()` 선언 (서브에이전트)
- `dma.cpp`: `struct map`에 dmabuf 필드 추가, helper 구현 (서브에이전트)
- `linux_dma.cpp`: typed close adapters, fd retain/close lifecycle (서브에이전트)

### Phase 3: Kernel ABI UGDS_HAVE_DMABUF (서브에이전트 진행 중)
- `internal/ioctl.h`: `_HIP` → `UGDS_HAVE_DMABUF` ✅
- `drv/ioctl.h`: 동일 ✅
- `drv/map.h`: 동일 ✅
- `drv/map.c`, `drv/pci.c`, `drv/Makefile`: (서브에이전트)

### Phase 4: Public API (완료 ✅)
- `ugds.h`:
  - `cuda_runtime.h` → conditional include (`__CUDACC__` / `__HIP_PLATFORM_AMD__`)
  - opaque `cudaStream_t` typedef (no SDK 시)
  - 새 에러 코드: `UGDS_RDMA_MR_STILL_ACTIVE`, `UGDS_BUSY`
  - `uGDSBufConfig_t`, `uGDSBufRegisterEx()`
  - `uGDSDmabufExport_t`, `uGDSExportDmabuf()`
  - `uGDSRDMARegion_t`, `uGDSRDMARegister/Unregister` (`#ifdef _RDMA`)

### Phase 5: RDMA 구현 (부분 완료)
- `ugds_internal.h`: `RDMARecordState` enum, `RDMARecord` struct, `rdma_records` map, `rdma_token_counter` ✅
- `ugds_buf.cpp`: `uGDSBufRegisterEx()`, `uGDSExportDmabuf()`, busy-fail `uGDSBufDeregister()` ✅
- `ugds_driver.cpp`: busy-fail `uGDSDriverClose()` ✅
- `ugds_rdma.cpp`: tracked RDMA MR API (서브에이전트)

### Phase 6: CMake (완료 ✅)
- `CUDA::cuda_driver` 링크 추가 (Driver API `cuMem*` 지원)
- `HAVE_CUDA_DMABUF` 감지 (`check_cxx_source_compiles`)
- `UGDS_ENABLE_RDMA` 옵션 + `libibverbs` 링크
- `UGDS_HAVE_DMABUF` userspace define (HIP 또는 CUDA+dmabuf 시)
- `__HIP_PLATFORM_AMD__` ugds target에 추가
- `ugds_rdma.cpp` 소스 조건부 추가

## 빌드 검증 결과

### 기본 빌드 (CUDA only, RDMA OFF)
- ✅ `libugds.so` 생성 완료
- 링크: `libcudart.so.13`, `libcuda.so.1`
- `HAVE_CUDA_DMABUF` 감지됨 (CUDA 13.3.33)

### RDMA 빌드 (CUDA + RDMA ON)
- ✅ `libugds.so` 생성 완료
- 링크: `libcudart.so.13`, `libcuda.so.1`, `libibverbs.so.1`

### 커밋 이력 (전체)
- `a2a8564` feat: GPUDirect RDMA support framework (Phase 1/3/4/5/6)
- `0377ad5` fix: `_CUDA` 매크로 사용
- `d83b303` fix: `unistd.h` for close() in ugds_rdma.cpp
- `1a3b166` feat: CUDA dma-buf export with full validation chain (Phase 2)
- `79b2345` feat: RDMA export tests, example, producer/consumer sync docs (Phase 7)

## Phase 7: Tests + Examples + Docs (완료 ✅)

### 테스트 (`test/functional/test_rdma_export.cu`)
- RDMA 등록 + dmabuf export lifecycle
- dup ownership 검증 (export fd != internal fd)
- offset 검증 (HIP non-zero, CUDA zero)
- NVMe I/O after export 정상 동작 확인
- non-RDMA 버퍼 export 시도 → IO_NOT_SUPPORTED
- unregistered 버퍼 export → MEMORY_NOT_REGISTERED
- 10회 반복 register/export/deregister 후 fd leak check

### 예제 (`examples/rdma_example.cu`)
- producer/consumer 패턴 시연
- `ibv_reg_dmabuf_mr` 직접 사용
- 올바른 teardown 순서: stop → drain CQ → dereg MR → close fd → deregister → free

### 문서 (`docs/installation.md`)
- 동기화 계약: producer/consumer 모델로 전면 개편
- 방향별 barrier 규칙 (NVMe/RDMA/GPU)
- RDMA MR lifetime 규칙
- consumer-consumer overlap safe 명시

## 남은 작업
- Phase 8: Codex 구현 리뷰 (백그라운드 실행 중)

## 기술 결정 사항

1. **fd lifecycle**: typed close adapter (`hsa_dmabuf_close_adapter`, `posix_close_adapter`)로 backend별 fd close 처리. `retain_fd=true` 시 fd 유지, `remove_mapping_descriptor()`에서 `close_fn`으로 해제.
2. **dmabuf metadata**: `struct map` 자체에 `dmabuf_fd/offset/length` 필드 추가 (container 확장 아님 — flexible array `ioaddrs[]` 충돌 회피).
3. **RDMA MR lifecycle**: unique token + state machine (PENDING→ACTIVE→DEREGISTERING). `ibv_dereg_mr`는 `uGDSRDMAUnregister` 내부에서 정확히 1회 호출. caller 직접 호출 금지.
4. **Kernel ABI**: `UGDS_HAVE_DMABUF` 단일 매크로로 HIP/CUDA dmabuf 경로 통일.
5. **동기화**: producer/consumer 방향별 barrier. `SYNC_MEMOPS` 실패 시 fatal.
