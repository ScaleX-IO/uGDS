cmake_minimum_required(VERSION 3.18)
project(ugds LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
find_package(Threads REQUIRED)

# - GPU Backend Selection ---------------------
option(UGDS_BACKEND_CUDA "Use NVIDIA CUDA backend" ON)
option(UGDS_BACKEND_HIP  "Use AMD HIP backend"     OFF)

# Both backends can be enabled simultaneously for dual-GPU systems.

# HIP language support requires CMake >= 3.21
if(UGDS_BACKEND_HIP AND CMAKE_VERSION VERSION_LESS "3.21")
    message(FATAL_ERROR "HIP backend requires CMake >= 3.21 (found ${CMAKE_VERSION})")
endif()

# - CUDA backend setup (independent) -----------
set(UGDS_CUDA_INCLUDES "")
set(UGDS_CUDA_LIBS "")
set(UGDS_CUDA_ARCHITECTURES "80;90")

if(UGDS_BACKEND_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(UGDS_CUDA_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif()

# - HIP backend setup (independent) ------------
set(UGDS_HIP_INCLUDES "")
set(UGDS_HIP_LIBS "")
set(UGDS_HIP_ARCHITECTURES "")

if(UGDS_BACKEND_HIP)
    # Set HIP architectures before enable_language to avoid CMake auto-probing
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942" CACHE STRING "HIP target architectures")  # MI250, MI300X
    endif()
    find_package(hip REQUIRED)
    # Respect CMAKE_PREFIX_PATH for ROCm install location; fallback to /opt/rocm
    if(NOT DEFINED ROCM_PATH)
        find_path(ROCM_PATH include/hsa/hsa_ext_amd.h PATHS /opt/rocm ${CMAKE_PREFIX_PATH})
    endif()
    set(UGDS_HIP_INCLUDES ${HIP_INCLUDE_DIRS})
    if(ROCM_PATH)
        list(APPEND UGDS_HIP_INCLUDES ${ROCM_PATH}/include)
    endif()
    set(UGDS_HIP_LIBS hip::host)
    set(UGDS_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

    # HSA runtime for hsa_amd_portable_export_dmabuf / close_dmabuf
    find_package(hsa-runtime64 CONFIG QUIET)
    if(hsa-runtime64_FOUND)
        list(APPEND UGDS_HIP_LIBS hsa-runtime64::hsa-runtime64)
    else()
        find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib /opt/rocm/lib REQUIRED)
        list(APPEND UGDS_HIP_LIBS ${HSA_RUNTIME_LIB})
    endif()

    # Detect hsa_amd_portable_export_dmabuf_v2 (ROCm >= 5.6 with PCIe P2P mapping)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${UGDS_HIP_INCLUDES}")
    set(CMAKE_REQUIRED_LIBRARIES "${UGDS_HIP_LIBS}")
    check_cxx_source_compiles("
        #include <hsa/hsa_ext_amd.h>
        int main() {
            int fd; uint64_t off;
            hsa_amd_portable_export_dmabuf_v2((const void*)0, 0, &fd, &off,
                                               HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
            return 0;
        }
    " HAVE_HSA_DMABUF_V2)
    set(CMAKE_REQUIRED_INCLUDES)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(NOT HAVE_HSA_DMABUF_V2)
        message(FATAL_ERROR "HIP backend requires hsa_amd_portable_export_dmabuf_v2 "
                            "(HSA_AMD_DMABUF_MAPPING_TYPE_PCIE). "
                            "Please install ROCm >= 5.6 or disable HIP backend (-DUGDS_BACKEND_HIP=OFF).")
    endif()
endif()

if(NOT UGDS_BACKEND_CUDA AND NOT UGDS_BACKEND_HIP)
    message(WARNING "No GPU backend selected, building library only")
endif()

# Library sources
set(UGDS_SOURCES
    src/libnvm/admin.cpp
    src/libnvm/ctrl.cpp
    src/libnvm/dma.cpp
    src/libnvm/error.cpp
    src/libnvm/linux_device.cpp
    src/libnvm/linux_dma.cpp
    src/libnvm/mutex.cpp
    src/libnvm/queue.cpp
    src/libnvm/rpc.cpp
    src/ugds_driver.cpp
    src/ugds_handle.cpp
    src/ugds_buf.cpp
    src/ugds_io.cpp
    src/ugds_batch.cpp
    src/ugds_async.cpp
    src/ugds_hugepage.cpp
)

add_library(ugds SHARED ${UGDS_SOURCES})

target_include_directories(ugds
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/libnvm
)

# Backend includes and compile definitions (independent)
target_link_libraries(ugds PRIVATE Threads::Threads)

if(UGDS_BACKEND_CUDA)
    target_include_directories(ugds PRIVATE ${UGDS_CUDA_INCLUDES})
    target_compile_definitions(ugds PRIVATE _CUDA)
    target_link_libraries(ugds PRIVATE CUDA::cudart)
endif()

if(UGDS_BACKEND_HIP)
    target_include_directories(ugds PRIVATE ${UGDS_HIP_INCLUDES})
    target_link_libraries(ugds PRIVATE ${UGDS_HIP_LIBS})
    target_compile_definitions(ugds PRIVATE _HIP)
    if(HAVE_HSA_DMABUF_V2)
        target_compile_definitions(ugds PRIVATE HAVE_HSA_DMABUF_V2)
    endif()
endif()

target_compile_options(ugds PRIVATE -Wall -O2)

# - Tests and benchmarks -----------------------
set(FUNCTIONAL_TESTS
    test_driver_lifecycle
    test_handle_register
    test_buf_register
    test_read_write_basic
    test_read_write_large
    test_read_write_unregistered
    test_alignment_errors
    test_multi_offset
    test_concurrent_qps
    test_batch_basic
    test_batch_mixed_sizes
    test_batch_mixed_ops
    test_batch_errors
    test_batch_reuse
    test_batch_deep
    test_async_basic
    test_async_stream_order
    test_async_late_binding
    test_async_errors
    test_async_multi_stream
    test_interrupt_ioctls
)

# CUDA tests (compiled with nvcc)
if(UGDS_BACKEND_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)

        # Suffix targets only in dual mode to preserve backward compatibility
        if(UGDS_BACKEND_HIP)
            set(CUDA_SUFFIX "_cuda")
        else()
            set(CUDA_SUFFIX "")
        endif()

        foreach(test_name ${FUNCTIONAL_TESTS})
            set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/test/functional/${test_name}.cu)
            add_executable(${test_name}${CUDA_SUFFIX} ${test_src})
            target_include_directories(${test_name}${CUDA_SUFFIX} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/test/functional
            )
            target_link_libraries(${test_name}${CUDA_SUFFIX} ugds CUDA::cudart)
            set_target_properties(${test_name}${CUDA_SUFFIX} PROPERTIES
                CUDA_ARCHITECTURES "${UGDS_CUDA_ARCHITECTURES}"
                BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
            )
        endforeach()

        # bench_ugds (CUDA)
        set(bench_src ${CMAKE_CURRENT_SOURCE_DIR}/test/perf/cufile_bench.cu)
        add_executable(bench_ugds${CUDA_SUFFIX} ${bench_src})
        target_include_directories(bench_ugds${CUDA_SUFFIX} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
        )
        target_link_libraries(bench_ugds${CUDA_SUFFIX} ugds CUDA::cudart)
        set_target_properties(bench_ugds${CUDA_SUFFIX} PROPERTIES
            CUDA_ARCHITECTURES "${UGDS_CUDA_ARCHITECTURES}"
            BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
        )

        # bench_overlap: async IO + GPU compute overlap benchmark
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/perf/bench_overlap.cu)
            add_executable(bench_overlap${CUDA_SUFFIX} test/perf/bench_overlap.cu)
            target_include_directories(bench_overlap${CUDA_SUFFIX} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
            )
            target_link_libraries(bench_overlap${CUDA_SUFFIX} ugds CUDA::cudart)
            set_target_properties(bench_overlap${CUDA_SUFFIX} PROPERTIES
                CUDA_ARCHITECTURES "${UGDS_CUDA_ARCHITECTURES}"
                BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
            )
        endif()

        # bench_gds: NVIDIA GDS comparison (optional)
        set(CUFILE_LIB "" CACHE FILEPATH "Path to NVIDIA libcufile.so for GDS comparison benchmark")
        set(CUFILE_INCLUDE "" CACHE PATH "Path to NVIDIA cuFile headers (if different from ours)")
        if(CUFILE_LIB)
            add_executable(bench_gds ${bench_src})
            target_include_directories(bench_gds PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
            )
            if(CUFILE_INCLUDE)
                target_include_directories(bench_gds PRIVATE ${CUFILE_INCLUDE})
            else()
                target_include_directories(bench_gds PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
            endif()
            target_link_libraries(bench_gds ${CUFILE_LIB} CUDA::cudart)
            target_compile_definitions(bench_gds PRIVATE USE_O_DIRECT USE_NVIDIA_GDS)
            set_target_properties(bench_gds PROPERTIES
                CUDA_ARCHITECTURES "${UGDS_CUDA_ARCHITECTURES}"
            )

            # bench_overlap_gds: overlap benchmark linked against libcufile.so
            add_executable(bench_overlap_gds test/perf/bench_overlap.cu)
            target_include_directories(bench_overlap_gds PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
            )
            if(CUFILE_INCLUDE)
                target_include_directories(bench_overlap_gds PRIVATE ${CUFILE_INCLUDE})
            else()
                target_include_directories(bench_overlap_gds PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
            endif()
            target_link_libraries(bench_overlap_gds ${CUFILE_LIB} CUDA::cudart)
            target_compile_definitions(bench_overlap_gds PRIVATE USE_O_DIRECT USE_NVIDIA_GDS)
            set_target_properties(bench_overlap_gds PROPERTIES
                CUDA_ARCHITECTURES "${UGDS_CUDA_ARCHITECTURES}"
            )
        endif()
    else()
        message(WARNING "CUDA compiler not found, CUDA tests will not be built")
    endif()
endif()

# HIP tests (compiled with hipcc)
if(UGDS_BACKEND_HIP)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)

        # Suffix targets only in dual mode to preserve backward compatibility
        if(UGDS_BACKEND_CUDA)
            set(HIP_SUFFIX "_hip")
        else()
            set(HIP_SUFFIX "")
        endif()

        foreach(test_name ${FUNCTIONAL_TESTS})
            set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/test/functional/${test_name}.cu)
            if(UGDS_BACKEND_CUDA)
                # Dual mode: copy .cu to .hip.cu to avoid LANGUAGE conflict
                set(hip_src ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.hip.cu)
                configure_file(${test_src} ${hip_src} COPYONLY)
                set_source_files_properties(${hip_src} PROPERTIES LANGUAGE HIP)
                add_executable(${test_name}${HIP_SUFFIX} ${hip_src})
            else()
                set_source_files_properties(${test_src} PROPERTIES LANGUAGE HIP)
                add_executable(${test_name}${HIP_SUFFIX} ${test_src})
            endif()
            target_include_directories(${test_name}${HIP_SUFFIX} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/test/functional
            )
            target_link_libraries(${test_name}${HIP_SUFFIX} ugds ${UGDS_HIP_LIBS})
            target_compile_definitions(${test_name}${HIP_SUFFIX} PRIVATE __HIP_PLATFORM_AMD__)
            set_target_properties(${test_name}${HIP_SUFFIX} PROPERTIES
                HIP_ARCHITECTURES "${UGDS_HIP_ARCHITECTURES}"
                BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
            )
        endforeach()

        # bench_ugds (HIP)
        set(bench_src_orig ${CMAKE_CURRENT_SOURCE_DIR}/test/perf/cufile_bench.cu)
        if(UGDS_BACKEND_CUDA)
            set(bench_src ${CMAKE_CURRENT_BINARY_DIR}/cufile_bench.hip.cu)
            configure_file(${bench_src_orig} ${bench_src} COPYONLY)
            set_source_files_properties(${bench_src} PROPERTIES LANGUAGE HIP)
        else()
            set(bench_src ${bench_src_orig})
            set_source_files_properties(${bench_src} PROPERTIES LANGUAGE HIP)
        endif()
        add_executable(bench_ugds${HIP_SUFFIX} ${bench_src})
        target_include_directories(bench_ugds${HIP_SUFFIX} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
        )
        target_link_libraries(bench_ugds${HIP_SUFFIX} ugds ${UGDS_HIP_LIBS})
        target_compile_definitions(bench_ugds${HIP_SUFFIX} PRIVATE __HIP_PLATFORM_AMD__)
        set_target_properties(bench_ugds${HIP_SUFFIX} PROPERTIES
            HIP_ARCHITECTURES "${UGDS_HIP_ARCHITECTURES}"
            BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
        )
    else()
        message(WARNING "HIP compiler not found, HIP tests will not be built")
    endif()
endif()

if(NOT UGDS_BACKEND_CUDA AND NOT UGDS_BACKEND_HIP)
    message(WARNING "No GPU backend enabled, tests will not be built")
endif()
