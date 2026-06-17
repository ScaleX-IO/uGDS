cmake_minimum_required(VERSION 3.18)
project(ugds LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dependencies
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
)

add_library(ugds SHARED ${UGDS_SOURCES})

target_include_directories(ugds
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/libnvm
    PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ugds
    PRIVATE Threads::Threads
)

target_compile_options(ugds PRIVATE -Wall -O2)

# Test program (needs CUDA for cudaMalloc/kernel launch)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    # ── Functional regression tests (one executable per test case) ──
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
    )

    foreach(test_name ${FUNCTIONAL_TESTS})
        add_executable(${test_name} test/functional/${test_name}.cu)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/test/functional
        )
        target_link_libraries(${test_name} ugds CUDA::cudart)
        set_target_properties(${test_name} PROPERTIES
            CUDA_ARCHITECTURES "80;90"
            BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
        )
    endforeach()

    # ── Performance benchmarks ──

    # bench_ugds: uGDS benchmark linked against libugds.so
    add_executable(bench_ugds test/perf/cufile_bench.cu)
    target_include_directories(bench_ugds PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test/perf
    )
    target_link_libraries(bench_ugds ugds CUDA::cudart)
    set_target_properties(bench_ugds PROPERTIES
        CUDA_ARCHITECTURES "80;90"
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}"
    )

    # bench_gds: same benchmark linked against NVIDIA's libcufile.so (optional)
    set(CUFILE_LIB "" CACHE FILEPATH "Path to NVIDIA libcufile.so for GDS comparison benchmark")
    set(CUFILE_INCLUDE "" CACHE PATH "Path to NVIDIA cuFile headers (if different from ours)")
    if(CUFILE_LIB)
        add_executable(bench_gds test/perf/cufile_bench.cu)
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
            CUDA_ARCHITECTURES "80;90"
        )
    endif()

else()
    message(WARNING "CUDA not found, test_ugds will not be built")
endif()
