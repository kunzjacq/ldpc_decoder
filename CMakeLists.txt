cmake_minimum_required(VERSION 3.20)

project(ldpc_decoder_cuda)

#if nvcc is not in PATH, it must be pointed to cmake by
# CUDACXX=<nvcc path>
#at configure stage, e.g
#CUDACXX=/usr/local/cuda-13/bin/nvcc CMAKE_EXPORT_COMPILE_COMMANDS=1 cmake -S ldpc/ -B ldpc_build/ -G "Unix Makefiles"

# CUDA settings
set(CMAKE_CUDA_ARCHITECTURES "86") #8.6 / ampere
#set(CMAKE_CUDA_ARCHITECTURES native)
# use float16 for computations on the GPU and transfers to it.
#reduces memory usage by almost 50% on the GPU and increases performance by ~60% on a consumer-grade GPU, probably more on a professional one.
set(CUDA_USE_FLOAT16_COMPUTE 1)

enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)

message("Build type: ${CMAKE_BUILD_TYPE}")
message("CMAKE version: ${CMAKE_VERSION}")
message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message("CMAKE_SIZEOF_VOID_P: ${CMAKE_SIZEOF_VOID_P}")
#set(CMAKE_VERBOSE_MAKEFILE ON)

option(LTO "Link-time optimization" ON)

if(LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23) #for float16_t
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  add_compile_options(-Wall -Wextra)
  ## architecture flags strictly needed to compile the code: -mavx2
  add_compile_options(-march=native -mtune=native)
  #add_compile_options(-mavx2)
  set (CMAKE_EXE_LINKER_FLAGS -Wl,--gc-sections)
  if(MINGW)
      add_compile_definitions(__USE_MINGW_ANSI_STDIO=1)
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/arch:AVX2 /favor:AMD64)
else()
  message(SEND_ERROR "No usable platform detected, aborting.")
endif()

set(SRC_PATH "src")

find_package(OpenCL REQUIRED)
find_package(CUDAToolkit REQUIRED)
message("OpenCL include dir: ${OpenCL_INCLUDE_DIR}")
##FIXME should not be manual
set(OPENCL_NVIDIA True)
#set(OPENCL_AMD True)
#set(OPENCL_INTEL True)

configure_file ("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/config.h")

add_executable("ldpc_decoder_opencl" 
  ${SRC_PATH}/common.cpp
  ${SRC_PATH}/main.cpp
  ${SRC_PATH}/test_report.cpp
  ${SRC_PATH}/ldpc_code.cpp
  ${SRC_PATH}/channel.cpp
  ${SRC_PATH}/prng_chacha.cpp
  ${SRC_PATH}/chacha_stream.cpp
  ${SRC_PATH}/opencl/opencl_manager.cpp
  ${SRC_PATH}/ldpc_decoder_gpu.cpp
  ${SRC_PATH}/transpose.cpp
)

target_compile_definitions(ldpc_decoder_opencl PRIVATE OPENCL_DECODER)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "MSYS")
  target_compile_definitions(ldpc_decoder_opencl PRIVATE OPENCL_NVIDIA)
endif()

target_link_libraries(ldpc_decoder_opencl  ${OpenCL_LIBRARIES})
target_include_directories(ldpc_decoder_opencl PRIVATE "${OpenCL_INCLUDE_DIR}")
target_include_directories(ldpc_decoder_opencl PRIVATE "${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/h/")

add_executable("ldpc_decoder_cuda" 
  ${SRC_PATH}/common.cpp
  ${SRC_PATH}/main.cpp
  ${SRC_PATH}/test_report.cpp
  ${SRC_PATH}/ldpc_code.cpp
  ${SRC_PATH}/channel.cpp
  ${SRC_PATH}/prng_chacha.cpp
  ${SRC_PATH}/chacha_stream.cpp
  ${SRC_PATH}/cuda/cuda_manager.cpp
  ${SRC_PATH}/cuda/flood.cu
  ${SRC_PATH}/ldpc_decoder_gpu.cu
  ${SRC_PATH}/transpose.cpp
)

target_compile_definitions(ldpc_decoder_cuda PRIVATE CUDA_DECODER)
if(CUDA_USE_FLOAT16_COMPUTE)
target_compile_definitions(ldpc_decoder_cuda PRIVATE USE_FLOAT16_COMPUTE)
endif()
target_include_directories(ldpc_decoder_cuda PRIVATE "${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/h/")
target_compile_features(ldpc_decoder_cuda PRIVATE cxx_std_20 cuda_std_20)
set_target_properties(ldpc_decoder_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(ldpc_decoder_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS} )
