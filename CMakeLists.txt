cmake_minimum_required(VERSION 3.9)

project(ldpc_decoder_gpu)

option(LTO "Link-time optimization" ON)

if(LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  add_compile_options(-Wall -Wextra)
  ## architecture flags strictly needed to compile the code: -mavx2
  add_compile_options(-march=native -mtune=native)
  set (CMAKE_EXE_LINKER_FLAGS -Wl,--gc-sections)
  if(MINGW)
      add_compile_definitions(__USE_MINGW_ANSI_STDIO=1)
  endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/arch:AVX2 /favor:AMD64)
else()
  message(SEND_ERROR "No supported compiler detected, aborting.")
endif()

if("${CMAKE_SYSTEM_NAME}" STREQUAL "MSYS")
  string(REGEX REPLACE "\\\\" "/" cp_mingw "$ENV{CUDA_PATH}")
  string(REGEX REPLACE "^([a-zA-Z])\:" "/\\1" cp_mingw "${cp_mingw}")
  set(OpenCL_INCLUDE_DIR "${cp_mingw}/include/")
  set(OpenCL_LIBRARIES "${cp_mingw}/lib/x64/OpenCL.lib")
else()
  find_package(OpenCL REQUIRED)
  message("OpenCL include dir: ${OpenCL_INCLUDE_DIR}")
  ##FIXME should not be manual
  set(OPENCL_NVIDIA True)
  #set(OPENCL_AMD True)
  #set(OPENCL_INTEL True)
endif()

configure_file ("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/config.h")

set(SRC_PATH "src")

add_executable("ldpc_decoder_gpu" 
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

if("${CMAKE_SYSTEM_NAME}" STREQUAL "MSYS")
  target_compile_definitions(ldpc_decoder_gpu PRIVATE OPENCL_NVIDIA)
endif()

target_link_libraries(ldpc_decoder_gpu  ${OpenCL_LIBRARIES})
target_include_directories(ldpc_decoder_gpu PRIVATE "${OpenCL_INCLUDE_DIR}")
target_include_directories(ldpc_decoder_gpu PRIVATE "${PROJECT_BINARY_DIR}" "${CMAKE_SOURCE_DIR}/h/")

