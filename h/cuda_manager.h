#pragma once

#include "config.h"

// CUDA runtime
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err)                                                                                                \
  do {                                                                                                                 \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error in %s at %d: %s ", __FILE__, __LINE__, cudaGetErrorString(err));                     \
      exit(err);                                                                                                       \
    }                                                                                                                  \
  } while (0)

#if 0
#define CUDA_SYNC_CHECK                                                                                                \
  do {                                                                                                                 \
    cudaDeviceSynchronize();                                                                                           \
    cudaError_t err = cudaGetLastError();                                                                              \
    if (err != cudaSuccess) {                                                                                          \
      fprintf(stderr, "CUDA error in %s at %d: %s ", __FILE__, __LINE__, cudaGetErrorString(err));                     \
      exit(err);                                                                                                       \
    }                                                                                                                  \
  } while (0)
#else
#define CUDA_SYNC_CHECK
#endif

using namespace std;

class cuda_manager {
  struct cudaDeviceProp	prop;
  uint64_t totalMemAllocated = 0;

public:
  enum _bufType { rw, r, w } bufType;

  class arg_k {
  public:
    size_t m_size;
    const void *m_ptr;
    arg_k(size_t p_size, const void *p_ptr) : m_size(p_size), m_ptr(p_ptr) {}
  };

  cuda_manager() {
    cudaError_t err = cudaSetDevice(0);
    check_output(err, "Could not set device 0");
    err = cudaGetDeviceProperties(&prop, 0);
    check_output(err, "Could not obtain device properties");
  };
  ~cuda_manager() = default;
  static void check_output(cudaError_t res, const char *p_error_message,
                           unsigned int argument_index = -1u);
  template <class T> T *create_device_buffer(enum _bufType p_dev_buf_type, size_t p_buffer_size);
  template <class T> T *create_host_pinned_buffer(enum _bufType p_devBufType, size_t p_buffer_size);

  template <class T>
  void create_device_buffer_from_host_data(enum _bufType p_dev_buf_type, size_t p_buffer_size, T *&po_buf,
                                           T *p_buf_addr);
  template <class T>
  void enqueue_write(T *p_buf, bool p_blocking_write, size_t p_offset, size_t p_copy_size,
                                 const T *p_ptr);
  void enqueue_barrier();
  template <class T> void enqueue_clear(T *p_buf, size_t p_offset, size_t p_copy_size);

  template <class T>
  void enqueue_read(const T *p_buf, bool p_blocking_read, size_t p_offset, size_t p_copy_size, T *p_ptr);

  void finish();
  void release_buffer(void *b);
  void release_pinned_buffer(void* b);
  uint64_t get_total_global_memory () const{
    return prop.totalGlobalMem;
  }
  uint64_t mem_usage(){
    return totalMemAllocated;
  }
};

template <class T>
T* cuda_manager::create_device_buffer( enum _bufType p_devBufType [[maybe_unused]], size_t p_buffer_size )
{
  T* ptr = nullptr;
  totalMemAllocated+= sizeof(T) * p_buffer_size;
  cudaError_t err = cudaMalloc(&ptr, sizeof(T) * p_buffer_size);
  CUDA_CHECK(err);
  return ptr;
}

template<class T>
T* cuda_manager::create_host_pinned_buffer( enum _bufType p_devBufType [[maybe_unused]], size_t p_buffer_size )
{
  T* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, sizeof(T) * p_buffer_size);
  CUDA_CHECK(err);
  return ptr;
}

template <class T>
void cuda_manager::create_device_buffer_from_host_data(enum _bufType p_dev_buf_type [[maybe_unused]],
                                                       size_t p_buffer_size, T *&po_buf, T *p_buf_addr) {
  cudaError_t err = cudaMalloc(&po_buf, sizeof(T) * p_buffer_size);
  CUDA_CHECK(err);
  err = cudaMemcpy(po_buf, p_buf_addr, sizeof(T) * p_buffer_size, cudaMemcpyHostToDevice);
  CUDA_CHECK(err);
}

template <class T> void cuda_manager::enqueue_clear(T *p_buf, size_t p_offset, size_t p_size) {
  cudaError_t err = cudaMemsetAsync(p_buf + p_offset, 0, sizeof(T) * p_size);
  CUDA_CHECK(err);
}

template <class T>
void cuda_manager::enqueue_read(const T *p_buf, bool p_blocking_read, size_t p_offset, size_t p_copy_size, T *p_ptr) {
  cudaError_t err;
  if (p_blocking_read) {
    err = cudaMemcpy(p_ptr, p_buf + p_offset, p_copy_size * sizeof(T), cudaMemcpyDeviceToHost);

  } else {
    err = cudaMemcpyAsync(p_ptr, p_buf + p_offset, p_copy_size * sizeof(T), cudaMemcpyDeviceToHost);
  }
  CUDA_CHECK(err);
}

template <class T>
void cuda_manager::enqueue_write(T *p_buf, bool p_blocking_write, size_t p_offset, size_t p_copy_size, const T *p_ptr) {
  cudaError_t err;
  if (p_blocking_write) {
    err = cudaMemcpy(p_buf, p_ptr + p_offset * sizeof(T), p_copy_size * sizeof(T), cudaMemcpyHostToDevice);

  } else {
    err = cudaMemcpyAsync(p_buf, p_ptr + p_offset * sizeof(T), p_copy_size * sizeof(T), cudaMemcpyHostToDevice);
  }
  CUDA_CHECK(err);
}
