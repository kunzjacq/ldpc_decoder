#include "cuda_manager.h"

#include "common.h"

#include <iostream>
#include <sstream>
#include <cstdlib>


void cuda_manager::check_output(cudaError_t res, const char* p_error_message, unsigned int argument_index)
{
  if (res != cudaSuccess)
  {
    stringstream s;
    s << p_error_message << endl << "Error code: " << res;
    if(argument_index != -1u) cout << "; argument " << argument_index;
    s << endl;
    error e(s.str().c_str());
    throw e;
  }
}

void cuda_manager::enqueue_barrier()
{
  // no-op, since kernels and cudaMemcpy are executed sequentially on the device
}

void cuda_manager::finish()
{
  cudaError_t err = cudaDeviceSynchronize();
  CUDA_CHECK(err);
}

void cuda_manager::release_buffer(void *b) {
  cudaError_t err = cudaFree(b);
  CUDA_CHECK(err);
}

void cuda_manager::release_pinned_buffer(void* b){
   cudaError_t err = cudaFreeHost(b);
  CUDA_CHECK(err);
}