#pragma once

#include "ldpc_code.h"

#include <cstdint>

class ldpc_decoder_gpu_static_parameters
{ 
public:
  // log of numbers of vectors to decode in parallel requested by the user
  // the actual number may be lower because of memory constraints
  uint32_t m_max_log_parallel_factor_user;
  int m_log2_local_threads;
  int m_log2_global_threads;

  // sensible default values for 2**20-bit vectors and rather large GPUs
  ldpc_decoder_gpu_static_parameters():
    m_max_log_parallel_factor_user(5),
    m_log2_local_threads(9),
    m_log2_global_threads(25)
{}
};

class ldpc_decoder_gpu_dynamic_parameters
{
public:
  // A value above which phi=x->-log(tanh(x/2)) is replaced by its taylor development 2 exp(-x)
  // for OpenCL code. CUDA code uses constexpr values baked in the code.
  transfer_llr_t m_infinity_threshold;
  // number of GPU decoding iterations
  uint32_t m_num_iter_max;
  // number of iterations between computation of parity checks
  // (to find vectors whose decoding is finished)
  uint32_t m_num_iter_check_parity;
  // number of vectors in one batch: will be set after decoder is created to the number of
  // vectors simultaneously present on the GPU, times the loading factor below.
  uint32_t m_num_vectors_per_run;
  // the number of decoded vectors is several times the number of vectors simultaneously
  // present on the GPU, to test the mechanism allowing progressive decoding and replacement
  // of fully decoded vectors by new, not-yet-error-corrected ones.
  uint32_t m_loading_factor;

  uint32_t m_target_errors;

  // sensible default values
  ldpc_decoder_gpu_dynamic_parameters():
    m_infinity_threshold (10),
    m_num_iter_max(100),
    m_num_iter_check_parity(10),
    m_num_vectors_per_run(0),
    m_loading_factor(4),
    m_target_errors(0)
  {}
};