#pragma once

#include "ldpc_code.h"
#include "opencl_manager.h"
#include "channel.h"
#include "test_report.h"

#include <cmath>
#include <cfloat>
#include <exception>
#include <iostream>
#include <algorithm>

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
    m_log2_local_threads(6),
    m_log2_global_threads(18)
{}
};

class ldpc_decoder_gpu_dynamic_parameters
{
public:
  // the LLR threshold above which a value is considered to be known with certainty
  llr_t m_infinity_threshold;
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
    m_infinity_threshold (25),
    m_num_iter_max(100),
    m_num_iter_check_parity(20),
    m_num_vectors_per_run(0),
    m_loading_factor(4),
    m_target_errors(0)
  {}
};

class ldpc_decoder_gpu
{
protected:
  static const unsigned int log2_simd_parallel_factor;
  static const char* m_program_name;

  llr_t* m_llrs;
  uint32_t* m_final_bits_packed;
  char* m_parity_violations;
  llr_t* m_message_buffer;
  uint32_t m_max_table_index;
  llr_t m_noise_factor;
  const int64_t m_n_edges; // a 64-bit integer is used to avoid overflow issues when multiplying,
  // without explicit cast to a larger type. However the value is assumed to fit in an unsigned
  // 32-bit integer.
  const int64_t m_n_inputs; // same as above
  const int64_t m_n_outputs; // same as above
  int64_t m_n_outputs_words_p;
  int64_t m_syndrome_uint32_sz;
  int64_t m_n_erased_inputs;
  unsigned int m_num_new_vectors;
  const int32_t m_max_out_degree;
  uint32_t* m_out_edge_to_in_bit;
  uint32_t* m_out_to_in_edge;
  uint32_t* m_in_to_out_edge;
  uint32_t* m_out_bit_to_edge;
  uint32_t* m_in_bit_to_edge;
  uint32_t* m_vector_places;

  // GPU-related variables
  //  global object for GPU management
  cl_manager m_cl_manager;
  //  kernels
  cl_kernel m_convert_llr_bsc_k, m_convert_llr_biawgn_k, m_flood_backward_k,
  m_flood_forward_k, m_flood_forward_w_final_bits_k, m_deinterlace_output_k,
  m_check_parity_k, m_refill_k, m_refill_vec_k, m_permute_k;
  //  device buffers

  // memory-mapped i/o device buffers (pinned memory)
  cl_mem m_in_mapped_dev_buf, m_out_mapped_dev_buf, m_out_packed_mapped_dev_buf, m_parities_mapped_dev_buf;
  // other device buffers
  cl_mem   m_syndrome_dev_buf, m_new_syndrome_dev_buf, m_message_dev_buf,
    m_initial_llrs_dev_buf, m_new_initial_llrs_dev_buf,
    m_final_bits_dev_buf, m_final_bits_packed_dev_buf, m_vec_places_dev_buf,
    m_parities_violated_dev_buf,
    m_out_bit_to_edge_dev_buf, m_out_to_in_edge_dev_buf, m_in_to_out_edge_dev_buf,
    m_in_bit_to_edge_dev_buf, m_out_edge_to_in_bit_dev_buf,
    m_swap_o_dev_buf, m_swap_d_dev_buf;

  int32_t m_log2_local_threads;
  int32_t m_log2_global_threads;
  int32_t m_log2_threads_per_simd_vector;

  size_t m_local_threads;
  size_t m_global_threads;

  uint32_t m_log2_parallel_factor;
  uint32_t m_parallel_factor;

  const noisy_channel &m_channel;

public:
  ldpc_decoder_gpu(
      const ldpc_code& p_code,
      const noisy_channel &p_channel,
      const ldpc_decoder_gpu_static_parameters &p_params);
  ~ldpc_decoder_gpu();

  ldpc_decoder_gpu() = delete;
  ldpc_decoder_gpu(const ldpc_decoder_gpu&) = delete;
  ldpc_decoder_gpu & operator=(const ldpc_decoder_gpu&) = delete;

  /**
   * @brief decode
   * decoding method. takes as input channel values in p_input, the syndromes in p_syndromes,
   * outputs corrected vectors in p_results.
   * @param p_dyn_params
   * @param p_num_vectors_to_process
   * @param p_input
   *  p_input[v + num_vectors * i] = i-th channel value of v-th vector
   * @param p_syndromes
   * @param p_results
   * @param report
   * @param p_log
   */

  void decode(
      const ldpc_decoder_gpu_dynamic_parameters& p_dyn_params,
      uint32_t p_num_vectors_to_process,
      llr_t *p_input,
      const uint32_t *p_syndromes,
      uint32_t *p_results,
      test_report &report,
      uint32_t p_log = 0
      );

  bool decoding_input_is_llr() const
  {
    bool has_kernel_for_llr_conversion = (m_channel.channel() == bsc) || (m_channel.channel() == awgn);
    return !has_kernel_for_llr_conversion;
  }

  uint32_t parallel_factor() const
  {
    return m_parallel_factor;
  }

  void set_erased_variables(unsigned int p_n_erased_inputs)
  {
    m_n_erased_inputs = p_n_erased_inputs;
  }
private:
  void prepare_vectors(
      llr_t *p_input_llrs,
      uint32_t p_input_stride,
      uint32_t p_output_stride,
      uint32_t p_first_vector_idx,
      uint32_t p_num_output_vectors);

  void transfer_vectors(
      uint32_t p_num_vectors,
      llr_t *p_llrs,
      const uint32_t *p_syndromes,
      float p_threshold);
};


