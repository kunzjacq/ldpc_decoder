#pragma once

#include "channel.h"
#include "cuda_manager.h"
#include "ldpc_decoder_gpu_common.h"
#include "test_report.h"

#include <cmath>
#include <cfloat>

class ldpc_decoder_gpu_cuda
{
protected:
  static const char* m_program_name;

  transfer_llr_t* m_llrs;
  uint32_t* m_final_bits_packed;
  char* m_parity_violations;
  transfer_llr_t* m_message_buffer;
  uint32_t m_max_table_index;
  transfer_llr_t m_noise_factor;
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
  cuda_manager m_device_manager;

  //  device buffers

  // memory-mapped i/o device buffers (pinned memory)

  uint32_t* m_syndrome_dev_buf;
  uint32_t* m_new_syndrome_dev_buf;
  uint32_t* m_final_bits_packed_dev_buf;
  uint32_t* m_vec_places_dev_buf;
  uint32_t* m_swap_o_dev_buf;
  uint32_t* m_swap_d_dev_buf;


  llr_t* m_message_dev_buf;
  llr_t* m_initial_llrs_dev_buf;
  transfer_llr_t* m_new_initial_llrs_dev_buf;
  char* m_final_bits_dev_buf;
  char* m_parities_violated_dev_buf;

  uint32_t* m_out_bit_to_edge_dev_buf;
  uint32_t* m_out_to_in_edge_dev_buf;
  uint32_t* m_in_to_out_edge_dev_buf;
  uint32_t* m_in_bit_to_edge_dev_buf;
  uint32_t* m_out_edge_to_in_bit_dev_buf;

  //for pinned memory
  //cl_mem m_in_mapped_dev_buf, m_out_mapped_dev_buf, m_out_packed_mapped_dev_buf, m_parities_mapped_dev_buf;

  int32_t m_log2_local_threads;
  int32_t m_log2_global_threads;
  int32_t m_log2_threads_per_vector;

  size_t m_local_threads;
  size_t m_global_threads;
  size_t m_tiles;

  uint32_t m_log2_parallel_factor;
  uint32_t m_parallel_factor;

  const noisy_channel &m_channel;

public:
  ldpc_decoder_gpu_cuda(
      const ldpc_code& p_code,
      const noisy_channel &p_channel,
      const ldpc_decoder_gpu_static_parameters &p_params);
  ~ldpc_decoder_gpu_cuda();

  ldpc_decoder_gpu_cuda() = delete;
  ldpc_decoder_gpu_cuda(const ldpc_decoder_gpu_cuda&) = delete;
  ldpc_decoder_gpu_cuda & operator=(const ldpc_decoder_gpu_cuda&) = delete;

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
      void *p_input,
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
      transfer_llr_t *p_input_llrs,
      uint32_t p_input_stride,
      uint32_t p_output_stride,
      uint32_t p_first_vector_idx,
      uint32_t p_num_output_vectors);

  void transfer_vectors(
      uint32_t p_num_vectors,
      transfer_llr_t *p_llrs,
      const uint32_t *p_syndromes);
};


