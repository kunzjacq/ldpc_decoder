#include "../h/ldpc_decoder_gpu_cuda.h"

#include "common.h"
#include "config.h"

#include "../h/flood.cuh"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;

// FIXME pinned memory not supported with cuda impl yet
constexpr bool g_use_pinned_memory = true;

ldpc_decoder_gpu_cuda::ldpc_decoder_gpu_cuda(const ldpc_code &p_code, const noisy_channel &p_channel,
                                             const ldpc_decoder_gpu_static_parameters &p_params)
    : m_llrs(nullptr), m_message_buffer(nullptr), m_max_table_index(0), m_noise_factor(), m_n_edges(p_code.n_edges()),
      m_n_inputs(p_code.n_inputs()), m_n_outputs(p_code.n_outputs()), m_n_erased_inputs(p_code.n_erased_inputs()),
      m_max_out_degree(p_code.max_degree_out()), m_out_edge_to_in_bit(nullptr), m_out_to_in_edge(nullptr),
      m_in_to_out_edge(nullptr), m_out_bit_to_edge(nullptr), m_in_bit_to_edge(nullptr), m_device_manager(),
      m_log2_local_threads(p_params.m_log2_local_threads), m_log2_global_threads(p_params.m_log2_global_threads),
      m_local_threads(1uL << m_log2_local_threads), m_global_threads(1uL << m_log2_global_threads),
      m_tiles(1uL << (m_log2_global_threads - m_log2_local_threads)), m_log2_parallel_factor(0), m_parallel_factor(1),
      m_channel(p_channel) {
  if (m_n_inputs & 0x1F) {
    throw error("This decoder only handles input sizes that are multiple of 32");
  }
  
  m_out_to_in_edge = new uint32_t[m_n_edges];
  m_in_to_out_edge = new uint32_t[m_n_edges];
  m_out_edge_to_in_bit = new uint32_t[m_n_edges];
  m_in_bit_to_edge = new uint32_t[m_n_inputs + 1];
  m_out_bit_to_edge = new uint32_t[m_n_outputs + 1];

  error e("Incorrect code structure\n");

  for (int32_t in_bit = 0; in_bit < m_n_inputs; in_bit++) {
    const uint32_t in_edge = p_code.in_bit_to_edge(in_bit);
    if (in_edge >= m_n_edges || (in_bit > 0 && in_edge <= m_in_bit_to_edge[in_bit - 1])) {
      throw e;
    }
    m_in_bit_to_edge[in_bit] = in_edge;
  }
  m_in_bit_to_edge[m_n_inputs] = static_cast<uint32_t>(m_n_edges);

  for (int32_t out_bit = 0; out_bit < m_n_outputs; out_bit++) {
    const uint32_t out_edge = p_code.out_bit_to_edge(out_bit);
    if (out_edge >= m_n_edges || (out_bit > 0 && out_edge <= m_out_bit_to_edge[out_bit - 1])) {
      throw e;
    }
    m_out_bit_to_edge[out_bit] = out_edge;
  }
  m_out_bit_to_edge[m_n_outputs] = static_cast<uint32_t>(m_n_edges);

  for (uint32_t out_edge = 0; out_edge < m_n_edges; out_edge++) {
    uint32_t in_edge = p_code.edge_out_to_in(out_edge);
    m_out_to_in_edge[out_edge] = in_edge;
    m_in_to_out_edge[in_edge] = out_edge;
    m_out_edge_to_in_bit[out_edge] = p_code.in_edge_to_bit(in_edge);
  }

  // get the total memory of the device
  uint64_t total_memory = m_device_manager.get_total_global_memory();

  cout << "Total device memory: " << total_memory << " bytes = " << (total_memory >> 20) << " MB" << endl;

  uint64_t code_repr_memory = (m_n_outputs + 3 * m_n_edges + m_n_inputs) * 4;

  cout << "Memory used to represent the error-correcting code graph: " << code_repr_memory
       << " bytes = " << (code_repr_memory >> 20) << " MB" << endl;

  // instance buffers : 2 bits per syndrome bit, 1 llr per edge, 2 llrs + 1 char + 1 bit per input bit.
  uint64_t instance_memory =
      2 * (m_n_outputs >> 3) + sizeof(llr_t) * m_n_edges + (2 * sizeof(llr_t) + 1) * m_n_inputs + (m_n_inputs >> 3);

  cout << "Memory used by one decoded vector: " << instance_memory << " bytes = " << (instance_memory >> 20) << " MB"
       << endl;

  const uint64_t security_memory = total_memory / 10;

  // compute the maximum parallel factor that fits into the device memory
  uint64_t max_parallel_factor_from_gpu = (total_memory - security_memory - code_repr_memory) / instance_memory;

  m_log2_parallel_factor = 0;
  while (1u << (m_log2_parallel_factor + 1) <= max_parallel_factor_from_gpu)
    m_log2_parallel_factor++;
  m_log2_parallel_factor = min(m_log2_parallel_factor, p_params.m_max_log_parallel_factor_user);
  m_parallel_factor = 1u << m_log2_parallel_factor;

  cout << "Chosen parallel factor: 2**" << m_log2_parallel_factor << " = " << m_parallel_factor
       << " vectors decoded in parallel" << endl;

  cout << "estimated GPU memory usage: " << ((code_repr_memory + m_parallel_factor * instance_memory) >> 20) << " MB"
       << endl;

  // size of syndrome buffer of gpu, in terms of 32-bit words. because of bit-level interlacing
  // of vectors, and since the buffer is a uint32_t buffer,
  // m_parallel_factor must be rounded to the next multiple of 32.
  m_syndrome_uint32_sz = (m_n_outputs + 0x1f) >> 5;
  m_n_outputs_words_p = m_syndrome_uint32_sz * m_parallel_factor;
  const int64_t n_inputs_p = m_parallel_factor * m_n_inputs;
  const int64_t n_edges_p = m_parallel_factor * m_n_edges;
  m_vector_places = new uint32_t[m_parallel_factor];
  m_message_buffer = new transfer_llr_t[n_edges_p];

  m_log2_threads_per_vector = m_log2_global_threads - m_log2_parallel_factor;

  const int64_t threads_per_vector = 1uLL << m_log2_threads_per_vector;
  cout << "# of GPU threads per vector: " << threads_per_vector << endl;
  cout << "# of GPU threads per block per vector: " << pow(2, (int)(m_log2_local_threads - m_log2_parallel_factor)) << endl;

  // declare and initialize GPU buffers
  const uint64_t input_uint32_sz = m_n_inputs >> 5; // m_n_inputs is a multiple of 32
  if(g_use_pinned_memory)
  {
    m_llrs = m_device_manager.create_host_pinned_buffer<transfer_llr_t>(cuda_manager::w, n_inputs_p);
    m_final_bits_packed =
        m_device_manager.create_host_pinned_buffer<uint32_t>(cuda_manager::r, input_uint32_sz * m_parallel_factor);
    m_parity_violations = m_device_manager.create_host_pinned_buffer<char>(cuda_manager::r, m_parallel_factor);
  } else {
    m_llrs = new transfer_llr_t[n_inputs_p];
    m_final_bits_packed = new uint32_t[input_uint32_sz * m_parallel_factor];
    m_parity_violations = new char[m_parallel_factor];
  }
  memset(m_parity_violations, 0, m_parallel_factor);  

  m_syndrome_dev_buf = m_device_manager.create_device_buffer<uint32_t>(cuda_manager::w, m_n_outputs_words_p);
  m_new_syndrome_dev_buf = m_device_manager.create_device_buffer<uint32_t>(cuda_manager::w, m_n_outputs_words_p);
  m_message_dev_buf = m_device_manager.create_device_buffer<llr_t>(cuda_manager::r, n_edges_p);
  m_initial_llrs_dev_buf = m_device_manager.create_device_buffer<llr_t>(cuda_manager::w, n_inputs_p);
  m_new_initial_llrs_dev_buf = m_device_manager.create_device_buffer<transfer_llr_t>(cuda_manager::w, n_inputs_p);
  m_final_bits_dev_buf = m_device_manager.create_device_buffer<char>(cuda_manager::r, n_inputs_p);
  m_final_bits_packed_dev_buf =
      m_device_manager.create_device_buffer<uint32_t>(cuda_manager::r, input_uint32_sz * m_parallel_factor);
  m_vec_places_dev_buf = m_device_manager.create_device_buffer<uint32_t>(cuda_manager::r, m_parallel_factor);
  m_parities_violated_dev_buf = m_device_manager.create_device_buffer<char>(cuda_manager::r, m_parallel_factor);
  m_swap_o_dev_buf = m_device_manager.create_device_buffer<uint32_t>(cuda_manager::w, m_parallel_factor);
  m_swap_d_dev_buf = m_device_manager.create_device_buffer<uint32_t>(cuda_manager::w, m_parallel_factor);

  m_device_manager.create_device_buffer_from_host_data(cuda_manager::r, m_n_outputs + 1, m_out_bit_to_edge_dev_buf,
                                                     m_out_bit_to_edge);
  m_device_manager.create_device_buffer_from_host_data(cuda_manager::r, m_n_edges, m_out_to_in_edge_dev_buf,
                                                     m_out_to_in_edge);
  m_device_manager.create_device_buffer_from_host_data(cuda_manager::r, m_n_edges, m_in_to_out_edge_dev_buf,
                                                     m_in_to_out_edge);
  m_device_manager.create_device_buffer_from_host_data(cuda_manager::r, m_n_inputs + 1, m_in_bit_to_edge_dev_buf,
                                                     m_in_bit_to_edge);
  m_device_manager.create_device_buffer_from_host_data(cuda_manager::r, m_n_edges, m_out_edge_to_in_bit_dev_buf,
                                                     m_out_edge_to_in_bit);

  cout << "Total memory allocated: " << (m_device_manager.mem_usage() >> 20) << " MB" << endl;
}

ldpc_decoder_gpu_cuda::~ldpc_decoder_gpu_cuda() {
  m_device_manager.release_buffer(m_syndrome_dev_buf);
  m_device_manager.release_buffer(m_new_syndrome_dev_buf);
  m_device_manager.release_buffer(m_message_dev_buf);
  m_device_manager.release_buffer(m_initial_llrs_dev_buf);
  m_device_manager.release_buffer(m_out_bit_to_edge_dev_buf);
  m_device_manager.release_buffer(m_out_to_in_edge_dev_buf);
  m_device_manager.release_buffer(m_in_to_out_edge_dev_buf);
  m_device_manager.release_buffer(m_in_bit_to_edge_dev_buf);
  m_device_manager.release_buffer(m_out_edge_to_in_bit_dev_buf);
  m_device_manager.release_buffer(m_swap_o_dev_buf);
  m_device_manager.release_buffer(m_swap_d_dev_buf);

  if(g_use_pinned_memory)
  {
    m_device_manager.release_pinned_buffer(m_llrs);
    m_device_manager.release_pinned_buffer(m_final_bits_packed);
    m_device_manager.release_pinned_buffer(m_parity_violations);
  }
  else
  {
    delete[] m_llrs;
    delete[] m_final_bits_packed;
    delete[] m_parity_violations;
  }
  delete[] m_vector_places;
  delete[] m_message_buffer;
  delete[] m_out_to_in_edge;
  delete[] m_in_to_out_edge;
  delete[] m_out_edge_to_in_bit;
  delete[] m_in_bit_to_edge;
  delete[] m_out_bit_to_edge;
}

/*
 * output in m_llr the vectors of index
 * p_first_vector_idx ... p_first_vector_idx + p_num_output_vectors - 1
 * interleaved with a stride of p_output_stride;
 * The input stride is p_input_stride.
 */
void ldpc_decoder_gpu_cuda::prepare_vectors(transfer_llr_t *p_input, uint32_t p_input_stride, uint32_t p_output_stride,
                                            uint32_t p_first_vector_idx, uint32_t p_num_output_vectors) {
  const int64_t n_regular_variables = m_n_inputs - m_n_erased_inputs;
  for (int64_t i = 0; i < n_regular_variables; i++) {
    transfer_llr_t *a = &m_llrs[i * p_output_stride];
    transfer_llr_t *b = &p_input[i * p_input_stride + p_first_vector_idx];
    for (int64_t v = 0; v < p_num_output_vectors; v++)
      a[v] = b[v];
  }

  const bool gpu_needs_llrs = decoding_input_is_llr();
  if (gpu_needs_llrs) {
    // llrs are expected, the values on the channel must be converted
    for (int64_t idx = 0; idx < n_regular_variables * p_num_output_vectors; idx++) {
      m_llrs[idx] = m_channel.llr(m_llrs[idx]);
    }
  }
}

void ldpc_decoder_gpu_cuda::transfer_vectors(uint32_t p_num_vectors, transfer_llr_t *p_llrs, const uint32_t *p_syndromes) {
  const int64_t n_regular_variables = m_n_inputs - m_n_erased_inputs;
  // transfer values or LLRs to GPU
  m_device_manager.enqueue_write(m_new_initial_llrs_dev_buf, false, 0, n_regular_variables * p_num_vectors, p_llrs);
  CUDA_SYNC_CHECK;
  
  // clear LLRs corresponding to erased variables
  m_device_manager.enqueue_clear(m_new_initial_llrs_dev_buf, n_regular_variables * p_num_vectors, m_n_erased_inputs * p_num_vectors);
  CUDA_SYNC_CHECK;

  // transfer syndromes to GPU
  m_device_manager.enqueue_write(m_new_syndrome_dev_buf, false, 0, m_syndrome_uint32_sz * p_num_vectors, p_syndromes);
  CUDA_SYNC_CHECK;
  m_device_manager.enqueue_barrier();  

  channelType c = m_channel.channel();
  if (c == bsc || c == awgn) {
    // To handle more channels types here, one must:
    // - add more cases below and the corresponding kernel (obviously)
    // - signal the corresponding channel type in method decoding_input_is_llr() so that the
    // conversion on CPU is disabled.

    if (c == bsc) {
      auto cc = dynamic_cast<const bsc_channel *>(&m_channel);
      assert(cc);
      m_noise_factor = static_cast<float>(cc->ref_llr());
      // sets arguments before enqueuing call because m_noise_factor is modified
      llr_bsc<<<m_tiles, m_local_threads>>>(m_new_initial_llrs_dev_buf, m_noise_factor, m_log2_parallel_factor,
                                            n_regular_variables, m_log2_global_threads);
      CUDA_SYNC_CHECK;
    } else if (c == awgn) {
      auto cc = dynamic_cast<const biawgn_channel *>(&m_channel);
      assert(cc);
      m_noise_factor = static_cast<float>(cc->factor());
      // sets arguments before enqueuing call because m_noise_factor is modified
      llr_biawgn<<<m_tiles, m_local_threads>>>(m_new_initial_llrs_dev_buf, m_noise_factor, m_log2_parallel_factor,
                                               n_regular_variables, m_log2_global_threads);
      CUDA_SYNC_CHECK;
    }
  }
  m_device_manager.enqueue_barrier();
  uint32_t offset = 0;
  for (int i = 31; i >= 0; i--) {
    uint32_t bit = 1u << i;
    if (bit & p_num_vectors) {
      uint32_t log2_num_new_vectors = i;
      flood_refill<<<m_tiles, m_local_threads>>>(
          m_message_dev_buf, m_initial_llrs_dev_buf, m_new_initial_llrs_dev_buf, m_syndrome_dev_buf,
          m_new_syndrome_dev_buf, m_in_to_out_edge_dev_buf, m_in_bit_to_edge_dev_buf, m_syndrome_uint32_sz, offset,
          p_num_vectors, log2_num_new_vectors, m_n_inputs, m_log2_parallel_factor, m_log2_global_threads);
      CUDA_SYNC_CHECK;
      offset += bit;
    }
  }
  m_device_manager.enqueue_barrier();
}

static void print_time(timer &t, const char *msg) {
  t.stop();
  cout << "time: " << t.time() << endl;
  cout << msg << endl;
  t.reset();
  t.start();
}

void ldpc_decoder_gpu_cuda::decode(const ldpc_decoder_gpu_dynamic_parameters &p_dyn_params,
                                   uint32_t p_num_vectors_to_process, void *p_input_llrs_void, const uint32_t *p_syndromes,
                                   uint32_t *p_results, test_report &report, uint32_t p_log) {
  ios cout_format(nullptr);
  transfer_llr_t *p_input_llrs = (transfer_llr_t *)p_input_llrs_void;
  if (p_log >= 1) {
    cout_format.copyfmt(cout);
    cout << fixed << setprecision(3);
  }

  if (p_num_vectors_to_process == 0)
    return;

  timer gpu_timer(true);
  const uint64_t input_uint32_sz = m_n_inputs >> 5; // m_n_inputs is a multiple of 32

  const bool gpu_needs_llrs = decoding_input_is_llr();
  uint32_t num_vectors_to_process_in_gpu = p_num_vectors_to_process; // numbers of vectors to retrieve from GPU
  const uint32_t num_vectors_gpu_batch = min(p_num_vectors_to_process, m_parallel_factor);
  uint32_t next_vector_to_load = num_vectors_gpu_batch;
  uint32_t *vectors_in_gpu = new uint32_t[p_num_vectors_to_process];
  uint32_t *iter_start = new uint32_t[p_num_vectors_to_process];
  uint32_t *iter_end = new uint32_t[p_num_vectors_to_process];
  for(size_t i=0;i < p_num_vectors_to_process;i++){
    iter_start[i] = -1u;
    iter_end[i]   = -1u;
  }
  unique_ptr<uint32_t[]> _1(vectors_in_gpu);
  unique_ptr<uint32_t[]> _2(iter_start);
  unique_ptr<uint32_t[]> _3(iter_end);
  for (uint32_t i = 0; i < num_vectors_gpu_batch; i++) {
    vectors_in_gpu[i] = i;
  }

  if (p_log >= 2) {
    cout << "LLR conversion done on " << (gpu_needs_llrs ? " CPU" : "GPU") << endl;
    cout << "Total number of vectors to process: " << p_num_vectors_to_process << endl;
    cout << "LLR input stride: " << p_num_vectors_to_process << endl;
    cout << "LLR stride for GPU: " << m_parallel_factor << endl;
    cout << "number of vectors in first group: " << num_vectors_gpu_batch << endl;
  }

  uint32_t num_swaps, log2_num_swaps, num_new_vectors;
  prepare_vectors(p_input_llrs, p_num_vectors_to_process, num_vectors_gpu_batch, 0, num_vectors_gpu_batch);

  if (p_log >= 1) {
    cout << "decoder: pre-CUDA time: " << gpu_timer.time() << "; starting CUDA kernels" << endl;
  }

  // note: for all GPU kernel calls, passing nullptr instead of &m_local_threads should enable the
  // GPU to choose the proper workgroup size. However, it locks up the system under linux
  // and does not work properly on windows either.

  // transfer llrs and syndromes to GPU, convert values to LLRs on GPUs if needed
  transfer_vectors(num_vectors_gpu_batch, m_llrs, p_syndromes);

  if (p_log >= 1) {
    m_device_manager.finish();
    cout << "decoder: time = " << gpu_timer.time() << "; data transfer complete" << endl;
  }
  uint32_t global_iter = 0;
  float iter_start_time = gpu_timer.time();
  float iter_end_time = 0;
  while (true) {
    flood_backward<<<m_tiles, m_local_threads>>>(m_syndrome_dev_buf, m_message_dev_buf, m_out_bit_to_edge_dev_buf,
                                                 m_syndrome_uint32_sz, m_log2_parallel_factor, m_n_outputs, m_log2_threads_per_vector);
    CUDA_SYNC_CHECK;
    m_device_manager.enqueue_barrier();
    const bool do_parity_check = (global_iter > 0) && ((global_iter % p_dyn_params.m_num_iter_check_parity) == 0);
    if (!do_parity_check) {
      flood_forward<<<m_tiles, m_local_threads>>>(m_message_dev_buf, m_initial_llrs_dev_buf, m_in_to_out_edge_dev_buf,
                                                  m_in_bit_to_edge_dev_buf, m_log2_parallel_factor, m_n_inputs, m_log2_global_threads);
      CUDA_SYNC_CHECK;
      m_device_manager.enqueue_barrier();
    } else {
      if (p_log >= 1) {
        cout << "time " << gpu_timer.time() << endl;
        cout << "Iteration " << global_iter << ":" << endl;
      }
      flood_forward_w_final_bits<<<m_tiles, m_local_threads>>>(
          m_message_dev_buf, m_initial_llrs_dev_buf, m_in_to_out_edge_dev_buf, m_in_bit_to_edge_dev_buf,
          m_final_bits_dev_buf, m_log2_parallel_factor, m_n_inputs, m_log2_global_threads);
      CUDA_SYNC_CHECK;
      m_device_manager.enqueue_barrier();
      m_device_manager.enqueue_clear(m_parities_violated_dev_buf, 0, 1uL << m_log2_parallel_factor);
      check_parity<<<m_tiles, m_local_threads>>>(m_syndrome_dev_buf, m_out_bit_to_edge_dev_buf,
                                                 m_out_edge_to_in_bit_dev_buf, m_final_bits_dev_buf,
                                                 m_parities_violated_dev_buf, m_syndrome_uint32_sz,
                                                 m_log2_parallel_factor, m_n_outputs, m_log2_threads_per_vector);
      CUDA_SYNC_CHECK;
      m_device_manager.enqueue_barrier();
      m_device_manager.enqueue_read(m_parities_violated_dev_buf, false, 0, m_parallel_factor, m_parity_violations);
      m_device_manager.finish();
      uint32_t num_errors = 0;
      for (uint32_t j = 0; j < m_parallel_factor; j++) {
        if (m_parity_violations[j]) {
          num_errors++;
        }
      }
      if (p_log >= 1) {
        cout << num_errors << " vectors with parity errors" << endl;
      }

      vector<bool> vectors_to_stop(m_parallel_factor, false);
      // number of vectors whose decoding can be stopped
      // a subset of the corresponding vectors will be removed from the GPU
      // and replaced by new vectors to decode
      uint32_t num_vectors_to_stop = 0;
      if (p_log >= 3) {
        cout << " vectors that should be retrieved:" << endl;
      }

      for (uint32_t j = 0; j < num_vectors_gpu_batch; j++) {
        uint32_t num_iter = global_iter - iter_start[vectors_in_gpu[j]];
        if (!m_parity_violations[j] || num_iter >= p_dyn_params.m_num_iter_max) {
          num_vectors_to_stop++;
          vectors_to_stop[j] = true;
          if (iter_end[vectors_in_gpu[j]] == -1u) {
            iter_end[vectors_in_gpu[j]] = global_iter;
          }
        }

        if (p_log >= 3) {
          if (vectors_to_stop[j])
            cout << " *";
          else
            cout << "  ";
          cout << " " << "gpu idx = " << j << "; real idx = " << vectors_in_gpu[j]
               << "; parity violations: " << (int)m_parity_violations[j] << "; iterations: " << num_iter << endl;
        }
      }
      if (next_vector_to_load == p_num_vectors_to_process && num_vectors_to_stop == num_vectors_gpu_batch) {
        iter_end_time = gpu_timer.time();
        if (p_log >= 2) {
          cout << " All vectors sent to the GPU and finished" << endl;
        }
        // crude finishing strategy for now:
        // if all vectors have finished, retrieve everything
        // otherwise, we do not enter in the current condition.
        // the code after 'if(num_new_vectors > 0)' below executes instead, but if
        // all vectors have been loaded (next_vector_to_load == p_num_vectors_to_process)
        // num_new_vectors will be 0 and no vector will be retreived.

        // this has the consequence that some vectors may have more iterations that the maximum
        // allowed: they will be processed until all vectors are above the maximum, or have
        // been fully error-corrected.

        // A more elaborate strategy would retire vectors as soon as they are finished,
        // and let the gpu allocate more threads per vector if there are less vectors to
        // process.

        // retrieve all remaining results and break
        // produce on the GPU a packed representation of the first num_new_vectors vectors
        // FIXME: done for all vectors as a first implementation since it is simpler.
        deinterlace_output<<<m_tiles, m_local_threads>>>(m_final_bits_dev_buf, m_final_bits_packed_dev_buf,
                                                         m_log2_parallel_factor, m_n_inputs, m_log2_threads_per_vector);
        CUDA_SYNC_CHECK;
        m_device_manager.enqueue_barrier();
        // retrieve the packed representation of all remaining vectors in gpu
        m_device_manager.enqueue_read(m_final_bits_packed_dev_buf, false, 0, input_uint32_sz * num_vectors_gpu_batch,
                                    m_final_bits_packed);
        m_device_manager.finish();
        // update p_results accordingly
        if (p_log >= 1)
          cout << "Retrieving the last " << num_vectors_gpu_batch << " vectors" << endl;
        if (p_log >= 3) {
          cout << " Indexes of vectors retrieved: ";
          for (uint32_t j = 0; j < num_vectors_gpu_batch; j++) {
            cout << vectors_in_gpu[j];
            if (j + 1 < num_vectors_gpu_batch)
              cout << ", ";
          }
          cout << endl;
        }
        for (uint32_t j = 0; j < num_vectors_gpu_batch; j++) {
          memcpy(p_results + vectors_in_gpu[j] * input_uint32_sz, m_final_bits_packed + j * input_uint32_sz,
                 4 * input_uint32_sz);
        }
        break;
      }

      num_new_vectors = min(p_num_vectors_to_process - next_vector_to_load, num_vectors_to_stop);

      // since the number of new vectors to introduce is potentially smaller than the number of vectors to stop,
      // some vectors may remain on the GPU although they should be stopped. Hence it is possible to see vectors
      // with a number of iterations higher than the maximum allowed.

      if (num_new_vectors > 0) {
        if (p_log >= 1)
          cout << "Introducing " << num_new_vectors << " new vectors" << endl;
        if (p_log >= 2) {
          cout << " New vector indexes: " << next_vector_to_load << "..." << next_vector_to_load + num_new_vectors - 1
               << endl;
        }
        timer t(false);
        // step 1
        // compute the permutation to perform on gpu vectors to put num_new_vectors finished vectors
        // in 1st position
        // count vectors that should be output and do not have to be moved
        // update vectors_in_gpu according to permutation
        if (p_log >= 2) {
          cout << "Step 1" << endl;
          t.start();
        }
        uint32_t ctr = 0;
        for (uint32_t i = 0; i < num_new_vectors; i++) {
          if (vectors_to_stop[i])
            ctr++;
        }

        // num_swaps = # of elts j with should_stop[j] = false with j = 0 ... num_new_vectors - 1
        // num_swaps = num_new_vectors - ctr <= num_should_stop - ctr =
        //  # of elts j with should_stop[j] = true with j = num_new_vectors ... m_parallel_factor - 1
        num_swaps = num_new_vectors - ctr;
        vector<uint32_t> origin(num_swaps);
        vector<uint32_t> dest(num_swaps);
        uint32_t o = 0, d = num_new_vectors;
        for (uint32_t i = 0; i < num_swaps; i++) {
          while (vectors_to_stop[o])
            o++;
          assert(o < num_new_vectors);
          while (!vectors_to_stop[d])
            d++;
          assert(d < num_vectors_init);
          // one has:
          //  should_stop[o]
          // !should_stop[d]
          origin[i] = o++;
          dest[i] = d++;
        }

        for (uint32_t i = 0; i < num_swaps; i++) {
          swap(vectors_in_gpu[origin[i]], vectors_in_gpu[dest[i]]);
        }
        if (p_log >= 2) {
          cout << "Retrieving results for " << num_new_vectors << " vectors" << endl;
        }
        if (p_log >= 3) {
          cout << "  Indexes of vectors retrieved: ";
          for (uint32_t j = 0; j < num_new_vectors; j++) {
            cout << vectors_in_gpu[j];
            if (j + 1 < num_new_vectors)
              cout << ", ";
          }
          cout << endl;
        }

        // step 2
        // exec kernel that puts num_new_vectors vectors to output in 1st positions
        if (p_log >= 2)
          print_time(t, "Step 2");

        if (num_swaps > 0) {
          m_device_manager.enqueue_write(m_swap_o_dev_buf, false, 0, num_swaps, origin.data());
          m_device_manager.enqueue_write(m_swap_d_dev_buf, false, 0, num_swaps, dest.data());
          m_device_manager.enqueue_barrier();

          log2_num_swaps = 0;
          while ((1u << log2_num_swaps) < num_swaps)
            log2_num_swaps++;
          flood_permute_vecs<<<m_tiles, m_local_threads>>>(
              m_message_dev_buf, m_initial_llrs_dev_buf, m_final_bits_dev_buf, m_syndrome_dev_buf,
              m_in_bit_to_edge_dev_buf, m_swap_o_dev_buf, m_swap_d_dev_buf, m_syndrome_uint32_sz, num_swaps, 
              log2_num_swaps, m_log2_parallel_factor, m_n_inputs, m_log2_global_threads);
          CUDA_SYNC_CHECK;
          m_device_manager.enqueue_barrier();
        }

        // step 3: produce on the GPU a packed representation of the first num_new_vectors vectors
        // FIXME: done for all vectors as a first implementation since it is simpler.
        if (p_log >= 2) {
          m_device_manager.finish();
          print_time(t, "Step 3");
        }
        deinterlace_output<<<m_tiles, m_local_threads>>>(m_final_bits_dev_buf, m_final_bits_packed_dev_buf, m_log2_parallel_factor,
                                                         m_n_inputs, m_log2_threads_per_vector);
        CUDA_SYNC_CHECK;
        m_device_manager.enqueue_barrier();
        // step 4
        // retrieve the packed representation of the num_new_vectors 1st vectors in gpu,
        // and use it to update p_results accordingly
        if (p_log >= 2) {
          m_device_manager.finish();
          print_time(t, "Step 4");
        }
        m_device_manager.enqueue_read(m_final_bits_packed_dev_buf, false, 0, input_uint32_sz * num_new_vectors,
                                    m_final_bits_packed);
        m_device_manager.finish();

        for (uint32_t j = 0; j < num_new_vectors; j++) {
          memcpy(p_results + vectors_in_gpu[j] * input_uint32_sz, m_final_bits_packed + j * input_uint32_sz,
                 4 * input_uint32_sz);
        }

        // steps 5, 6 and 7 below could be split into multiple smaller steps in order to reduce
        // the size of GPU buffers m_new_initial_llrs_dev_buf and m_new_syndrome_dev_buf

        // step 5
        // fill host buffer with new vectors
        if (p_log >= 2) {
          m_device_manager.finish();
          t.stop();
          print_time(t, "Step 5");
        }

        prepare_vectors(p_input_llrs, p_num_vectors_to_process, num_new_vectors, next_vector_to_load, num_new_vectors);

        // step 6
        // transfer new vectors to GPU
        if (p_log >= 2)
          print_time(t, "Step 6");

        const uint32_t *syndrome_block = p_syndromes + next_vector_to_load * m_syndrome_uint32_sz;
        transfer_vectors(num_new_vectors, m_llrs, syndrome_block);

        if (p_log >= 2) {
          m_device_manager.finish();
          print_time(t, "Step 7");
        }

        // step 7: set statistics for new vectors
        for (uint32_t j = 0; j < num_new_vectors; j++) {
          vectors_in_gpu[j] = next_vector_to_load + j;
          iter_start[next_vector_to_load + j] = global_iter;
        }
        next_vector_to_load += num_new_vectors;
        num_vectors_to_process_in_gpu -= num_new_vectors;
        t.stop();
      }
    }
    global_iter++;
  }

  report.max_iter = 0;
  report.min_iter = -1u;
  report.avg_iter = 0;

  for (uint32_t j = 0; j < p_num_vectors_to_process; j++) {
    assert(iter_end[j] != -1u);
    uint32_t num_iter = iter_end[j] - iter_start[j];
    report.max_iter = max(report.max_iter, num_iter);
    report.min_iter = min(report.min_iter, num_iter);
    report.avg_iter += num_iter;
  }
  report.avg_iter /= p_num_vectors_to_process;
  report.iter_time_per_vector = (iter_end_time - iter_start_time) / (global_iter * num_vectors_gpu_batch);
  double total_time = gpu_timer.stop();
  if (p_log) {
    cout << "decoder: time = " << total_time << "; final transfer done" << endl;
  }
  cout.copyfmt(cout_format);
}
