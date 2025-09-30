#include "bool_vec.h"
#include "channel.h"
#include "common.h"
#include "ldpc_decoder_gpu.h"
#include "ldpc_decoder_gpu_cuda.h"
#include "test_report.h"
#include "transpose.h"

#include <bitset>
#include <cstring>
#include <memory>
#include <string>

#define USE_CHACHA_PRNG

#ifdef USE_CHACHA_PRNG
#include "prng_chacha.h"
#define prng prng_chacha
#else
#include "prng_aes.h"
#define prng prng_aes
#endif

/*
 sample command-line for ldpc program (must be run from root source directory to be able to find OpenCL Kernels)
 ldpc_decoder_gpu -f ../ldpc_codes/code_0_1048576_6.alist -c 1 -n 0.91 -i 200 -p 9 -m 4 -e 15
 ldpc_decoder_gpu -f ../ldpc_codes/code_14_1048576_6.alist -c  1 -n 0.94 -p 9 -m 4 -e 15
 biawgn example:
 ./ldpc_decoder_gpu -f ../ldpc_codes/code_14_1048576_6.alist -c 1 -n 0.94 -p 9 -m 4 -b 1e-5
 bsc example:
 ./ldpc_decoder_gpu  -f ../lpdc_codes/code_29_1048576_6.alist -c 0 -n 0.008 -p 9 -m 4 -b 1e-5
*/

void print_usage();

void do_test(
    const ldpc_code& code,
    noisy_channel& p_channel,
    uint32_t p_num_runs,
    const ldpc_decoder_gpu_static_parameters& p_decoder_params,
    ldpc_decoder_gpu_dynamic_parameters& p_dyn_params,
    uint32_t p_start_index, uint32_t p_logLevel);

void create_data(
    const ldpc_code& p_code,
    uint32_t p_vector_start_idx,
    uint32_t p_num_vec_per_batch,
    noisy_channel* p_channel_ptr,
    uint32_t p_batch_idx,
    transfer_llr_t* po_noisy_vec,
    uint32_t* po_ref_frames_deinterlaced,
    uint32_t* po_syndromes_deinterlaced);

int main(int argc, char** argv)
{
  string code_filename;
  transfer_llr_t noise = 0;
  uint32_t num_runs = 1;
  uint32_t vec_start_index = 0;
  int channel_idx = 0;
  uint32_t target_errors = 0;
  double target_ber = 0;
#ifdef EXTRA_CHANNELS
  uint32_t group_size = 16;
#endif

  ldpc_decoder_gpu_static_parameters static_p;
  ldpc_decoder_gpu_dynamic_parameters dyn_p;
  bool channel_defined = false;
  bool noise_defined = false;
  bool error_defined = false;
  bool ber_defined = false;

  int curr_arg = 1;
  bool err = false;
  int logLevel = 1;
  while (curr_arg < argc)
  {
	  size_t len = strlen(argv[curr_arg]);
    if (len != 2 || argv[curr_arg][0] != '-')
	  {
      err = true;
      break;
    }
    char c = argv[curr_arg][1];
    char* param = curr_arg + 1 < argc ? argv[curr_arg + 1] : nullptr;
    switch (c)
    {
    case 'b':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      ber_defined = true;
      target_ber = atof(param);
      break;
    case 'c':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      channel_defined = true;
      channel_idx = atoi(param);
      break;
    case 'e':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      error_defined = true;
      target_errors = atoi(param);
      break;
    case 'f':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      code_filename = string(param);
      break;
#ifdef EXTRA_CHANNELS
    case 'g':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      group_size = static_cast<uint32_t>(atoi(param));
      break;
#endif
    case 'h':
      print_usage();
      exit(EXIT_SUCCESS);
    case 'i':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      dyn_p.m_num_iter_max = static_cast<uint32_t>(atoi(param));
      break;
    case 'l':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      logLevel = atoi(param);
      if (logLevel < 1 || logLevel > 3) {
        err = true;
        break;
      }
      break;
    case 'm':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      dyn_p.m_loading_factor = atoi(param);
      break;
    case 'n':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      noise_defined = true;
      noise = static_cast<transfer_llr_t>(atof(param));
      break;
    case 'p':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      static_p.m_max_log_parallel_factor_user = static_cast<uint32_t>(atoi(param));
      break;
    case 'r':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      num_runs = static_cast<uint32_t>(atoi(param));
      break;
    case 's':
      if (!param) {
        err = true; break;
      }
      curr_arg++;
      vec_start_index = static_cast<uint32_t>(atoi(param));
      break;
    default:
      cout << "unrecognized argument" << endl;
      exit(EXIT_FAILURE);
    }
    curr_arg++;
  }
  if (err)
  {
	  print_usage();
	  exit(EXIT_FAILURE);
  }

  cout << "Code file name:" << code_filename << endl;

  if (num_runs == 0)
  {
    cout << "0 runs to perform, exiting" << endl;
    exit(EXIT_SUCCESS);
  }
  bool user_error = false;
  if(error_defined && ber_defined)
  {
    cout << "Cannot define both bit error rate and bit error count" << endl;
    user_error = true;
  }
  if(dyn_p.m_loading_factor == 0)
  {
    cout << "Invalid overloading factor" << endl;
    user_error = true;
  }
  if (!channel_defined || !noise_defined)
  {
    cout << "Missing mode and/or channel parameters" << endl;
    user_error = true;
  }

  if (code_filename.empty())
  {
    cout << "You have to enter a filename with option -f (filename)." << endl;
    user_error = true;
  }

  noisy_channel* channel_ptr = nullptr;
  switch(channel_idx)
  {
  case 0:
    channel_ptr = new bsc_channel(noise);
    break;
  case 1:
    channel_ptr = new biawgn_channel(noise);
    break;
#ifdef EXTRA_CHANNELS
  case 2:
    channel_ptr = new multigauss_channel(noise, group_size);
    break;
#endif
  default:
    cout << "Unknown channel type specified" << endl;
    user_error = true;
    break;
  }
  unique_ptr<noisy_channel> _(channel_ptr);

  if (user_error)
  {
    print_usage();
    exit(EXIT_FAILURE);
  }

  try
  {
    // read code
    const ldpc_code code(code_filename, true);
    const uint32_t frame_sz = static_cast<uint32_t>(code.n_inputs());

    dyn_p.m_target_errors = target_errors > 0 ? target_errors :
          static_cast<uint32_t>(static_cast<double>(frame_sz) * target_ber);
    cout << "Target number of errors per frame: " << dyn_p.m_target_errors << endl << endl;
    do_test(code, *channel_ptr, num_runs, static_p, dyn_p, vec_start_index, logLevel);
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }
  return (EXIT_SUCCESS);
}

void deinterlace(
    uint32_t p_num_vecs,
    int64_t p_vec_uint32_sz,
    bool_vec& p_vecs,
    uint32_t* p_vecs_deinterlaced)
{
  ALIGN(32) uint64_t mat[16];
  uint32_t* mat_32 = (uint32_t*) mat;

  for(uint32_t vgroup = 0; vgroup < (p_num_vecs + 0x1F) >> 5; vgroup++)
  {
    // deinterlace input data
    for(int64_t igroup = 0; igroup < p_vec_uint32_sz; igroup++)
    {
      for(int i = 0; i < 0x20; i++)
      {
        mat_32[i] = p_vecs.word_ref(vgroup, i + (igroup << 5));
      }
      transpose_32x32_AVX2(mat, mat);

      for(int64_t v = vgroup << 5; v < min((vgroup + 1) << 5, p_num_vecs); v++)
      {
        p_vecs_deinterlaced[igroup + v * p_vec_uint32_sz] = mat_32[v - (vgroup << 5)];
      }
    }
  }
}

void do_test(
    const ldpc_code& code,
    noisy_channel& p_channel,
    uint32_t p_num_runs,
    const ldpc_decoder_gpu_static_parameters& p_decoder_params,
    ldpc_decoder_gpu_dynamic_parameters& p_dyn_params,
    uint32_t p_start_index, uint32_t p_logLevel)
{
  //const uint32_t logging = 1;
  // 0: no logging
  // 1: standard
  // 2: verbose
  // 3: more verbose

#ifdef CUDA_DECODER
  ldpc_decoder_gpu_cuda dec(code, p_channel, p_decoder_params);
#else
  ldpc_decoder_gpu dec(code, p_channel, p_decoder_params);
#endif
  p_dyn_params.m_num_vectors_per_run = dec.parallel_factor() * p_dyn_params.m_loading_factor;
  const uint32_t n_vec_per_run = p_dyn_params.m_num_vectors_per_run;
  const uint32_t frame_sz = static_cast<uint32_t>(code.n_inputs());
  const int64_t num_data_bits_per_batch = code.n_inputs() * n_vec_per_run;
  const int64_t num_syndrome_bits_per_batch = n_effective_outputs(code) * n_vec_per_run;

  // compute and log channel and code parameters
  stringstream desc;
  describe_run(p_num_runs, n_vec_per_run, desc);

  stringstream s;
  describe_code_and_channel(code, p_channel, s);
  test_report report;
  report.code_and_channel_specs = s.str();
  report.num_runs = p_num_runs;
  report.num_vectors_per_run = n_vec_per_run;
  report.frame_size = static_cast<uint32_t>(frame_sz);
  report.target_errors = p_dyn_params.m_target_errors;

  // FIXME do not duplicate this definition
  const int64_t vec_uint32_sz = (frame_sz + 0x1F) >> 5;
  // round the vector sizes below to the upper multiple of 32 bits to be able to use
  // vectorized transpose
  int64_t syndrome_uint32_sz = (n_effective_outputs(code) + 0x1F) >> 5;

  uint32_t* ref_frames_deinterlaced = new uint32_t[vec_uint32_sz * n_vec_per_run];
  unique_ptr<uint32_t[]> _1(ref_frames_deinterlaced);

  uint32_t* result_frames_deinterlaced = new uint32_t[vec_uint32_sz * n_vec_per_run];
  unique_ptr<uint32_t[]> _2(result_frames_deinterlaced);

  uint32_t* syndromes_deinterlaced = new uint32_t[syndrome_uint32_sz * n_vec_per_run];
  unique_ptr<uint32_t[]> _3(syndromes_deinterlaced);

  vector<transfer_llr_t> noisy_frames(num_data_bits_per_batch);

  cout << desc.str();
  cout << "Total syndrome size per batch: " << num_syndrome_bits_per_batch << " bits" << endl;
  cout << "Total data size per batch: "     << num_data_bits_per_batch     << " bits" << endl;
  cout << endl;

  // timer summing all test times of all batches
  timer t(false);
  for (uint32_t i = 0; i < report.num_runs; i++)
  {
    cout << "Creating and processing frame batch " << i << " / " << report.num_runs << endl;

    cout << " Creating test vectors" << endl;
    t.start();

    create_data(
          code,
          p_start_index,
          n_vec_per_run,
          &p_channel,
          i,
          &noisy_frames[0],
          ref_frames_deinterlaced,
          syndromes_deinterlaced);

    cout << " Test vector computation time: " << t.stop() << endl;
    t.reset();
    vector<uint32_t> errors(n_vec_per_run, 0);
    const uint32_t offset = p_start_index + report.num_vectors_per_run * i;
    if(p_logLevel >= 3)
    {
      cout << " Computing errors before EC" << endl;
      for (uint32_t v = 0; v < n_vec_per_run; v++)
      {
        errors[v] = 0;
        for (uint32_t j = 0; j < frame_sz; j++)
        {
          bool noisy_bool = llr_to_bool(noisy_frames[v + j * n_vec_per_run]);
          bool ref_bool = (ref_frames_deinterlaced[(j >> 5) + vec_uint32_sz * v] >> (j&0x1F)) & 1;
          if (noisy_bool != ref_bool) errors[v]++;
        }
      }
      cout << "  Errors before error correction ";
      describe_error_stats(report.num_vectors_per_run, offset, errors, frame_sz, cout, p_logLevel);
    }
    cout << " Decoding" << endl;
    // decode
    t.start();
    dec.decode(
        p_dyn_params, n_vec_per_run, (void*) noisy_frames.data(), syndromes_deinterlaced,
        result_frames_deinterlaced, report, p_logLevel);
    report.elapsed_time = t.stop();

    if(p_logLevel >= 1)
    {
      cout << "Iterations (avg / max / min): " << report.avg_iter << " " << report.max_iter
           << " " << report.min_iter << endl;
    }

    // compute number of errors after decoding
    cout << " Computing errors after EC" << endl;
    for(int64_t v = 0; v < n_vec_per_run; v++)
    {
      errors[v] = 0;
      for(int64_t i = 0; i < vec_uint32_sz; i++)
      {
        uint32_t ref    =    ref_frames_deinterlaced[i + v * vec_uint32_sz];
        uint32_t result = result_frames_deinterlaced[i + v * vec_uint32_sz];
        if(ref != result)
        {
          uint32_t cnt =
              static_cast<uint32_t>(bitset<numeric_limits<uint32_t>::digits>(ref^result).count());
          errors[v]+= cnt;
          report.num_bit_errors += cnt;
        }
      }
    }

    cout << "  Errors after error correction ";
    describe_error_stats(report.num_vectors_per_run, offset, errors, frame_sz, cout, p_logLevel);

    for(uint32_t vector = 0 ; vector < report.num_vectors_per_run; vector++)
    {
      if(errors[vector] > 0)              report.vectors_with_errors++;
      if(errors[vector] > report.target_errors) report.vectors_with_error_above_target++;
      report.max_bit_error = max(report.max_bit_error, errors[vector]);
    }
    cout << endl;
  }
  cout << "End of decoding test" << endl << endl;

  report.gen_summary();
  cout << report.report.str();
}

void create_data(
    const ldpc_code& p_code,
    uint32_t p_vector_start_idx,
    uint32_t p_num_vec_per_batch,
    noisy_channel* p_channel_ptr,
    uint32_t p_batch_idx,
    transfer_llr_t* po_noisy_vec,
    uint32_t* po_ref_frames_deinterlaced,
    uint32_t* po_syndromes_deinterlaced)
{
  const int64_t vec_uint32_sz = (p_code.n_inputs() + 0x1F) >> 5;
  // round the vector sizes below to the upper multiple of 32 bits to be able to use
  // vectorized transpose
  int64_t syndrome_uint32_sz = (n_effective_outputs(p_code) + 0x1F) >> 5;
  bool_vec po_ref_vec(p_num_vec_per_batch, p_code.n_inputs());
  bool_vec syndrome_batch(p_num_vec_per_batch, syndrome_uint32_sz << 5);

  const int64_t vec_sz         = p_code.n_inputs();
  const int64_t transmitted_sz = p_code.n_inputs() - p_code.n_erased_inputs();
#ifdef EXTRA_CHANNELS
  const int64_t block_size = p_channel_ptr->block_size();
#endif
  const uint32_t num_words = po_ref_vec.num_words_per_bit();
  const uint32_t num_vec_rounded = num_words * bool_t_bit_size;
  // index of the first vector generated. all seeds for pseudorandom generation are
  // computed from that value, in order to be able to seek into sequences
  const uint64_t vec_start_idx = p_vector_start_idx + p_batch_idx * p_num_vec_per_batch;
  prng r(0);
  for (uint32_t v_group = 0; v_group < num_words; v_group++)
  {
    r.reset_seed(vec_start_idx + v_group * bool_t_bit_size);
    prng r(vec_start_idx + v_group * bool_t_bit_size);
    // generate bool_t_bit_size vectors in one pass
    for (int64_t i = 0; i < vec_sz; i++)
    {
      po_ref_vec.word_ref(i * num_words + v_group) = r.random_int();
    }
  }
#ifdef EXTRA_CHANNELS
  if(block_size > 1)
  {
    llr_t buf[max_channel_group_size];
    for (uint32_t v = 0; v < p_num_vec_per_batch; v++)
    {
      r.reset_seed((vec_start_idx + v) | (1uLL << 32));
      prng r((vec_start_idx + v) | (1uLL << 32));
      const size_t num_blocks = transmitted_sz / block_size;
      for (size_t b = 0; b < num_blocks; b++)
      {
        const size_t p_offset_ref   = v + b * block_size * num_vec_rounded;
        const size_t p_offset_noisy = v + b * block_size * p_num_vec_per_batch;
        for(int64_t j = 0; j < block_size; j++)
        {
          buf[j] = bool_to_llr(po_ref_vec[p_offset_ref + j * num_vec_rounded]);
        }
        p_channel_ptr->add_noise(r, buf);
        for(int64_t j = 0; j < block_size; j++)
        {
          
          po_noisy_vec[p_offset_noisy + j * p_num_vec_per_batch] = buf[j];
        }
      }
      for (int64_t i = transmitted_sz; i < vec_sz; i++)
      {
        po_noisy_vec[v + p_num_vec_per_batch * i] = 0;
      }
    }
  }
  else {
#endif
    for (uint32_t v = 0; v < p_num_vec_per_batch; v++)
    {
      r.reset_seed((vec_start_idx + v) | (1uLL << 32));
      int64_t i;
      for (i = 0; i < transmitted_sz; i++)
      {
        transfer_llr_t val = bool_to_llr(po_ref_vec[v + i * num_vec_rounded]);
        po_noisy_vec[v + p_num_vec_per_batch * i] = p_channel_ptr->add_noise(r, val);
      }
      // erased bits: channel output is 0
      for (; i < vec_sz; i++) po_noisy_vec[v + p_num_vec_per_batch * i] = 0;
    }
#ifdef EXTRA_CHANNELS
  }
#endif
  deinterlace(p_num_vec_per_batch, vec_uint32_sz, po_ref_vec, po_ref_frames_deinterlaced);
  compute_syndrome(p_code, po_ref_vec, syndrome_batch);
  deinterlace(p_num_vec_per_batch, syndrome_uint32_sz, syndrome_batch, po_syndromes_deinterlaced);
}

void print_usage()
{
  cout << "options: " << endl;
  cout << " -b f where f is the bit error rate above which a frame is considered to be in error; alternative to -e; default is 0" << endl;
#ifdef EXTRA_CHANNELS
  cout << " -c n where n defines the channel: 0 for bsc, 1 for awgn, 2 for grouped Gauss" << endl;
#else
  cout << " -c n where n defines the channel: 0 for bsc, 1 for awgn" << endl;
#endif
  cout << " -e n where n is the number of bit errors above which a frame is considered to be in error; alternative to -b; default is 0" << endl;
  cout << " -f s where s is the name of the code file" << endl;
#ifdef EXTRA_CHANNELS
  cout << " -g n where n is the group size for grouped Gauss channel (default 16)" << endl;
#endif
  cout << " -h to display this help" << endl;
  cout << " -i n where n is the maximum number of iterations per vector of the decoding algorithm; default is 100" << endl;
  cout << " -l n where n is the log level, from 1 to 3 included. default 1." << endl;
  cout << " -m n where, if k vectors are decoded in parallel by the GPU, n*k vectors are decoded in each run; default is 4" << endl;
  cout << " -n f where f is the noise level of the simulated channel" << endl;
  cout << " -p n where n is the log2 of the maximum number of vectors decoded in parallel by the GPU; default is 5" << endl;
  cout << " -r n where n is the number of decoding runs; default is 1" << endl;
  cout << " -s n where n is the first vector sequence index (seed for rngs), in order to reproduce a test" << endl;
  cout << " Option parameters are either i(n)tegers, (f)loating-point values or (s)trings" << endl;
}
