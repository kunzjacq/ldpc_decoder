#include "../../h/flood.cuh"

#ifdef USE_FLOAT16_COMPUTE
#define abs __habs
#define signbit(x) (*((uint16_t*)(&(x)))>>15)
#define copysign_(x,y,res) uint16_t tmp=(((*((uint32_t*)(&(x)))&0x7FFF)|(*((uint16_t*)(&(y)))&0x8000))); res = reinterpret_cast<llr_t&>(tmp)
//#define signbit(x) (((x)<static_cast<llr_t>(0))?1:0) //does not work!!!
//#define copysign_(x,y,res) res = ((y)>static_cast<llr_t>(0)) ? abs(x) : -abs(x);
#define fmax(x,y) (x)>(y)?(x):(y)
#else
#define abs fabs
#define signbit(x) (*((uint32_t*)(&(x)))>>31)
#define copysign_(x,y,res) uint32_t tmp=(((*((uint32_t*)(&(x)))&0x7FFFFFFF)|(*((uint32_t*)(&(y)))&0x80000000))); res = reinterpret_cast<llr_t&>(tmp)
constexpr float pre_threshold = 1.e-5;
#endif

// phi_abs is the function x -> -log(tanh(x/2)) on R_+ 
// phi is x -> phi_abs(|x|) . sign(x)

#ifdef USE_FLOAT16_COMPUTE
inline __device__ __half phi_abs(__half x) {
  // as half, 3 is represented by 0x4200, 4<=n<=8 is represented by 0x4n00, 9 by 0x4880
  constexpr __half c_phi_taylor_limit(__half_raw{0x4500}); // =5; 8 is more accurate but a bit slower
  constexpr __half pre_threshold_h(__half_raw {0x3f});
  constexpr __half two(__half_raw {0x4000});
  constexpr __half half_one(__half_raw {0x3800});
  __half xm = fmax(x, pre_threshold_h);
  return (xm > c_phi_taylor_limit) ? two * hexp(-xm) : -hlog(htanh(xm*half_one));
}
#else
inline __device__ llr_t phi_abs(llr_t x) {
  const float c_phi_taylor_limit = 5.f;
  float xm = fmax(x, pre_threshold);
  float e = exp(-xm);
  return xm > c_phi_taylor_limit ? 2.f * e : log(-(e + 1.f) / expm1(-xm)); // log((1+e)/(1-e))
  //return xm > c_phi_taylor_limit ? 2.f * e : -log(tanh(xm/2)); 
}
#endif

inline __device__ llr_t phi(llr_t x) { 
  llr_t pa = phi_abs(abs(x));
  float res;
  copysign_(pa, x, res); 
  return res;
}

__global__ void llr_bsc(transfer_llr_t *p_initial_llrs, const llr_t p_noise_factor, const uint p_log_num_vecs,
                        const long p_vec_input_bitsize, const uint p_log_threads) {
  const ulong id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint log_vec_gps = p_log_num_vecs;
  const uint num_vec_gps = 1 << log_vec_gps;
  const uint vec_id = id & (num_vec_gps - 1);
  const ulong thread_id = id >> log_vec_gps;
  const long bit_min = (thread_id * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  const long bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  for (long in_bit = bit_min; in_bit < bit_max; in_bit++) {
    long idx = vec_id + num_vec_gps * in_bit;
    copysign_(p_noise_factor, p_initial_llrs[idx], p_initial_llrs[idx]);
  }
}

__global__ void llr_biawgn(transfer_llr_t *p_initial_llrs, const llr_t p_noise_factor, const uint p_log_num_vecs,
                           const long p_vec_input_bitsize, const uint p_log_threads) {
  const ulong id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint log_vec_gps = p_log_num_vecs;
  const uint num_vec_gps = 1 << log_vec_gps;
  const uint vec_id = id & (num_vec_gps - 1);
  const ulong thread_id = id >> log_vec_gps;
  const long bit_min = (thread_id * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  const long bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  for (long in_bit = bit_min; in_bit < bit_max; in_bit++) {
    long idx = vec_id + num_vec_gps * in_bit;
    p_initial_llrs[idx] *= p_noise_factor;
  }
}

__global__ void flood_backward(uint *p_syndrome, llr_t *p_edge_buffer, uint *p_out_bit_to_edge,
                               const uint p_syndrome_uint32_sz, const uint p_log_num_vecs,
                               const long p_vec_output_bitsize, const uint p_log_threads_per_vec) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_vecs = 1 << (p_log_num_vecs);
  const uint vec_id = id & (num_vecs - 1);
  const ulong thread_id = id >> (p_log_num_vecs);
  const uint out_word_min = (thread_id * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  const uint out_word_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  uint a = p_out_bit_to_edge[out_word_min << 5];
  uint b;
  for (uint out_word = out_word_min; out_word < out_word_max; out_word++) {
    uint syndrome_word = p_syndrome[vec_id + out_word * num_vecs];
    for (uint i = 0; i < 32; i++) {
      uint out_bit = (out_word << 5) + i;
      if (out_bit < p_vec_output_bitsize) {
        int syndrome_bit = syndrome_word & 1;
        syndrome_word >>= 1; // prepare syndrome_word for next value of i
        llr_t ext_llr = 0.f;
        b = p_out_bit_to_edge[out_bit + 1];
        for (uint out_edge = a; out_edge < b; out_edge++) {
          const llr_t edge_llr = p_edge_buffer[vec_id + num_vecs * out_edge];
          ext_llr += abs(edge_llr);
          syndrome_bit ^= (signbit(edge_llr) == 0); // positive llrs correspond to bits equal to 1, which we xor
        }
        for (uint out_edge = a; out_edge < b; out_edge++) {
          uint idx = vec_id + num_vecs * out_edge;
          const llr_t edge_llr = p_edge_buffer[idx];
          const llr_t phi_preimage = ext_llr - abs(edge_llr);
          const llr_t res = phi_abs(phi_preimage);
          const int is_neg =
              signbit(edge_llr) ^ syndrome_bit; // or equivalently: bit most likely equal to 1 <=> positive LLR <=> ((signbit(edge_llr) == 0)^sgn) == 0
          p_edge_buffer[idx] = is_neg ? -res : res;
        }
        a = b;
      }
    }
  }
}

__global__ void flood_forward(llr_t *p_edge_buffer, llr_t *p_initial_llrs, uint *p_in_to_out_edge,
                              uint *p_in_bit_to_edge, const uint p_log_num_vecs,
                              const long p_vec_input_bitsize, const uint p_log_threads) {
  // uncomment to try to avoid some computations of phi.
  // does not really speed up computations and has a slight impact on the decoding result.
  // #define USE_KNOWN_BITS
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint vec_id = id & (num_vecs - 1);
  const ulong thread_id = id >> p_log_num_vecs;
  const uint log_threads_per_vec_gp = p_log_threads - p_log_num_vecs;
  const uint bit_min = (thread_id * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  const uint bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  uint a, b;
  a = p_in_bit_to_edge[bit_min];
  for (uint in_bit = bit_min; in_bit < bit_max; in_bit++) {
    const uint idx = vec_id + num_vecs * in_bit;
    llr_t val = p_initial_llrs[idx];
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint in_edge = a; in_edge < b; in_edge++) {
      const uint idx = vec_id + num_vecs * p_in_to_out_edge[in_edge];
      val += p_edge_buffer[idx];
    }
#ifdef USE_KNOWN_BITS
    const llr_t zero = 0.f;
    // zero is phi(infinity) : infinite certainty if bit is known, zero uncertainty
    const llr_t sign = copysign(zero, val);
    const int is_known = abs(val) > threshold;
#endif
    for (uint in_edge = a; in_edge < b; in_edge++) {
      const uint edge_idx = vec_id + num_vecs * p_in_to_out_edge[in_edge];
      const llr_t preimage = val - p_edge_buffer[edge_idx];
#ifdef USE_KNOWN_BITS
      p_edge_buffer[edge_idx] = is_known ? sign : phi(preimage);
#else
      p_edge_buffer[edge_idx] = phi(preimage);
#endif
    }
    a = b;
  }
}

__global__ void flood_forward_w_final_bits(llr_t *p_edge_buffer, llr_t *p_initial_llrs, uint *p_in_to_out_edge,
                                           uint *p_in_bit_to_edge, char *p_final_bits, const uint p_log_num_vecs,
                                           const long p_vec_input_bitsize, const uint p_log_threads) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint vec_id = id & (num_vecs - 1);
  const ulong thread_id = id >> p_log_num_vecs;
  const uint log_threads_per_vec_gp = p_log_threads - p_log_num_vecs;
  const uint bit_min = (thread_id * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  const uint bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  uint a, b;
  a = p_in_bit_to_edge[bit_min];
  for (uint in_bit = bit_min; in_bit < bit_max; in_bit++) {
    const uint idx = vec_id + num_vecs * in_bit;
    llr_t val = p_initial_llrs[idx];
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint in_edge = a; in_edge < b; in_edge++) {
      const uint idx = vec_id + num_vecs * p_in_to_out_edge[in_edge];
      val += p_edge_buffer[idx];
    }
    //int bits = 1 - signbit(val); // LLR >= 0 <=> signbit = 0 <=> bit = 1, LLR < 0 <=> signbit = 1 <=> bit = 0
    p_final_bits[idx] = signbit(val) == 0;

    for (uint in_edge = a; in_edge < b; in_edge++) {
      const uint edge_idx = vec_id + num_vecs * p_in_to_out_edge[in_edge];
      const llr_t preimage = val - p_edge_buffer[edge_idx];
      p_edge_buffer[edge_idx] = phi(preimage);
    }
    a = b;
  }
}

__global__ void check_parity(uint *p_syndrome, uint *p_out_bit_to_edge, uint *p_out_edge_to_in_bit, char *p_final_bits,
                             char *p_parities_violated, const uint p_syndrome_uint32_sz, const uint p_log_num_vecs,
                             const long p_vec_output_bitsize, const uint p_log_threads_per_vec) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_vecs = 1 << (p_log_num_vecs);
  const uint vec_id = id & (num_vecs - 1);
  const ulong thread_id = id >> (p_log_num_vecs);
  const uint out_word_min = (thread_id * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  const uint out_word_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  uint a, b;
  char parities = 0;
  a = p_out_bit_to_edge[out_word_min << 5];
  for (uint out_word = out_word_min; out_word < out_word_max; out_word++) {
    uint s = p_syndrome[vec_id + out_word * num_vecs];
    for (uint i = 0; i < 32; i++) {
      uint out_bit = (out_word << 5) + i;
      if (out_bit < p_vec_output_bitsize) {
        char sgn = s & 1;
        s >>= 1; //shift s for next read
        b = p_out_bit_to_edge[out_bit + 1];
        for (uint out_edge = a; out_edge < b; out_edge++) {
          const char bit = p_final_bits[vec_id + num_vecs * p_out_edge_to_in_bit[out_edge]];
          sgn ^= bit;
        }
        a = b;
        parities |= sgn;
      }
    }
  }
  if (parities == 1 && p_parities_violated[vec_id] == 0) {
    p_parities_violated[vec_id] = 1;
  }
}

__global__ void flood_permute_vecs(llr_t *p_edge_buffer, llr_t *p_initial_llrs, char *p_final_bits, uint *p_syndrome,
                                   uint *p_in_bit_to_edge, uint *p_vec_origin, uint *p_vec_dest, const uint p_syndrome_uint32_sz,
                                   const uint p_num_transp,
                                   const uint p_log_num_transp, // 1 << p_log_num_transp >=  p_num_transp (and
                                                                // p_log_num_transp is the smallest possible)
                                   const uint p_log_num_vecs, const long p_vec_input_bitsize,
                                   const uint p_log_threads) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint transp_mask = (1 << p_log_num_transp) - 1;
  const uint transp_idx = id & transp_mask;
  if (transp_idx < p_num_transp) {
    // destination vectors are going to be decoded further, but not the origin vectors
    // for the origin vectors, we only need the final bit values
    const uint vec_id_o = p_vec_origin[transp_idx];
    const uint vec_id_d = p_vec_dest[transp_idx];
    const ulong thread_id = id >> p_log_num_transp;
    const uint log_threads_per_transp = p_log_threads - p_log_num_transp;
    {
      const uint bit_min_in = (thread_id * p_vec_input_bitsize) >> log_threads_per_transp;
      const uint bit_max_in = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_transp;
      uint a, b;
      a = p_in_bit_to_edge[bit_min_in];
      for (uint in_bit = bit_min_in; in_bit < bit_max_in; in_bit++) {
        const uint idx_o = vec_id_o + num_vecs * in_bit;
        const uint idx_d = vec_id_d + num_vecs * in_bit;

        // do the full swap for p_final_bits
        char bit = p_final_bits[idx_o];
        p_final_bits[idx_o] = p_final_bits[idx_d];
        p_final_bits[idx_d] = bit;

        p_initial_llrs[idx_d] = p_initial_llrs[idx_o];
        b = p_in_bit_to_edge[in_bit + 1];
        for (uint in_edge = a; in_edge < b; in_edge++) {
          const uint idx_o = vec_id_o + num_vecs * in_edge;
          const uint idx_d = vec_id_d + num_vecs * in_edge;
          p_edge_buffer[idx_d] = p_edge_buffer[idx_o];
        }
        a = b;
      }
    }
    {
      const uint out_word_min = (thread_id * p_syndrome_uint32_sz) >> log_threads_per_transp;
      const uint out_word_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_transp;
      for (uint out_word = out_word_min; out_word < out_word_max; out_word++) {
        p_syndrome[vec_id_d + out_word * num_vecs] = p_syndrome[vec_id_o + out_word * num_vecs];
      }
    }
  }
}

__global__ void deinterlace_output(char *p_final_bits, uint *p_final_bits_packed, const uint p_log_num_vecs,
                                   const long p_vec_input_bitsize, const uint p_log_threads_per_vec) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint vec_id = id & (num_vecs - 1);
  const ulong thread_id = id >> p_log_num_vecs;
  const uint vec_uint32_size = p_vec_input_bitsize >> 5;
  const uint dec = p_log_threads_per_vec + 5;
  const uint word_min = (thread_id * p_vec_input_bitsize) >> dec;
  const uint word_max = ((thread_id + 1) * p_vec_input_bitsize) >> dec;
  for (uint in_word_idx = word_min; in_word_idx < word_max; in_word_idx++) {
    uint v = 0;
    for (uint i = 0; i < 32; i++) {
      uint in_bit_idx = (in_word_idx << 5) + i;
      v |= ((uint)(p_final_bits[vec_id + num_vecs * in_bit_idx])) << i;
    }
    p_final_bits_packed[in_word_idx + vec_uint32_size * vec_id] = v;
  }
}

__global__ void flood_refill(llr_t *p_edge_buffer, llr_t *p_initial_llrs, llr_t *p_new_initial_llrs, uint *p_syndrome,
                             uint *p_new_syndrome, uint *p_in_to_out_edge, uint *p_in_bit_to_edge,
                             const uint p_syndrome_uint32_sz, const uint p_vec_offset, const uint p_num_new_vecs,
                             const uint p_log_num_new_vecs, const long p_vec_input_bitsize, const uint p_log_num_vecs,
                             const uint p_log_threads) {
  const uint id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint num_new_vec_mask = (1 << p_log_num_new_vecs) - 1;
  const ulong thread_id = id >> p_log_num_new_vecs; // thread id for given vector or vector group
  const uint new_vec_id = (id & num_new_vec_mask) + p_vec_offset;
  const uint log_threads_per_new_vec = p_log_threads - p_log_num_new_vecs;
  const uint bit_min_in = ((thread_id * p_vec_input_bitsize) >> log_threads_per_new_vec) & (~0x1f);
  const uint bit_max_in = (((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_new_vec) & (~0x1f);
  uint a, b;
  a = p_in_bit_to_edge[bit_min_in];
  for (uint in_bit = bit_min_in; in_bit < bit_max_in; in_bit++) {
    const uint idx = new_vec_id + in_bit * p_num_new_vecs;
    const uint idx_prime = new_vec_id + (in_bit << p_log_num_vecs);
    const llr_t llr = p_new_initial_llrs[idx];
    p_initial_llrs[idx_prime] = llr;
    const llr_t new_val = phi(llr);
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint in_edge = a; in_edge < b; in_edge++) {
      const uint out_edge = p_in_to_out_edge[in_edge];
      p_edge_buffer[new_vec_id + (out_edge << p_log_num_vecs)] = new_val;
    }
    a = b;
  }
  const uint out_word_min = (thread_id * p_syndrome_uint32_sz) >> log_threads_per_new_vec;
  const uint out_word_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_new_vec;
  for (uint out_word = out_word_min; out_word < out_word_max; out_word++) {
    p_syndrome[new_vec_id + (out_word << p_log_num_vecs)] = p_new_syndrome[new_vec_id * p_syndrome_uint32_sz + out_word];
  }
}
