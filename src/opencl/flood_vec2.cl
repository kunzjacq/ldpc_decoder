__constant uint float_vec_log_size = 1;

/* to change vector size from 2 to 4:
replace float2 by float4, int2 by int4, char2 by char4
change float_vec_log_size to 2 
in host code, change ldpc_decoder_gpu::log_simd_parallel_factor accordingly
there are some places where these vectors are manipulated coordinate-by-coordinate for the lack of a better implementation:
these must be modified by hand. Correct compile errors and search for .x, .y to complete with cases .z, .w
The impact performance is measured to be negligible on a NV 1080 Ti
*/

/* !! behavior of signbit and logical arithmetic with vector types

    component of signbit(x) 
    * on vector types:
    -1 if negative bit is set (i.e. x < 0 or x = -0), 0 otherwise
    * on scalar types:
    1 if negative bit is set, 0 otherwise

    ==> with vector types, if a positive number encodes a 1 and a negative number encodes a 0
    1 + signbit(x) is the bit associated to x
    BUT with scalar types, one should use 
    1 - signbit(x)
*/

/*
thread ids : old formula used was
 get_group_id(0) * get_local_size(0) + get_local_id(0)

now we use 

 get_global_id(0)

we assume that the LSBs of this value represent the local thread id, as before. 
This is important as consecutive threads manage consecutive vectors, hence benefit of synchronized memory accesses
such as coalesced reads on NVIDIA hardware. 

In any case the chance of formula mentioned above did not change the observed performance.
*/


/* for x > 0, phi(x) = -log(tanh(x/2)) > 0 where log is the natural logarithm.
   it is a self-inverse function because e -> (1-e)/(1+e) = 2/(1+e) - 1 is self-inverse. 
   indeed 2/(1 + 2/(1 + e) - 1) - 1 = 1 + e - 1 = e.
   for arbitrary x, phi(x) = sgn(x) phi(|x|) whith sgn(x) = x/|x| for x != 0, 1 for x = 0.

    with floating-point values, the computation of log(tanh(x)) has poor precision when x is 
    large because tanh(x) is very close to 1. Hence for x > c_phi_taylor_limit, 
    a 1st order taylor expansion in e = exp(-x) is used instead:
    1 / tanh(x/2) = (1+e)/(1-e) = (1+e) * (1+e+O(e^2)) = 1 + 2*e + O(e^2) 
    hence -log(tanh(x/2)) = 2*e + O(e^2)

    one could also compute phi(x) as 
    logp1(u) = log(1 + u) with u = 2e/(1-e) = 2/(1/e - 1) = 2 / (exp(x) - 1)
    indeed,
    phi(x) = -log(tanh(x/2)) = log((1+e)/(1-e)) = log(1+(1+e-(1-e))/(1-e)) = log(1 + 2e/(1-e))
    curiously, this is slightly slower than the formulas above on intel OpenCL (i7-8550u).
    on nvidia hardware, changing the formula does not make any difference w.r.t performance.
   */

/*
  NVIDIA BUG 
  triggered by both flood_refill and flood_refill_vec, second version, 
  under linux or windows (observed with nvidia driver 418.56 or 431.60), 
  if thread_id is declared as a uint, the expression  
   thread_id * p_vec_input_bitsize 
  in the computation of work boundaries bit_min_in and bit_min_out is not properly promoted to a long.
  Because of this, when p_log_new_num_vecs = 1, this overflows and hence the kernels fails to work in that case
*/

  __constant float c_threshold = 15.;
  __constant float c_pre_threshold = 0.000000225;//2.f * exp(-(c_threshold + 1.f));
  // i.e phi(c_threshold + 1) = c_pre_threshold (phi being approximated by its taylor expansion)
  // i.e. ph(c_pre_threshold) = c_threshold + 1 (since phi is self-inverse)
  __constant float c_phi_taylor_limit = 5.f;

inline float2 phi_abs_vec(float pre_threshold, float2 x)
{
#if 1

#if 0
  return  select(select(log(-(exp(-x) + 1.f)/expm1(-x)), 2.f * exp(-x) , x > c_phi_taylor_limit), c_threshold, x < pre_threshold);
  //return  select(select(log(1.f + 2.f/expm1(x)), 2.f * exp(-x) , x > c_phi_taylor_limit), c_threshold, x < pre_threshold);
#else
  float2 xm = fmax(x, pre_threshold);
  return  select(log(-(exp(-xm) + 1.f)/expm1(-xm)), 2.f * exp(-xm) , xm > c_phi_taylor_limit);
  //return  select(log(exp(-xm) + 1.f)-log(-expm1(-xm)), 2.f * exp(-xm) , xm > c_phi_taylor_limit);
  //return  select(log(1.f + 2.f/expm1(xm)), 2.f * exp(-xm) , xm > c_phi_taylor_limit);
#endif
#else
  float2 xm = fmax(x, pre_threshold);
  return  log1p(2.f/expm1(xm));
#endif
}

inline float2 phi_vec(float pre_threshold, float2 x)
{
  return copysign(phi_abs_vec(pre_threshold, fabs(x)), x);
}

inline float phi_abs(float pre_threshold, float x)
{
#if 0
  if(x < pre_threshold) return c_threshold;
  float xm = x;
#else
  float xm = fmax(x, pre_threshold);
#endif

#if 1
  float e = exp(-xm);
  return  select(log(-(e + 1.f)/expm1(-xm)), 2.f * e , xm > c_phi_taylor_limit);
#else
  return  log1p(2.f/expm1(xm));
#endif
}

inline float phi(float pre_threshold, float x)
{
  return copysign(phi_abs(pre_threshold, fabs(x)), x);
}

__kernel void llr_bsc(
  __global float2* p_initial_llrs,
  const float p_noise_factor,
  const uint p_log_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_threads
)
{
  const uint id = get_global_id(0);
  const uint log_vec_gps = p_log_num_vecs - float_vec_log_size;
  const uint num_vec_gps = 1 << log_vec_gps;
  const uint vec_gp_id = id & (num_vec_gps - 1);
  const uint thread_id = id >> log_vec_gps;
  const long bit_min = ( thread_id      * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  const long bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  for (
    long in_bit = bit_min;
         in_bit < bit_max;
         in_bit++)
  {
    long idx = vec_gp_id + num_vec_gps * in_bit;
    p_initial_llrs[idx] = copysign(p_noise_factor, p_initial_llrs[idx]);
  }
}

__kernel void llr_biawgn(
  __global float2* p_initial_llrs,
  const float p_noise_factor,
  const uint p_log_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_threads
)
{
  const uint id = get_global_id(0);
  const uint log_vec_gps = p_log_num_vecs - float_vec_log_size;
  const uint num_vec_gps = 1 << log_vec_gps;
  const uint vec_gp_id = id & (num_vec_gps - 1);
  const uint thread_id = id >> log_vec_gps;
  const long bit_min = ( thread_id      * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  const long bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> (p_log_threads - log_vec_gps);
  for (
    long in_bit = bit_min;
         in_bit < bit_max;
         in_bit++)
  {
    long idx = vec_gp_id + num_vec_gps * in_bit;
    p_initial_llrs[idx] *= p_noise_factor;
  }
}

/* updates llr values of all edges using output bits */
__kernel void flood_backward(
  __global int2* p_syndrome,
  __global float2* p_edge_buffer,
  __global uint* p_out_bit_to_edge,
  const float p_threshold,
  const uint p_syndrome_uint32_sz,
  const uint p_log_num_vecs,
  const long p_vec_output_bitsize,
  const uint p_log_threads_per_vec
)
{
  const float2 threshold = p_threshold;
  const float2 over_threshold = threshold + 1.f;
  const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
  const uint id = get_global_id(0);
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint num_vec_gps = 1 << (p_log_num_vecs - float_vec_log_size);
  const uint vec_gp_id = id & (num_vec_gps - 1);
  const uint vec_id = vec_gp_id << float_vec_log_size;
  const ulong thread_id = id >> (p_log_num_vecs - float_vec_log_size);
  const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  uint a = p_out_bit_to_edge[out_bit_gp_min << 5];
  uint b;
  for (
    uint out_bit_gp = out_bit_gp_min;
         out_bit_gp < out_bit_gp_max; 
         out_bit_gp++)
  {
    int2 s = p_syndrome[vec_gp_id + out_bit_gp * num_vec_gps];
    for (uint i = 0; i < 32; i++)
    {
      uint out_bit = (out_bit_gp << 5) + i;
      if(out_bit < p_vec_output_bitsize)
      {
        int2 sgn = -(s&1);
        s >>= 1;
        float2 ext_llr = 0.f;
        b =  p_out_bit_to_edge[out_bit + 1];
        for (uint  out_edge = a; out_edge < b; out_edge++)
        {
          const float2 edge_llr = p_edge_buffer[vec_gp_id + num_vec_gps * out_edge];
          ext_llr += fabs(edge_llr);
          sgn ^= (signbit(edge_llr) == 0); // positive llrs correspond to bits equal to 1, which we xor
        }
        for (uint  out_edge = a; out_edge < b; out_edge++)
        {
          uint idx = vec_gp_id + num_vec_gps * out_edge;
          const float2 edge_llr = p_edge_buffer[idx];
          const float2 phi_preimage = ext_llr - fabs(edge_llr);
          const float2 res = phi_abs_vec(pre_threshold, phi_preimage);
          const int2 is_neg = (signbit(edge_llr) == -1)^sgn; // or equivalently: bit most likely equal to 1 <=> positive LLR <=> (signbit(edge_llr) == 0)^sgn == 0
          p_edge_buffer[idx] = select(res, -res, is_neg);
        }
        a = b;
      }
    }
  }
}

/* flood_forward: forward pass of belief propagation algorithm.
   updates llr values of input bits and re-propagates this information on all edges */
__kernel void flood_forward(
  __global float2* p_edge_buffer,
  __global float2* p_initial_llrs,
  __global uint* p_in_to_out_edge,
  __global uint* p_in_bit_to_edge,
  const float p_threshold,
  const uint p_log_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_threads
)
{
// uncomment to try to avoid some computations of phi.
// does not really speed up computations and has a slight impact on the decoding result.
//#define USE_KNOWN_BITS
  const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
  const uint id = get_global_id(0);
  const uint log_vec_gps = p_log_num_vecs - float_vec_log_size;
  const uint num_vec_gps = 1 << log_vec_gps;
  const uint vec_gp_id = id & (num_vec_gps - 1);
  const ulong thread_id = id >> log_vec_gps;
  const uint log_threads_per_vec_gp = p_log_threads - log_vec_gps;
  const uint bit_min = ( thread_id      * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  const uint bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  uint a, b;
  a = p_in_bit_to_edge[bit_min];
  for (
    uint in_bit = bit_min;
         in_bit < bit_max;
         in_bit++)
  {
    const uint idx = vec_gp_id + num_vec_gps * in_bit;
    float2 val = p_initial_llrs[idx];
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint in_edge = a;  in_edge < b; in_edge++)
    {
      const uint idx = vec_gp_id + num_vec_gps * p_in_to_out_edge[in_edge];
      val += p_edge_buffer[idx];
    }
#ifdef USE_KNOWN_BITS
    const float2 zero = 0.f;
    // zero is phi_vec(infinity) : infinite certainty if bit is known, zero uncertainty
    const float2 sign = copysign(zero, val);
    const int2 is_known = fabs(val) > threshold;
#endif
    for (uint in_edge = a; in_edge < b; in_edge++)
    {
      const uint edge_idx = vec_gp_id + num_vec_gps * p_in_to_out_edge[in_edge];
      const float2 preimage = val - p_edge_buffer[edge_idx];
#ifdef USE_KNOWN_BITS
      p_edge_buffer[edge_idx] = select(phi_vec(pre_threshold, preimage), sign, is_known);
#else
      p_edge_buffer[edge_idx] = phi_vec(pre_threshold, preimage);
#endif
    }
    a = b;
  }
}

/* flood_forward_w_final_bits: forward pass of belief propagation algorithm.
   updates llr values of input bits and re-propagates this information on all edges
   variant that also updates the final bits, and that can be chained with check_parity */
__kernel void flood_forward_w_final_bits(
  __global float2* p_edge_buffer,
  __global float2* p_initial_llrs,
  __global uint* p_in_to_out_edge,
  __global uint* p_in_bit_to_edge,
  __global char2* p_final_bits,
  __global char2* p_parities_violated, 
  const float p_threshold,
  const uint p_log_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_threads
)
{
  const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
  const uint id = get_global_id(0);
  const uint log_vec_gps = p_log_num_vecs - float_vec_log_size;
  const uint num_vec_gps = 1 << log_vec_gps;
  const uint vec_gp_id = id & (num_vec_gps - 1);
  p_parities_violated[vec_gp_id] = 0; // to allow chaining with check_parity  
  const ulong thread_id = id >> log_vec_gps;
  const uint log_threads_per_vec_gp = p_log_threads - log_vec_gps;
  const uint bit_min = ( thread_id      * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  const uint bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_vec_gp;
  uint a, b;
  a = p_in_bit_to_edge[bit_min];
  for (
    uint in_bit = bit_min;
         in_bit < bit_max;
         in_bit++)
  {
    const uint idx = vec_gp_id + num_vec_gps * in_bit;
    float2 val = p_initial_llrs[idx];
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint in_edge = a;  in_edge < b; in_edge++)
    {
      const uint idx = vec_gp_id + num_vec_gps * p_in_to_out_edge[in_edge];
      val += p_edge_buffer[idx];
    }
    int2 bits = 1 + signbit(val); // LLR >= 0 <=> signbit = 0 <=> bit = 1, LLR < 0 <=> signbit = -1 <=> bit = 0
    p_final_bits[idx] = convert_char2(bits);

    for (uint in_edge = a; in_edge < b; in_edge++)
    {
      const uint edge_idx = vec_gp_id + num_vec_gps * p_in_to_out_edge[in_edge];
      const float2 preimage = val - p_edge_buffer[edge_idx];
      p_edge_buffer[edge_idx] = phi_vec(pre_threshold, preimage);

    }
    a = b;
  }
}

/*
  deinterlace_output computes the packed output of decoding, based on the unpacked one
  i.e. it builds p_final_bits_packed from p_final_bits.
 */
__kernel void deinterlace_output(
  __global char* p_final_bits,
  __global uint* p_final_bits_packed,
  __global char* p_parities_violated, 
  const uint p_log_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_threads_per_vec
)
{
  const uint id = get_global_id(0);
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint vec_id = id & (num_vecs - 1);
  const ulong thread_id = id >> p_log_num_vecs;
  const uint vec_uint32_size = p_vec_input_bitsize >> 5;
  const uint dec = p_log_threads_per_vec - float_vec_log_size + 5;
  const uint bit_min = ( thread_id      * p_vec_input_bitsize) >> dec;
  const uint bit_max = ((thread_id + 1) * p_vec_input_bitsize) >> dec;
  for (
    uint in_bit_gp = bit_min;
         in_bit_gp < bit_max;
         in_bit_gp++)
  {
    uint v = 0;
    for(uint i = 0; i < 32; i++)
    {
      uint in_bit = (in_bit_gp << 5) + i;
      v |= (uint)(p_final_bits[vec_id + num_vecs * in_bit]) << i;
    }
    p_final_bits_packed[in_bit_gp + vec_uint32_size * vec_id] = v;
  }
}

/*
check_parity: computes parity bits for all vectors from p_final_bits, and outputs in p_parities_violated
a boolean for each input vector indicating whether there is at least one parity equation violated. 
*/
__kernel void check_parity(
  __global int2* p_syndrome,
  __global uint* p_out_bit_to_edge,
  __global uint* p_out_edge_to_in_bit,
  __global char2* p_final_bits, 
  __global char2* p_parities_violated, 
  const uint p_syndrome_uint32_sz,
  const uint p_log_num_vecs,
  const long p_vec_output_bitsize,
  const uint p_log_threads_per_vec
)
{
  const uint id = get_global_id(0);
  const uint num_vec_gps = 1 << (p_log_num_vecs - float_vec_log_size);
  const uint vec_gp_id = id & (num_vec_gps - 1);
  const ulong thread_id = id >> (p_log_num_vecs - float_vec_log_size);
  const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> p_log_threads_per_vec;
  uint a, b;
  char2 parities = 0;
  a = p_out_bit_to_edge[out_bit_gp_min << 5];
  for (
    uint out_bit_gp = out_bit_gp_min;
         out_bit_gp < out_bit_gp_max; 
         out_bit_gp++)
  {
    int2 s = p_syndrome[vec_gp_id + out_bit_gp * num_vec_gps];
    for (uint i = 0; i < 32; i++)
    {
      uint out_bit = (out_bit_gp << 5) + i;
      if(out_bit < p_vec_output_bitsize)
      {
        char2 sgn = convert_char2(s&1);
        s >>= 1;
        b = p_out_bit_to_edge[out_bit + 1];
        for (uint  out_edge = a; out_edge < b; out_edge++)
        {
          const char2 buf = p_final_bits[vec_gp_id + num_vec_gps * p_out_edge_to_in_bit[out_edge]];
          sgn ^= buf;
        }
        a = b;
        parities |= sgn;
      }
    }
  }
  //cannot be written as
  //  p_parities_violated[vec_gp_id] |= parities
  // because there is a race between threads
  // atomic_or does not apply directly to a char2 and would probably slow down the code
  // whereas the writing below does not. 
  if (parities.x == 1 && p_parities_violated[vec_gp_id].x == 0) p_parities_violated[vec_gp_id].x = 1;
  if (parities.y == 1 && p_parities_violated[vec_gp_id].y == 0) p_parities_violated[vec_gp_id].y = 1;
}

/*
Performs a series of transpositions of the vectors in the internal state.
updates edge buffer, initial_llrs, current_llrs, syndrome and final_bits,
final_bits_packed is NOT updated; this can be done with flood_exit_deinterlace_scalar.
used to regroup finished vectors at the beginning of the arrays, in order to simplify 
the introduction of new vectors.
*/
__kernel void flood_permute_vecs(
  __global float* p_edge_buffer,
  __global float* p_initial_llrs,
  __global char* p_final_bits,
  __global uint* p_syndrome,
  __global uint* p_in_bit_to_edge,
  __global uint* p_vec_origin,
  __global uint* p_vec_dest,
  const float p_threshold,
  const uint p_syndrome_uint32_sz,
  const uint p_num_transp,
  const uint p_log_num_transp, // 1 << p_log_num_transp >=  p_num_transp (and p_log_num_transp is the smallest possible)
  const uint p_log_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_threads
)
{
  const uint id = get_global_id(0);
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint transp_mask = (1 << p_log_num_transp) - 1;
  const uint transp_idx = id & transp_mask;
  if(transp_idx < p_num_transp)
  {
    // destination vectors are going to be decoded further, but not the origin vectors
    // for the origin vectors, we only need the final bit values
    const uint vec_id_o = p_vec_origin[transp_idx];
    const uint vec_id_d = p_vec_dest[transp_idx];
    const ulong thread_id = id >> p_log_num_transp;
    const uint log_threads_per_transp = p_log_threads - p_log_num_transp;
    {
      const uint bit_min_in = ( thread_id      * p_vec_input_bitsize) >> log_threads_per_transp;
      const uint bit_max_in = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_transp;
      uint a, b;
      a = p_in_bit_to_edge[bit_min_in];
      for (
        uint in_bit = bit_min_in;
            in_bit < bit_max_in;
            in_bit++)
      {
        const uint idx_o = vec_id_o + num_vecs * in_bit;
        const uint idx_d = vec_id_d + num_vecs * in_bit;

        // do the full swap for p_final_bits
        char bit = p_final_bits[idx_o];
        p_final_bits[idx_o] = p_final_bits[idx_d];
        p_final_bits[idx_d] = bit;

        p_initial_llrs[idx_d] = p_initial_llrs[idx_o];
        b = p_in_bit_to_edge[in_bit + 1];
        for (uint in_edge = a;  in_edge < b; in_edge++)
        {
          const uint idx_o = vec_id_o + num_vecs * in_edge;
          const uint idx_d = vec_id_d + num_vecs * in_edge;
          p_edge_buffer[idx_d] = p_edge_buffer[idx_o];
        }
        a = b;
      }
    }
    {
      const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> log_threads_per_transp;
      const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_transp;
      for (
        uint out_bit_gp = out_bit_gp_min;
             out_bit_gp < out_bit_gp_max; 
             out_bit_gp++)
      {
        p_syndrome[vec_id_d + out_bit_gp * num_vecs] = p_syndrome[vec_id_o + out_bit_gp * num_vecs];
      }
    }
  }
}

/* puts new vectors in their final location. 
   this kernel is scalar as we wish to avoid interlacing issues and since a SIMD kernel would force us 
   to overwrite frames that have not finished decoding.
   Nevertheless, if p_num_new_vecs were a multiple of the number of vectors in a SIMD group,
   a SIMD kernel could work (see kernel variant below)
   */
#if 0
__kernel void flood_refill(
  __global float* p_edge_buffer,
  __global float* p_initial_llrs,
  __global float* p_new_initial_llrs,
  __global uint* p_syndrome,
  __global uint* p_new_syndrome,
  __global uint* p_in_to_out_edge,
  __global uint* p_in_bit_to_edge,
  const uint p_syndrome_uint32_sz,
  const float p_threshold,
  const uint p_vec_offset,
  const uint p_num_new_vecs,
  const uint p_log_new_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_num_vecs,
  const uint p_log_threads)
{
  const uint id = get_global_id(0);
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint num_new_vecs_mask = (1 << p_log_new_num_vecs) - 1;
  const ulong thread_id = id >> p_log_new_num_vecs;
  const uint vec_id = (id & num_new_vecs_mask) + p_vec_offset;
  const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
  const uint log_threads_per_new_vec = p_log_threads - p_log_new_num_vecs;

  if(vec_id < p_num_new_vecs)
  {
    const uint bit_min_in = ( thread_id      * p_vec_input_bitsize) >> log_threads_per_new_vec;
    const uint bit_max_in = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_new_vec;
    uint a, b;
    a = p_in_bit_to_edge[bit_min_in];
    for (
      uint in_bit = bit_min_in;
           in_bit < bit_max_in;
           in_bit++)
    {
      const uint idx       = vec_id + p_num_new_vecs * in_bit;
      const uint idx_prime = vec_id + (in_bit << p_log_num_vecs);
      const float llr =  p_new_initial_llrs[idx];
      p_initial_llrs[idx_prime] = llr;
      const float new_val = phi(pre_threshold, llr);
      b = p_in_bit_to_edge[in_bit + 1];
      for (uint  in_edge = a; in_edge < b; in_edge++)
      {
        const uint out_edge = p_in_to_out_edge[in_edge];
        const uint e = vec_id + (out_edge << p_log_num_vecs);
        p_edge_buffer[e] = new_val;
      }
      a = b;
    }

    const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> log_threads_per_new_vec;
    const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_new_vec;
    for (
      uint out_bit_gp = out_bit_gp_min;
           out_bit_gp < out_bit_gp_max; 
           out_bit_gp++)
    {
      p_syndrome[vec_id + out_bit_gp * num_vecs] = p_new_syndrome[vec_id * p_syndrome_uint32_sz + out_bit_gp];
    }
  }
}

/* vector version of the above, that works if p_num_new_vecs is a multiple of the SIMD size */
__kernel void flood_refill_vec(
  __global float2* p_edge_buffer,
  __global float2* p_initial_llrs,
  __global float2* p_new_initial_llrs,
  __global uint2* p_syndrome,
  __global uint* p_new_syndrome,
  __global uint* p_in_to_out_edge,
  __global uint* p_in_bit_to_edge,
  const uint p_syndrome_uint32_sz,
  const float p_threshold,
  const uint p_vec_offset,
  const uint p_num_new_vecs,
  const uint p_log_num_new_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_num_vecs,
  const uint p_log_threads)
{
  const uint id = get_global_id(0);
  const uint log_vec_gps = p_log_num_vecs - float_vec_log_size;
  const uint log_new_vec_gps = p_log_num_new_vecs - float_vec_log_size;
  const uint num_new_vec_gp_mask = (1 << log_new_vec_gps) - 1;
  const uint thread_id = id >> log_new_vec_gps; //thread id for given vector or vector group
  const uint vec_gp_id = (id & num_new_vec_gp_mask) + (p_vec_offset >> float_vec_log_size);
  const uint num_new_vec_gps = p_num_new_vecs >> float_vec_log_size;
  if(vec_gp_id < num_new_vec_gps)
  {
    const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
    const uint log_threads_per_new_vec_gp = p_log_threads - log_new_vec_gps;
    const uint bit_min_in = (( thread_id      * p_vec_input_bitsize) >> log_threads_per_new_vec_gp) & (~0x1f);
    const uint bit_max_in = (((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_new_vec_gp) & (~0x1f);
    uint a, b;
    a = p_in_bit_to_edge[bit_min_in];
    for (
      uint in_bit = bit_min_in;
            in_bit < bit_max_in;
            in_bit ++)
    {
      const uint idx       = vec_gp_id + in_bit * num_new_vec_gps;
      const uint idx_prime = vec_gp_id + (in_bit << log_vec_gps);
      const float2 llr =  p_new_initial_llrs[idx];
      p_initial_llrs[idx_prime] = llr;
      const float2 new_val = phi_vec(pre_threshold, llr);
      b = p_in_bit_to_edge[in_bit + 1];
      for (uint in_edge = a; in_edge < b; in_edge++)
      {
        const uint out_edge = p_in_to_out_edge[in_edge];
        const uint e = vec_gp_id + (out_edge << (p_log_num_vecs - float_vec_log_size));
        p_edge_buffer[e] = new_val;
      }
      a = b;
    }

    const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> log_threads_per_new_vec_gp;
    const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_new_vec_gp;
    for (
      uint out_bit_gp = out_bit_gp_min;
            out_bit_gp < out_bit_gp_max; 
            out_bit_gp++)
    {
      uint offset = (1 << float_vec_log_size) * vec_gp_id * p_syndrome_uint32_sz + out_bit_gp;
      uint2 s = (uint2) (
        p_new_syndrome[offset], 
        p_new_syndrome[offset + 1 * p_syndrome_uint32_sz]
      );
      p_syndrome[vec_gp_id + (out_bit_gp << log_vec_gps)] = s;
    }
  }
}


#else
// version of flood_refill that works with a number of input vectors that is a power of 2.
// is called once for every bit set to 1 in the number of new vectors to transfer.
// should be faster in theory because no thread is idle, contrary to the version above.
// in practice, the performance is the same.

__kernel void flood_refill(
  __global float* p_edge_buffer,
  __global float* p_initial_llrs,
  __global float* p_new_initial_llrs,
  __global uint* p_syndrome,
  __global uint* p_new_syndrome,
  __global uint* p_in_to_out_edge,
  __global uint* p_in_bit_to_edge,
  const uint p_syndrome_uint32_sz,
  const float p_threshold,
  const uint p_vec_offset,
  const uint p_num_new_vecs,
  const uint p_log_new_num_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_num_vecs,
  const uint p_log_threads)
{
  const uint id = get_global_id(0);
  const uint num_vecs = 1 << p_log_num_vecs;
  const uint num_new_vecs_mask = (1 << p_log_new_num_vecs) - 1;
  const ulong thread_id = id >> p_log_new_num_vecs; //thread id for given vector or vector group
  // uint should be large enough above but see NVIDIA_BUG at the top of this file
  const uint vec_id = (id & num_new_vecs_mask) + p_vec_offset;
  const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
  const uint log_threads_per_new_vec = p_log_threads - p_log_new_num_vecs;
  const uint bit_min_in = ( thread_id      * p_vec_input_bitsize) >> log_threads_per_new_vec; // (*)
  const uint bit_max_in = ((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_new_vec;
  uint a = p_in_bit_to_edge[bit_min_in];
  uint b;
  for (
    uint in_bit = bit_min_in;
          in_bit < bit_max_in;
          in_bit++)
  {
    const uint idx       = vec_id + p_num_new_vecs * in_bit;
    const uint idx_prime = vec_id + (in_bit << p_log_num_vecs);
    const float llr =  p_new_initial_llrs[idx];
    p_initial_llrs[idx_prime] = llr;
    const float new_val = phi(pre_threshold, llr);
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint  in_edge = a; in_edge < b; in_edge++)
    {
      const uint out_edge = p_in_to_out_edge[in_edge];
      const uint e = vec_id + (out_edge << p_log_num_vecs);
      p_edge_buffer[e] = new_val;
    }
    a = b;
  }

  const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> log_threads_per_new_vec;
  const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_new_vec;
  for (
    uint out_bit_gp = out_bit_gp_min;
          out_bit_gp < out_bit_gp_max; 
          out_bit_gp++)
  {
    p_syndrome[vec_id + out_bit_gp * num_vecs] = p_new_syndrome[vec_id * p_syndrome_uint32_sz + out_bit_gp];
  }
}

/* vector version of the above, that works if p_num_new_vecs is a multiple of the SIMD size */
__kernel void flood_refill_vec(
  __global float2* p_edge_buffer,
  __global float2* p_initial_llrs,
  __global float2* p_new_initial_llrs,
  __global uint2* p_syndrome,
  __global uint* p_new_syndrome,
  __global uint* p_in_to_out_edge,
  __global uint* p_in_bit_to_edge,
  const uint p_syndrome_uint32_sz,
  const float p_threshold,
  const uint p_vec_offset,
  const uint p_num_new_vecs,
  const uint p_log_num_new_vecs,
  const long p_vec_input_bitsize,
  const uint p_log_num_vecs,
  const uint p_log_threads)
{
  const uint id = get_global_id(0);
  const uint log_vec_gps = p_log_num_vecs - float_vec_log_size;
  const uint log_new_vec_gps = p_log_num_new_vecs - float_vec_log_size;
  const uint num_new_vec_gp_mask = (1 << log_new_vec_gps) - 1;
  const ulong thread_id = id >> log_new_vec_gps; //thread id for given vector or vector group
  // uint should be large enough above but see NVIDIA_BUG at the top of this file
  const uint vec_gp_id = (id & num_new_vec_gp_mask) + (p_vec_offset >> float_vec_log_size);
  const float pre_threshold = 2.f * exp(-(p_threshold + 1.f));
  const uint log_threads_per_new_vec_gp = p_log_threads - log_new_vec_gps;

  const uint bit_min_in = (( thread_id      * p_vec_input_bitsize) >> log_threads_per_new_vec_gp) & (~0x1f);
  const uint bit_max_in = (((thread_id + 1) * p_vec_input_bitsize) >> log_threads_per_new_vec_gp) & (~0x1f);
  uint a, b;
  a = p_in_bit_to_edge[bit_min_in];
  for (
    uint in_bit = bit_min_in;
          in_bit < bit_max_in;
          in_bit ++)
  {
    const uint idx       = vec_gp_id + in_bit * (p_num_new_vecs >> float_vec_log_size);
    const uint idx_prime = vec_gp_id + (in_bit << log_vec_gps);
    const float2 llr =  p_new_initial_llrs[idx];
    p_initial_llrs[idx_prime] = llr;
    const float2 new_val = phi_vec(pre_threshold, llr);
    b = p_in_bit_to_edge[in_bit + 1];
    for (uint in_edge = a; in_edge < b; in_edge++)
    {
      const uint out_edge = p_in_to_out_edge[in_edge];
      const uint e = vec_gp_id + (out_edge << (p_log_num_vecs - float_vec_log_size));
      p_edge_buffer[e] = new_val;
    }
    a = b;
  }

  const uint out_bit_gp_min = ( thread_id      * p_syndrome_uint32_sz) >> log_threads_per_new_vec_gp;
  const uint out_bit_gp_max = ((thread_id + 1) * p_syndrome_uint32_sz) >> log_threads_per_new_vec_gp;
  for (
    uint out_bit_gp = out_bit_gp_min;
          out_bit_gp < out_bit_gp_max; 
          out_bit_gp++)
  {
    uint2 s = (uint2) (p_new_syndrome[2 * vec_gp_id * p_syndrome_uint32_sz + out_bit_gp], p_new_syndrome[(2 * vec_gp_id + 1) * p_syndrome_uint32_sz + out_bit_gp]);
    p_syndrome[vec_gp_id + (out_bit_gp << log_vec_gps)] = s;
  }
}
#endif
