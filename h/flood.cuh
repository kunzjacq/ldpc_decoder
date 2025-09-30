#pragma once

#include "common.h"

#include <cstdint>

#include <cuda_runtime_api.h>

using uint = uint32_t;

/*
Computes llrs from channel values for a bsc channel.
*/
__global__ void llr_bsc(transfer_llr_t *p_initial_llrs, const llr_t p_noise_factor, const uint p_log_num_vecs,
                        const long p_vec_input_bitsize, const uint p_log_threads);

/*
Computes llrs from channel values for a channel with awgn noise.
*/
__global__ void llr_biawgn(transfer_llr_t *p_initial_llrs, const llr_t p_noise_factor, const uint p_log_num_vecs,
                           const long p_vec_input_bitsize, const uint p_log_threads);

/* 
Backward pass of belief propagation algorithm:
updates llr values of all edges using output bits.
*/
__global__ void flood_backward(uint *p_syndrome, llr_t *p_edge_buffer, uint *p_out_bit_to_edge,
                               const uint p_syndrome_uint32_sz, const uint p_log_num_vecs,
                               const long p_vec_output_bitsize, const uint p_log_threads_per_vec);


/* 
Forward pass of belief propagation algorithm:
updates llr values of input bits from edge buffer and re-propagates this information on all edges. 
*/
__global__ void flood_forward(llr_t *p_edge_buffer, llr_t *p_initial_llrs, uint *p_in_to_out_edge,
                              uint *p_in_bit_to_edge, const uint p_log_num_vecs,
                              const long p_vec_input_bitsize, const uint p_log_threads);

/* 
Forward pass of belief propagation algorithm:
updates llr values of input bits and re-propagates this information on all edges
variant that also updates the final bits, and that can be chained with check_parity. 
*/

__global__ void flood_forward_w_final_bits(llr_t *p_edge_buffer, llr_t *p_initial_llrs, uint *p_in_to_out_edge,
                                           uint *p_in_bit_to_edge, char *p_final_bits, const uint p_log_num_vecs,
                                           const long p_vec_input_bitsize, const uint p_log_threads);

/*
Computes parity bits for all vectors from p_final_bits, and outputs in p_parities_violated
a boolean for each input vector indicating whether there is at least one parity equation violated.
*/
__global__ void check_parity(uint *p_syndrome, uint *p_out_bit_to_edge, uint *p_out_edge_to_in_bit, char *p_final_bits,
                             char *p_parities_violated, const uint p_syndrome_uint32_sz, const uint p_log_num_vecs,
                             const long p_vec_output_bitsize, const uint p_log_threads_per_vec);

/*
Performs a series of transpositions of the vectors in the internal state. Updates edge buffer, initial_llrs, current_llrs, syndrome and final_bits.
*/
__global__ void flood_permute_vecs(llr_t *p_edge_buffer, llr_t *p_initial_llrs, char *p_final_bits, uint *p_syndrome,
                                   uint *p_in_bit_to_edge, uint *p_vec_origin, uint *p_vec_dest,
                                   const uint p_syndrome_uint32_sz, const uint p_num_transp, const uint p_log_num_transp, 
                                   // 1 << p_log_num_transp >=  p_num_transp (and p_log_num_transp is the smallest possible)
                                   const uint p_log_num_vecs, const long p_vec_input_bitsize, const uint p_log_threads);

/*
Computes the packed output of decoding, based on the unpacked one.
i.e. it builds p_final_bits_packed from p_final_bits.
 */                                           
__global__ void deinterlace_output(char *p_final_bits, uint *p_final_bits_packed, const uint p_log_num_vecs,
                                   const long p_vec_input_bitsize, const uint p_log_threads_per_vec);

/*
reads new llrs in p_new_initial_llrs and writes them at the appropriate place in p_initial_llrs 
(i.e. properly interlaced based on the total number of vectors). Also populates the corresponding values in 
edge_buffer with phi(llrs), and syndrome values in p_syndrome (from p_new_syndrome).
Is called once for every bit set to 1 in the number of new vectors to transfer with the appropriate p_offset.
This is marginally more efficient than having one loop over new vectors as the number of threads per block can
adapt to the number of vectors to process which is a power of two.
*/
__global__ void flood_refill(llr_t *p_edge_buffer, llr_t *p_initial_llrs, transfer_llr_t *p_new_initial_llrs, uint *p_syndrome,
                             uint *p_new_syndrome, uint *p_in_to_out_edge, uint *p_in_bit_to_edge,
                             const uint p_syndrome_uint32_sz, const uint p_vec_offset, const uint p_num_new_vecs, 
                             const uint p_log_new_num_vecs, const long p_vec_input_bitsize,
                             const uint p_log_num_vecs, const uint p_log_threads);
