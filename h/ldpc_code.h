#pragma once

#include "common.h"

#include <vector>
#include <string>

class bool_vec;

class ldpc_code
{
private:
  uint32_t* m_in_bit_to_edge;
  uint32_t* m_in_edge_to_bit;

  uint32_t* m_out_bit_to_edge;
  uint32_t* m_out_edge_to_bit;

  uint32_t* m_edge_in_to_out;
  uint32_t* m_edge_out_to_in;

  int64_t m_n_inputs;
  int64_t m_n_outputs;
  uint32_t m_n_edges;
  int64_t m_n_erased_variables;
  int64_t m_n_erased_check_bits;

  int32_t m_max_degree_in;
  int32_t m_max_degree_out;

public:
  ldpc_code(const std::string & p_alist, bool p_is_filename = true);
  ldpc_code(const ldpc_code& p_code) = delete;

  virtual ~ldpc_code();
  void release_memory();

  void init_from_alist_file(std::istream & p_alist_stream);

  void set_n_erased_in_bits(int32_t p_n_erased_bits);
  int64_t n_inputs() const;
  int64_t n_outputs() const;
  int64_t n_erased_inputs() const;
  int64_t n_erased_outputs() const;
  uint32_t n_edges() const;
  uint32_t edge_int_to_out(uint32_t p_in_edge) const;
  uint32_t edge_out_to_in(uint32_t p_output_edge) const;
  uint32_t out_bit_to_edge(uint32_t p_output_bit) const;
  uint32_t out_edge_to_bit(uint32_t p_output_edge) const;
  uint32_t in_bit_to_edge(uint32_t p_input_bit) const;
  uint32_t in_edge_to_bit(uint32_t p_input_edge) const;
  int32_t max_degree_in() const;
  int32_t max_degree_out() const;
};

int64_t n_effective_inputs(const ldpc_code& p_code);
int64_t n_effective_outputs(const ldpc_code& p_code);
float rate(const ldpc_code& p_code);
void compute_syndrome(
    const ldpc_code& p_code,
    const bool_vec& p_input_vector,
    bool_vec& po_output_vector);
