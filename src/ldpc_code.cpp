#include "ldpc_code.h"
#include "bool_vec.h"

#include <fstream>
#include <iostream>
#include <memory> // for unique_ptr
#include <string> // for string
#include <sstream> // for stringstream

ldpc_code::ldpc_code(const string & p_alist, bool p_is_filename) :
  m_in_bit_to_edge(nullptr),
  m_in_edge_to_bit(nullptr),
  m_out_bit_to_edge(nullptr),
  m_out_edge_to_bit(nullptr),
  m_edge_in_to_out(nullptr),
  m_edge_out_to_in(nullptr),
  m_n_edges(),
  m_n_erased_variables(),
  m_n_erased_check_bits(),
  m_max_degree_in(),
  m_max_degree_out()
{
  if (p_is_filename)
  {
    ifstream f(p_alist.c_str());
    if (!f.good())
    {
      throw error("Alist file could not be opened for reading");
    }
    init_from_alist_file(f);
    f.close();
  }
  else
  {
    stringstream alist_content(p_alist);
    init_from_alist_file(alist_content);
  }
}

ldpc_code::~ldpc_code()
{
  release_memory();
}

void ldpc_code::init_from_alist_file(istream & p_alist_stream)
{
  m_n_erased_variables = 0;

  char* buffer = new char[1024];
  unique_ptr<char[]> _(buffer);

  while (p_alist_stream.peek() == '#')
  {
    string line_buf;
    p_alist_stream >> line_buf;
    size_t eq_pos = line_buf.find("=");
    string param = line_buf.substr(1, eq_pos - 1);
    string val = line_buf.substr(eq_pos + 1);

    if (param == "e")
    {
      stringstream value(val);
      value >> m_n_erased_variables;
    }
    else if (param == "ec")
    {
      stringstream value(val);
      value >> m_n_erased_check_bits;
    }
    else
    {
      cout << " " << param << " = " << val << endl;
    }

    p_alist_stream.ignore(-1u, '\n');
  }

  p_alist_stream >> m_n_outputs;
  p_alist_stream >> m_n_inputs;

  m_in_bit_to_edge = new uint32_t[m_n_inputs + 1];
  m_out_bit_to_edge = new uint32_t[m_n_outputs + 1];

  p_alist_stream.ignore(-1u, '\n');
  p_alist_stream.ignore(-1u, '\n');

  m_n_edges = 0;

  for (int32_t i = 0; i < m_n_outputs; i++)
  {
    m_out_bit_to_edge[i] = m_n_edges;
    int32_t vertices_in_row = 0;
    p_alist_stream >> vertices_in_row;
    m_max_degree_out = max(m_max_degree_out, vertices_in_row);
    m_n_edges += vertices_in_row;
  }
  m_out_bit_to_edge[m_n_outputs] = m_n_edges;

  p_alist_stream.ignore(-1u, '\n');

  uint32_t n_edges_check = 0;
  for (int32_t i = 0; i < m_n_inputs; i++)
  {
    m_in_bit_to_edge[i] = n_edges_check;
    int32_t edges_in_column = 0;
    p_alist_stream >> edges_in_column;
    m_max_degree_in = max(m_max_degree_in, edges_in_column);
    n_edges_check += edges_in_column;
  }
  m_in_bit_to_edge[m_n_inputs] = n_edges_check;

  p_alist_stream.ignore(-1u, '\n');

  if (m_n_edges != n_edges_check)
  {
    throw error("PrecomputedCode::init_from_alist_file(): malformed alist file");
  }

  m_in_edge_to_bit = new uint32_t[m_n_edges];
  m_out_edge_to_bit = new uint32_t[m_n_edges];

  uint32_t c_in = 0;
  uint32_t c_out = 0;
  for (uint32_t i = 0; i < m_n_edges; i++)
  {
    if (i >= m_in_bit_to_edge[c_in + 1])   c_in++;
    if (i >= m_out_bit_to_edge[c_out + 1]) c_out++;
    m_in_edge_to_bit[i] = c_in;
    m_out_edge_to_bit[i] = c_out;
  }

  uint32_t* tmp_column_indexes = new uint32_t[m_n_inputs];
  unique_ptr<uint32_t[]> _2(tmp_column_indexes);
  m_edge_out_to_in = new uint32_t[m_n_edges];
  m_edge_in_to_out = new uint32_t[m_n_edges];

  for (int32_t i = 0; i < m_n_inputs; i++) tmp_column_indexes[i] = 0;

  for (int32_t i = 0; i < m_n_outputs; i++)
  {
    for (uint32_t j = m_out_bit_to_edge[i]; j < m_out_bit_to_edge[i + 1]; j++)
    {
      uint32_t col_index = 0;
      p_alist_stream >> col_index;
      col_index--;
      m_edge_out_to_in[j] = m_in_bit_to_edge[col_index] + tmp_column_indexes[col_index];
      m_edge_in_to_out[m_edge_out_to_in[j]] = j;
      tmp_column_indexes[col_index]++;
    }
    p_alist_stream.ignore(-1u, '\n');
  }
}

void ldpc_code::release_memory()
{
  delete [] m_in_bit_to_edge;
  delete [] m_out_bit_to_edge;
  delete [] m_in_edge_to_bit;
  delete [] m_out_edge_to_bit;
  delete [] m_edge_out_to_in;
  delete [] m_edge_in_to_out;
}

void ldpc_code::set_n_erased_in_bits(int32_t p_n_erased_bits)
{
  m_n_erased_variables = p_n_erased_bits;
}

uint32_t ldpc_code::edge_int_to_out(uint32_t p_in_edge) const
{
  return m_edge_in_to_out[p_in_edge];
}

uint32_t ldpc_code::edge_out_to_in(uint32_t p_output_edge) const
{
  return m_edge_out_to_in[p_output_edge];
}

uint32_t ldpc_code::out_bit_to_edge(uint32_t p_output_bit) const
{
  return m_out_bit_to_edge[p_output_bit];
}

uint32_t ldpc_code::out_edge_to_bit(uint32_t p_output_edge) const
{
  return m_out_edge_to_bit[p_output_edge];
}

uint32_t ldpc_code::in_bit_to_edge(uint32_t p_input_bit) const
{
  return m_in_bit_to_edge[p_input_bit];
}

uint32_t ldpc_code::in_edge_to_bit(uint32_t p_input_edge) const
{
  return m_in_edge_to_bit[p_input_edge];
}

int64_t ldpc_code::n_inputs() const
{
  return m_n_inputs;
}

int64_t ldpc_code::n_erased_inputs() const
{
  return m_n_erased_variables;
}

int64_t ldpc_code::n_erased_outputs() const
{
  return m_n_erased_check_bits;
}

int64_t ldpc_code::n_outputs() const
{
  return m_n_outputs;
}

uint32_t ldpc_code::n_edges() const
{
  return m_n_edges;
}

int32_t ldpc_code::max_degree_in() const
{
  return m_max_degree_in;
}

int32_t ldpc_code::max_degree_out() const
{
  return m_max_degree_out;
}

int64_t n_effective_inputs(const ldpc_code &p_code)
{
  return p_code.n_inputs() - p_code.n_erased_inputs();
}

int64_t n_effective_outputs(const ldpc_code &p_code)
{
  return p_code.n_outputs() - p_code.n_erased_outputs();
}

float rate(const ldpc_code &p_code)
{
  // erased variables: input signals that are not sent, but still decoded without
  // any initial information about them. With i input variables among which e are erased,
  // and o output variables (parity bits), the number of independent symbols decoded is i-o,
  // and the number of symbols transmitted on the wire is i-e. hence the rate is (i-o)/(i-e).
  float code_rate =
      static_cast < float > (p_code.n_inputs() - p_code.n_outputs()) /
      static_cast < float > (p_code.n_inputs() - p_code.n_erased_inputs());
  return code_rate;
}

void compute_syndrome(
    const ldpc_code& p_code,
    const bool_vec& p_input_vector,
    bool_vec& po_output_vector)
{
  const uint32_t n_outputs = static_cast<uint32_t>(p_code.n_outputs());
  const uint32_t num_input_words = p_input_vector.num_words_per_bit();
  const uint32_t num_output_words = po_output_vector.num_words_per_bit();
  const int64_t vec_bit_sz = po_output_vector.vec_bit_sz();
  assert(num_input_words == num_output_words);
  assert(vec_bit_sz >= n_outputs);
  // vec_bit_sz can be larger that n_outputs if it is rounded above
  for (uint32_t i = 0; i < vec_bit_sz * num_output_words; i++)
  {
    po_output_vector.word_ref(i) = 0;
  }
  for (uint32_t out_bit_idx = 0; out_bit_idx < n_outputs; out_bit_idx++)
  {
    for (uint32_t out_edge = p_code.out_bit_to_edge(out_bit_idx);
                  out_edge < p_code.out_bit_to_edge(out_bit_idx + 1);
                  out_edge++)
    {
      const uint32_t in_bit_idx = p_code.in_edge_to_bit(p_code.edge_out_to_in(out_edge));
      for (uint32_t i = 0; i < num_input_words; i++)
      {
        const bool_t in_bit_val = p_input_vector.word_ref(in_bit_idx * num_input_words + i);
        po_output_vector.word_ref(out_bit_idx * num_input_words + i) ^= in_bit_val;
      }
    }
  }
}

