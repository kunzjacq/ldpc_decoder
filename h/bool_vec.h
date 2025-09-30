#pragma once

#include "common.h"
#include "channel.h"

#include <vector>
#include <cstdlib>
#include <cstring>

typedef uint32_t bool_t;

static constexpr bool_t bool_t_unit = 1;
static constexpr uint32_t bool_t_bit_size = 8*sizeof(bool_t);
static constexpr uint32_t bool_t_log_bit_size = 5;

class bool_vec
{
private:
  uint32_t m_num_words_per_bit;
  uint32_t m_num_vec;
  int64_t m_vec_bit_sz;
  bool_t* m_ref;

public:
  bool_vec(uint32_t num_vec, int64_t vec_bit_sz, bool erase=false):
    m_num_words_per_bit((num_vec + bool_t_bit_size - 1) / bool_t_bit_size),
    m_num_vec(num_vec),
    m_vec_bit_sz(vec_bit_sz),
    m_ref((bool_t*) util::aligned_alloc(32, m_num_words_per_bit*vec_bit_sz*sizeof(bool_t)))
  {
    if(erase) memset(m_ref, 0, m_num_words_per_bit*vec_bit_sz*sizeof(bool_t));
  }

  ~bool_vec()
  {
    util::aligned_free(m_ref);
  }

  // access with idx = vec_idx + word_sz * num_words_per_bit * bit_idx
  bool operator[](size_t idx) const
  {

    auto a = idx >> bool_t_log_bit_size;
    auto b = idx - (a << bool_t_log_bit_size);
    return (m_ref[a] & (bool_t_unit << b));
  }
  bool bit(size_t vec_idx, size_t bit_idx) const
  {
    auto a = vec_idx >> bool_t_log_bit_size;
    auto b = vec_idx - (a << bool_t_log_bit_size);
    return m_ref[a + m_num_words_per_bit * bit_idx] & (bool_t_unit << b);
  }

  void set_bit(size_t vec_idx, size_t bit_idx)
  {
    auto a = vec_idx >> bool_t_log_bit_size;
    auto b = vec_idx - (a << bool_t_log_bit_size);
    m_ref[a + m_num_words_per_bit * bit_idx] |= (bool_t_unit << b);
  }


  bool_t& word_ref(size_t idx){ return m_ref[idx];}
  const bool_t& word_ref(size_t idx) const { return m_ref[idx];}
  bool_t& word_ref(size_t vec_group_idx, size_t bit_idx)
  {
    return m_ref[vec_group_idx + m_num_words_per_bit * bit_idx];
  }
  const bool_t& word_ref(size_t vec_group_idx, size_t bit_idx) const
  {
    return m_ref[vec_group_idx + m_num_words_per_bit * bit_idx];
  }
  const bool_t* array() const { return m_ref;}
  uint32_t num_words_per_bit () const {return m_num_words_per_bit;}
  uint32_t vector_stride () const {return m_num_words_per_bit * bool_t_bit_size;}
  int32_t num_vectors() const {return m_num_vec;}
  int64_t vec_bit_sz() const {return m_vec_bit_sz;}
};
