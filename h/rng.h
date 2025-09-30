#pragma once

#include <cassert>

#include "common.h"

class rng
{
private:
  uint32_t bool_num_bits;
  uint32_t g_val_exists;
  float g_val_next;

public:

  rng() :
    bool_num_bits(0),
    g_val_exists(false),
    g_val_next(0)
  {
  }

  virtual ~rng() = default;

  // methods that must be implemented by subclasses:
  // reset rng seed
  virtual void reset_seed_internal(uint64_t p_seed) = 0;
  // provide a random 32-bit integer
  virtual uint32_t random_int() = 0;

  void reset_seed(uint64_t p_seed)
  {
    reset_seed_internal(p_seed);
    bool_num_bits = 0;
    g_val_exists = false;
  }

  float unit()
  {
    static const float normalizer = pow(2.f, -32.f);
    return (static_cast<float>(random_int()) + .5f) * normalizer;
  }

  bool biased_bool(float p_proba)
  {
    return unit() < p_proba;
  }

  transfer_llr_t gaussian()
  {
    float result = 0;
    if (g_val_exists) result = g_val_next;
    else
    {
      float x, y, sqnorm;
      do
      {
        x = 2.f * unit() - 1.f;
        y = 2.f * unit() - 1.f;
        sqnorm = x * x + y * y;
      }
      while (sqnorm >= 1 || sqnorm == 0);
      const float modulus = sqrt((-2 * log(sqnorm)) / sqnorm);
      // {x, y} * modulus are the two produced gaussian values
      result     = x * modulus;
      g_val_next = y * modulus;
    }
    g_val_exists = !g_val_exists;
    return static_cast<transfer_llr_t>(result);
  }
};
