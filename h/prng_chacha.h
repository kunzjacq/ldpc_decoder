#pragma once

#include "rng.h"

class prng_chacha : public rng
{
private:
  uint32_t* buf;
  unsigned char* buf_c;
  unsigned char* key_c;
  unsigned char* iv_c;
  alignas(32) uint32_t key[8]; //__attribute__ ((aligned (32)));
  alignas(32) uint64_t iv; // __attribute__((aligned(32)));
  unsigned int idx;
public:

  prng_chacha(uint64_t p_seed);
  virtual ~prng_chacha();

  prng_chacha() = delete;
  prng_chacha(const prng_chacha&) = delete;
  prng_chacha(prng_chacha&&) = delete;
  prng_chacha& operator= (prng_chacha &&) = delete;
  prng_chacha& operator= (const prng_chacha &) = delete;

  uint32_t random_int();

protected:
  void reset_seed_internal(uint64_t p_seed);
private:
  void refill();
};
