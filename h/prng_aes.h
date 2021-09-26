#pragma once

#include "rng.h"

#include <wmmintrin.h>  //for AES-NI intrinsics and types


class prng_aes : public rng
{
private:
  uint32_t enc[4];
  unsigned char* enc_c;
  unsigned char* plain_c;
  unsigned char* key_c;
  alignas(32) uint64_t key[2];// __attribute__((aligned(32)));
  alignas(32) uint64_t iv[2];// __attribute__((aligned(32)));
  alignas(32) __m128i key_schedule[20]; // __attribute__((aligned(32)));
  unsigned int idx;
public:

  prng_aes(uint64_t p_seed);
  virtual ~prng_aes() = default;

  prng_aes() = delete;
  prng_aes(const prng_aes&) = delete;
  prng_aes(prng_aes&&) = delete;
  prng_aes& operator= (prng_aes &&) = delete;
  prng_aes& operator= (const prng_aes &) = delete;

  uint32_t random_int();

protected:
  void reset_seed_internal(uint64_t p_seed);

private:
  void refill();
};
