#include "prng_chacha.h"

#include <iostream>
#include <memory>
#include <cstring> // for memset
#include <cstdlib>


int chacha_avx(
    unsigned char *out,
    unsigned int inlen,
    const unsigned char *nonce,
    const unsigned char *key,
    const unsigned long long counter,
    const unsigned int num_rounds
    );

int chacha_stream(
    unsigned char *out,
    unsigned long outlen,
    const unsigned char *n, // 8 bytes
    const unsigned char *k  // 32 bytes
    )
{
  return chacha_avx(out, outlen, n, k, 0, 8);
}

static constexpr size_t buffer_size = 1536;

prng_chacha::prng_chacha(uint64_t p_seed):
  buf((uint32_t*) aligned_alloc(32, buffer_size)),
  buf_c((unsigned char*)buf),
  key_c((unsigned char*)key),
  iv_c((unsigned char*)&iv)
{
  reset_seed_internal(p_seed);
}

void prng_chacha::reset_seed_internal(uint64_t p_seed)
{
  iv = 0;
  idx = 0;
  memset(buf, 0, buffer_size);
  memset(key, 0, 32);
  key[0] = p_seed & 0xFFFFFFFF;
  p_seed >>= 32;
  key[1] = p_seed & 0xFFFFFFFF;
  refill();
}

prng_chacha::~prng_chacha()
{
  aligned_free(buf);
}

uint32_t prng_chacha::random_int()
{
  if(idx == (buffer_size >> 2)) refill();
  return buf[idx++];
}

void prng_chacha::refill()
{
  chacha_stream(buf_c, buffer_size, iv_c, key_c);
  idx = 0;
  iv++;
}



