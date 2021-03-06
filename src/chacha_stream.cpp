/* ChaCha implementation using 256-bit (512-bit) vectorization by the authors of [1].
* This is a public domain implementation, which improves the slightly modified implementations
* of Ted Krovetz in the Chromium Project by using the Advanced Vector Extensions AVX2 and AVX512
* to widen the vectorization. Further details and measurement results are provided in:
* [1] Goll, M., and Gueron,S.: Vectorization of ChaCha Stream Cipher. Cryptology ePrint Archive,
* Report 2013/759, November, 2013, http://eprint.iacr.org/2013/759.pdf
*/

// if MSVC is used, version 16.3 or above must be used because of a bug with previous versions
// see https://developercommunity.visualstudio.com/t/avx2-optimization-bug-with-cl-versions-1916270301/549774


#include <iostream>
#include <iomanip>
#include <cstring>
#include <immintrin.h>

using namespace std;

#ifndef __AVX2__
#error -- Implementation supports only microarchitectures with support for Advanced Vector Extensions 2
#endif

#ifdef _MSC_VER
#define ALIGN(x) __declspec(align (x))
#else
#define ALIGN(x) __attribute__ ((aligned (x)))
#endif

#define XOR128(a,b)	_mm_xor_si128(a, b)
#define LOAD128(m)	_mm_loadu_si128((__m128i*)(m))
#define STORE128(m,r)	_mm_storeu_si128((__m128i*)(m),  (r))
#define WRITE_XOR_128(ip, op, d, v0, v1, v2, v3)	\
  STORE128(op + d + 0, XOR128(LOAD128(ip + d + 0), v0));	\
  STORE128(op + d + 4, XOR128(LOAD128(ip + d + 4), v1));	\
  STORE128(op + d + 8, XOR128(LOAD128(ip + d + 8), v2));	\
  STORE128(op + d +12, XOR128(LOAD128(ip + d +12), v3));
#define WRITE_128(op, d, v0, v1, v2, v3)	\
  STORE128(op + d + 0, v0);	\
  STORE128(op + d + 4, v1);	\
  STORE128(op + d + 8, v2);	\
  STORE128(op + d +12, v3);
#define TWO	_mm256_set_epi64x(0,2,0,2)
#define LOAD256(m)		_mm256_loadu_si256((__m256i*)(m))
#define STORE256(m,r)	_mm256_storeu_si256((__m256i*)(m),  (r))
#define LOW128(x)	_mm256_castsi256_si128( (x))
#define HIGH128(x)	_mm256_extractf128_si256( (x), 1)
#define ADD256_32(a,b)	_mm256_add_epi32(a, b)
#define ADD256_64(a,b)	_mm256_add_epi64(a, b)
#define XOR256(a,b)	_mm256_xor_si256(a, b)
#define ROR256_V1(x)	_mm256_shuffle_epi32(x,_MM_SHUFFLE(0,3,2,1))
#define ROR256_V2(x)	_mm256_shuffle_epi32(x,_MM_SHUFFLE(1,0,3,2))
#define ROR256_V3(x)	_mm256_shuffle_epi32(x,_MM_SHUFFLE(2,1,0,3))
#define ROL256_7(x)		XOR256(_mm256_slli_epi32(x, 7), _mm256_srli_epi32(x,25))
#define ROL256_12(x)	XOR256(_mm256_slli_epi32(x,12), _mm256_srli_epi32(x,20))
#define ROL256_8(x)		_mm256_shuffle_epi8(x,_mm256_set_epi8(14,13,12,15,	\
  10, 9, 8,11,	\
  6, 5, 4, 7,	\
  2, 1, 0, 3,	\
  14,13,12,15,	\
  10, 9, 8,11,	\
  6, 5, 4, 7,	\
  2, 1, 0, 3))
#define ROL256_16(x)	_mm256_shuffle_epi8(x,_mm256_set_epi8(13,12,15,14,	\
  9, 8,11,10,	\
  5, 4, 7, 6,	\
  1, 0, 3, 2,	\
  13,12,15,14,	\
  9, 8,11,10,	\
  5, 4, 7, 6,	\
  1, 0, 3, 2))
#define DQROUND_VECTORS_256(a,b,c,d)						\
  a = ADD256_32(a,b); d = XOR256(d,a); d = ROL256_16(d);	\
  c = ADD256_32(c,d); b = XOR256(b,c); b = ROL256_12(b);	\
  a = ADD256_32(a,b); d = XOR256(d,a); d = ROL256_8(d);	\
  c = ADD256_32(c,d); b = XOR256(b,c); b = ROL256_7(b);	\
  b = ROR256_V1(b); c = ROR256_V2(c); d = ROR256_V3(d);	\
  a = ADD256_32(a,b); d = XOR256(d,a); d = ROL256_16(d);	\
  c = ADD256_32(c,d); b = XOR256(b,c); b = ROL256_12(b);	\
  a = ADD256_32(a,b); d = XOR256(d,a); d = ROL256_8(d);	\
  c = ADD256_32(c,d); b = XOR256(b,c); b = ROL256_7(b);	\
  b = ROR256_V3(b); c = ROR256_V2(c); d = ROR256_V1(d);
#define WRITE_XOR_256(ip, op, d, v0, v1, v2, v3)							\
  STORE256(op + d + 0, XOR256(LOAD256(ip + d + 0), _mm256_permute2x128_si256(v0, v1, 0x20)));	\
  STORE256(op + d + 8, XOR256(LOAD256(ip + d + 8), _mm256_permute2x128_si256(v2, v3, 0x20)));	\
  STORE256(op + d +16, XOR256(LOAD256(ip + d +16), _mm256_permute2x128_si256(v0, v1, 0x31)));	\
  STORE256(op + d +24, XOR256(LOAD256(ip + d +24), _mm256_permute2x128_si256(v2, v3, 0x31)));

#define WRITE_256(op, d, v0, v1, v2, v3)							\
  STORE256(op + d + 0, _mm256_permute2x128_si256(v0, v1, 0x20));	\
  STORE256(op + d + 8, _mm256_permute2x128_si256(v2, v3, 0x20));	\
  STORE256(op + d +16, _mm256_permute2x128_si256(v0, v1, 0x31));	\
  STORE256(op + d +24, _mm256_permute2x128_si256(v2, v3, 0x31));

// round selector, specified values:
//  8:  low security - high speed
// 12:  mid security -  mid speed
// 20: high security -  low speed

// Change * and ** to 'unsigned long long' if there is a need to encrypt/decrypt more than 2^32-1 bytes (~4GB) using a single call.
// This will slightly slow down the implementation due to all loop iterators become 64-bit.

int chacha_avx(
    unsigned char *out,
    unsigned int inlen, // *
    const unsigned char *nonce,
    const unsigned char *key,
    const unsigned long long counter,
    const unsigned int num_rounds
    )
{
  unsigned int i, j; // **
  unsigned int *op = (unsigned int *)out;

  ALIGN(16) unsigned int chacha_const[] = {
    0x61707865,0x3320646E,0x79622D32,0x6B206574
  };

  ALIGN(16) __m128i s3 = _mm_set_epi32(((unsigned int *)nonce)[1], ((unsigned int *)nonce)[0], counter >> 32, counter & 0xffffffff);

  ALIGN(32) __m256i d0 = _mm256_broadcastsi128_si256(*(__m128i*)chacha_const);
  ALIGN(32) __m256i d1 = _mm256_broadcastsi128_si256(((__m128i*)key)[0]);
  ALIGN(32) __m256i d2 = _mm256_broadcastsi128_si256(((__m128i*)key)[1]);
  ALIGN(32) __m256i d3 = ADD256_64(_mm256_broadcastsi128_si256(s3), _mm256_set_epi64x(0,1,0,0));

  for (j = 0; j < inlen/384; j++)
  {
    ALIGN(32) __m256i v0=d0, v1=d1, v2=d2, v3=d3;
    ALIGN(32) __m256i v4=d0, v5=d1, v6=d2, v7=ADD256_64(d3, TWO);
    ALIGN(32) __m256i v8=d0, v9=d1, v10=d2, v11=ADD256_64(v7, TWO);

    for (i = num_rounds/2; i; i--)
    {
      DQROUND_VECTORS_256(v0,v1,v2,v3);
      DQROUND_VECTORS_256(v4,v5,v6,v7);
      DQROUND_VECTORS_256(v8,v9,v10,v11);
    }

    WRITE_256(op, 0, ADD256_32(v0,d0), ADD256_32(v1,d1), ADD256_32(v2,d2), ADD256_32(v3,d3));
    d3 = ADD256_64(d3, TWO);
    WRITE_256(op,32, ADD256_32(v4,d0), ADD256_32(v5,d1), ADD256_32(v6,d2), ADD256_32(v7,d3));
    d3 = ADD256_64(d3, TWO);
    WRITE_256(op,64, ADD256_32(v8,d0), ADD256_32(v9,d1), ADD256_32(v10,d2), ADD256_32(v11,d3));
    d3 = ADD256_64(d3, TWO);
    op += 96;
  }
  inlen = inlen % 384;

  if(inlen >= 256)
  {
    ALIGN(32) __m256i v0 = d0, v1 = d1, v2 = d2, v3 = d3;
    ALIGN(32) __m256i v4 = d0, v5 = d1, v6 = d2, v7 = ADD256_64(d3, TWO);

    for (i = num_rounds/2; i; i--)
    {
      DQROUND_VECTORS_256(v0,v1,v2,v3);
      DQROUND_VECTORS_256(v4,v5,v6,v7);
    }

    WRITE_256(op, 0, ADD256_32(v0,d0), ADD256_32(v1,d1), ADD256_32(v2,d2), ADD256_32(v3,d3));
    d3 = ADD256_64(d3, TWO);
    WRITE_256(op, 32, ADD256_32(v4,d0), ADD256_32(v5,d1), ADD256_32(v6,d2), ADD256_32(v7,d3));
    d3 = ADD256_64(d3, TWO);
    op += 64;
    inlen = inlen % 256;
  }

  if (inlen >= 128)
  {
    ALIGN(32) __m256i v0=d0, v1=d1, v2=d2, v3=d3;
    for (i = num_rounds/2; i; i--)
    {
      DQROUND_VECTORS_256(v0,v1,v2,v3);
    }

    WRITE_256(op, 0, ADD256_32(v0,d0), ADD256_32(v1,d1), ADD256_32(v2,d2), ADD256_32(v3,d3));
    d3 = ADD256_64(d3, TWO);
    op += 32;
    inlen = inlen % 128;
  }

  if (inlen)
  {
    ALIGN(32) __m256i v0=d0, v1=d1, v2=d2, v3=d3;
    ALIGN(16) __m128i buf[4];

    for (i = num_rounds/2; i; i--) {
      DQROUND_VECTORS_256(v0,v1,v2,v3)
    }
    v0 = ADD256_32(v0,d0); v1 = ADD256_32(v1,d1);
    v2 = ADD256_32(v2,d2); v3 = ADD256_32(v3,d3);

    if (inlen >= 64)
    {
      WRITE_128(op, 0, LOW128(v0), LOW128(v1), LOW128(v2), LOW128(v3));
      buf[0] = HIGH128(v0); j = 64;
      if (inlen >= 80) {
        STORE128(op + 16, buf[0]);
        buf[1] = HIGH128(v1);
        if (inlen >= 96) {
          STORE128(op + 20, buf[1]);
          buf[2] = HIGH128(v2);
          if (inlen >= 112) {
            STORE128(op + 24, buf[2]);
            buf[3] = HIGH128(v3);
          }
        }
      }
    }
    else
    {
      buf[0] = LOW128(v0);  j = 0;
      if (inlen >= 16) {
        STORE128(op + 0, buf[0]);
        buf[1] = LOW128(v1);
        if (inlen >= 32) {
          STORE128(op + 4, buf[1]);
          buf[2] = LOW128(v2);
          if (inlen >= 48) {
            STORE128(op + 8, buf[2]);
            buf[3] = LOW128(v3);
          }
        }
      }
    }

    for (i=(inlen & ~0xF); i < inlen; i++)
    {
      ((unsigned char *)op)[i] = ((unsigned char *)buf)[i-j];
    }
  }

  return 0;
}

