#pragma once

#include <immintrin.h>

#include <cstdint>

using namespace std;

#ifndef __AVX2__
#error -- Implementation supports only microarchitectures with support for Advanced Vector Extensions (AVX2 or AVX512).
#endif

#ifdef _MSC_VER
#define ALIGN(x) __declspec(align (x))
#else
#define ALIGN(x) __attribute__ ((aligned (x)))
#endif

void transpose_16x16_SSE(const __m128i &aIn, const __m128i &bIn, __m128i & cOut, __m128i &dOut);

__m256i transpose_4x8x8_AVX2(const __m256i &x);

__m256i transpose_16x16_AVX2(const __m256i &x);

void transpose_32x32_AVX2(const uint64_t*in, uint64_t* out);

void transpose_32x32_AVX2_alt(const uint64_t*in, uint64_t* out);
