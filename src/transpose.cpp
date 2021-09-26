#include "transpose.h"

// this code does not work with mingw under windows because of
// https://stackoverflow.com/questions/5983389/how-to-align-stack-at-32-byte-boundary-in-gcc
// see also https://gcc.gnu.org/bugzilla/show_bug.cgi?id=49001
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412

#define MAYBE_STATIC static

// inspired from
// http://microperf.blogspot.com/2017/10/transposing-16x16-bit-matrix-using-sse2.html

// All matrices are represented in row-major order.

/*
// The book "Hacker's Delight" has an ansi algorithm for transposing a 8x8 bit matrix (64 bits total):
unsigned __int64 transpose8(unsigned __int64 x)
{
    x = (x & 0xAA55AA55AA55AA55LL) | ((x & 0x00AA00AA00AA00AALL) << 7) | ((x >> 7) & 0x00AA00AA00AA00AALL);
    x = (x & 0xCCCC3333CCCC3333LL) | ((x & 0x0000CCCC0000CCCCLL) << 14) | ((x >> 14) & 0x0000CCCC0000CCCCLL);
    x = (x & 0xF0F0F0F00F0F0F0FLL) | ((x & 0x00000000F0F0F0F0LL) << 28) | ((x >> 28) & 0x00000000F0F0F0F0LL);
    return x;

    // the first line operates on the 2x2 submatrices and transposes them:
    // - one term copies the diagonal terms unchanged
    // - one term deals with terms above the diagonals
    // - one term deals with terms below the diagonals.

    // second and third line do the same, respectively, for 4x4 submatrices and for the whole matrix.
    // 'terms' in these cases are smaller submatrices (2x2 in the 4x4 case and 4x4 in the 8x8 case)
    // for instance, 0xF0F0F0F00F0F0F0F selects the 2 diagonal 4x4 submatrices.
}*/

#if 1
// transposes two 64-bit 8x8 matrices in each half of the input and output __m128i
// based on the algorithm above performed twice on the two 64-bit lanes
__m128i transpose_8x8_SSE(const __m128i &x)
{
  const __m128i c1 = _mm_set1_epi64x(0xAA55AA55AA55AA55LL);
  const __m128i c2 = _mm_set1_epi64x(0x00AA00AA00AA00AALL);
  const __m128i c3 = _mm_set1_epi64x(0xCCCC3333CCCC3333LL);
  const __m128i c4 = _mm_set1_epi64x(0x0000CCCC0000CCCCLL);
  const __m128i c5 = _mm_set1_epi64x(0xF0F0F0F00F0F0F0FLL);
  const __m128i c6 = _mm_set1_epi64x(0x00000000F0F0F0F0LL);

  const __m128i x1 = _mm_or_si128(
        _mm_and_si128(x, c1),
        _mm_or_si128(
          _mm_slli_epi64(_mm_and_si128(x, c2), 7),
          _mm_and_si128(_mm_srli_epi64(x, 7), c2)
          )
        );
  const __m128i x2 = _mm_or_si128(
        _mm_and_si128(x1, c3),
        _mm_or_si128(
          _mm_slli_epi64(_mm_and_si128(x1, c4), 14),
          _mm_and_si128(_mm_srli_epi64(x1, 14), c4)
          )
        );
  const __m128i x3 = _mm_or_si128(
        _mm_and_si128(x2, c5),
        _mm_or_si128(
          _mm_slli_epi64(_mm_and_si128(x2, c6), 28),
          _mm_and_si128(_mm_srli_epi64(x2, 28), c6)
          )
        );
  return x3;
}

// from byte input [0, 1, ..., f]
// outputs [0, 8, 1, 9, 2, a, 3, b, 4, c, 5, d, 6, e, 7, f].
// enables to merge two 8x8 matrices into a 8x16 matrix.
__m128i merge_8x8_SSE(const __m128i&xIn)
{
  __m128i mRet = _mm_unpacklo_epi8(xIn, _mm_shuffle_epi32(xIn, _MM_SHUFFLE(3, 2, 3, 2)));
  return mRet;
}

// inverse of previous function
// from byte order [0, 1, ..., f]
// outputs [0, 2, 4, 6, 8, a, c, e, 1, 3, 5, 7, 9, b, d, f].
// enables to split a 8x16 matrix into two 8x8 matrices.

#if 0
// the permutation of merge_8x8_SSE has order 4, hence merge_8x8_SSE^3 = merge_8x8_SSE^-1
__m128i split_8x8_SSE(const __m128i& x)
{
  return merge_8x8_SSE(merge_8x8_SSE(merge_8x8_SSE(x)));
}
#else
__m128i split_8x8_SSE(const __m128i& x)
{
  const __m128i shuffle_8x16_to_2x8x8 = _mm_set_epi64x(
        0x0f0d0b0907050301LL,
        0x0e0c0a0806040200LL);
  return _mm_shuffle_epi8(x, shuffle_8x16_to_2x8x8);
}
#endif

// Transposes a matrix 16x8 matrix m.
// The two 8x8 submatrices of m lie in each half of the 128-bit value x.
// They are both transposed, then the two resulting matrices are interleaved
// with merge_8x8_SSE to form the 16-bit rows of the result.
__m128i transpose_16x8_SSE(const __m128i& x)
{
  __m128i t = transpose_8x8_SSE(x);
  return merge_8x8_SSE(t);
}

// Inverse of the preceding function: transposes a 8x16 matrix.
__m128i transpose_8x16_SSE(const __m128i& x)
{
  __m128i t = split_8x8_SSE( x);
  __m128i mRet = transpose_8x8_SSE(t);
  return mRet;
}

// 16x16 transpose. Input and output matrix are in row order, and split between 2 sse registers,
// with the 1st register containing the matrix upper half and the second the lower half.
void transpose_16x16_SSE(const __m128i &in_top, const __m128i &in_bottom, __m128i & out_top, __m128i &out_bottom)
{

  __m128i left  = transpose_8x16_SSE(in_top);
  __m128i right = transpose_8x16_SSE(in_bottom);
  out_top    = _mm_unpacklo_epi8(left, right);
  out_bottom = _mm_unpackhi_epi8(left, right);
}
#endif

// Transposes 2 pairs of 64-bit 8x8 matrices (one pair in each AVX2 lane)
// into 2 8x16 matrices, that is, one 16x16 matrix.
__m256i merge_8x8_AVX2(const __m256i& x)
{
  return _mm256_unpacklo_epi8(x, _mm256_shuffle_epi32(x, _MM_SHUFFLE(3, 2, 3, 2)));
}

inline void merge_8x8_AVX2_in_place(__m256i& x)
{
  x = _mm256_unpacklo_epi8(x, _mm256_shuffle_epi32(x, _MM_SHUFFLE(3, 2, 3, 2)));
}

// splits 2 8x16 matrices, or one 16x16 matrix, into 4 8x8 matrices
// inverse of merge_8x8_AVX2
#if 0
// implementation that leverages the fact that merge_8x8_AVX2^4 = identity, hence
// merge_8x8_AVX2^3 = merge_8x8_AVX2^-1
__m256i split_8x8_AVX2(const __m256i& x)
{
  return merge_8x8_AVX2(merge_8x8_AVX2(merge_8x8_AVX2(x)));
}
#else
__m256i split_8x8_AVX2(const __m256i& x)
{
  // variable not declared static since it appears that MSVC may not handle static
  // __m256i variables well. Performance is not improved with gcc 8 by using static.
  MAYBE_STATIC const __m256i shuffle_8x16_to_2x8x8 = _mm256_set_epi64x(
        0x0f0d0b0907050301LL,
        0x0e0c0a0806040200LL,
        0x0f0d0b0907050301LL,
        0x0e0c0a0806040200LL);
  return _mm256_shuffle_epi8(x, shuffle_8x16_to_2x8x8);
}

inline void split_8x8_AVX2_in_place(__m256i& x)
{
  // variable not declared static since it appears that MSVC may not handle static
  // __m256i variables well. Performance is not improved with gcc 8 by using static.
  MAYBE_STATIC const __m256i shuffle_8x16_to_2x8x8 = _mm256_set_epi64x(
        0x0f0d0b0907050301LL,
        0x0e0c0a0806040200LL,
        0x0f0d0b0907050301LL,
        0x0e0c0a0806040200LL);
  x = _mm256_shuffle_epi8(x, shuffle_8x16_to_2x8x8);
}
#endif

// transposes 4 64-bit 8x8 matrices in each quarter of the input __m256i
// based on the algorithm on top performed 4 times
inline __m256i transpose_4x8x8_AVX2(const __m256i& x)
{
  MAYBE_STATIC const __m256i c1 = _mm256_set1_epi64x(0xAA55AA55AA55AA55LL);
  MAYBE_STATIC const __m256i c2 = _mm256_set1_epi64x(0x00AA00AA00AA00AALL);
  MAYBE_STATIC const __m256i c3 = _mm256_set1_epi64x(0xCCCC3333CCCC3333LL);
  MAYBE_STATIC const __m256i c4 = _mm256_set1_epi64x(0x0000CCCC0000CCCCLL);
  MAYBE_STATIC const __m256i c5 = _mm256_set1_epi64x(0xF0F0F0F00F0F0F0FLL);
  MAYBE_STATIC const __m256i c6 = _mm256_set1_epi64x(0x00000000F0F0F0F0LL);

  const __m256i x1 = _mm256_or_si256(
        _mm256_and_si256(x, c1),
        _mm256_or_si256(
          _mm256_slli_epi64(_mm256_and_si256(x, c2), 7),
          _mm256_and_si256(_mm256_srli_epi64(x, 7), c2)
          )
        );
  const __m256i x2 =
      _mm256_or_si256(
        _mm256_and_si256(x1, c3),
        _mm256_or_si256(
          _mm256_slli_epi64(_mm256_and_si256(x1, c4), 14),
          _mm256_and_si256(_mm256_srli_epi64(x1, 14), c4)
          )
        );
  const __m256i x3 =
      _mm256_or_si256(
        _mm256_and_si256(x2, c5),
        _mm256_or_si256(
          _mm256_slli_epi64(
            _mm256_and_si256(x2, c6), 28),
          _mm256_and_si256(_mm256_srli_epi64(x2, 28), c6)
          )
        );

  return x3;
}

inline void transpose_4x8x8_AVX2_in_place(__m256i& x)
{
  MAYBE_STATIC const __m256i c1 = _mm256_set1_epi64x(0xAA55AA55AA55AA55LL);
  MAYBE_STATIC const __m256i c2 = _mm256_set1_epi64x(0x00AA00AA00AA00AALL);
  MAYBE_STATIC const __m256i c3 = _mm256_set1_epi64x(0xCCCC3333CCCC3333LL);
  MAYBE_STATIC const __m256i c4 = _mm256_set1_epi64x(0x0000CCCC0000CCCCLL);
  MAYBE_STATIC const __m256i c5 = _mm256_set1_epi64x(0xF0F0F0F00F0F0F0FLL);
  MAYBE_STATIC const __m256i c6 = _mm256_set1_epi64x(0x00000000F0F0F0F0LL);

  const __m256i x1 = _mm256_or_si256(
        _mm256_and_si256(x, c1),
        _mm256_or_si256(
          _mm256_slli_epi64(_mm256_and_si256(x, c2), 7),
          _mm256_and_si256(_mm256_srli_epi64(x, 7), c2)
          )
        );
  const __m256i x2 =
      _mm256_or_si256(
        _mm256_and_si256(x1, c3),
        _mm256_or_si256(
          _mm256_slli_epi64(_mm256_and_si256(x1, c4), 14),
          _mm256_and_si256(_mm256_srli_epi64(x1, 14), c4)
          )
        );
  x =
      _mm256_or_si256(
        _mm256_and_si256(x2, c5),
        _mm256_or_si256(
          _mm256_slli_epi64(
            _mm256_and_si256(x2, c6), 28),
          _mm256_and_si256(_mm256_srli_epi64(x2, 28), c6)
          )
        );
}

// 16x16 AVX2 transpose
inline __m256i transpose_16x16_AVX2(const __m256i &x)
{

  // view the matrix as 2 8x16 matrices.
  // split each of these matrices into 2 8x8 matrices.
  __m256i mUnshuffle;
  mUnshuffle = split_8x8_AVX2(x);
  // transpose the 4 8x8 matrices.
  __m256i mSmallTranspose = transpose_4x8x8_AVX2(mUnshuffle);
  // swap non-diagonal 8x8 matrices.
  __m256i mSmallTransposePrime = _mm256_permute4x64_epi64(mSmallTranspose, _MM_SHUFFLE(3,1,2,0));
  // merge each pair of 8x8 matrices to form 2 8x16 matrices again, i.e. the final 16x16 matrix.
  __m256i mRet = merge_8x8_AVX2(mSmallTransposePrime);
  return mRet;
}

inline void transpose_16x16_AVX2_in_place(__m256i &x)
{

  split_8x8_AVX2_in_place(x);
  transpose_4x8x8_AVX2_in_place(x);
  x = _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3,1,2,0));
  merge_8x8_AVX2_in_place(x);
}

void transpose_32x32_AVX2(const uint64_t*in, uint64_t* out)
{
  MAYBE_STATIC const __m256i shuffle_4x32_to_4x4x8 = _mm256_set_epi64x(
        0x0f0b07030e0a0602LL,
        0x0d0905010c080400LL,
        0x0f0b07030e0a0602LL,
        0x0d0905010c080400LL);

  MAYBE_STATIC const __m256i shuffle_4x4x8_to_4x32 = shuffle_4x32_to_4x4x8;

  MAYBE_STATIC const __m256i permute_2x4x4x8_to_4x8x8 = _mm256_set_epi64x(
        0x0000000700000003LL,
        0x0000000600000002LL,
        0x0000000500000001LL,
        0x0000000400000000LL);

  MAYBE_STATIC const __m256i permute_4x8x8_to_2x4x4x8 = _mm256_set_epi64x(
        0x0000000700000005LL,
        0x0000000300000001LL,
        0x0000000600000004LL,
        0x0000000200000000LL);

  __m256i* res = (__m256i*) out;

  __m256i v[4];
  for(int i = 0; i < 4; i++)
  {
    // read a 8x32 submatrix (seen below as 2 4x32 matrices, one in each lane)
    __m256i t = _mm256_set_epi64x(in[3 + 4 * i], in[2 + 4 * i], in[1 + 4 * i], in[0 + 4 * i]);
    // split the 4x32 matrices into 4 4x8 matrices
    t = _mm256_shuffle_epi8(t, shuffle_4x32_to_4x4x8);
    // interleave the 32-bit words of the two 128-bit lanes to form 4 8x8 matrices from
    // the two sets of 4 4x8 matrices
    t = _mm256_permutevar8x32_epi32(t, permute_2x4x4x8_to_4x8x8);
    // transpose all 8x8 matrices
    //v[i] = transpose_4x8x8_AVX2(t);
    transpose_4x8x8_AVX2_in_place(t);
    v[i] = t;
  }
  // now for each i, v[i] is a sequence of 4 8x8 matrices that corresponds to column i in the output.
  // v[i] has matrices m(i,0), m(i,1), m(i,2), m(i,3).
  // the eight operations below block-transpose these matrices into the correct order for output.
  __m256i w[4];
  w[0] = _mm256_unpacklo_epi64(v[0], v[1]); // m(0,0), m(1,0), m(0,2), m(1,2)
  w[1] = _mm256_unpacklo_epi64(v[2], v[3]); // m(2,0), m(3,0), m(2,2), m(3,2)
  w[2] = _mm256_unpackhi_epi64(v[0], v[1]); // m(0,1), m(1,1), m(0,3), m(1,3)
  w[3] = _mm256_unpackhi_epi64(v[2], v[3]); // m(2,1), m(3,1), m(2,3), m(3,3)

  v[0] = _mm256_permute2f128_si256(w[0], w[1], 0x20); // lower lanes of inputs: m(0,0), m(1,0), m(2,0), m(3,0)
  v[1] = _mm256_permute2f128_si256(w[2], w[3], 0x20); // lower lanes of inputs: m(0,1), m(1,1), m(2,1), m(3,1)
  v[2] = _mm256_permute2f128_si256(w[0], w[1], 0x31); // upper lanes of inputs: m(0,2), m(1,2), m(2,2), m(3,2)
  v[3] = _mm256_permute2f128_si256(w[2], w[3], 0x31); // upper lanes of inputs: m(0,3), m(1,3), m(2,3), m(3,3)

  // now v[i] for i = 0 ... 3 is a set of 4 8x8 matrices corresponding to a horizontal 8x32 submatrix
  // of the output. Let's merge them again.
  for(int i = 0; i < 4; i++)
  {
    // reorganize 4 8x8 matrices into two sequences of 4 4x8 matrices,
    // one sequence in each 128-bit lane
    __m256i t = _mm256_permutevar8x32_epi32(v[i], permute_4x8x8_to_2x4x4x8);
    // in each lane, merge the 4 4x8 matrices to form one 4x32 matrix
    res[i] = _mm256_shuffle_epi8(t, shuffle_4x4x8_to_4x32);
  }
}

void transpose_32x32_AVX2_alt(const uint64_t*in, uint64_t* out)
{
  MAYBE_STATIC const __m256i shuffle_4x32_to_2x4x16 = _mm256_set_epi64x(
        0x0f0e0b0a07060302LL,
        0x0d0c090805040100LL,
        0x0f0e0b0a07060302LL,
        0x0d0c090805040100LL);

  MAYBE_STATIC const __m256i shuffle_2x4x16_to_4x32 = _mm256_set_epi64x(
        0x0f0e07060d0c0504LL,
        0x0b0a030209080100LL,
        0x0f0e07060d0c0504LL,
        0x0b0a030209080100LL);

  __m256i v[4];
  __m256i* res = (__m256i*) out;
  for(int i = 0; i < 4; i++)
  {
    // read a 8x32 submatrix (seen below as 2 4x32 matrices, one in each lane)
    __m256i t = _mm256_set_epi64x(in[3 + 4 * i], in[2 + 4 * i], in[1 + 4 * i], in[0 + 4 * i]);
    // split the 4x32 matrices into 2 4x16 matrices
    t = _mm256_shuffle_epi8(t, shuffle_4x32_to_2x4x16);
    // interleave the 64-bit words of the two 128-bit lanes to form 2 8x16 matrices from
    // the two sets of 2 4x16 matrices
    v[i] = _mm256_permute4x64_epi64 (t,  _MM_SHUFFLE(3,1,2,0));
  }
  // now v[i] contains 2 8x16 matrices corresponding to a 8x32 horizontal slice of the initial matrix.

  __m256i w[4];
  // interleave 128-bit 8x16 matrices to form 16x16 matrices, in the correct output order
  w[0] = _mm256_permute2f128_si256(v[0], v[1], 0x20);
  w[2] = _mm256_permute2f128_si256(v[0], v[1], 0x31);
  w[1] = _mm256_permute2f128_si256(v[2], v[3], 0x20);
  w[3] = _mm256_permute2f128_si256(v[2], v[3], 0x31);
#if 0
  // transpose each 16x16 submatrix
  v[0] = transpose_16x16_AVX2(w[0]);
  v[1] = transpose_16x16_AVX2(w[1]);
  v[2] = transpose_16x16_AVX2(w[2]);
  v[3] = transpose_16x16_AVX2(w[3]);

  // form 8x16 matrices again
  w[0] = _mm256_permute2f128_si256(v[0], v[1], 0x20);
  w[1] = _mm256_permute2f128_si256(v[0], v[1], 0x31);
  w[2] = _mm256_permute2f128_si256(v[2], v[3], 0x20);
  w[3] = _mm256_permute2f128_si256(v[2], v[3], 0x31);
  for(int i = 0; i < 4; i++)
  {
    // interleave the 64-bit words of the two 128-bit lanes to form 4 4x16 matrices from
    // 2 8x16 matrices (same permutation as before)
    __m256i t = _mm256_permute4x64_epi64 (w[i], _MM_SHUFFLE(3,1,2,0));
    // merge pairs of successive 4x16 matrices into 4x32 matrices
    res[i] = _mm256_shuffle_epi8(t, shuffle_2x4x16_to_4x32);
  }
#else
  // transpose each 16x16 submatrix
  transpose_16x16_AVX2_in_place(w[0]);
  transpose_16x16_AVX2_in_place(w[1]);
  transpose_16x16_AVX2_in_place(w[2]);
  transpose_16x16_AVX2_in_place(w[3]);

  // form 8x16 matrices again
  v[0] = _mm256_permute2f128_si256(w[0], w[1], 0x20);
  v[1] = _mm256_permute2f128_si256(w[0], w[1], 0x31);
  v[2] = _mm256_permute2f128_si256(w[2], w[3], 0x20);
  v[3] = _mm256_permute2f128_si256(w[2], w[3], 0x31);
  for(int i = 0; i < 4; i++)
  {
    // interleave the 64-bit words of the two 128-bit lanes to form 4 4x16 matrices from
    // 2 8x16 matrices (same permutation as before)
    __m256i t = _mm256_permute4x64_epi64 (v[i], _MM_SHUFFLE(3,1,2,0));
    // merge pairs of successive 4x16 matrices into 4x32 matrices
    res[i] = _mm256_shuffle_epi8(t, shuffle_2x4x16_to_4x32);
  }
#endif
}
