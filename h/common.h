#pragma once

#define _USE_MATH_DEFINES
#include <cmath> 

#include <string>
#include <exception>
#include <chrono>

using namespace std;
using namespace std::chrono;

#ifdef CUDA_DECODER

#ifdef __CUDACC__
#include <cuda_fp16.h>
#ifdef USE_FLOAT16_COMPUTE
using llr_t = __half;
#else
using llr_t = float;
#endif
#else
#include <stdfloat>
#ifdef USE_FLOAT16_COMPUTE
using llr_t = float16_t;
#else
using llr_t = float;
#endif
#endif

#else
// opencl decoder only supports floats.
using llr_t = float;
#endif

// format used for transfer to the gpu could differ from the format used for computation,
// however for simplicity it is set to the latter.
using transfer_llr_t = llr_t;

namespace util{
void* aligned_alloc(size_t align, size_t sz);
void aligned_free(void* ptr);
}

typedef enum _channelType
{
  awgn, bsc, groupGauss, erasure
} channelType;

// boths functions below depend on LLR sign convention
inline bool llr_to_bool (const transfer_llr_t &val)
{
  return val > transfer_llr_t(0);
}

inline transfer_llr_t bool_to_llr(const bool& val)
{
  return val ? 1 : -1;
}

class error : public exception
{
  string m_message;
public:
  error() = delete;
  error(const error&) = default;
  error & operator=(const error&) = default;

  error(const char* p_message);
  error(const string & p_message);
  ~error() throw () {}
  virtual const char* what() const noexcept;
};

class timer
{
private:
    time_point<high_resolution_clock> m_begin, m_end;
    bool m_is_running;
    double m_total_time;

public:
    // creates the timer. the bool argument tells whether the timer should be started or not.
    timer(bool);
    // stops the timer and returns the elapsed time, in seconds.
    double stop();
    // starts or restarts the timer.
    void start();
    // obtains accumulated time.
    double time();
    // resets the timer.
    void reset();
};


