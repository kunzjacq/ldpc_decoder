#include "common.h"

#include <cmath>

#ifdef _WIN32
#include <malloc.h>

void* aligned_alloc(size_t align, size_t sz)
{
  return _aligned_malloc(sz, align);
}

void aligned_free(void* ptr)
{
  _aligned_free(ptr);
}
#else
#include <stdlib.h>
void aligned_free(void* ptr)
{
  std::free(ptr);
}
#endif

error::error(const char *p_message) :
  m_message(p_message)
{}

error::error(const string &p_message) :
  m_message(p_message)
{}

const char *error::what() const noexcept
{
  return m_message.c_str();
}

timer::timer(bool p_start) :
  m_is_running(false),
  m_total_time(0.)
{
  if (p_start) start();
}

void timer::start()
{
  if (!m_is_running)
  {
    m_begin = high_resolution_clock::now();
    m_is_running = true;
  }
}

double timer::stop()
{
  double e = time();
  m_is_running = false;
  m_total_time = e;
  return m_total_time;
}

double timer::time()
{
  double elapsed_time = m_total_time;
  if (m_is_running)
  {
    m_end = high_resolution_clock::now();
    constexpr double factor = 1e-9;// pow(10, -9);
    elapsed_time +=
        static_cast<double>(duration_cast<chrono::nanoseconds>(m_end-m_begin).count()) * factor;
  }
  return elapsed_time;
}

void timer::reset()
{
  stop();
  m_total_time = 0;
}
