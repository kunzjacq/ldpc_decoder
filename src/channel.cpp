#include "channel.h"

static float bsc_capacity(float p);
static float biawgn_capacity(float s, float step, float range);

bsc_channel::bsc_channel(float p):
  m_p(p),
  m_llr_ref(log(1 - m_p) - log(m_p)),
  m_capacity(bsc_capacity(p))
{}


float bsc_channel::capacity() const
{
  return m_capacity;
}

transfer_llr_t bsc_channel::llr(float val) const
{
  // depends on LLR sign convention
  return static_cast<transfer_llr_t>(val > 0 ? m_llr_ref : -m_llr_ref);
}

void bsc_channel::description(ostream &p_stream) const
{
  p_stream << "Binary channel with bit error probability: " << m_p << endl;
}

void bsc_channel::add_noise(rng& r, float* val)
{
  if(r.unit() < m_p) *val *= -1;
}

transfer_llr_t bsc_channel::add_noise(rng& r, float val)
{
  if(r.unit() < m_p) val *= -1;
  return static_cast<transfer_llr_t>(val);
}

biawgn_channel::biawgn_channel(float s): // noise standard deviation
  m_s(s),
  m_snr(1 / (m_s * m_s)),
  m_capacity(biawgn_capacity(m_s, 0.05f, 16.f)){}

float biawgn_channel::capacity() const
{
  return m_capacity;
}

transfer_llr_t biawgn_channel::llr(const float val) const
{
  return static_cast<transfer_llr_t>(2 * m_snr * val);
}

void biawgn_channel::description(ostream &p_stream) const
{
  p_stream << "Binary channel with Gaussian noise of std. deviation " << m_s << "; SNR = " << m_snr << endl;
}

void biawgn_channel::add_noise(rng& r, float* val)
{
  *val += static_cast<transfer_llr_t>(r.gaussian()) * m_s;
}

transfer_llr_t biawgn_channel::add_noise(rng &r, float val)
{
  return static_cast<transfer_llr_t>(val + static_cast<transfer_llr_t>(r.gaussian()) * m_s);
}

static float bsc_capacity(float p)
{
  return 1 + p * (log2(p)) + (1 - p) * (log2(1 - p));
}

// log(cosh(x)), with an approximation for large |x| wherelog(cosh(x)) = |x| - log(2)
static float log_cosh(float x, float range)
{
  float abs_x = abs(x);
  if(abs_x > range) return abs_x - log(static_cast<transfer_llr_t>(2));
  else return log(cosh(x));
}

static float biawgn_capacity(float s, float step, float range)
{
  float c = 0.;
  // if s is small enough, capacity is 1
  if(s < static_cast<float>(0.001)) c = 1.;
  else
  {
    const float inv_s = 1 / s;
    const float sq_inv_s = inv_s * inv_s;
    const float norm_factor =
        step / (log(2.f) * sqrt(2. * M_PI));
    for(float x = -range; x < range; x += step)
    {
      c += exp(-x * x / 2) * (sq_inv_s - log_cosh(x * inv_s + sq_inv_s, range));
    }
    c *= norm_factor;
  }
  return c;
}


