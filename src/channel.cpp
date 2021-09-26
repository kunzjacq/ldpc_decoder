#include "channel.h"

static llr_t bsc_capacity(llr_t p);
static llr_t biawgn_capacity(llr_t s, llr_t step, llr_t range);

bsc_channel::bsc_channel(llr_t p):
  m_p(p),
  m_llr_ref(log(1 - m_p) - log(m_p)),
  m_capacity(bsc_capacity(p))
{}


llr_t bsc_channel::capacity() const
{
  return m_capacity;
}

llr_t bsc_channel::llr(llr_t val) const
{
  // depends on LLR sign convention
  return val > 0 ? m_llr_ref : -m_llr_ref;
}

void bsc_channel::description(ostream &p_stream) const
{
  p_stream << "Binary channel with bit error probability: " << m_p << endl;
}

void bsc_channel::add_noise(rng& r, llr_t* val)
{
  if(r.unit() < m_p) *val *= -1;
}

llr_t bsc_channel::add_noise(rng& r, llr_t val)
{
  if(r.unit() < m_p) val *= -1;
  return(val);
}

biawgn_channel::biawgn_channel(llr_t s): // noise standard deviation
  m_s(s),
  m_snr(1 / (m_s * m_s)),
  m_capacity(biawgn_capacity(m_s, 0.05f, 16.f)){}

llr_t biawgn_channel::capacity() const
{
  return m_capacity;
}

llr_t biawgn_channel::llr(const llr_t val) const
{
  return 2 * m_snr * val;
}

void biawgn_channel::description(ostream &p_stream) const
{
  p_stream << "Binary channel with Gaussian noise of std. deviation " << m_s << "; SNR = " << m_snr << endl;
}

void biawgn_channel::add_noise(rng& r, llr_t* val)
{
  *val += static_cast<llr_t>(r.gaussian()) * m_s;
}

llr_t biawgn_channel::add_noise(rng &r, llr_t val)
{
  return(val + static_cast<llr_t>(r.gaussian()) * m_s);
}

static llr_t bsc_capacity(llr_t p)
{
  return 1 + p * (log2(p)) + (1 - p) * (log2(1 - p));
}

// log(cosh(x)), with an approximation for large |x| where log(cosh(x)) = |x| - log(2)
static llr_t log_cosh(llr_t x, llr_t range)
{
  llr_t abs_x = abs(x);
  if(abs_x > range) return abs_x - log(static_cast<llr_t>(2));
  else return log(cosh(x));
}

static llr_t biawgn_capacity(llr_t s, llr_t step, llr_t range)
{
  llr_t c = 0.;
  // if s is small enough, capacity is 1
  if(s < static_cast<llr_t>(0.001)) c = 1.;
  else
  {
    const llr_t inv_s = 1 / s;
    const llr_t sq_inv_s = inv_s * inv_s;
    const llr_t norm_factor =
        step / (log(2.f) * sqrt(static_cast<llr_t>(2. * M_PI)));
    for(llr_t x = -range; x < range; x += step)
    {
      c += exp(-x * x / 2) * (sq_inv_s - log_cosh(x * inv_s + sq_inv_s, range));
    }
    c *= norm_factor;
  }
  return c;
}
