#pragma once

#include "common.h"
#include "rng.h"

#include <sstream>
#include <string>
#include <fstream>
#include <istream>
#include <iostream>

class noisy_channel
{
public:
  noisy_channel() = default;
  virtual ~noisy_channel() = default;
  virtual void add_noise(rng& , llr_t*) = 0;
  virtual llr_t add_noise(rng&, llr_t) = 0;
  virtual llr_t llr(const llr_t) const = 0;
  virtual llr_t capacity() const = 0;
  virtual void description(ostream&) const = 0;
  virtual channelType channel() const = 0;
};

class bsc_channel : public noisy_channel
{
private:
  llr_t m_p; // error probability
  llr_t m_llr_ref;
  llr_t m_capacity;
public:
  bsc_channel(llr_t p);
  void add_noise(rng& r, llr_t *val);
  llr_t add_noise(rng& r, llr_t val);
  llr_t llr(llr_t val) const;
  llr_t capacity()const;
  void description(ostream& p_stream) const;
  llr_t ref_llr() const
  {
    return m_llr_ref;
  }
  channelType channel() const
  {
    return bsc;
  }
};

class biawgn_channel : public noisy_channel
{
private:
  llr_t m_s; // noise standard deviation (for a modulation ±1)
  llr_t m_snr;
  llr_t m_capacity;
public:
  biawgn_channel(llr_t s);
  void add_noise(rng& r, llr_t *val);
  llr_t add_noise(rng& r, llr_t val);
  llr_t llr(const llr_t val) const;
  llr_t capacity()const;
  void description(ostream& p_stream) const;
  llr_t factor() const
  {
    return 2 * m_snr;
  }
  channelType channel() const
  {
    return awgn;
  }
};
