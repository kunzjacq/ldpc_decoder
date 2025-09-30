#pragma once

#include "common.h"
#include "rng.h"

#include <sstream>
#include <string>
#include <fstream>
#include <istream>
#include <iostream>

//#define EXTRA_CHANNELS

#ifdef EXTRA_CHANNELS
constexpr uint32_t max_channel_group_size = 64;
#endif

class noisy_channel
{
public:
  noisy_channel() = default;
  virtual ~noisy_channel() = default;
  virtual void add_noise(rng& , float*) = 0;
  virtual transfer_llr_t add_noise(rng&, float) = 0;
  virtual transfer_llr_t llr(const float) const = 0;
  virtual float capacity() const = 0;
  virtual void description(ostream&) const = 0;
#ifdef EXTRA_CHANNELS
  virtual size_t block_size() const;
#endif
  virtual channelType channel() const = 0;
};

class bsc_channel : public noisy_channel
{
private:
  float m_p; // error probability
  float m_llr_ref;
  float m_capacity;
public:
  bsc_channel(float p);
  void add_noise(rng& r, float *val) override;
  transfer_llr_t add_noise(rng& r, float val) override;
  transfer_llr_t llr(float val) const override;
  float capacity()const override;
  void description(ostream& p_stream) const override;
  float ref_llr() const
  {
    return m_llr_ref;
  }
  channelType channel() const override
  {
    return bsc;
  }
};

class biawgn_channel : public noisy_channel
{
private:
  float m_s; // noise standard deviation (for a modulation ±1)
  float m_snr;
  float m_capacity;
public:
  biawgn_channel(float s);
  void add_noise(rng& r, float *val) override;
  transfer_llr_t add_noise(rng& r, float val) override;
  transfer_llr_t llr(const float val) const override;
  float capacity()const override;
  void description(ostream& p_stream) const override;
  float factor() const 
  {
    return 2 * m_snr;
  }
  channelType channel() const override
  {
    return awgn;
  }
};

#ifdef EXTRA_CHANNELS
llr_t ngauss_capacity(llr_t p_sigma, uint32_t p_n);
class multigauss_channel: public noisy_channel
{
private:
  llr_t m_noise;
  uint32_t m_group_size;
public:
  multigauss_channel(llr_t p_noise, uint32_t p_group_size):
    m_noise(p_noise),
    m_group_size(p_group_size)
  {
    if(p_group_size >= max_channel_group_size)
    {
      m_group_size = 0;
      throw error("Group size too big");
    }
  }
  llr_t llr(const llr_t p_value) const;
  llr_t capacity()const;
  void description(ostream& p_stream) const;
  void add_noise(rng& p_rng, llr_t *pio_data);
  llr_t add_noise(rng& p_rng, llr_t pio_data);
  size_t block_size() const
  {
    return m_group_size;
  }
  channelType channel() const
  {
    return groupGauss;
  }
};

class erasure_channel : public noisy_channel
{
private:
  llr_t m_erasure_probability;
public:
  erasure_channel(llr_t p_erasure_probability):m_erasure_probability(p_erasure_probability){}
  llr_t llr(const llr_t p_value) const;
  llr_t capacity() const;
  void description(ostream& p_stream) const;
  void add_noise(rng& p_rng, llr_t *pio_data);
  llr_t add_noise(rng& p_rng, llr_t pio_data);
  llr_t noise() const
  {
    return m_erasure_probability;
  }
  channelType channel() const
  {
    return erasure;
  }
};
#endif

