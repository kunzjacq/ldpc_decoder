#pragma once

#include "config.h"

#include "common.h"
#include "ldpc_code.h"
#include "channel.h"

#include <vector>
#include <sstream>
#include <iomanip>

class test_report
{
public:
  string code_and_channel_specs;
  uint32_t num_vectors_per_run;
  double ber;
  uint32_t num_runs;
  float avg_iter;
  float iter_time_per_vector;
  uint32_t min_iter;
  uint32_t max_iter;
  uint32_t frame_size;
  uint32_t target_errors;
  double elapsed_time;
  double mbits_processed;
  uint32_t vectors_with_errors;
  uint32_t max_bit_error;
  uint32_t num_bit_errors;
  uint32_t numGroupsWithErrors;
  uint32_t vectors_with_error_above_target;
  stringstream report;

  test_report():
    ber(),
    num_runs(),
    avg_iter(),
    min_iter(-1u),
    max_iter(),
    frame_size(),
    target_errors(),
    elapsed_time(),
    mbits_processed(),
    vectors_with_errors(),
    max_bit_error(),
    num_bit_errors(),
    numGroupsWithErrors(),
    vectors_with_error_above_target(),
    report()
  {}
  void gen_summary();
};


void describe_error_stats(uint32_t p_num_frames_per_test,
    uint32_t p_offset,
    const vector<uint32_t>& p_errors,
    uint32_t p_vec_size,
    ostream& po_description, uint32_t p_log);

void describe_channel(
    const noisy_channel& p_channel,
    ostream &po_description);

void describe_code(
    const ldpc_code& p_code,
    ostream& po_description);

void describe_code_and_channel(
    const ldpc_code& p_code,
    const noisy_channel& p_channel,
    ostream& po_description);

void describe_run(size_t p_num_batches,
    size_t p_num_frames_per_batch,
    ostream &p_iterType);

void describe_error_stats(
    uint32_t p_num_frames_per_test,
    uint32_t p_offset,
    const vector<uint32_t> &p_errors);

