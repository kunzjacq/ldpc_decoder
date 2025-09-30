#include "test_report.h"

#include <algorithm> // for min, max

void describe_error_stats(
    uint32_t p_num_frames_per_test,
    uint32_t p_offset,
    const vector<uint32_t>& p_errors,
    uint32_t p_vec_size,
    ostream& po_description,
    uint32_t p_log)
{
  if(p_num_frames_per_test > 1)
  {
    uint32_t min_errors = -1u;
    uint32_t max_errors = 0;
    double total_errors = 0;
    double avg_errors = 0;
    for(uint32_t v = 0 ; v < p_num_frames_per_test; v++)
    {
      total_errors += p_errors[v];
      min_errors = min(p_errors[v], min_errors);
      max_errors = max(p_errors[v], max_errors);
    }
    avg_errors = total_errors / p_num_frames_per_test;
    po_description
        << "on vectors " << p_offset << " ... " << p_offset + p_num_frames_per_test - 1 << ":" << endl;
    po_description
        << "  total = " << total_errors << ", average = " << avg_errors << ", min = " << min_errors
        << ", max = " << max_errors << endl;

    if(p_log >= 3)
    {
      for(uint32_t v = 0 ; v < p_num_frames_per_test; v++)
      {
        cout << "errors on vector " << v << ": " << p_errors[v] << "; p = " <<
                float(p_errors[v]) / float(p_vec_size) << endl;
      }
    }
  }
  else
  {
    po_description << "on frame " << p_offset << ": " << p_errors[0] << endl;
  }
}

void describe_code_and_channel(
    const ldpc_code& p_code,
    const noisy_channel& p_channel,
    ostream& po_description)
{
  describe_channel(p_channel, po_description);
  describe_code(p_code, po_description);
  auto eff = rate(p_code) / static_cast<float>(p_channel.capacity()) * 100;
  ios out_format(nullptr);
  out_format.copyfmt(po_description);
  po_description << fixed << setprecision(2);
  po_description << "Code efficiency over channel = rate/channel capacity = " << eff << "%" << endl;
  po_description.copyfmt(out_format);

}

void describe_code(
    const ldpc_code& p_code,
    ostream& po_description)
{
  po_description << "Error-correcting code:" << endl;
  po_description << p_code.n_inputs() << " variables" << endl;
  po_description << p_code.n_outputs() << " parity bits" << endl;
  po_description << p_code.n_erased_inputs() << " erased variables (not sent, but recovered)" << endl;
  po_description << "maximum input bit arity: " << p_code.max_degree_in() << endl;
  po_description << "maximum output/check bit arity: " << p_code.max_degree_out() << endl;  
  po_description << "Rate = " << rate(p_code) << endl;
  po_description << endl;
}

void describe_channel(const noisy_channel& p_channel, ostream& po_description)
{
  po_description << "Channel:" << endl;
  p_channel.description(po_description);
  float channelCapacity = p_channel.capacity();
  po_description << "capacity: " << channelCapacity << " bits/symbol" << endl;
  po_description << endl;
}

void describe_run(
    size_t p_num_batches,
    size_t p_num_frames_per_batch,
    ostream& po_description)
{
  po_description << "Performing a test with " << p_num_batches << " run(s)" << endl;
  po_description << "Number of vectors (or frames) per run: " << p_num_frames_per_batch << endl;
  cout << endl;
}

void test_report::gen_summary()
{
  report << "                                            ***" << endl;
  report << "                                          Summary " << endl << endl;
  report << "* Channel and code description" << endl << endl;
  report << code_and_channel_specs;
  report << endl << endl;

  report << "* Test result" << endl;
  report << endl;

  const size_t num_bits_per_test =
      static_cast<size_t>(frame_size) * static_cast<size_t>(num_vectors_per_run);
  size_t bits_processed = size_t(num_runs) * num_bits_per_test;
  ber = double(num_bit_errors) / (double(bits_processed));
  mbits_processed = double(bits_processed >> 20);
  uint32_t frames_decoded = num_runs * num_vectors_per_run;

  report <<   "# of frames decoded:              " << frames_decoded << endl;
  report <<   "Frame size:                       " << frame_size << " bits" << endl;
  report <<   "Total # of errors:                " << num_bit_errors << endl;
  report <<   "Bit error rate (BER):             " << ber << endl;
  report <<   "Maximum # of errors / frame:      " << max_bit_error << endl;
  if(target_errors > 0)
  {
    report << "Frames with more than " << target_errors <<
              " errors:  " << vectors_with_error_above_target << " (corresponding FER: " <<
              double(vectors_with_error_above_target) / double(frames_decoded)  << ")" << endl;
  }  report <<   "Frames with at least one error:   " << vectors_with_errors <<
              " (corresponding FER: " << double(vectors_with_errors) / double(frames_decoded) <<
              ")" << endl;
  report << endl;
  report <<   "Mbits processed:                  "                << mbits_processed << endl;
  report <<   "Elapsed system time:              "                << elapsed_time << " sec." << endl;
  report <<   "Throughput including transfers and finish: "       << mbits_processed / elapsed_time << " Mbits/sec." << endl;
  report <<   "Max/min/average number of iterations per vector: " << max_iter << "/" << min_iter << "/" << avg_iter << endl;
  report <<   "Iteration time per vector (i.e. iteration time / vector batch size): " << iter_time_per_vector << " sec" << endl;
  report <<   "Decoding throughput: " <<  frame_size / (avg_iter * iter_time_per_vector * 1048576.) << " Mbits/sec." << endl;
  report << endl;
}




