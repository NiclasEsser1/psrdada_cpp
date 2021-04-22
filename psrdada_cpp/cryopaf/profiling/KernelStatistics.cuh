#ifndef KERNEL_STATISTICS_CUH
#define KERNEL_STATISTICS_CUH

#include <fstream>
#include <cuda.h>
#include "psrdada_cpp/cuda_utils.hpp"


struct ProfileConfig{
   int device_id;
   std::size_t n_samples;
   std::size_t n_channel;
   std::size_t n_elements;
   std::size_t n_beam;
   std::size_t integration;
   std::string precision;
   std::string protocol;
   std::string out_dir;
   const std::size_t n_pol = 2;
 };

class KernelProfiler
{
public:
  KernelProfiler(ProfileConfig& config,
    cudaStream_t& cuda_stream,
    std::string kernel_name,
    std::size_t complexity,
    std::size_t read_size,
    std::size_t write_size,
    std::size_t input_size);
  ~KernelProfiler();
  void measure_start();
  void measure_stop();
  void update(float time_ms);
  void finialize();
  void export_to_csv(std::string filename);

private:
  ProfileConfig& conf;
  cudaStream_t& stream;
  cudaDeviceProp prop;

  std::string name;
  std::string head_line;

  cudaEvent_t start, stop;
  std::vector<float> elapsed_time;
  std::vector<float> compute_tput;
  std::vector<float> memory_bw;
  std::vector<float> input_bw;
  std::vector<float> output_bw;
  float peak_mem_bandwidth;
  float ms = 0;
  float avg_time = 0;
  float min_time = 0;
  float max_time = 0;
  float avg_tput = 0;
  float min_tput = 0;
  float max_tput = 0;
  float avg_m_bw = 0;
  float min_m_bw = 0;
  float max_m_bw = 0;
  float avg_m_bw_perc = 0;
  float min_m_bw_perc = 0;
  float max_m_bw_perc = 0;
  float avg_i_bw = 0;
  float min_i_bw = 0;
  float max_i_bw = 0;
  float avg_o_bw = 0;
  float min_o_bw = 0;
  float max_o_bw = 0;

  std::size_t iterations = 0;
  std::size_t reads;
  std::size_t writes;
  std::size_t input_sz;
  std::size_t compute_complexity;
};

#include "psrdada_cpp/cryopaf/profiling/src/KernelStatistics.cu"

#endif
