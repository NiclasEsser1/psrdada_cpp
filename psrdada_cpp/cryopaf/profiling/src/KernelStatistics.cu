#ifdef KERNEL_STATISTICS_CUH

KernelProfiler::KernelProfiler(ProfileConfig& config,
  cudaStream_t& cuda_stream,
  std::string kernel_name,
  std::size_t complexity,
  std::size_t read_size,
  std::size_t write_size,
  std::size_t input_size)
  : conf(config),
    stream(cuda_stream),
    name(kernel_name),
    compute_complexity(complexity),
    reads(read_size),
    writes(write_size),
    input_sz(input_size)
{
  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));
  CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, conf.device_id));
  peak_mem_bandwidth = prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
}

KernelProfiler::~KernelProfiler()
{
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));
}

void KernelProfiler::measure_start()
{
	CUDA_ERROR_CHECK(cudaEventRecord(start, stream));
}
void KernelProfiler::measure_stop()
{
	CUDA_ERROR_CHECK(cudaEventRecord(stop, stream));
	CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
	CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));
  update(ms);
}

void KernelProfiler::update(float time_ms)
{
  elapsed_time.push_back(time_ms);
  compute_tput.push_back(compute_complexity / (time_ms * 1e6));
  memory_bw.push_back((reads + writes) / (time_ms * 1e6));
  input_bw.push_back(input_sz / (time_ms * 1e6));
  output_bw.push_back(writes / (time_ms * 1e6));
  iterations = elapsed_time.size();
}

void KernelProfiler::finialize()
{
  BOOST_LOG_TRIVIAL(debug) << "Finializing profiling results..";
  if(iterations > 0)
  {
    avg_time = std::accumulate(elapsed_time.begin(), elapsed_time.end(), .0) / iterations;
    max_time = *(std::max_element(elapsed_time.begin(), elapsed_time.end()));
    min_time = *(std::min_element(elapsed_time.begin(), elapsed_time.end()));
    avg_tput = std::accumulate(compute_tput.begin(), compute_tput.end(), .0) / iterations;
    min_tput = *(std::min_element(compute_tput.begin(), compute_tput.end()));
    max_tput = *(std::max_element(compute_tput.begin(), compute_tput.end()));
    avg_m_bw = std::accumulate(memory_bw.begin(), memory_bw.end(), .0) / iterations;
    min_m_bw = *(std::min_element(memory_bw.begin(), memory_bw.end()));
    max_m_bw = *(std::max_element(memory_bw.begin(), memory_bw.end()));
    avg_m_bw_perc = avg_m_bw / peak_mem_bandwidth * 100;
    min_m_bw_perc = min_m_bw / peak_mem_bandwidth * 100;
    max_m_bw_perc = max_m_bw / peak_mem_bandwidth * 100;
    avg_i_bw = std::accumulate(input_bw.begin(), input_bw.end(), .0) / iterations;
    min_i_bw = *(std::min_element(input_bw.begin(), input_bw.end()));
    max_i_bw = *(std::max_element(input_bw.begin(), input_bw.end()));
    avg_o_bw = std::accumulate(output_bw.begin(), output_bw.end(), .0) / iterations;
    min_o_bw = *(std::min_element(output_bw.begin(), output_bw.end()));
    max_o_bw = *(std::max_element(output_bw.begin(), output_bw.end()));
  }
  else
  {
    BOOST_LOG_TRIVIAL(error) << "0 iterations, no kernel was profiled..";
  }
}

void KernelProfiler::export_to_csv(std::string filename)
{
  std::ofstream csv;
  std::ifstream test(filename.c_str());
  BOOST_LOG_TRIVIAL(debug) << "Exporting results to " << filename;

  // If file does not exists we add the header line
  if(!test.good())
  {
    BOOST_LOG_TRIVIAL(debug) << "Writing CSV header";
    csv.open(filename.c_str(), std::ios::out);
    csv << "devicename,"
        << "kernelname,"
        << "samples,"
        << "channels,"
        << "elements,"
        << "beams,"
        << "integration,"
        << "precision,"
        << "reads,"
        << "writes,"
        << "avg_time,"
        << "min_time,"
        << "max_time,"
        << "avg_throughput,"
        << "min_throughput,"
        << "max_throughput,"
        << "avg_bandwidth,"
        << "min_bandwidth,"
        << "max_bandwidth,"
        << "percentage_avg_bandwidth,"
        << "percentage_min_bandwidth,"
        << "percentage_max_bandwidth,"
        << "input_avg_bandwidth,"
        << "input_min_bandwidth,"
        << "input_max_bandwidth,"
        << "output_avg_bandwidth,"
        << "output_min_bandwidth,"
        << "output_max_bandwidth\n";
  }
  else
  {
    BOOST_LOG_TRIVIAL(debug) << "Appending data to CSV";
    csv.open(filename.c_str(), std::ios::app);
  }
  csv << prop.name << ","
      << name << ","
      << conf.n_samples << ","
      << conf.n_channel << ","
      << conf.n_elements << ","
      << conf.n_beam << ","
      << conf.integration << ","
      << conf.precision << ","
      << reads << ","
      << writes << ","
      << avg_time << ","
      << min_time << ","
      << max_time << ","
      << avg_tput << ","
      << min_tput << ","
      << max_tput << ","
      << avg_m_bw << ","
      << min_m_bw << ","
      << max_m_bw << ","
      << avg_m_bw_perc << ","
      << min_m_bw_perc << ","
      << max_m_bw_perc << ","
      << avg_i_bw << ","
      << min_i_bw << ","
      << max_i_bw << ","
      << avg_o_bw << ","
      << min_o_bw << ","
      << max_o_bw << "\n";
  csv.close();
}

#endif
