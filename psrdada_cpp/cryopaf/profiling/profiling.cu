#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <cuda.h>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/multilog.hpp"
#include "psrdada_cpp/cryopaf/Unpacker.cuh"
#include "psrdada_cpp/cryopaf/Beamformer.cuh"
#include "psrdada_cpp/cryopaf/profiling/KernelStatistics.cuh"

const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;
using namespace std::chrono;

template<typename T>
void profile(ProfileConfig conf, std::size_t iter)
{
  // If template parameter is not of a complex dtype profiling has to be aborted
  if(  !(std::is_same<T,  float2>::value)
    && !(std::is_same<T, __half2>::value))
  {
    BOOST_LOG_TRIVIAL(error) << "ProfilingError: Template type not supported";
    exit(1);
  }
  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
  // Instantiate processor objects
  Beamformer<T> beamformer(stream,
    conf.n_samples,
    conf.n_channel,
    conf.n_elements,
    conf.n_beam,
    conf.integration);

  Unpacker<T> unpacker(stream,
		conf.n_samples,
		conf.n_channel,
		conf.n_elements,
		conf.protocol);

  // Calulate memory size for input, weights and output
  std::size_t input_size = conf.n_samples
    * conf.n_elements
    * conf.n_channel
    * conf.n_pol;
  std::size_t weight_size = conf.n_beam
    * conf.n_elements
    * conf.n_channel
    * conf.n_pol;
  std::size_t output_size = conf.n_samples
    * conf.n_beam
    * conf.n_channel;
  std::size_t required_mem = input_size * sizeof(T)
    + weight_size * sizeof(T)
    + output_size * sizeof(decltype(T::x));
  BOOST_LOG_TRIVIAL(debug) << "Required device memory: " << std::to_string(required_mem / (1024*1024)) << "MiB";

  // Allocate device memory
  thrust::device_vector<char> input_up(input_size * unpacker.sample_size(),0);
  thrust::device_vector<T> input_bf(input_size, {.1, .1});
  thrust::device_vector<T> weights_bf(weight_size, {.1, .1});
  thrust::device_vector<T> output_bf_voltage(output_size * conf.n_pol);
  thrust::device_vector<decltype(T::x)> output_bf_power(output_size / conf.integration, 0);

  // Calculation of compute complexity
  std::size_t n = conf.n_beam
    * conf.n_channel
    * conf.n_elements
    * conf.n_pol
    * conf.n_samples;
  std::size_t complexity_bf_vol = 8 * n;
  std::size_t complexity_bf_pow = 8 * n + 4 * n /** + accumulation **/;
  std::size_t complexity_unpack = n / conf.n_beam;

  // Create KernelStatistics object for each kernel
  KernelProfiler voltage_profiler(conf, stream,
    "Voltage",
    complexity_bf_vol,
    (input_bf.size() + weights_bf.size()) * sizeof(T),
    output_bf_voltage.size() * sizeof(T),
    input_bf.size() * sizeof(T));

  KernelProfiler power_profiler(conf, stream,
    "StokesI",
    complexity_bf_pow,
    (input_bf.size() + weights_bf.size()) * sizeof(T),
    output_bf_power.size() * sizeof(decltype(T::x)),
    input_bf.size() * sizeof(T));

  KernelProfiler unpack_profiler(conf, stream,
    "Unpacker",
    complexity_unpack,
    input_up.size() * sizeof(uint64_t),
    input_bf.size() * sizeof(T),
    input_up.size() * sizeof(uint64_t));

  // Run all used kernels i-times
  for(int i = 0; i < iter; i++)
  {
    // Call Stokes I detection beamformer kernel
    power_profiler.measure_start();
  	beamformer.process(
      thrust::raw_pointer_cast(input_bf.data()),
      thrust::raw_pointer_cast(weights_bf.data()),
      thrust::raw_pointer_cast(output_bf_power.data()));
    power_profiler.measure_stop();
    // Call to voltage beamformer kernel
    voltage_profiler.measure_start();
  	beamformer.process(
      thrust::raw_pointer_cast(input_bf.data()),
      thrust::raw_pointer_cast(weights_bf.data()),
      thrust::raw_pointer_cast(output_bf_voltage.data()));
    voltage_profiler.measure_stop();
    // Call to unpacking kernel
    unpack_profiler.measure_start();
    unpacker.unpack(
      thrust::raw_pointer_cast(input_up.data()),
      thrust::raw_pointer_cast(input_bf.data()));
    unpack_profiler.measure_stop();
  }
  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));

  power_profiler.finialize();
  voltage_profiler.finialize();
  unpack_profiler.finialize();
  power_profiler.export_to_csv(conf.out_dir + "power_kernel.csv");
  voltage_profiler.export_to_csv(conf.out_dir + "voltage_kernel.csv");
  unpack_profiler.export_to_csv(conf.out_dir + "unpacker_kernel.csv");
}


int main(int argc, char** argv)
{
  // Variables to store command line options
  ProfileConfig conf;
  int iter;
  std::string precision;
  std::string filename;

  // Parse command line
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()

  ("help,h", "Print help messages")
  ("samples", po::value<std::size_t>(&conf.n_samples)->default_value(1024), "Number of samples within one batch")
  ("channels", po::value<std::size_t>(&conf.n_channel)->default_value(14), "Number of channels")
  ("elements", po::value<std::size_t>(&conf.n_elements)->default_value(WARP_SIZE*4), "Number of elements")
  ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(64), "Number of beams")
  ("integration", po::value<std::size_t>(&conf.integration)->default_value(1), "Integration interval; must be multiple 2^n and smaller 32")
  ("device", po::value<int>(&conf.device_id)->default_value(0), "ID of GPU device")
  ("protocol", po::value<std::string>(&conf.protocol)->default_value("codif"), "Protocol of input data; supported protocol 'codif'")
  ("iteration", po::value<int>(&iter)->default_value(5), "Iterations to run")
  ("precision", po::value<std::string>(&conf.precision)->default_value("half"), "Compute type of GEMM operation; supported precisions 'half' and 'single'")
  ("outdir", po::value<std::string>(&conf.out_dir)->default_value("Results/"), "Output directory to store csv files");

  po::variables_map vm;
  try
  {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if ( vm.count("help")  )
      {
          std::cout << "Beamform Profiling" << std::endl
          << desc << std::endl;
          return SUCCESS;
      }
      po::notify(vm);
  }
  catch(po::error& e)
  {
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return ERROR_IN_COMMAND_LINE;
  }

  if(conf.precision == "half")
  {
    profile<__half2>(conf, iter);
  }
  else if(conf.precision == "single")
  {
    profile<float2>(conf, iter);
  }
  else
  {
    BOOST_LOG_TRIVIAL(error) << "Compute type " << precision << " not implemented";
  }

  return 0;
}
