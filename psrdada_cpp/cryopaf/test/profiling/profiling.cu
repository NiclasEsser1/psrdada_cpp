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
#include "psrdada_cpp/cryopaf/Pipeline.cuh"

const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;
using namespace std::chrono;

template<typename T>
void profile(PipelineConfig conf, std::size_t iter)
{
  // If template parameter is not of a complex dtype profiling has to be aborted
  if(  !(std::is_same<T,  float2>::value)
    && !(std::is_same<T, __half2>::value))
  {
    BOOST_LOG_TRIVIAL(error) << "ProfilingError: Template type not supported";
    exit(1);
  }
  // Kernel characteristic variables
  dim3 grid;
  dim3 block;

  // Calulate memory size for input, weights and output
  std::size_t input_size = conf.n_samples * conf.n_elements * conf.n_channel * conf.n_pol;
  std::size_t weight_size =  conf.n_beam * conf.n_elements * conf.n_channel * conf.n_pol;
  std::size_t output_size = conf.n_samples * conf.n_beam * conf.n_channel;
  std::size_t required_mem = input_size * sizeof(T)
    + weight_size * sizeof(T)
    + output_size * sizeof(decltype(T::x));
  BOOST_LOG_TRIVIAL(debug) << "Required device memory: " << std::to_string(required_mem / (1024*1024)) << "MiB";

  // Allocate device memory & assign test samples
  // Input and weights are equal for host and device vector
  thrust::device_vector<uint64_t> input_up(input_size / conf.n_pol,0);
  thrust::device_vector<T> input_bf(input_size, {.1, .1});
  thrust::device_vector<T> weights_bf(weight_size, {.1, .1});
  thrust::device_vector<T> output_bf_voltage(output_size * conf.n_pol);
  thrust::device_vector<decltype(T::x)> output_bf_power(output_size / conf.integration, 0);

  // Kernel calls placed below
  for(int i = 0; i < iter; i++)
  {
    // Layout for Beamformer (input: FPTE, weights: FPBE, output: FBT)
    // Stokes I and voltage beamformer have same layout
    grid.x = ceil(conf.n_beam / (double)TILE_SIZE);
    grid.y = ceil(conf.n_samples / (double)TILE_SIZE);
    grid.z = conf.n_channel;
    block.x = TILE_SIZE;
    block.y = TILE_SIZE;
    block.z = 1;
    // Kernel call Stokes I detection beamformer
  	beamformer_power_fpte_fpbe_ftb<<<grid, block>>>(
      thrust::raw_pointer_cast(input_bf.data()),
      thrust::raw_pointer_cast(weights_bf.data()),
      thrust::raw_pointer_cast(output_bf_power.data()),
      conf.n_samples, conf.n_elements, conf.n_beam, conf.integration);
    // Kernel call voltage beamformer
  	beamformer_voltage_fpte_fpbe_fptb<<<grid, block>>>(
      thrust::raw_pointer_cast(input_bf.data()),
      thrust::raw_pointer_cast(weights_bf.data()),
      thrust::raw_pointer_cast(output_bf_voltage.data()),
      conf.n_samples, conf.n_elements, conf.n_beam);

    // Layout for unpacker kernels
    grid.x = conf.n_samples / NSAMP_DF;
    grid.y = conf.n_elements;
    grid.z = conf.n_channel / NCHAN_CHK;
    block.x = NSAMP_DF;
    block.y = 1;
    block.z = NCHAN_CHK;
    // Kernel call to CODIF unpacker (Output format: TFEP)
    unpack_codif_to_fpte<<<grid, block>>>(
      thrust::raw_pointer_cast(input_up.data()),
      thrust::raw_pointer_cast(input_bf.data()));
  }
}


int main(int argc, char** argv)
{
  // Variables to store command line options
  PipelineConfig conf;
  int iter;
  int precision;
  std::string filename;

  // Parse command line
  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
  ("help,h", "Print help messages")
  ("samples", po::value<std::size_t>(&conf.n_samples)->default_value(1024), "Number of samples within one heap")
  ("channels", po::value<std::size_t>(&conf.n_channel)->default_value(256), "Number of channels")
  ("elements", po::value<std::size_t>(&conf.n_elements)->default_value(WARP_SIZE*4), "Number of antennas")
  ("pol", po::value<std::size_t>(&conf.n_pol)->default_value(2), "Polarisation")
  ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(64), "Number of beams")
  ("integration", po::value<std::size_t>(&conf.integration)->default_value(1), "Beamform type:")
  ("id", po::value<int>(&conf.device_id)->default_value(0), "Device id")
  ("iteration", po::value<int>(&iter)->default_value(5), "Iterations to run")
  ("precision", po::value<int>(&precision)->default_value(1), "0 = half; 1 = single")
  ("outputfile", po::value<std::string>(&filename)->default_value(""), "Store profile data to csv file");

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

      if(precision == 0)
      {
        profile<__half2>(conf, iter);
      }
      else if(precision == 1)
      {
        profile<float2>(conf, iter);
      }

  return 0;
}
