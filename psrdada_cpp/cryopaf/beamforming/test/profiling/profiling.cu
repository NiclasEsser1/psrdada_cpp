#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <cuda.h>
#include <random>
#include <cmath>

#include "psrdada_cpp/cryopaf/beamforming/cu_beamformer.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include "boost/program_options.hpp"
#include "psrdada_cpp/cryopaf/types.cuh"

using namespace psrdada_cpp;
using namespace psrdada_cpp::cryopaf;

namespace
{
  const size_t ERROR_IN_COMMAND_LINE = 1;
  const size_t SUCCESS = 0;
  const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace




int main(int argc, char** argv)
{
    try
    {
        // Variables to store command line options
        cryopaf::bf_config_t conf;
        std::size_t iter;

        // Parse command line
        namespace po = boost::program_options;
        po::options_description desc("Options");
        desc.add_options()
        ("help,h", "Print help messages")
        ("samples", po::value<std::size_t>(&conf.n_samples)->default_value(1024), "Number of samples within one heap")
        ("channels", po::value<std::size_t>(&conf.n_channel)->default_value(256), "Number of channels")
        ("antennas", po::value<std::size_t>(&conf.n_antenna)->default_value(WARP_SIZE*2), "Number of antennas")
        ("pol", po::value<std::size_t>(&conf.n_pol)->default_value(2), "Polarisation")
        ("beams", po::value<std::size_t>(&conf.n_beam)->default_value(16), "Number of beams")
        ("type", po::value<std::size_t>(&conf.bf_type)->default_value(0), "Beamform type:")
        ("iteration", po::value<std::size_t>(&iter)->default_value(1), "Beamform type:");

        po::variables_map vm;
        try
        {
            po::store(po::parse_command_line(argc, argv, desc), vm);
            if ( vm.count("help")  )
            {
                std::cout << "Cryopaf -- The CryoPAF beamformer implementations" << std::endl
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

        // Set up normal distributed sample and weight generator
        const float input_level = 4.0f;
        const double pi = std::acos(-1);
        std::default_random_engine generator;
        std::normal_distribution<float> normal_dist(0.0, input_level);
        std::uniform_real_distribution<float> uniform_dist(0.0, 2*pi);

        // Calulate memory size for input, weights and output
        std::size_t input_size = conf.n_samples * conf.n_antenna * conf.n_channel * conf.n_pol;
        std::size_t weights_size =  conf.n_beam * conf.n_antenna * conf.n_channel * conf.n_pol;
        std::size_t output_size = conf.n_samples * conf.n_beam * conf.n_channel;
        std::size_t required_mem = input_size * sizeof(float2)
            + weights_size * sizeof(float2)
            + output_size * sizeof(float2) * conf.n_pol
            + output_size * sizeof(float2);

        std::cout << "Required device memory: " << std::to_string(required_mem/(1024*1024)) << "MiB" << std::endl;
        std::cout << "Required host memory: " << std::to_string(2*required_mem/(1024*1024)) << "MiB" << std::endl;

        // Allocate host vectors
        thrust::host_vector<float2> host_input(input_size);
        thrust::host_vector<float2> host_weights(weights_size);
        thrust::host_vector<float2> host_output(output_size * conf.n_pol);
        thrust::host_vector<float> host_output_stokesI(output_size);

        // Generate test samples / normal distributed noise for input signal
        for (size_t i = 0; i < host_input.size(); i++)
        {
            host_input[i] = {normal_dist(generator), normal_dist(generator)};
        }
        // Build complex weight as C * exp(i * theta).
        for (size_t i = 0; i < host_weights.size(); i++)
        {
            host_weights[i] = {normal_dist(generator), normal_dist(generator)};
        }

        // Allocate device memory & assign test samples
        // Input and weights are equal for host and device vector
        thrust::device_vector<float2> dev_input = host_input;
        thrust::device_vector<float2> dev_weights = host_weights;
        thrust::device_vector<float2> dev_output(output_size * conf.n_pol);

        // thrust::device_vector<thrust::complex<__half>> dev_input_fp16 = __float2half(host_input);
        // thrust::device_vector<thrust::complex<__half>> dev_weights_fp16 = __float2half(host_weights);
        // thrust::device_vector<thrust::complex<__half>> dev_output_fp16(output_size * conf.n_pol);

        thrust::device_vector<float> dev_output_stokesI(output_size);

        psrdada_cpp::cryopaf::beamforming::CudaBeamformer<float2> bf(&conf);

        for(int i = 0; i < iter; i++)
        {
            bf.process(dev_input, dev_output, dev_weights);    // launches CUDA kernel
            CUDA_ERROR_CHECK(cudaDeviceSynchronize());
            // bf.process(dev_input, dev_output_stokesI, dev_weights);    // launches CUDA kernel
            // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
        }
    }
    catch(std::exception& e)
    {
        std::cerr << "Unhandled Exception reached the top of main: "
            << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
  return 0;
}
