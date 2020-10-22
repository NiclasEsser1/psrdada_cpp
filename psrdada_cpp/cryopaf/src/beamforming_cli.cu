#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuComplex.h>
#include <complex>
#include <random>
#include <cmath>

#include "psrdada_cpp/cryopaf/VoltageBeamformer.cuh"
#include "psrdada_cpp/cryopaf/PowerBeamformer.cuh"
#include "psrdada_cpp/cryopaf/types.cuh"
#include "psrdada_cpp/cuda_utils.hpp"

int main(int argc, char** argv)
{
  // Load & set configuration from cryopaf_conf.hpp
  // psrdada_cpp::cryopaf::bf_config_t _conf;
  // psrdada_cpp::cryopaf::beamforming::CudaBeamformer bf(&_conf);
  //
  //
  // // Set up normal distributed sample and weight generator
  // const float input_level = 32.0f;
  // const double pi = std::acos(-1);
  // std::default_random_engine generator;
  // std::normal_distribution<float> normal_dist(0.0, input_level);
  // std::uniform_real_distribution<float> uniform_dist(0.0, 2*pi);
  //
  // // Calulate memory size for input, weights and output
  // std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  // std::size_t weights_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  // std::size_t output_size = _conf.n_samples * _conf.n_beam * _conf.n_channel;
  // std::size_t required_mem = input_size * sizeof(cuComplex)
  //   + weights_size * sizeof(cuComplex)
  //   + output_size * sizeof(float);
  //
  // std::cout << "Required device memory: " << std::to_string(required_mem/(1024*1024)) << "MiB" << std::endl;
  // std::cout << "Required host memory: " << std::to_string(2*required_mem/(1024*1024)) << "MiB" << std::endl;
  //
  // // Allocate host vectors
  // thrust::host_vector<cuComplex> host_input(input_size, {0, 0});
  // thrust::host_vector<cuComplex> host_weights(weights_size, {0, 0});
  // thrust::host_vector<float> host_output(output_size, 0);
  //
  // // Generate test samples / normal distributed noise for input signal
  // for (size_t i = 0; i < host_input.size(); i++)
  // {
  //     host_input[i].x = normal_dist(generator);
  //     host_input[i].y = normal_dist(generator);
  // }
  // // Build complex weight as C * exp(i * theta).
  // for (size_t i = 0; i < host_weights.size(); i++)
  // {
  //     std::complex<float> awgn = 12.0f * std::exp(std::complex<float>(0.0f, uniform_dist(generator)));
  //     host_weights[i].x = awgn.real();
  //     host_weights[i].y = awgn.imag();
  // }
  //
  // // Allocate device memory & assign test samples
  // // Input and weights are equal for host and device vector
  // thrust::device_vector<cuComplex> dev_input = host_input;
  // thrust::device_vector<cuComplex> dev_weights = host_weights;
  // thrust::device_vector<float> dev_output(output_size);
  //
  // bf.process(dev_input, dev_output, dev_weights);    // launches CUDA kernel
  return 0;
}
