#include "psrdada_cpp/cryopaf/beamforming/test/beamformer_tester.cuh"


namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{
namespace test{


BeamformerTester::BeamformerTester()
  : ::testing::TestWithParam<bf_config_t>()
  , _conf(GetParam()) // GetParam() is inherited from base TestWithParam
  , _stream(0)
{
}

void BeamformerTester::SetUp()
{
    //CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}


void BeamformerTester::TearDown()
{
    //CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}


void BeamformerTester::compare(thrust::host_vector<float> cpu,
  thrust::device_vector<float> gpu, float tol)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;
  // Copy CUDA results to host
  thrust::host_vector<float> host_gpu_result = gpu;
  float *h_out = thrust::raw_pointer_cast(cpu.data());
  float *g_out = thrust::raw_pointer_cast(host_gpu_result.data());

  // Check each element of CPU and GPU implementation
  for(std::size_t ii = 0; ii < gpu.size(); ii++)
  {
      //std::cout << "Tolerate: " << std::to_string(std::abs(host_gpu_result[ii])*tol/2) << "Sample: " << std::to_string(ii) << std::endl;
      ASSERT_TRUE(std::abs(cpu[ii] - host_gpu_result[ii]) <= std::abs(host_gpu_result[ii])*tol/2)
      << "Beamformer with Stokes I: CPU and GPU result is unequal for element " << std::to_string(ii) << std::endl
      << "  CPU result: " << std::to_string(cpu[ii]) << std::endl
      << "  GPU result: " << std::to_string(host_gpu_result[ii]) << std::endl;
  }
}

void BeamformerTester::compare(thrust::host_vector<cuComplex> cpu,
  thrust::device_vector<cuComplex> gpu)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;
  // Copy CUDA results to host
  thrust::host_vector<cuComplex> host_gpu_result = gpu;
  cuComplex *h_out = thrust::raw_pointer_cast(cpu.data());
  cuComplex *g_out = thrust::raw_pointer_cast(host_gpu_result.data());
  cuComplex max_deviation = {0,0};
  int arg_max_deviation;

  // Check each element of CPU and GPU implementation
  for(std::size_t ii = 0; ii < gpu.size(); ii++)
  {
      if(std::abs(cpu[ii].x - host_gpu_result[ii].x) > max_deviation.x
        && std::abs(cpu[ii].y - host_gpu_result[ii].y) > max_deviation.y){
          arg_max_deviation = ii;
          max_deviation.y = std::abs(cpu[arg_max_deviation].y - host_gpu_result[arg_max_deviation].y);
          max_deviation.x = std::abs(cpu[arg_max_deviation].x - host_gpu_result[arg_max_deviation].x);
      }

      ASSERT_TRUE(std::abs(cpu[ii].x - host_gpu_result[ii].x) <= 1 && std::abs(cpu[ii].y - host_gpu_result[ii].y) <= 1)
      << "Beamformer: CPU and GPU result is unequal for element " << std::to_string(ii) << std::endl
      << "  CPU result: " << std::to_string(cpu[ii].x) << "+ i*" << std::to_string(cpu[ii].y) << std::endl
      << "  GPU result: " << std::to_string(host_gpu_result[ii].x) << "+ i*" << std::to_string(host_gpu_result[ii].y) << std::endl;
  }

  std::cout << "Maximum deviation detected for element " << std::to_string(arg_max_deviation) << std::endl
  << "Deviation abs(cpu[" << std::to_string(arg_max_deviation)
  << "] - gpu[" << std::to_string(arg_max_deviation) << "]) = "
  << std::to_string(max_deviation.x) << " + i* " << std::to_string(max_deviation.y) << std::endl << std::endl;
}

template<typename T, typename U>
void BeamformerTester::gpu_process(const thrust::device_vector<T>& in,
  thrust::device_vector<U>& out,
  const thrust::device_vector<T>& weights)
{
  CudaBeamformer cu_bf(&_conf);
  cu_bf.process(in, out, weights);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
template void BeamformerTester::gpu_process<cuComplex, float>
  (const thrust::device_vector<cuComplex>& in, thrust::device_vector<float>& out, const thrust::device_vector<cuComplex>& weights);
template void BeamformerTester::gpu_process<cuComplex, cuComplex>
  (const thrust::device_vector<cuComplex>& in, thrust::device_vector<cuComplex>& out, const thrust::device_vector<cuComplex>& weights);



void BeamformerTester::cpu_process(thrust::host_vector<cuComplex>& in,
  thrust::host_vector<float>& out,
  thrust::host_vector<cuComplex>& weights)
{
  const int pt = _conf.n_pol;
  const int fpt = _conf.n_channel * pt;
  const int afpt = _conf.n_antenna * fpt;
  float real, imag;
  int out_idx;

  for(int t = 0; t < _conf.n_samples; t++)
  {
    for(int b = 0; b < _conf.n_beam; b++)
    {
      for(int a = 0; a < _conf.n_antenna; a++)
      {
        for(int f = 0; f < _conf.n_channel; f++)
        {
          real = 0; imag = 0;
          for(int p = 0; p < _conf.n_pol; p++)
          {
            int in_idx = t * afpt + a * fpt + f * pt + p;
            int weight_idx = b * afpt + a * fpt + f * pt + p;
            real += in[in_idx].x * weights[weight_idx].x
              - in[in_idx].y * weights[weight_idx].y;
            imag += in[in_idx].x * weights[weight_idx].y
              + in[in_idx].y * weights[weight_idx].x;
          }
          out_idx = b * _conf.n_samples * _conf.n_channel + t * _conf.n_channel + f;
          out[out_idx] += real*real + imag*imag;
        }
      }
    }
  }
}



void BeamformerTester::cpu_process(thrust::host_vector<cuComplex>& in,
  thrust::host_vector<cuComplex>& out,
  thrust::host_vector<cuComplex>& weights)
{
  const int pt = _conf.n_pol;
  const int fpt = _conf.n_channel * pt;
  const int afpt = _conf.n_antenna * fpt;
  int out_idx, in_idx, weight_idx;

  for(int t = 0; t < _conf.n_samples; t++)
  {
    for(int b = 0; b < _conf.n_beam; b++)
    {
      for(int a = 0; a < _conf.n_antenna; a++)
      {
        for(int f = 0; f < _conf.n_channel; f++)
        {
          for(int p = 0; p < _conf.n_pol; p++)
          {
            out_idx = b * _conf.n_samples * fpt + t * fpt + f * pt + p;
            in_idx = t * afpt + a * fpt + f * pt + p;
            weight_idx = b * afpt + a * fpt + f * pt + p;
            out[out_idx].x += in[in_idx].x * weights[weight_idx].x
              - in[in_idx].y * weights[weight_idx].y;
            out[out_idx].y += in[in_idx].x * weights[weight_idx].y
              + in[in_idx].y * weights[weight_idx].x;
          }
        }
      }
    }
  }
}


TEST_P(BeamformerTester, BeamformerWithoutStokesIdetection){
  // Retrive configuration setting for test case
  bf_config_t _conf = GetParam();

  // Set up normal distributed sample and weight generator
  const float input_level = 32.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);
  std::uniform_real_distribution<float> uni_real(0.0, 2*pi);


  // Calulate memory size for input, weights and output
  std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t weights_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t output_size = _conf.n_samples * _conf.n_beam * _conf.n_channel * _conf.n_pol;
  std::size_t required_mem = input_size * sizeof(cuComplex)
    + weights_size * sizeof(cuComplex)
    + output_size * sizeof(cuComplex);

  std::cout << "Required device memory: " << std::to_string(required_mem/(1024*1024)) << "MiB" << std::endl;
  std::cout << "Required host memory: " << std::to_string(2*required_mem/(1024*1024)) << "MiB" << std::endl;

  // Allocate host vectors
  thrust::host_vector<cuComplex> host_input(input_size, {0, 0});
  thrust::host_vector<cuComplex> host_weights(weights_size, {0, 0});
  thrust::host_vector<cuComplex> host_output(output_size, {0, 0});

  // Generate test samples / normal distributed noise for input signal
  for (size_t i = 0; i < host_input.size(); i++)
  {
      host_input[i].x = normal_dist(generator);
      host_input[i].y = normal_dist(generator);
  }
  // Build complex weight as C * exp(i * theta).
  for (size_t i = 0; i < host_weights.size(); i++)
  {
      std::complex<float> awgn = 12.0f * std::exp(std::complex<float>(0.0f, uni_real(generator)));
      host_weights[i].x = awgn.real();
      host_weights[i].y = awgn.imag();
  }

  // Allocate device memory & assign test samples
  // Input and weights are equal for host and device vector
  thrust::device_vector<cuComplex> dev_input = host_input;
  thrust::device_vector<cuComplex> dev_weights = host_weights;
  thrust::device_vector<cuComplex> dev_output(output_size);

  // Test beamforming by comparing results of a single threaded CPU implementation
  // and CUDA kernel.
  cpu_process(host_input, host_output, host_weights); // launches cpu beamforming
  gpu_process(dev_input, dev_output, dev_weights);    // launches CUDA kernel
  compare(host_output, dev_output);                   // compare results of both outputs
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


TEST_P(BeamformerTester, BeamformerWithStokesIdetection){
  // Retrive configuration setting for test case
  bf_config_t _conf = GetParam();

  const float input_level = 32.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);
  std::uniform_real_distribution<float> uni_real(0.0, 2*pi);

  // Calulate memory size for input, weights and output
  std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t weights_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t output_size = _conf.n_samples * _conf.n_beam * _conf.n_channel;
  std::size_t required_mem = input_size * sizeof(cuComplex)
    + weights_size * sizeof(cuComplex)
    + output_size * sizeof(float);

  std::cout << "Required device memory: " << std::to_string(required_mem/(1024*1024)) << "MiB" << std::endl;
  std::cout << "Required host memory: " << std::to_string(2*required_mem/(1024*1024)) << "MiB" << std::endl;

  // Allocate host vectors
  thrust::host_vector<cuComplex> host_input(input_size, {0, 0});
  thrust::host_vector<cuComplex> host_weights(weights_size, {0, 0});
  thrust::host_vector<float> host_output(output_size, 0);

  // Generate test samples / normal distributed noise for input signal
  for (size_t i = 0; i < host_input.size(); i++)
  {
      host_input[i].x = normal_dist(generator);
      host_input[i].y = normal_dist(generator);
  }
  // Build complex weight as C * exp(i * theta).
  for (size_t i = 0; i < host_weights.size(); i++)
  {
      std::complex<float> awgn = 12.0f * std::exp(std::complex<float>(0.0f, uni_real(generator)));
      host_weights[i].x = awgn.real();
      host_weights[i].y = awgn.imag();
  }

  // Allocate device memory & assign test samples
  // Input and weights are equal for host and device vector
  thrust::device_vector<cuComplex> dev_input = host_input;
  thrust::device_vector<cuComplex> dev_weights = host_weights;
  thrust::device_vector<float> dev_output(output_size);


  // Test beamforming by comparing results of a single threaded CPU implementation
  // and CUDA kernel.
  cpu_process(host_input, host_output, host_weights); // launches cpu beamforming
  gpu_process(dev_input, dev_output, dev_weights);    // launches CUDA kernel
  compare(host_output, dev_output);                   // compare results of both outputs
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, BeamformerTester, ::testing::Values(

  // samples | channels | antenna | polarisation | beam | threads | warp_size | beamformer type

  bf_config_t{1024, 16, 32, 2, 16, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{2048, 8, 1, 2, 4, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{16, 64, 163, 2, 32, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{512, 16, 288, 2, 64, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{1024, 16, 48, 2, 32, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{2048, 16, 16, 1, 32, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{4096, 8, 8, 2, 32, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{1024, 1024, 8, 2, 8, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{64, 12, 144, 3, 64, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{64, 8, 8, 2, 128, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT},
  bf_config_t{4096, 512, 188, 2, 128, NTHREAD, WARP_SIZE, SIMPLE_BF_TAFPT}
));


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test
