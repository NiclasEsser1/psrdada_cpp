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




template<typename T>
void BeamformerTester::test(T type)
{
  // Set up normal distributed sample and weight generator
  const float input_level = 32.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);
  std::uniform_real_distribution<float> uni_real(0.0, 2*pi);


  // Calulate memory size for input, weights and output
  std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t weights_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t output_size_voltg = _conf.n_samples * _conf.n_beam * _conf.n_channel * _conf.n_pol;
  std::size_t output_size_power = _conf.n_samples * _conf.n_beam * _conf.n_channel;
  std::size_t required_mem = input_size * sizeof(thrust::complex<T>)
    + weights_size * sizeof(thrust::complex<T>)
    + output_size_voltg * sizeof(thrust::complex<T>)
    + output_size_power * sizeof(T);

  std::cout << "Required device memory: " << std::to_string(required_mem/(1024*1024)) << "MiB" << std::endl;
  std::cout << "Required host memory: " << std::to_string(2*required_mem/(1024*1024)) << "MiB" << std::endl;

  // Allocate host vectors
  thrust::host_vector<thrust::complex<T>> host_input(input_size, {0, 0});
  thrust::host_vector<thrust::complex<T>> host_weights(weights_size, {0, 0});
  thrust::host_vector<thrust::complex<T>> host_output_voltg(output_size_voltg, {0, 0});
  thrust::host_vector<T> host_output_power(output_size_power, 0);

  // Generate test samples / normal distributed noise for input signal
  for (size_t i = 0; i < host_input.size(); i++)
  {
    if(std::is_same<T, short>::value)
    {
      host_input[i] = thrust::complex<T>(__float2half(normal_dist(generator)), __float2half(normal_dist(generator)));
    }else{
      host_input[i] = thrust::complex<T>(normal_dist(generator), normal_dist(generator));
    }
  }
  // Build complex weight as C * exp(i * theta).
  for (size_t i = 0; i < host_weights.size(); i++)
  {
    if(std::is_same<T, short>::value)
    {
      host_weights[i] = 12.0f * std::exp(std::complex<float>(0.0f, __float2half(normal_dist(generator))));
    }else{
      host_weights[i] = 12.0f * std::exp(std::complex<float>(0.0f, normal_dist(generator)));
    }

  }
  // Allocate device memory & assign test samples
  // Input and weights are equal for host and device vector
  thrust::device_vector<thrust::complex<T>> dev_input = host_input;
  thrust::device_vector<thrust::complex<T>> dev_weights = host_weights;
  thrust::device_vector<thrust::complex<T>> dev_output_voltg(output_size_voltg);
  thrust::device_vector<T> dev_output_power(output_size_power);

  /* ----
  *  TEST
  *  ----
  * 1.) Compare CPU and GPU based beamformer. Output is voltage
  * 2.) Compare CPU and GPU based beamformer. Output is power / Stokes I detection
  * 3.) TODO: Full Stokes mode
  */

  // Voltage beamformer
  cpu_process(host_input, host_output_voltg, host_weights);  // launches cpu beamforming
  gpu_process(dev_input, dev_output_voltg, dev_weights);     // launches CUDA kernel
  compare(host_output_voltg, dev_output_voltg);              // compare results of both outputs

  // Power beamformer
  // cpu_process(host_input, host_output_power, host_weights); // launches cpu beamforming
  // gpu_process(dev_input, dev_output_power, dev_weights);    // launches CUDA kernel
  // compare(host_output_power, dev_output_power);             // compare results of both outputs
}
// template void BeamformerTester::test<short>(short type);
template void BeamformerTester::test<float>(float type);
template void BeamformerTester::test<double>(double type);





template <typename T>
void BeamformerTester::compare(
  const thrust::host_vector<T> cpu,
  const thrust::device_vector<T> gpu,
  const float tol)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;

  // Copy CUDA results to host
  thrust::host_vector<float> host_gpu_result = gpu;

  // Check each element of CPU and GPU implementation
  for(std::size_t ii = 0; ii < gpu.size(); ii++)
  {
      ASSERT_TRUE(std::abs(cpu[ii] - host_gpu_result[ii]) <= std::abs(host_gpu_result[ii])*tol/2)
        << "Beamformer with Stokes I: CPU and GPU result is unequal for element " << std::to_string(ii) << std::endl
        << "  CPU result: " << std::to_string(cpu[ii]) << std::endl
        << "  GPU result: " << std::to_string(host_gpu_result[ii]) << std::endl;
  }
}
template void BeamformerTester::compare<float>(
  const thrust::host_vector<float> cpu, const thrust::device_vector<float> gpu, const float tol);
template void BeamformerTester::compare<double>(
  const thrust::host_vector<double> cpu, const thrust::device_vector<double> gpu, const float tol);





template<typename T>
void BeamformerTester::compare(
  const thrust::host_vector<thrust::complex<T>>& cpu,
  const thrust::device_vector<thrust::complex<T>>& gpu)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;

  // Copy CUDA results to host
  thrust::host_vector<thrust::complex<T>> host_gpu_result = gpu;
  thrust::complex<T> max_deviation(0,0);
  int arg_max_deviation = 0;

  // Check each element of CPU and GPU implementation
  for(std::size_t ii = 0; ii < gpu.size(); ii++)
  {
      if(thrust::abs(cpu[ii] - host_gpu_result[ii]) > thrust::abs(max_deviation)){
          arg_max_deviation = ii;
          max_deviation = thrust::abs(cpu[arg_max_deviation] - host_gpu_result[arg_max_deviation]);
      }
      ASSERT_TRUE(thrust::abs(cpu[ii] - host_gpu_result[ii]) <= 1)
        << "Beamformer: CPU and GPU result is unequal for element " << std::to_string(ii) << std::endl
        << "  CPU result: " << std::to_string(cpu[ii].real()) << "+ i*" << std::to_string(cpu[ii].imag()) << std::endl
        << "  GPU result: " << std::to_string(host_gpu_result[ii].real()) << "+ i*" << std::to_string(host_gpu_result[ii].imag()) << std::endl;
  }

  std::cout << "Maximum deviation detected for element " << std::to_string(arg_max_deviation) << std::endl
    << "Deviation abs(cpu[" << std::to_string(arg_max_deviation)
    << "] - gpu[" << std::to_string(arg_max_deviation) << "]) = "
    << std::to_string(max_deviation.real()) << " + i* " << std::to_string(max_deviation.imag()) << std::endl << std::endl;
}
template void BeamformerTester::compare<float>(
  const thrust::host_vector<thrust::complex<float>>& cpu, const thrust::device_vector<thrust::complex<float>>& gpu);
template void BeamformerTester::compare<double>(
  const thrust::host_vector<thrust::complex<double>>& cpu, const thrust::device_vector<thrust::complex<double>>& gpu);





// Voltage mode
template<typename T>
void BeamformerTester::gpu_process(
  thrust::device_vector<thrust::complex<T>>& in,
  thrust::device_vector<thrust::complex<T>>& out,
  thrust::device_vector<thrust::complex<T>>& weights)
{
  CudaBeamformer<T> cu_bf(&_conf);
  cu_bf.process(in, out, weights);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
// template void BeamformerTester::gpu_process<short>(
//   thrust::device_vector<thrust::complex<short>>& in, thrust::device_vector<thrust::complex<short>>& out, thrust::device_vector<thrust::complex<short>>& weights);
template void BeamformerTester::gpu_process<float>(
  thrust::device_vector<thrust::complex<float>>& in, thrust::device_vector<thrust::complex<float>>& out, thrust::device_vector<thrust::complex<float>>& weights);
template void BeamformerTester::gpu_process<double>(
  thrust::device_vector<thrust::complex<double>>& in, thrust::device_vector<thrust::complex<double>>& out, thrust::device_vector<thrust::complex<double>>& weights);






// Power mode / Stokes I dector
template<typename T>
void BeamformerTester::gpu_process(
  thrust::device_vector<thrust::complex<T>>& in,
  thrust::device_vector<T>& out,
  thrust::device_vector<thrust::complex<T>>& weights)
{
  CudaBeamformer<T> cu_bf(&_conf);
  cu_bf.process(in, out, weights);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
// template void BeamformerTester::gpu_process<short>(
//   thrust::device_vector<thrust::complex<short>>& in, thrust::device_vector<short>& out, thrust::device_vector<thrust::complex<short>>& weights);
template void BeamformerTester::gpu_process<float>(
  thrust::device_vector<thrust::complex<float>>& in, thrust::device_vector<float>& out, thrust::device_vector<thrust::complex<float>>& weights);
template void BeamformerTester::gpu_process<double>(
  thrust::device_vector<thrust::complex<double>>& in, thrust::device_vector<double>& out, thrust::device_vector<thrust::complex<double>>& weights);






template <typename T>
void BeamformerTester::cpu_process(
  thrust::host_vector<thrust::complex<T>>& in,
  thrust::host_vector<T>& out,
  thrust::host_vector<thrust::complex<T>>& weights)
{
  const int pt = _conf.n_pol;
  const int fpt = _conf.n_channel * pt;
  const int afpt = _conf.n_antenna * fpt;
  int out_idx;

  for(int t = 0; t < _conf.n_samples; t++)
  {
    for(int b = 0; b < _conf.n_beam; b++)
    {
      for(int a = 0; a < _conf.n_antenna; a++)
      {
        for(int f = 0; f < _conf.n_channel; f++)
        {
          thrust::complex<float> tmp(0,0);
          for(int p = 0; p < _conf.n_pol; p++)
          {
            int in_idx = t * afpt + a * fpt + f * pt + p;
            int weight_idx = b * afpt + a * fpt + f * pt + p;
            tmp += in[in_idx] * weights[weight_idx];
          }
          out_idx = b * _conf.n_samples * _conf.n_channel + t * _conf.n_channel + f;
          out[out_idx] += tmp.real()*tmp.real() + tmp.imag()*tmp.imag();
        }
      }
    }
  }
}
template void BeamformerTester::cpu_process<float>(
  thrust::host_vector<thrust::complex<float>>& in, thrust::host_vector<float>& out, thrust::host_vector<thrust::complex<float>>& weights);
template void BeamformerTester::cpu_process<double>(
  thrust::host_vector<thrust::complex<double>>& in, thrust::host_vector<double>& out, thrust::host_vector<thrust::complex<double>>& weights);






template <typename T>
void BeamformerTester::cpu_process(
  thrust::host_vector<thrust::complex<T>>& in,
  thrust::host_vector<thrust::complex<T>>& out,
  thrust::host_vector<thrust::complex<T>>& weights)
{
  int pt = _conf.n_pol;
  int fpt = _conf.n_channel * pt;
  int afpt = _conf.n_antenna * fpt;
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
            /*
            * INPUT: TAFP(t)
            * WEIGHTS: BFAP
            * OUTPUT: TBF
            */
            if(_conf.bf_type == SIMPLE_BF_TAFPT)
            {
              out_idx = b * _conf.n_samples * fpt + t * fpt + f * pt + p;
              in_idx = t * afpt + a * fpt + f * pt + p;
              weight_idx = b * afpt + a * fpt + f * pt + p;
              out[out_idx] = in[in_idx] * weights[weight_idx];
            }
            /* ! Slightly different output dataproduct !
            * INPUT: TFAP(t)
            * WEIGHTS: BFAP
            * OUTPUT: BFT
            */
            else if(_conf.bf_type == BF_TFAP)
            {
              out_idx = b * _conf.n_samples * _conf.n_channel * _conf.n_pol
                + f * _conf.n_samples * _conf.n_pol + t * _conf.n_pol + p;
              in_idx = t * _conf.n_channel * _conf.n_antenna * _conf.n_pol
                + f * _conf.n_antenna * _conf.n_pol + a * _conf.n_pol + p;
              weight_idx = b * _conf.n_channel * _conf.n_antenna * _conf.n_pol
                + f * _conf.n_antenna * _conf.n_pol + a * _conf.n_pol + p;
              out[out_idx] += in[in_idx] * weights[weight_idx];
            }
          }
        }
      }
    }
  }
}
template void BeamformerTester::cpu_process<float>(
  thrust::host_vector<thrust::complex<float>>& in, thrust::host_vector<thrust::complex<float>>& out, thrust::host_vector<thrust::complex<float>>& weights);
template void BeamformerTester::cpu_process<double>(
  thrust::host_vector<thrust::complex<double>>& in, thrust::host_vector<thrust::complex<double>>& out, thrust::host_vector<thrust::complex<double>>& weights);






/**
* Testing with Google Test Framework
*/
// TEST_P(BeamformerTester, BeamformerHalf){
//   std::cout << std::endl
//     << "----------------------------------------------" << std::endl
//     << "Testing with T=short (half precision (16bit))"  << std::endl
//     << "----------------------------------------------" << std::endl << std::endl;
//   short t = 0.0f;
//   test(t);
// }
TEST_P(BeamformerTester, BeamformerSingle){
  std::cout << std::endl
    << "-----------------------------------------------" << std::endl
    << "Testing with T=float (single precision (32bit))"  << std::endl
    << "-----------------------------------------------" << std::endl << std::endl;
  float t = 0.0f;
  test(t);
}
TEST_P(BeamformerTester, BeamformerDouble){
  std::cout << std::endl
    << "---------------------..........................." << std::endl
    << "Testing with T=double (double precision (64bit))"  << std::endl
    << "---------------------..........................." << std::endl << std::endl;
  double t = 0.0f;
  test(t);
}
INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, BeamformerTester, ::testing::Values(

  // samples | channels | antenna | polarisation | beam | threads | warp_size | beamformer type

  bf_config_t{32, 8, 16, 2, 8, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{2048, 1, 16, 2, 4, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{1024, 64, 32, 1, 32, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{1024, 16, 32, 1, 64, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{1024, 16, 32, 2, 32, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{2048, 15, 32, 2, 33, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{1024, 32, 64, 2, 32, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{1024, 1024, 64, 2, 8, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{4096, 12, 32, 2, 64, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{2048, 5, 64, 2, 10, NTHREAD, WARP_SIZE, BF_TFAP},
  bf_config_t{1024, 2, 64, 2, 2, NTHREAD, WARP_SIZE, BF_TFAP}
));


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test
