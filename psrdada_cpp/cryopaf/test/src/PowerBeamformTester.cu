#ifdef POWERBEAMFORMER_TESTER_H_

namespace psrdada_cpp{
namespace cryopaf{
namespace beamforming{
namespace test{




PowerBeamformTester::PowerBeamformTester()
  : ::testing::TestWithParam<bf_config_t>()
  , _conf(GetParam()) // GetParam() is inherited from base TestWithParam
  , _stream(0)
{
  // std::cout << "Creating instance of PowerBeamformTester" << std::endl;
}


PowerBeamformTester::~PowerBeamformTester()
{
  // std::cout << "Destroying instance of PowerBeamformTester" << std::endl;
}


void PowerBeamformTester::SetUp()
{
    //CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}


void PowerBeamformTester::TearDown()
{
    //CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}


template<typename T, typename U>
void PowerBeamformTester::test(int device_id)
{
  // Set up normal distributed sample and weight generator
  const float input_level = 2.0f;
  const double pi = std::acos(-1);
  std::default_random_engine generator;
  std::normal_distribution<float> normal_dist(0.0, input_level);
	id = device_id;

	CUDA_ERROR_CHECK(cudaSetDevice(id));

  // Calulate memory size for input, weights and output
  std::size_t input_size = _conf.n_samples * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t weight_size =  _conf.n_beam * _conf.n_antenna * _conf.n_channel * _conf.n_pol;
  std::size_t output_size = _conf.n_samples / _conf.interval * _conf.n_beam * _conf.n_channel;
  std::size_t required_mem = input_size * sizeof(T)
    + weight_size * sizeof(T)
    + output_size * sizeof(U);

  std::cout << "Required device memory: " << std::to_string(required_mem / (1024*1024)) << "MiB" << std::endl;
  std::cout << "Required host memory: " << std::to_string((required_mem
		+ (input_size + weight_size) * sizeof(T)/sizeof(float2)) / (1024*1024)) << "MiB" << std::endl;

  // Allocate host vectors
	std::cout << "Allocating host memory... " << std::flush;
	thrust::host_vector<U> host_output(output_size, 0);
	thrust::host_vector<T> host_input(input_size, {0, 0});
	thrust::host_vector<T> host_weights(weight_size, {0, 0});
	std::cout << "done" << std::endl;

  // Generate test samples / normal distributed noise for input signal
	std::cout << "Generating test samples... " << std::flush;
	for (size_t i = 0; i < host_input.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
        host_input[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
      else
				host_input[i] = {normal_dist(generator), normal_dist(generator)};
	}
	std::cout << "done" << std::endl;

	// Generate test weights
	std::cout << "Generated test weights... " << std::flush;
  for (size_t i = 0; i < host_weights.size(); i++)
  {
      if constexpr (std::is_same<T, __half2>::value)
        host_weights[i] = __float22half2_rn({normal_dist(generator), normal_dist(generator)});
			else
				host_weights[i] = {normal_dist(generator), normal_dist(generator)};
  }
	std::cout << "done" << std::endl;

  // Allocate device memory & assign test samples
	std::cout << "Allocating and transfering to device memory... " << std::flush;
	thrust::device_vector<T> dev_input = host_input;
  	thrust::device_vector<T> dev_weights = host_weights;
  	thrust::device_vector<U> dev_output = host_output;
	std::cout << "done" << std::endl;

	cudaEvent_t start, stop;
	float ms = 0;

	CUDA_ERROR_CHECK(cudaEventCreate(&start));
	CUDA_ERROR_CHECK(cudaEventCreate(&stop));

	// launches CUDA kernel
	CUDA_ERROR_CHECK(cudaEventRecord(start));
  	gpu_process(dev_input, dev_output, dev_weights);
	CUDA_ERROR_CHECK(cudaEventRecord(stop));
	CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
	CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));
	std::cout << "Elapsed time for gpu_process(): " << std::to_string(ms) << " ms" << std::endl;

	// launches cpu beamforming
	CUDA_ERROR_CHECK(cudaEventRecord(start));
	cpu_process(host_input, host_output, host_weights);
	CUDA_ERROR_CHECK(cudaEventRecord(stop));
	CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
	CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));
	std::cout << "Elapsed time for cpu_process(): " << std::to_string(ms) << " ms" << std::endl;

  // compare results of both outputs
	CUDA_ERROR_CHECK(cudaEventRecord(start));
  	compare(host_output, dev_output);
	CUDA_ERROR_CHECK(cudaEventRecord(stop));
	CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
	CUDA_ERROR_CHECK(cudaEventElapsedTime(&ms, start, stop));
	std::cout << "Elapsed time for compare(): " << std::to_string(ms) << " ms" << std::endl;

}


template <typename T>
void PowerBeamformTester::compare(
  const thrust::host_vector<T> cpu,
  const thrust::device_vector<T> gpu,
  const float tol)
{
  // Check if vectors have the same size
  ASSERT_TRUE(cpu.size() == gpu.size())
    << "Host and device vector size not equal" << std::endl;

  // Copy CUDA results to host
	thrust::host_vector<T> host_gpu_result = gpu;
	float abs_diff;
	int arg_max_deviation = 0;

	float gpu_element;
	float cpu_element;
	float max_deviation = .0;

  for(std::size_t i = 0; i < gpu.size(); i++)
  {
			if constexpr (std::is_same<T, __half>::value)
			{
				gpu_element = __half2float(host_gpu_result[i]);
				cpu_element = __half2float(cpu[i]);
			}
			else
			{
				gpu_element = host_gpu_result[i];
				cpu_element = cpu[i];
			}

			abs_diff = std::abs(cpu_element - gpu_element);
			if(abs_diff > max_deviation)
			{
					arg_max_deviation = i;
					max_deviation = abs_diff;
			}

      ASSERT_TRUE(std::abs(cpu_element - gpu_element) <= std::abs(gpu_element)*tol/2)
      	<< "Beamformer with Stokes I: CPU and GPU results are unequal for element " << std::to_string(i) << std::endl
        << "  CPU result: " << std::to_string(cpu_element) << std::endl
        << "  GPU result: " << std::to_string(gpu_element) << std::endl;
  }
	std::cout << "Maximum deviation detected for element " << std::to_string(arg_max_deviation) << std::endl
		<< "Deviation abs(cpu[" << std::to_string(arg_max_deviation)
		<< "] - gpu[" << std::to_string(arg_max_deviation) << "]) = "
		<< std::to_string(max_deviation) << std::endl << std::endl;
}


template<typename T, typename U>
void PowerBeamformTester::gpu_process(
  thrust::device_vector<T>& in,
  thrust::device_vector<U>& out,
  thrust::device_vector<T>& weights)
{
	std::cout << "Calculating GPU results... " << std::endl;
  PowerBeamformer<T, U> bf(&_conf, id);
  bf.process(in, out, weights);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}


template<typename T, typename U>
void PowerBeamformTester::cpu_process(
  thrust::host_vector<T>& in,
  thrust::host_vector<U>& out,
  thrust::host_vector<T>& weights)
{
	std::cout << "Calculating CPU results... " << std::flush;
	int ap = _conf.n_antenna * _conf.n_pol;
	int fap = _conf.n_channel * ap;
	int out_idx, in_idx, weight_idx;
	float2 voltage = {0,0};
	float power;
	for(int t = 0; t < _conf.n_samples; t+=_conf.interval)
	{
		for(int b = 0; b < _conf.n_beam; b++)
		{
			for(int f = 0; f < _conf.n_channel; f++)
			{
				out_idx = b * _conf.n_channel * _conf.n_samples/_conf.interval + f * _conf.n_samples/_conf.interval + t/_conf.interval;
				power = .0;
				for(int i = 0; i < _conf.interval; i++)
				{
					for(int p = 0; p < _conf.n_pol; p++)
					{
						for(int a = 0; a < _conf.n_antenna; a++)
						{
							in_idx = (t+i) * fap + f * ap + a * _conf.n_pol + p;
							weight_idx = b * fap + f * ap + a * _conf.n_pol + p;

							if constexpr (std::is_same<T, float2>::value)
							{
								voltage = cmadd(in[in_idx], weights[weight_idx], voltage);
							}
							else if constexpr (std::is_same<T, __half2>::value)
							{
								voltage = cmadd(__half22float2(in[in_idx]), __half22float2(weights[weight_idx]), voltage);
							}
						}
						power += (voltage.x * voltage.x + voltage.y * voltage.y) / _conf.interval;
						voltage = {0,0};
					}
				}
				// printf("CPU: out[%d][%d][%d] = %f\n", b, f, t/_conf.interval, power/ _conf.interval);
				if constexpr (std::is_same<T, float2>::value)
					out[out_idx] = power;
				else if constexpr (std::is_same<T, __half2>::value)
					out[out_idx] = __float2half(power);
			}
		}
	}
	std::cout << "done" << std::endl;
}



/**
* Testing with Google Test Framework
*/

TEST_P(PowerBeamformTester, BeamformerPowerSingle){
  std::cout << std::endl
    << "-------------------------------------------------------------" << std::endl
    << " Testing power mode with T=float2 (single precision 2x32bit) " << std::endl
    << "-------------------------------------------------------------" << std::endl << std::endl;
  test<float2, float>();
}


// TEST_P(PowerBeamformTester, BeamformerPowerHalf)
// {
//   std::cout << std::endl
//     << "-------------------------------------------------------------" << std::endl
//     << " Testing power mode with T=__half2 (half precision 2x16bit)  " << std::endl
//     << "-------------------------------------------------------------" << std::endl << std::endl;
//   test<__half2, __half>();
// }



INSTANTIATE_TEST_CASE_P(BeamformerTesterInstantiation, PowerBeamformTester, ::testing::Values(
	// samples | channels | antenna | polarisation | beam | interval/integration | beamformer type

	// bf_config_t{1024, 64, 64, 2, 32, 1, SIMPLE_BF_TAFPT},
  // bf_config_t{1024, 1, 1024, 2, 1, 1, SIMPLE_BF_TAFPT},
	// bf_config_t{2048, 1, 16, 2, 4, 1, SIMPLE_BF_TAFPT},
	bf_config_t{1024, 32, 512, 2, 64, 32, BF_TFAP},
	bf_config_t{24, 1, 512, 2, 32, 8, BF_TFAP},
	bf_config_t{256, 8, 512, 2, 64, 64, BF_TFAP},
	bf_config_t{4096, 32, 512, 2, 1024, 64, BF_TFAP},
	bf_config_t{128, 256, 512, 2, 64, 64, BF_TFAP},
	bf_config_t{1024, 32, 512, 2, 64, 32, BF_TFAP},
	bf_config_t{24, 1, 512, 2, 32, 8, BF_TFAP_TEX},
	bf_config_t{256, 8, 512, 2, 64, 64, BF_TFAP_TEX},
	bf_config_t{4096, 32, 512, 2, 1024, 64, BF_TFAP_TEX},
	bf_config_t{128, 256, 512, 2, 64, 64, BF_TFAP_TEX}
	// bf_config_t{N_SAMPLES, N_CHANNEL, N_ANTENNA, N_POL, N_BEAM, INTERVAL, BF_TFAP_V2}
  // bf_config_t{1024, 64, 32, 2, 32, 8, BF_TFAP},
  // bf_config_t{1024, 64, 32, 1, 64, 8, BF_TFAP},
  // bf_config_t{1024, 64, 64, 2, 32, 8, BF_TFAP},
	// bf_config_t{4, 4, 4, 1, 4, 8, BF_TFAP},
));


} // namespace psrdada_cpp
} // namespace cryopaf
} // namespace beamforming
} // namespace test

#endif
